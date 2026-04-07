"""
indexer.py
----------
FAISS index lifecycle: build, save, load, search.

Index is built once from DOCS_URL, persisted to FAISS_INDEX_DIR.
Search uses cosine similarity (IndexFlatIP + L2 normalization).

Metadata is stored alongside the index to detect embedding
provider/model changes that require a full rebuild.
"""

import os
import json
import re
import numpy as np
import faiss

from config import settings
from embedder import embed, embed_one

_INDEX_FILE    = lambda: os.path.join(settings.faiss_index_dir, "index.faiss")
_METADATA_FILE = lambda: os.path.join(settings.faiss_index_dir, "metadata.json")

# ── In-memory state (populated by load_index or build_index) ──────────────────
_index:  faiss.Index | None = None
_chunks: list[str] = []

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "what", "when",
    "where", "which", "with", "show", "all",
}

_NOISE_PATTERNS = [
    "skip to content",
    "skip to search",
    "skip to footer",
    "bias-free language",
    "was this document helpful",
    "open a support case",
    "book table of contents",
    "download options",
    "available languages",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def index_exists() -> bool:
    return os.path.exists(_INDEX_FILE()) and os.path.exists(_METADATA_FILE())


def _scrape_and_chunk(url: str) -> list[str]:
    """Scrape URL with BeautifulSoup and split into overlapping text chunks."""
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": settings.user_agent}
    print(f"[indexer] Fetching: {url}")
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove nav/header/footer noise
    for tag in soup(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    # Recursive character splitter (inline — no LangChain needed)
    return _recursive_split(text, settings.chunk_size, settings.chunk_overlap)


def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    """Split text recursively on paragraph → newline → space boundaries."""
    separators = ["\n\n", "\n", " ", ""]
    chunks = []
    _split(text, size, overlap, separators, chunks)
    return [c.strip() for c in chunks if c.strip() and not _is_noisy_chunk(c)]


def _is_noisy_chunk(chunk: str) -> bool:
    """Drop chunks dominated by site chrome/navigation text."""
    lowered = chunk.lower()
    noise_hits = sum(1 for p in _NOISE_PATTERNS if p in lowered)
    if noise_hits >= 2:
        return True
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    short_lines = sum(1 for ln in lines if len(ln) <= 18)
    if short_lines / len(lines) > 0.7:
        return True
    return False


def _query_terms(query: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-zA-Z0-9_-]+", query.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


def _lexical_score(query: str, chunk: str) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    lowered = chunk.lower()
    matches = sum(1 for term in terms if term in lowered)
    return matches / len(terms)


def _command_bonus(chunk: str) -> float:
    """Boost chunks that appear to contain concrete NX-OS CLI syntax."""
    lowered = chunk.lower()
    bonus = 0.0
    if re.search(r"(?m)^\s*show\s+[a-z0-9-]+", lowered):
        bonus += 0.08
    if "show bgp" in lowered:
        bonus += 0.1
    if "neighbor" in lowered or "neigh" in lowered:
        bonus += 0.05
    if "uptime" in lowered:
        bonus += 0.03
    return bonus


def _noise_penalty(chunk: str) -> float:
    lowered = chunk.lower()
    hits = sum(1 for p in _NOISE_PATTERNS if p in lowered)
    return min(0.2, hits * 0.04)


def _split(text: str, size: int, overlap: int, separators: list[str], out: list[str]):
    if len(text) <= size:
        out.append(text)
        return
    sep = separators[0] if separators else ""
    parts = text.split(sep) if sep else list(text)
    current = ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                out.append(current)
            # Start next chunk with overlap from the end of current
            overlap_text = current[-overlap:] if overlap and current else ""
            current = overlap_text + (sep if overlap_text else "") + part
            if len(current) > size and len(separators) > 1:
                # Recurse with next separator
                _split(current, size, overlap, separators[1:], out)
                current = ""
    if current:
        out.append(current)


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build_index() -> int:
    """
    Scrape DOCS_URL, embed chunks, build FAISS index, persist to disk.
    Returns the number of chunks indexed.
    """
    global _index, _chunks

    chunks = _scrape_and_chunk(settings.docs_url)
    if not chunks:
        raise RuntimeError("No content extracted from DOCS_URL")

    print(f"[indexer] Embedding {len(chunks)} chunks with "
          f"{settings.embedding_provider}/{settings.embedding_model} ...")
    vectors = embed(chunks)

    # L2-normalize for cosine similarity via IndexFlatIP
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    os.makedirs(settings.faiss_index_dir, exist_ok=True)
    faiss.write_index(index, _INDEX_FILE())
    with open(_METADATA_FILE(), "w") as f:
        json.dump({
            "docs_url":           settings.docs_url,
            "embedding_provider": settings.embedding_provider,
            "embedding_model":    settings.embedding_model,
            "dim":                dim,
            "total_chunks":       len(chunks),
            "chunk_size":         settings.chunk_size,
            "chunk_overlap":      settings.chunk_overlap,
        }, f, indent=2)
    with open(os.path.join(settings.faiss_index_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    _index  = index
    _chunks = chunks
    print(f"[indexer] Index built: {len(chunks)} chunks, dim={dim}")
    return len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_index() -> bool:
    """
    Load persisted FAISS index into memory.
    Returns False and logs a warning if provider/model mismatch detected
    (caller should rebuild).
    """
    global _index, _chunks

    with open(_METADATA_FILE()) as f:
        meta = json.load(f)

    # Detect embedding provider/model change — requires rebuild
    if (meta.get("embedding_provider") != settings.embedding_provider or
            meta.get("embedding_model") != settings.embedding_model):
        print(
            f"[indexer] WARNING: Embedding mismatch. "
            f"Index was built with {meta['embedding_provider']}/{meta['embedding_model']}, "
            f"but config says {settings.embedding_provider}/{settings.embedding_model}. "
            f"Rebuilding..."
        )
        return False

    _index = faiss.read_index(_INDEX_FILE())
    with open(os.path.join(settings.faiss_index_dir, "chunks.json")) as f:
        _chunks = json.load(f)

    print(f"[indexer] Index loaded: {len(_chunks)} chunks, dim={meta['dim']}")
    return True


def get_index_meta() -> dict:
    """Return metadata from the persisted index, or empty dict if not built."""
    if not index_exists():
        return {}
    with open(_METADATA_FILE()) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

def search(query: str, k: int | None = None) -> list[dict]:
    """
    Semantic search over the FAISS index.

    Returns list of dicts: [{"text": str, "score": float}, ...]
    Scores are cosine similarities in [0, 1] (higher = more relevant).
    """
    if _index is None:
        raise RuntimeError("Index is not loaded. Call load_index() or build_index() first.")

    k = k or settings.faiss_top_k
    # Oversample, then rerank so final top-k are more command-bearing.
    candidate_k = min(len(_chunks), max(k * 4, 12))
    query_vec = embed_one(query).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)

    distances, indices = _index.search(query_vec, candidate_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx >= 0:
            chunk = _chunks[idx]
            semantic = float(score)
            lexical = _lexical_score(query, chunk)
            hybrid = (semantic * 0.75) + (lexical * 0.25) + _command_bonus(chunk) - _noise_penalty(chunk)
            results.append({
                "text":  chunk,
                "score": round(max(0.0, hybrid), 4),
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]
