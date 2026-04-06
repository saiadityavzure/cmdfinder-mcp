"""
embedder.py
-----------
Embedding abstraction layer.

Supports:
  - sentence_transformers (local, no API key)
  - openai (OpenAI API)
  - ollama (local Ollama server)

All providers return numpy float32 arrays of shape (N, D).
Models are lazy-loaded and cached at module level.
"""

import numpy as np
from config import settings

# ── Module-level model cache (lazy-loaded once) ────────────────────────────────
_st_model = None       # SentenceTransformer instance
_openai_client = None  # openai.OpenAI instance


def _embed_sentence_transformers(texts: list[str]) -> np.ndarray:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading sentence-transformers model: {settings.embedding_model}")
        _st_model = SentenceTransformer(settings.embedding_model)
        print("[embedder] Model loaded.")
    vectors = _st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32)


def _embed_openai(texts: list[str]) -> np.ndarray:
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.openai_api_key)

    # OpenAI allows up to 2048 inputs per call — batch if needed
    all_vectors = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _openai_client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        all_vectors.extend([d.embedding for d in resp.data])
    return np.array(all_vectors, dtype=np.float32)


def _embed_ollama(texts: list[str]) -> np.ndarray:
    """
    Calls Ollama's /api/embed endpoint.
    Ollama >= 0.1.27 supports batched input; older versions need one call per text.
    Falls back to per-text calls if batched request fails.
    """
    import httpx

    base_url = settings.ollama_base_url.rstrip("/")
    model = settings.ollama_embedding_model

    with httpx.Client(timeout=60) as client:
        # Try batched first (Ollama >= 0.1.27)
        try:
            resp = client.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" in data:
                return np.array(data["embeddings"], dtype=np.float32)
        except Exception:
            pass

        # Fallback: one call per text (older Ollama, /api/embeddings)
        vectors = []
        for text in texts:
            resp = client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return np.array(vectors, dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using the configured provider.
    Returns shape (N, D) float32 numpy array.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    provider = settings.embedding_provider
    if provider == "sentence_transformers":
        return _embed_sentence_transformers(texts)
    elif provider == "openai":
        return _embed_openai(texts)
    elif provider == "ollama":
        return _embed_ollama(texts)
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}")


def embed_one(text: str) -> np.ndarray:
    """
    Embed a single string.
    Returns shape (D,) float32 numpy array.
    """
    return embed([text])[0]
