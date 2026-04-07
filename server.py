"""
server.py — CMDFinder MCP Server
----------------------------------
FastMCP SSE server on port 9008.

Startup:
  - Validates config (.env)
  - Loads FAISS index from disk (or builds it on first run)
  - Registers tools and starts SSE server

Tools:
  - find_command(query)   → clean JSON with semantic search results
  - get_index_info()      → current index + config status
  - refresh_index()       → [PLANNED] rebuilds index from live docs

Usage:
    python server.py
    → SSE URL: http://0.0.0.0:9008/sse
"""

import json
import logging
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "cmdfinder-mcp/2.0")

from config import settings
settings.validate()

from indexer import build_index, load_index, index_exists, search, get_index_meta
from fastmcp import FastMCP


def _build_tool_logger() -> logging.Logger:
    """Create a dedicated logger for MCP tool responses."""
    logger = logging.getLogger("cmdfinder_mcp.tools")
    if logger.handlers:
        return logger

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "tool_responses.log"

    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


TOOL_LOGGER = _build_tool_logger()
MAX_LOG_CHARS = 12000


def _log_tool_response(tool_name: str, response: dict) -> None:
    """Log tool responses as JSON with bounded size."""
    serialized = json.dumps(response, ensure_ascii=True, default=str)
    if len(serialized) > MAX_LOG_CHARS:
        serialized = serialized[:MAX_LOG_CHARS] + "...<truncated>"
    TOOL_LOGGER.info("tool=%s response=%s", tool_name, serialized)


def _extract_command_candidates(results: list[dict]) -> list[str]:
    """Extract likely NX-OS command lines from top search chunks."""
    candidates: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"(?im)^\s*(show\s+[a-z0-9][a-z0-9\s\-./:()]{2,120})$",
        r"(?im)\b(show\s+bgp\s+neighbors?(?:\s+\S+)*)",
    ]

    for item in results:
        text = str(item.get("text", ""))
        for pat in patterns:
            for match in re.findall(pat, text):
                cmd = " ".join(match.strip().split())
                lowered = cmd.lower()
                if lowered in seen:
                    continue
                if len(cmd) < 6 or len(cmd) > 150:
                    continue
                seen.add(lowered)
                candidates.append(cmd)
                if len(candidates) >= 10:
                    return candidates
    return candidates

# ── Load or build FAISS index at startup ──────────────────────────────────────
print(f"\n{'='*50}")
print(f"  CMDFinder MCP")
print(f"  Embedding : {settings.embedding_provider}/{settings.embedding_model}")
print(f"  Docs URL  : {settings.docs_url}")
print(f"{'='*50}")

if not index_exists():
    print("[startup] No index found — building for the first time...")
    build_index()
else:
    ok = load_index()
    if not ok:
        # Embedding provider/model changed — rebuild
        build_index()

# ─────────────────────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="cmdfinder-mcp",
    instructions=(
        "cmdfinder-mcp is a semantic CLI command finder for Cisco NX-OS documentation. "
        "\n\n"
        "When find_command returns results, examine the 'results' array. Each entry has "
        "'text' (the raw documentation chunk) and 'score' (cosine similarity, 0-1). "
        "From the highest-scoring chunks, extract and present: "
        "(1) the exact command syntax, "
        "(2) what the command displays or does, "
        "(3) any important optional parameters or variants. "
        "\n\n"
        "Use get_index_info to confirm which documentation is indexed and which "
        "embedding model is in use. "
        "\n\n"
        "refresh_index is planned but not yet implemented — inform the user if they request it."
    ),
)


@mcp.tool()
async def find_command(query: str, top_k: int = settings.faiss_top_k) -> dict:
    """
    Semantically search the indexed CLI documentation and return the most
    relevant chunks for the given natural language query.

    Args:
        query:  Natural language description. E.g. "show BGP neighbor status"
        top_k:  Number of result chunks to return (default from config).
    """
    results = search(query, k=top_k)
    response = {
        "query":   query,
        "source":  settings.docs_url,
        "top_k":   top_k,
        "results": results,   # list of {"text": ..., "score": ...}
        "command_candidates": _extract_command_candidates(results),
    }
    _log_tool_response("find_command", response)
    return response


@mcp.tool()
async def get_index_info() -> dict:
    """
    Returns the current FAISS index status and server configuration.
    Useful to confirm which documentation is indexed and which
    embedding model and LLM provider are configured.
    """
    meta = get_index_meta()
    response = {
        "index_built":        index_exists(),
        "docs_url":           settings.docs_url,
        "embedding_provider": settings.embedding_provider,
        "embedding_model":    settings.embedding_model,
        "llm_provider":       settings.llm_provider,
        "llm_model":          settings.llm_model,
        "faiss_index_dir":    settings.faiss_index_dir,
        "total_chunks":       meta.get("total_chunks", "unknown"),
        "chunk_size":         settings.chunk_size,
        "chunk_overlap":      settings.chunk_overlap,
        "top_k_default":      settings.faiss_top_k,
    }
    _log_tool_response("get_index_info", response)
    return response


@mcp.tool()
async def refresh_index() -> dict:
    """
    [PLANNED] Rebuild the FAISS index by re-scraping the live documentation URL.
    This allows picking up documentation updates without restarting the server.
    Not yet implemented.
    """
    response = {
        "status": "not_implemented",
        "message": (
            "refresh_index is planned but not yet implemented. "
            "To rebuild the index manually, delete the faiss_index/ directory "
            f"({settings.faiss_index_dir}) and restart the server."
        ),
    }
    _log_tool_response("refresh_index", response)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  CMDFinder MCP  —  SSE Server")
    print(f"  SSE URL   : http://{settings.mcp_host}:{settings.mcp_port}/sse")
    print(f"  Tools     : find_command, get_index_info, refresh_index")
    print(f"{'='*50}\n")
    mcp.run(transport="sse", host=settings.mcp_host, port=settings.mcp_port)
