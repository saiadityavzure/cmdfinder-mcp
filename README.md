# CMDFinder MCP

A semantic CLI command finder MCP server for Cisco NX-OS documentation.

Uses FAISS + OpenAI embeddings to semantically index documentation and expose it as MCP tools for AI agents (Claude Desktop, etc.).

---

## How It Works

```
DOCS_URL (Cisco docs page)
        │
        ▼
Scrape + Chunk (BeautifulSoup)
        │
        ▼
Embed (OpenAI text-embedding-3-small)
        │
        ▼
FAISS Index (persisted to disk)
        │
        ▼
FastMCP SSE Server → Claude Desktop
```

On first startup the server scrapes the configured docs URL, embeds all chunks, and saves the FAISS index to `faiss_index/`. Every subsequent start loads from disk instantly (~1 second).

---

## Project Structure

```
CMDFinder_MCP/
├── server.py          # FastMCP SSE entry point
├── config.py          # All settings loaded from .env
├── embedder.py        # Embedding abstraction (openai / sentence_transformers / ollama)
├── indexer.py         # FAISS build, load, search
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
├── .env               # Your local secrets (never commit)
├── .env.example.local # Template — copy to .env and fill in
├── faiss_index/       # Auto-generated on first run (gitignored)
└── claude_desktop_config.json  # Claude Desktop SSE connection config
```

---

## Prerequisites

- Python 3.11+
- Docker (for containerized run)
- OpenAI API key

---

## Setup

### 1. Clone and configure

```bash
cd CMDFinder_MCP
cp .env.example.local .env
```

Edit `.env` and fill in your values:

```env
DOCS_URL=https://your-cisco-docs-url.com
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-proj-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_LLM_API_KEY=sk-proj-...
```

> Keys in `.env` must have **no quotes** around values — Docker passes them literally.

---

## Option A: Run Locally (Python)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the server

```bash
python server.py
```

On first run this will:
1. Scrape `DOCS_URL`
2. Embed all chunks via OpenAI
3. Build and save the FAISS index to `faiss_index/`
4. Start the SSE server

```
==================================================
  CMDFinder MCP  —  SSE Server
  SSE URL   : http://0.0.0.0:9008/sse
  Tools     : find_command, get_index_info, refresh_index
==================================================
```

---

## Option B: Run with Docker Compose (recommended)

### 2. Start the server

```bash
docker compose up -d
```

This will:
- Build the image if not already built
- Start the container with port `9008` exposed
- Bind-mount `./faiss_index` and `./logs` so index and tool logs are visible on host and in container
- Auto-restart the container unless manually stopped

### Useful Docker Compose commands

```bash
# View live logs
docker compose logs -f

# Stop the server
docker compose down

# Rebuild image after code changes
docker compose up -d --build

# Rebuild the FAISS index (delete volume + restart)
docker compose down -v
docker compose up -d
```

---

## Option C: Run with Docker (manual)

### 2. Build the image

```bash
docker build -t cmdfinder-mcp .
```

### 3. Run the container

```bash
docker run -d \
  --name cmdfinder-mcp \
  -p 9008:9008 \
  --env-file .env \
  -v $(pwd)/faiss_index:/app/faiss_index \
  cmdfinder-mcp
```

> The `-v` mount persists the FAISS index so it survives container restarts and is not rebuilt every time.

### Useful Docker commands

```bash
# View logs
docker logs -f cmdfinder-mcp

# Stop and remove
docker rm -f cmdfinder-mcp
```

---

## Connect Claude Desktop

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cmdfinder-mcp": {
      "url": "http://localhost:9008/sse"
    }
  }
}
```

Restart Claude Desktop. The server will appear as a connected MCP tool.

---

## Available Tools

| Tool | Description |
|---|---|
| `find_command(query)` | Semantic search — returns top-k relevant doc chunks for a natural language query |
| `get_index_info()` | Returns current index stats and config (embedding provider, model, chunk count) |
| `refresh_index()` | [Planned] Rebuild index from live docs without restarting |

### Example queries

- `"show BGP neighbor status"`
- `"display all VLANs configured on the switch"`
- `"check interface errors and packet drops"`
- `"view OSPF routing table"`

---

## Embedding Providers

Switch provider by changing `EMBEDDING_PROVIDER` in `.env`:

| Provider | `EMBEDDING_PROVIDER` | Required env vars |
|---|---|---|
| OpenAI (default) | `openai` | `OPENAI_API_KEY`, `EMBEDDING_MODEL` |
| Local (no API key) | `sentence_transformers` | `EMBEDDING_MODEL` (e.g. `all-MiniLM-L6-v2`) |
| Ollama (local LLM) | `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL` |

> If you change the embedding provider or model, delete `faiss_index/` and restart — the server will detect the mismatch and rebuild automatically.

---

## Rebuild the Index

If the docs change or you switch embedding models:

```bash
# Delete the index and restart — server rebuilds automatically
rm -rf faiss_index/
python server.py
```

Or with Docker:

```bash
rm -rf faiss_index/
docker restart cmdfinder-mcp
```
