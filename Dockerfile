FROM python:3.11-slim

WORKDIR /app

# Install system deps for faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py embedder.py indexer.py server.py registry_builder.py build_registry.py ./

# faiss_index is mounted or auto-built on first run
# command_registry is mounted so registry edits on the host are visible in-container
VOLUME ["/app/faiss_index"]
VOLUME ["/app/command_registry"]

EXPOSE 9008

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=9008

CMD ["python", "server.py"]
