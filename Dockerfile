FROM python:3.11-slim

WORKDIR /app

# Install system deps for faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py embedder.py indexer.py server.py ./

# faiss_index will be mounted or auto-built on first run
VOLUME ["/app/faiss_index"]

EXPOSE 9008

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=9008

CMD ["python", "server.py"]
