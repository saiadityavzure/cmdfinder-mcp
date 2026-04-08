"""
build_registry.py — standalone script to build/refresh command_registry.json

Run this BEFORE starting the server (or when docs change) to produce the
configurable command registry that the FAISS index is built from.

Usage:
    python build_registry.py

After it finishes, edit command_registry/command_registry.json if needed:
  - Set "enabled": false  to exclude commands from the index
  - Edit/add "tags"       for better semantic retrieval
  - Add entries manually  for commands not covered by the scraped docs

Then start the server (python server.py) — it will detect the registry and
build the FAISS index from it automatically.
"""

import os
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "cmdfinder-mcp/2.0")

from config import settings
settings.validate()

from registry_builder import build_registry, load_registry, chunks_from_registry

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  CMDFinder MCP — Registry Builder")
    print(f"  Docs URL : {settings.docs_url}")
    print(f"  Max pages: {settings.docs_max_pages}")
    print(f"  Output   : {settings.command_registry_dir}/command_registry.json")
    print(f"{'='*55}\n")

    count = build_registry()

    print(f"\n{'='*55}")
    print(f"  Done. {count} command entries written.")
    print(f"\n  Next steps:")
    print(f"  1. Review/edit: {settings.command_registry_dir}/command_registry.json")
    print(f"     - Set \"enabled\": false to exclude commands from the index")
    print(f"     - Edit \"tags\" for better retrieval")
    print(f"  2. Run: python server.py  (builds FAISS index from registry)")
    print(f"{'='*55}\n")
