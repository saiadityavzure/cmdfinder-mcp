"""
config.py
---------
Single source of truth for all settings.
Every other module imports `settings` from here — nothing else calls os.getenv.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── Documentation ──────────────────────────────────────────────────────────
    docs_url: str = field(default_factory=lambda: os.getenv("DOCS_URL", ""))
    docs_max_pages: int = field(
        default_factory=lambda: int(os.getenv("DOCS_MAX_PAGES", "120"))
    )

    # ── Embedding ──────────────────────────────────────────────────────────────
    # Providers: sentence_transformers | openai | ollama
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    # OpenAI embeddings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # Ollama embeddings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_embedding_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    )

    # ── LLM (configured for future use / non-Claude clients) ───────────────────
    # Providers: anthropic | openai | ollama
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    )
    # Anthropic LLM
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    # OpenAI LLM (separate key from embedding)
    openai_llm_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_LLM_API_KEY", "")
    )
    # Ollama LLM
    ollama_llm_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_LLM_BASE_URL", "http://localhost:11434")
    )
    ollama_llm_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
    )

    # ── Command Registry ───────────────────────────────────────────────────────
    command_registry_dir: str = field(
        default_factory=lambda: os.getenv("COMMAND_REGISTRY_DIR", "command_registry")
    )

    # ── FAISS ──────────────────────────────────────────────────────────────────
    faiss_index_dir: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_DIR", "faiss_index")
    )
    faiss_top_k: int = field(
        default_factory=lambda: int(os.getenv("FAISS_TOP_K", "5"))
    )

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1500"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )

    # ── Server ─────────────────────────────────────────────────────────────────
    mcp_host: str = field(default_factory=lambda: os.getenv("MCP_HOST", "0.0.0.0"))
    mcp_port: int = field(default_factory=lambda: int(os.getenv("MCP_PORT", "9008")))
    user_agent: str = field(
        default_factory=lambda: os.getenv("USER_AGENT", "cmdfinder-mcp/2.0")
    )

    def validate(self) -> None:
        """Raise ValueError for any missing required configuration."""
        if not self.docs_url:
            raise ValueError("DOCS_URL is required in .env")
        if self.docs_max_pages < 1:
            raise ValueError("DOCS_MAX_PAGES must be >= 1")

        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        if self.embedding_provider not in ("sentence_transformers", "openai", "ollama"):
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER: {self.embedding_provider!r}. "
                "Choose: sentence_transformers | openai | ollama"
            )
        if self.llm_provider not in ("anthropic", "openai", "ollama"):
            raise ValueError(
                f"Unknown LLM_PROVIDER: {self.llm_provider!r}. "
                "Choose: anthropic | openai | ollama"
            )
        if self.llm_provider == "openai" and not self.openai_llm_api_key:
            raise ValueError(
                "OPENAI_LLM_API_KEY is required when LLM_PROVIDER=openai"
            )


settings = Settings()
