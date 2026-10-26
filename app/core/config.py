"""Application settings, loaded from LKA_* environment variables and an optional .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration.

    Data paths other than ``data_dir`` are derived from it when not set explicitly.
    No value here triggers any download, network call or file creation.
    """

    model_config = SettingsConfigDict(
        env_prefix="LKA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data locations
    data_dir: Path = Path("data")
    documents_dir: Optional[Path] = None
    vector_db_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    metadata_db_path: Optional[Path] = None
    categories_file: Path = _PROJECT_ROOT / "config" / "categories.json"

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = Field(default="qwen2.5:3b", min_length=1)
    embedding_model: str = Field(default="nomic-embed-text", min_length=1)
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    # nomic-embed-text expects task prefixes. Part of the index identity: changing them
    # (or the model) requires reindexing. Use empty strings for models without prefixes.
    embedding_document_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "

    # Ingestion (sizes are in characters)
    chunk_size: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)
    embed_batch_size: int = Field(default=16, ge=1, le=256)

    # API
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    admin_token: Optional[SecretStr] = None

    @model_validator(mode="after")
    def _derive_data_paths(self) -> "Settings":
        if self.documents_dir is None:
            self.documents_dir = self.data_dir / "documents"
        if self.vector_db_dir is None:
            self.vector_db_dir = self.data_dir / "vector_db"
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        if self.metadata_db_path is None:
            self.metadata_db_path = self.data_dir / "metadata.db"
        return self

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("chunk_overlap must be smaller than half of chunk_size")
        return self


def load_settings() -> Settings:
    """Build a fresh Settings instance (no hidden global state)."""
    return Settings()
