import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def test_defaults_are_safe_and_local():
    s = Settings(_env_file=None)
    assert s.api_host == "127.0.0.1"
    assert s.api_port == 8000
    assert s.ollama_base_url == "http://127.0.0.1:11434"
    assert s.chat_model == "qwen2.5:3b"
    assert s.embedding_model == "nomic-embed-text"
    assert s.admin_token is None


def test_data_paths_derive_from_data_dir():
    s = Settings(_env_file=None, data_dir=Path("/tmp/lka"))
    assert s.documents_dir == Path("/tmp/lka/documents")
    assert s.vector_db_dir == Path("/tmp/lka/vector_db")
    assert s.cache_dir == Path("/tmp/lka/cache")
    assert s.metadata_db_path == Path("/tmp/lka/metadata.db")


def test_env_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("LKA_DATA_DIR", "/srv/lka")
    monkeypatch.setenv("LKA_VECTOR_DB_DIR", "/mnt/fast/vectors")
    monkeypatch.setenv("LKA_CHAT_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("LKA_API_PORT", "9001")
    s = load_settings()
    assert s.documents_dir == Path("/srv/lka/documents")
    assert s.vector_db_dir == Path("/mnt/fast/vectors")
    assert s.chat_model == "qwen2.5:1.5b"
    assert s.api_port == 9001


def test_env_file_is_read(tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text("LKA_EMBEDDING_MODEL=other-embed\n", encoding="utf-8")
    s = Settings(_env_file=env_file)
    assert s.embedding_model == "other-embed"


@pytest.mark.parametrize("name,value", [
    ("LKA_API_PORT", "0"),
    ("LKA_API_PORT", "70000"),
    ("LKA_OLLAMA_TIMEOUT_SECONDS", "0"),
    ("LKA_CHAT_MODEL", ""),
])
def test_invalid_values_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_admin_token_is_not_exposed_in_repr(monkeypatch):
    monkeypatch.setenv("LKA_ADMIN_TOKEN", "super-secret-value")
    s = Settings(_env_file=None)
    assert "super-secret-value" not in repr(s)
    assert s.admin_token.get_secret_value() == "super-secret-value"


def test_settings_creation_does_not_touch_filesystem(tmp_path):
    Settings(_env_file=None, data_dir=tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_ingestion_defaults():
    s = Settings(_env_file=None)
    assert (s.chunk_size, s.chunk_overlap, s.embed_batch_size) == (1200, 200, 16)
    assert s.embedding_document_prefix == "search_document: "
    assert s.embedding_query_prefix == "search_query: "
    assert s.categories_file.is_file()


def test_ingestion_settings_from_env(monkeypatch):
    monkeypatch.setenv("LKA_CHUNK_SIZE", "800")
    monkeypatch.setenv("LKA_CHUNK_OVERLAP", "100")
    monkeypatch.setenv("LKA_EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("LKA_EMBEDDING_DOCUMENT_PREFIX", "")
    s = load_settings()
    assert (s.chunk_size, s.chunk_overlap, s.embed_batch_size) == (800, 100, 4)
    assert s.embedding_document_prefix == ""


@pytest.mark.parametrize("name,value", [
    ("LKA_CHUNK_SIZE", "50"),
    ("LKA_EMBED_BATCH_SIZE", "0"),
    ("LKA_CHUNK_OVERLAP", "-1"),
    ("LKA_CHUNK_OVERLAP", "600"),  # not smaller than half of the default chunk size
])
def test_invalid_ingestion_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_categories_file_is_valid():
    data = json.loads((REPO_ROOT / "config" / "categories.json").read_text(encoding="utf-8"))
    slugs = [c["slug"] for c in data["categories"]]
    assert len(slugs) == len(set(slugs))
    assert all(SLUG_RE.match(s) for s in slugs)
    assert {"programming", "software-architecture", "distributed-systems", "devops-sre",
            "cloud", "containers-kubernetes", "iac-automation", "databases-messaging",
            "security", "ai-ml", "general"} == set(slugs)


def test_retrieval_defaults():
    s = Settings(_env_file=None)
    assert (s.retrieval_k, s.retrieval_max_k, s.retrieval_overfetch) == (5, 50, 4)
    assert s.dedupe_adjacent is True and s.max_query_chars == 1000
    assert s.relevance_max_distance == 0.45


def test_retrieval_settings_from_env(monkeypatch):
    monkeypatch.setenv("LKA_RETRIEVAL_K", "8")
    monkeypatch.setenv("LKA_RELEVANCE_MAX_DISTANCE", "0.5")
    monkeypatch.setenv("LKA_DEDUPE_ADJACENT", "false")
    s = load_settings()
    assert (s.retrieval_k, s.relevance_max_distance, s.dedupe_adjacent) == (8, 0.5, False)


@pytest.mark.parametrize("name,value", [
    ("LKA_RETRIEVAL_K", "0"),
    ("LKA_RETRIEVAL_K", "51"),          # above the default max_k of 50
    ("LKA_RELEVANCE_MAX_DISTANCE", "0"),
    ("LKA_RELEVANCE_MAX_DISTANCE", "2.5"),
    ("LKA_RETRIEVAL_OVERFETCH", "0"),
    ("LKA_MAX_QUERY_CHARS", "5"),
])
def test_invalid_retrieval_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
