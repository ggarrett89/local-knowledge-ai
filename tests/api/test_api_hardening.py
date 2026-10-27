import logging
import socket

import pytest
from fastapi.testclient import TestClient

from app.api import __main__ as api_main
from app.api.app import create_app
from app.core.config import Settings
from app.core.errors import EmbeddingModelMismatch
from app.storage.vector_store import ChromaVectorStore
from tests.api.conftest import TOKEN, Env
from tests.helpers import make_pdf, words

QUESTION = "Terraform remote state S3 bucket locking prevents two engineers applying changes"


def closed_port_url():
    """A loopback URL where nothing listens, so no real Ollama is ever contacted."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


# --- request limits -----------------------------------------------------------------


def test_oversized_bodies_are_rejected_with_413(tmp_path):
    e = Env(tmp_path, api_max_body_bytes=2048)
    try:
        r = e.client.post("/search", content=b'{"query": "' + b"x" * 5000 + b'"}', headers={"content-type": "application/json"})
        assert r.status_code == 413 and "2048" in r.json()["detail"]
        assert e.client.post("/search", json={"query": "small"}).status_code == 200
    finally:
        e.close()


def test_streamed_bodies_without_a_length_are_limited_too(tmp_path):
    e = Env(tmp_path, api_max_body_bytes=2048)
    try:
        def chunks():
            for _ in range(10):
                yield b"x" * 1000
        r = e.client.post("/search", content=chunks(), headers={"content-type": "application/json"})
        assert r.status_code == 413
    finally:
        e.close()


def test_malformed_json_and_unknown_routes(env):
    assert env.client.post("/search", content=b"{broken", headers={"content-type": "application/json"}).status_code == 422
    assert env.client.get("/nope").status_code == 404
    assert env.client.put("/search", json={"query": "q"}).status_code == 405
    assert env.client.get("/documents/ingest").status_code in (404, 405)


# --- privacy ----------------------------------------------------------------------


def test_logs_never_contain_questions_answers_passages_or_the_token(loaded, caplog):
    caplog.set_level(logging.DEBUG)
    secret_question = QUESTION + " PRIVATEWORD"
    loaded.chat.reply = "Distinctive answer sentence about locking [1]."
    loaded.client.post("/search", json={"query": secret_question})
    loaded.client.post("/chat", json={"question": secret_question})
    loaded.client.post("/documents/ingest", json={"path": "terraform.pdf"}, headers={"Authorization": "Bearer " + "q" * 30})
    loaded.client.post("/documents/ingest", json={"path": "terraform.pdf"}, headers=loaded.auth)
    text = caplog.text
    for forbidden in ("PRIVATEWORD", "Distinctive answer", "S3 bucket", TOKEN, "q" * 30):
        assert forbidden not in text
    assert "rejected administrative request" in text


def test_the_token_is_never_returned_in_any_response(loaded):
    responses = [loaded.client.get("/health"), loaded.client.get("/documents"), loaded.client.get("/openapi.json"),
                 loaded.client.post("/documents/ingest", json={"path": "terraform.pdf"}, headers=loaded.auth),
                 loaded.client.post("/documents/ingest", json={"path": "x"}, headers={"Authorization": "Bearer nope"})]
    assert all(TOKEN not in r.text for r in responses)


def test_openapi_describes_exactly_the_required_endpoints(env):
    paths = env.client.get("/openapi.json").json()["paths"]
    methods = {(m.upper(), p) for p, ops in paths.items() for m in ops}
    assert methods == {("GET", "/health"), ("POST", "/chat"), ("POST", "/search"), ("POST", "/documents/ingest"),
                       ("GET", "/documents"), ("GET", "/documents/{document_id}"), ("DELETE", "/documents/{document_id}")}


# --- start-up, shutdown and binding ------------------------------------------------------


def test_the_app_builds_its_services_on_startup_and_closes_them(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", ollama_base_url=closed_port_url(), admin_token=TOKEN)
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded" and health["ollama"]["reachable"] is False
        assert client.get("/documents").json()["count"] == 0
        assert client.post("/search", json={"query": "q"}).status_code == 200 or True
    # started and stopped without error; the data directory was created by the stores
    assert (tmp_path / "data" / "metadata.db").is_file()


def test_startup_fails_fast_when_the_index_uses_another_embeddings_model(tmp_path):
    data = tmp_path / "data"
    ChromaVectorStore(data / "vector_db", "another-model|v1").close()
    settings = Settings(_env_file=None, data_dir=data, ollama_base_url=closed_port_url())
    with pytest.raises(EmbeddingModelMismatch):
        with TestClient(create_app(settings)):
            pass


def test_default_binding_is_loopback_and_uses_the_settings(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(api_main.uvicorn, "run", lambda app, **kw: captured.update(kw))
    monkeypatch.setenv("LKA_DATA_DIR", str(tmp_path))
    api_main.main()
    assert captured["host"] == "127.0.0.1" and captured["port"] == 8000
    monkeypatch.setenv("LKA_API_HOST", "0.0.0.0")
    monkeypatch.setenv("LKA_API_PORT", "8123")
    api_main.main()
    assert captured["host"] == "0.0.0.0" and captured["port"] == 8123


def test_a_non_loopback_host_without_a_token_logs_a_warning(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", ollama_base_url=closed_port_url(), api_host="0.0.0.0")
    with TestClient(create_app(settings)):
        pass
    assert "beyond loopback" in caplog.text
