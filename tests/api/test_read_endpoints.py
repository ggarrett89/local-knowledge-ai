import pytest

from app.core.errors import EmbeddingModelMismatch, GenerationError, ModelNotInstalled, OllamaUnavailable
from tests.helpers import FakeCatalog, FakeChat
from tests.api.conftest import TOKEN, Env

QUESTION = "Terraform remote state S3 bucket locking prevents two engineers applying changes"


# --- health ---------------------------------------------------------------------


def test_health_reports_ok_with_models_and_index_counts(loaded):
    r = loaded.client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["admin_endpoints_enabled"] is True
    assert body["ollama"] == {"reachable": True, "embedding_model": "nomic-embed-text", "embedding_model_installed": True,
                              "chat_model": "fake-chat", "chat_model_installed": True}


def test_health_is_degraded_when_a_model_is_missing(env):
    env.catalog.installed = ["nomic-embed-text:latest"]
    body = env.client.get("/health").json()
    assert body["status"] == "degraded" and body["ollama"]["reachable"] is True
    assert body["ollama"]["embedding_model_installed"] is True and body["ollama"]["chat_model_installed"] is False


def test_health_is_degraded_but_still_200_when_ollama_is_unreachable(env):
    env.catalog.error = OllamaUnavailable("cannot list models")
    r = env.client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["ollama"]["reachable"] is False


def test_health_counts_the_index_and_never_exposes_the_token(loaded):
    r = loaded.client.get("/health")
    body = r.json()
    assert body["status"] == "ok" and body["index"]["documents"] == 2 and body["index"]["chunks"] > 2
    assert TOKEN not in r.text


def test_health_says_when_admin_endpoints_are_disabled(tmp_path):
    e = Env(tmp_path, token=None)
    try:
        assert e.client.get("/health").json()["admin_endpoints_enabled"] is False
    finally:
        e.close()


# --- search -----------------------------------------------------------------------


def test_search_returns_ranked_hits_with_real_references(loaded):
    r = loaded.client.post("/search", json={"query": QUESTION, "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(body["results"]) <= 3
    top = body["results"][0]
    assert top["rank"] == 1 and top["title"] == "Terraform Handbook" and "S3 bucket" in top["text"]
    assert top["reference"] in ("Terraform Handbook, p. 2", "Terraform Handbook, pp. 1-2")
    assert [h["distance"] for h in body["results"]] == sorted(h["distance"] for h in body["results"])


def test_search_filters_and_overrides(loaded):
    body = loaded.client.post("/search", json={"query": "partitions topic modules",
                                               "filters": {"category": "databases-messaging"}}).json()
    assert body["count"] > 0 and {h["title"] for h in body["results"]} == {"Kafka Handbook"}
    body = loaded.client.post("/search", json={"query": "partitions", "filters": {"topics": ["terraform"]}}).json()
    assert {h["title"] for h in body["results"]} <= {"Terraform Handbook"}
    assert loaded.client.post("/search", json={"query": "chocolate cake sugar", "max_distance": 0.1}).json()["count"] == 0
    assert loaded.client.post("/search", json={"query": "q", "filters": {"category": "none-here"}}).json()["count"] == 0


def test_search_on_an_empty_index_returns_nothing(env):
    r = env.client.post("/search", json={"query": "anything"})
    assert r.status_code == 200 and r.json() == {"count": 0, "results": []}


@pytest.mark.parametrize("payload", [
    {}, {"query": ""}, {"query": "q", "k": 0}, {"query": "q", "k": 101}, {"query": "q", "max_distance": 0},
    {"query": "q", "filters": {"category": "Not Valid!"}}, {"query": "q", "filters": {"topics": ["../etc"]}},
    {"query": "q", "filters": {"source_type": "video"}}, {"query": "x" * 4001},
])
def test_search_rejects_invalid_requests(env, payload):
    assert env.client.post("/search", json=payload).status_code == 422


def test_search_uses_the_configured_question_limit(tmp_path):
    e = Env(tmp_path, max_query_chars=50)
    try:
        assert e.client.post("/search", json={"query": "x" * 51}).status_code == 422
        assert e.client.post("/search", json={"query": "x" * 50}).status_code == 200
    finally:
        e.close()


def test_search_whitespace_question_is_a_422(env):
    assert env.client.post("/search", json={"query": "    "}).status_code == 422


# --- chat -------------------------------------------------------------------------


def test_chat_returns_an_answer_with_verified_sources(loaded):
    r = loaded.client.post("/chat", json={"question": QUESTION, "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["answer"] == "Locking stops concurrent applies [1]."
    assert body["sources"][0]["title"] == "Terraform Handbook" and body["sources"][0]["number"] == 1
    assert body["retrieved"] and body["model"] == "fake-chat"


def test_chat_does_not_pass_invented_references_through(loaded):
    loaded.chat.reply = "Per 'Fake Book' page 999 and https://evil.example/x it works [1]."
    body = loaded.client.post("/chat", json={"question": QUESTION}).json()
    assert "evil.example" not in body["answer"] and "page 999" not in body["answer"]
    assert body["removed_references"] and body["sources"][0]["title"] == "Terraform Handbook"


def test_chat_without_evidence_does_not_call_the_model(env):
    r = env.client.post("/chat", json={"question": "anything at all"})
    assert r.status_code == 200 and r.json()["status"] == "insufficient_evidence" and env.chat.calls == []


def test_chat_filters_apply(loaded):
    body = loaded.client.post("/chat", json={"question": QUESTION, "filters": {"category": "none-here"}}).json()
    assert body["status"] == "insufficient_evidence" and loaded.chat.calls == []


@pytest.mark.parametrize("error,status", [(OllamaUnavailable("cannot reach Ollama at http://127.0.0.1:11434"), 503),
                                          (ModelNotInstalled("model 'qwen2.5:3b' is not installed"), 503),
                                          (GenerationError("Ollama returned HTTP 500: secret detail"), 502)])
def test_chat_model_failures_map_to_generic_http_errors(loaded, error, status):
    loaded.chat.error = error
    r = loaded.client.post("/chat", json={"question": QUESTION})
    assert r.status_code == status
    for leaked in ("127.0.0.1", "11434", "qwen2.5", "secret detail", "Traceback"):
        assert leaked not in r.text


def test_unexpected_errors_are_a_generic_500_without_details(loaded, caplog):
    loaded.chat.error = RuntimeError("the user's private question text is inside this message")
    r = loaded.client.post("/chat", json={"question": QUESTION})
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error."}
    assert "private question" not in r.text and "private question" not in caplog.text


def test_embeddings_mismatch_is_a_409_with_fixed_text(loaded):
    loaded.embedder.error = EmbeddingModelMismatch("index: 'secret-model-name' vs configured: 'x'")
    r = loaded.client.post("/search", json={"query": "anything"})
    assert r.status_code == 409 and "secret-model-name" not in r.text and "reindexing" in r.json()["detail"]


def test_embeddings_service_down_is_a_503_on_search(loaded):
    loaded.embedder.error = OllamaUnavailable("cannot reach Ollama at http://127.0.0.1:11434")
    r = loaded.client.post("/search", json={"query": "anything"})
    assert r.status_code == 503 and "127.0.0.1" not in r.text


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "q", "k": 0}, {"question": "x" * 4001},
                                     {"question": "q", "filters": {"category": "Bad Slug"}}])
def test_chat_rejects_invalid_requests(env, payload):
    assert env.client.post("/chat", json=payload).status_code == 422
    assert env.chat.calls == []


# --- documents (read) ---------------------------------------------------------------


def test_list_and_filter_documents(loaded):
    body = loaded.client.get("/documents").json()
    assert body["count"] == 2 and {d["title"] for d in body["documents"]} == {"Terraform Handbook", "Kafka Handbook"}
    assert loaded.client.get("/documents", params={"category": "iac-automation"}).json()["count"] == 1
    assert loaded.client.get("/documents", params={"topic": "kafka"}).json()["documents"][0]["title"] == "Kafka Handbook"
    assert loaded.client.get("/documents", params={"status": "error"}).json()["count"] == 0


def test_documents_never_expose_absolute_host_paths(loaded):
    text = loaded.client.get("/documents").text
    doc_id = loaded.client.get("/documents").json()["documents"][0]["document_id"]
    text += loaded.client.get(f"/documents/{doc_id}").text
    assert str(loaded.root) not in text and "file://" not in text and "file_path" not in text
    assert "terraform.pdf" in text or "kafka.pdf" in text


def test_document_detail_includes_errors(env):
    (env.docs / "broken.pdf").write_bytes(b"not a pdf")
    assert env.ingest("broken.pdf").json()["results"][0]["outcome"] == "error"
    doc = env.client.get("/documents", params={"status": "error"}).json()["documents"][0]
    detail = env.client.get(f"/documents/{doc['document_id']}").json()
    assert detail["status"] == "error" and detail["errors"][0]["stage"] == "extract"
    assert detail["errors"][0]["message"] and detail["file_name"] == "broken.pdf"


def test_unknown_document_is_a_404(env):
    r = env.client.get("/documents/pdf-unknown")
    assert r.status_code == 404 and r.json() == {"detail": "Document not found."}


@pytest.mark.parametrize("params", [{"category": "Bad Slug"}, {"status": "weird"}, {"topic": "../x"}])
def test_document_list_rejects_invalid_filters(env, params):
    assert env.client.get("/documents", params=params).status_code == 422
