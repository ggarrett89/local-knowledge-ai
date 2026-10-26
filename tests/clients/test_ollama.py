import json

import httpx
import pytest

from app.clients.ollama import OllamaEmbeddingClient
from app.core.errors import EmbeddingError, ModelNotInstalled, OllamaUnavailable


def make_client(handler, **kwargs):
    return OllamaEmbeddingClient(
        "http://127.0.0.1:11434",
        "nomic-embed-text",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def ok_handler(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in body["input"]]})
    return handler


def test_embed_documents_applies_prefix_and_disables_truncation():
    seen = []
    client = make_client(ok_handler(seen), document_prefix="search_document: ", query_prefix="search_query: ")
    vectors = client.embed_documents(["alpha", "beta"])
    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    path, body = seen[0]
    assert path == "/api/embed"
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == ["search_document: alpha", "search_document: beta"]
    assert body["truncate"] is False


def test_embed_query_applies_query_prefix():
    seen = []
    client = make_client(ok_handler(seen), query_prefix="search_query: ")
    assert client.embed_query("how?") == [0.1, 0.2, 0.3]
    assert seen[0][1]["input"] == ["search_query: how?"]


def test_empty_input_makes_no_request():
    seen = []
    client = make_client(ok_handler(seen))
    assert client.embed_documents([]) == []
    assert seen == []


def test_identity_includes_model_and_prefixes():
    a = make_client(ok_handler([]), document_prefix="d: ").identity
    b = make_client(ok_handler([]), document_prefix="x: ").identity
    assert "nomic-embed-text" in a
    assert a != b


def test_connection_failure_is_ollama_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(OllamaUnavailable):
        make_client(handler).embed_documents(["x"])


def test_timeout_is_ollama_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("slow")
    with pytest.raises(OllamaUnavailable):
        make_client(handler).embed_documents(["x"])


def test_missing_model_is_reported_and_never_pulled():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(404, json={"error": 'model "nomic-embed-text" not found, try pulling it first'})

    with pytest.raises(ModelNotInstalled) as info:
        make_client(handler).embed_documents(["x"])
    assert "ollama pull nomic-embed-text" in str(info.value)
    assert paths == ["/api/embed"]  # no /api/pull or any other call


@pytest.mark.parametrize("status", [400, 500])
def test_http_errors_become_embedding_error(status):
    def handler(request):
        return httpx.Response(status, json={"error": "input length exceeds the context length"})
    with pytest.raises(EmbeddingError) as info:
        make_client(handler).embed_documents(["x"])
    assert str(status) in str(info.value)


@pytest.mark.parametrize("payload", [
    {"unexpected": 1},
    {"embeddings": [[0.1]]},          # wrong count for 2 inputs
    {"embeddings": [[0.1], []]},      # empty vector
    {"embeddings": "nope"},
])
def test_malformed_responses_become_embedding_error(payload):
    def handler(request):
        return httpx.Response(200, json=payload)
    with pytest.raises(EmbeddingError):
        make_client(handler).embed_documents(["a", "b"])


def test_non_json_response_becomes_embedding_error():
    def handler(request):
        return httpx.Response(200, content=b"<html>not json</html>")
    with pytest.raises(EmbeddingError):
        make_client(handler).embed_documents(["a"])
