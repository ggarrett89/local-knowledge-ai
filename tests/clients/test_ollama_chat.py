import json
import logging

import httpx
import pytest

from app.clients.ollama import OllamaCatalog, OllamaChatClient
from app.core.config import Settings
from app.core.errors import GenerationError, ModelNotInstalled, OllamaUnavailable
from app.core.models import ChatMessage

MESSAGES = [ChatMessage(role="system", content="rules"), ChatMessage(role="user", content="question")]


def chat_client(handler, **kwargs):
    return OllamaChatClient("http://127.0.0.1:11434", "qwen2.5:3b", transport=httpx.MockTransport(handler), **kwargs)


def reply(content="Hello [1].", **extra):
    body = {"message": {"role": "assistant", "content": content}, "done": True}
    body.update(extra)
    return lambda request: httpx.Response(200, json=body)


def test_sends_the_conversation_with_non_streaming_options():
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "  Hi [1].  "}})

    text = chat_client(handler, num_ctx=2048, temperature=0.3, max_tokens=123).complete(MESSAGES)
    assert text == "Hi [1]."
    path, body = seen[0]
    assert path == "/api/chat" and body["model"] == "qwen2.5:3b" and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]
    assert body["options"] == {"num_ctx": 2048, "temperature": 0.3, "num_predict": 123}


def test_model_property_and_from_settings():
    s = Settings(_env_file=None, chat_model="other:1b", chat_num_ctx=2048, chat_max_tokens=99, chat_temperature=0.0)
    c = OllamaChatClient.from_settings(s)
    assert c.model == "other:1b" and c._options == {"num_ctx": 2048, "temperature": 0.0, "num_predict": 99}
    c.close()


@pytest.mark.parametrize("payload", [{"message": {"content": ""}}, {"message": {"content": "   "}}, {"x": 1},
                                     {"message": "text"}, {"message": {"content": 5}}])
def test_empty_or_malformed_replies_are_generation_errors(payload):
    with pytest.raises(GenerationError):
        chat_client(lambda r: httpx.Response(200, json=payload)).complete(MESSAGES)


def test_non_json_reply_is_a_generation_error():
    with pytest.raises(GenerationError):
        chat_client(lambda r: httpx.Response(200, content=b"<html>")).complete(MESSAGES)


@pytest.mark.parametrize("exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_connection_problems_are_ollama_unavailable(exc):
    def handler(request):
        raise exc
    with pytest.raises(OllamaUnavailable):
        chat_client(handler).complete(MESSAGES)


def test_missing_model_is_reported_and_never_pulled():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(404, json={"error": 'model "qwen2.5:3b" not found, try pulling it first'})

    with pytest.raises(ModelNotInstalled) as info:
        chat_client(handler).complete(MESSAGES)
    assert "ollama pull qwen2.5:3b" in str(info.value) and paths == ["/api/chat"]


def test_http_error_becomes_generation_error():
    with pytest.raises(GenerationError, match="500"):
        chat_client(lambda r: httpx.Response(500, json={"error": "boom"})).complete(MESSAGES)


BIG = [ChatMessage(role="system", content="rules"), ChatMessage(role="user", content="x" * 6000)]  # ~2000 tokens


def test_a_prompt_that_cannot_fit_is_refused_before_it_is_sent():
    sent = []

    def handler(request):
        sent.append(1)
        return httpx.Response(200, json={"message": {"content": "x"}})

    with pytest.raises(GenerationError, match="does not fit"):
        chat_client(handler, num_ctx=2048, max_tokens=600).complete(BIG)
    assert sent == [], "nothing is sent when the prompt cannot fit"


def test_a_prompt_that_ollama_silently_truncated_is_detected():
    # Measured with Ollama 0.40: a ~3500-token prompt in a 512-token window is reported as 258 tokens.
    with pytest.raises(GenerationError, match="was cut by the model's context window"):
        chat_client(reply(prompt_eval_count=258), num_ctx=4096, max_tokens=100).complete(BIG)


def test_a_normally_processed_prompt_passes_the_truncation_check():
    assert chat_client(reply(prompt_eval_count=1900), num_ctx=4096, max_tokens=100).complete(BIG) == "Hello [1]."
    assert chat_client(reply(), num_ctx=4096, max_tokens=100).complete(BIG) == "Hello [1].", "no count reported"


def test_a_reply_cut_at_the_token_limit_is_returned_with_a_warning(caplog):
    caplog.set_level(logging.WARNING)
    assert chat_client(reply(done_reason="length")).complete(MESSAGES) == "Hello [1]."
    assert "token limit" in caplog.text


# --- catalog --------------------------------------------------------------------


def catalog(handler):
    return OllamaCatalog("http://127.0.0.1:11434", transport=httpx.MockTransport(handler))


def test_catalog_lists_installed_models():
    c = catalog(lambda r: httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}, {"name": "nomic-embed-text:latest"}]}))
    assert c.installed_models() == ["qwen2.5:3b", "nomic-embed-text:latest"]


@pytest.mark.parametrize("handler", [
    lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")),
    lambda r: httpx.Response(500, json={}),
    lambda r: httpx.Response(200, content=b"not json"),
    lambda r: httpx.Response(200, json={"models": [{"no_name": 1}]}),
])
def test_catalog_failures_are_ollama_unavailable(handler):
    with pytest.raises(OllamaUnavailable):
        catalog(handler).installed_models()


def test_model_name_matching_handles_the_default_tag():
    installed = ["nomic-embed-text:latest", "qwen2.5:3b"]
    assert OllamaCatalog.matches(installed, "nomic-embed-text")
    assert OllamaCatalog.matches(installed, "nomic-embed-text:latest")
    assert OllamaCatalog.matches(installed, "qwen2.5:3b")
    assert not OllamaCatalog.matches(installed, "qwen2.5:7b")
    assert not OllamaCatalog.matches(installed, "qwen2.5")


def test_last_usage_records_numbers_only():
    c = chat_client(reply(prompt_eval_count=1342, eval_count=107, load_duration=5_000_000_000,
                          prompt_eval_duration=700_000_000, eval_duration=21_400_000_000))
    assert c.last_usage == {}
    c.complete(MESSAGES)
    assert c.last_usage == {"prompt_tokens": 1342, "reply_tokens": 107, "load_s": 5.0,
                            "prompt_eval_s": 0.7, "generation_s": 21.4}
    assert all(isinstance(v, (int, float)) for v in c.last_usage.values())
