"""Ollama HTTP clients: embeddings (``EmbeddingClient``), chat (``ChatClient``) and a model catalog.

Never pulls models: a missing model raises ``ModelNotInstalled`` with the manual command.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Type
from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.core.errors import (
    EmbeddingError,
    GenerationError,
    ModelNotInstalled,
    OllamaError,
    OllamaUnavailable,
)
from app.core.models import ChatMessage

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

# Rough size of a prompt in tokens. Measured on this stack: ~2.5 characters per token for technical
# text, so dividing by 3 slightly underestimates, which keeps the truncation check free of false alarms.
_CHARS_PER_TOKEN = 3
_TRUNCATION_RATIO = 0.6


def estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        text = str(body.get("error", "")) if isinstance(body, dict) else ""
    except ValueError:
        text = ""
    return (text or "no details")[:200]


class _OllamaHttp:
    """Shared HTTP plumbing and error mapping."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        host = urlparse(base_url).hostname
        if host not in _LOCAL_HOSTS:
            logger.warning("Ollama base URL is not a loopback address; document text will be sent to it")
        self._base_url = base_url
        self._model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout_seconds, transport=transport)

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, payload: Dict[str, Any], failure: Type[OllamaError]) -> Any:
        try:
            response = self._http.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable(f"Ollama timed out at {self._base_url}") from exc
        except httpx.TransportError as exc:
            raise OllamaUnavailable(f"cannot reach Ollama at {self._base_url}") from exc

        if response.status_code >= 400:
            message = _error_message(response)
            if response.status_code == 404 and "not found" in message.lower():
                raise ModelNotInstalled(
                    f"model {self._model!r} is not installed in Ollama; "
                    f"install it explicitly with: ollama pull {self._model}"
                )
            raise failure(f"Ollama returned HTTP {response.status_code}: {message}")
        try:
            return response.json()
        except ValueError as exc:
            raise failure(f"unexpected response from Ollama {path}") from exc


class OllamaEmbeddingClient(_OllamaHttp):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 120.0,
        document_prefix: str = "",
        query_prefix: str = "",
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        super().__init__(base_url, model, timeout_seconds=timeout_seconds, transport=transport)
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaEmbeddingClient":
        return cls(
            settings.ollama_base_url,
            settings.embedding_model,
            timeout_seconds=settings.ollama_timeout_seconds,
            document_prefix=settings.embedding_document_prefix,
            query_prefix=settings.embedding_query_prefix,
        )

    @property
    def identity(self) -> str:
        return f"{self._model}|doc={self._document_prefix!r}|query={self._query_prefix!r}"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed([self._document_prefix + t for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._embed([self._query_prefix + text])[0]

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        if not inputs:
            return []
        # truncate=False: fail loudly instead of silently embedding a cut-off chunk.
        data = self._post("/api/embed", {"model": self._model, "input": inputs, "truncate": False}, EmbeddingError)
        try:
            embeddings = data["embeddings"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("unexpected response from Ollama /api/embed") from exc
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(inputs)
            or any(not isinstance(v, list) or not v for v in embeddings)
        ):
            raise EmbeddingError("Ollama returned a different number of embeddings than inputs")
        return [[float(x) for x in vector] for vector in embeddings]


class OllamaChatClient(_OllamaHttp):
    """Chat completion through ``/api/chat`` (non-streaming)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 300.0,
        num_ctx: int = 4096,
        temperature: float = 0.1,
        max_tokens: int = 600,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        super().__init__(base_url, model, timeout_seconds=timeout_seconds, transport=transport)
        self._options = {"num_ctx": num_ctx, "temperature": temperature, "num_predict": max_tokens}
        self.last_usage: Dict[str, float] = {}  # numbers only, from the latest reply (for measurements)

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaChatClient":
        return cls(
            settings.ollama_base_url,
            settings.chat_model,
            timeout_seconds=settings.chat_timeout_seconds,
            num_ctx=settings.chat_num_ctx,
            temperature=settings.chat_temperature,
            max_tokens=settings.chat_max_tokens,
        )

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        prompt_text = " ".join(m.content for m in messages)
        needed = estimate_tokens(prompt_text) + self._options["num_predict"]
        if needed >= self._options["num_ctx"]:
            raise GenerationError(
                f"the prompt (about {needed} tokens with the reply) does not fit the {self._options['num_ctx']}-token "
                "context window; lower LKA_RAG_CONTEXT_CHARS or raise LKA_CHAT_NUM_CTX"
            )
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": self._options,
        }
        data = self._post("/api/chat", payload, GenerationError)
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise GenerationError("unexpected response from Ollama /api/chat") from exc
        if not isinstance(text, str) or not text.strip():
            raise GenerationError("the model returned an empty reply")
        self.last_usage = {
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "reply_tokens": int(data.get("eval_count") or 0),
            "load_s": round((data.get("load_duration") or 0) / 1e9, 2),
            "prompt_eval_s": round((data.get("prompt_eval_duration") or 0) / 1e9, 2),
            "generation_s": round((data.get("eval_duration") or 0) / 1e9, 2),
        }
        logger.info("chat usage %s", self.last_usage)
        # Measured: Ollama silently truncates a prompt that exceeds num_ctx and reports a token count BELOW
        # the window (3529 tokens in a 512 window gave 258), so "count >= num_ctx" never fires. Compare with
        # the size estimate instead; a truncated prompt may have lost the instructions.
        reported = int(data.get("prompt_eval_count") or 0)
        expected = estimate_tokens(prompt_text)
        if reported and reported < expected * _TRUNCATION_RATIO:
            raise GenerationError(
                "the prompt was cut by the model's context window "
                f"(sent about {expected} tokens, the model read {reported}); lower LKA_RAG_CONTEXT_CHARS"
            )
        if data.get("done_reason") == "length":
            logger.warning("chat reply was cut at the token limit (num_predict=%d)", self._options["num_predict"])
        return text.strip()


class OllamaCatalog:
    """Which models are installed (read-only; used by the health check)."""

    def __init__(
        self, base_url: str, *, timeout_seconds: float = 3.0, transport: Optional[httpx.BaseTransport] = None
    ) -> None:
        self._base_url = base_url
        self._http = httpx.Client(base_url=base_url, timeout=timeout_seconds, transport=transport)

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaCatalog":
        return cls(settings.ollama_base_url)

    def installed_models(self) -> List[str]:
        try:
            response = self._http.get("/api/tags")
            response.raise_for_status()
            return [str(m["name"]) for m in response.json().get("models", [])]
        except (httpx.TransportError, httpx.HTTPStatusError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise OllamaUnavailable("cannot list models from Ollama") from exc

    @staticmethod
    def matches(installed: Sequence[str], model: str) -> bool:
        wanted = model if ":" in model else f"{model}:latest"
        return any(name == wanted or name == model for name in installed)

    def close(self) -> None:
        self._http.close()
