"""Ollama HTTP client for embeddings (implements ``EmbeddingClient``).

Never pulls models: a missing model raises ``ModelNotInstalled`` with the manual command.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence
from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.core.errors import EmbeddingError, ModelNotInstalled, OllamaUnavailable

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class OllamaEmbeddingClient:
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
        host = urlparse(base_url).hostname
        if host not in _LOCAL_HOSTS:
            logger.warning("Ollama base URL is not a loopback address; document text will be sent to it")
        self._base_url = base_url
        self._model = model
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._http = httpx.Client(base_url=base_url, timeout=timeout_seconds, transport=transport)

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

    def close(self) -> None:
        self._http.close()

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        if not inputs:
            return []
        # truncate=False: fail loudly instead of silently embedding a cut-off chunk.
        payload = {"model": self._model, "input": inputs, "truncate": False}
        try:
            response = self._http.post("/api/embed", json=payload)
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
            raise EmbeddingError(f"Ollama returned HTTP {response.status_code}: {message}")

        try:
            data: Any = response.json()
            embeddings = data["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("unexpected response from Ollama /api/embed") from exc
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(inputs)
            or any(not isinstance(v, list) or not v for v in embeddings)
        ):
            raise EmbeddingError("Ollama returned a different number of embeddings than inputs")
        return [[float(x) for x in vector] for vector in embeddings]


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        text = str(body.get("error", "")) if isinstance(body, dict) else ""
    except ValueError:
        text = ""
    return (text or "no details")[:200]
