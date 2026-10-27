"""Map domain errors to HTTP responses. Only this layer knows about HTTP status codes.

Responses use fixed, generic text for infrastructure failures: no URLs, model identities, paths or
tracebacks. Details go to the logs as exception class names only (never document or question text).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import (
    DocumentNotFound,
    EmbeddingModelMismatch,
    InvalidDocument,
    InvalidQuery,
    InvalidSlug,
    LkaError,
    ModelNotInstalled,
    OllamaError,
    OllamaUnavailable,
)

logger = logging.getLogger(__name__)


def _json(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidQuery)
    @app.exception_handler(InvalidSlug)
    @app.exception_handler(InvalidDocument)
    async def _unprocessable(request: Request, exc: LkaError) -> JSONResponse:
        return _json(422, str(exc))

    @app.exception_handler(DocumentNotFound)
    async def _not_found(request: Request, exc: DocumentNotFound) -> JSONResponse:
        return _json(404, "Document not found.")

    @app.exception_handler(EmbeddingModelMismatch)
    async def _mismatch(request: Request, exc: EmbeddingModelMismatch) -> JSONResponse:
        logger.error("embeddings configuration differs from the index")
        return _json(409, "The index was built with a different embeddings configuration; reindexing is required.")

    @app.exception_handler(OllamaUnavailable)
    @app.exception_handler(ModelNotInstalled)
    async def _unavailable(request: Request, exc: OllamaError) -> JSONResponse:
        logger.error("language model service unavailable: %s", type(exc).__name__)
        return _json(503, "The language model service is unavailable or a required model is not installed. "
                          "Check GET /health.")

    @app.exception_handler(OllamaError)
    async def _bad_gateway(request: Request, exc: OllamaError) -> JSONResponse:
        logger.error("language model service error: %s", type(exc).__name__)
        return _json(502, "The language model service returned an error.")

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error: %s", type(exc).__name__)  # class only: messages may hold user text
        return _json(500, "Internal server error.")
