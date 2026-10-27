"""Explicit domain errors. HTTP mapping happens only in the api layer."""

from __future__ import annotations


class LkaError(Exception):
    """Base class for all domain errors.

    ``systemic`` errors affect every document (service down, wrong model), so batch
    operations stop instead of failing once per file.
    """

    systemic = False


class InvalidSlug(LkaError, ValueError):
    """A category or topic is not a valid slug."""


class InvalidDocument(LkaError):
    """The file is corrupted, encrypted, unsupported or unreadable."""


class InvalidQuery(LkaError, ValueError):
    """The search question is empty, too long or otherwise unusable."""


class DocumentNotFound(LkaError):
    """No document with the given id exists."""


class EmbeddingModelMismatch(LkaError):
    """The configured embeddings model differs from the one the index was built with."""

    systemic = True


class OllamaError(LkaError):
    """Base class for problems talking to Ollama."""


class OllamaUnavailable(OllamaError):
    """Ollama cannot be reached or timed out."""

    systemic = True


class ModelNotInstalled(OllamaError):
    """The requested model is not installed in Ollama (it is never pulled automatically)."""

    systemic = True


class EmbeddingError(OllamaError):
    """Ollama answered, but the embeddings request failed or returned unusable data."""


class GenerationError(OllamaError):
    """Ollama answered, but the chat request failed or returned no usable text."""
