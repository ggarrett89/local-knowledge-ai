"""Small interfaces for external clients, so they can be mocked and swapped."""

from __future__ import annotations

from typing import List, Protocol, Sequence

from app.core.models import ChatMessage


class EmbeddingClient(Protocol):
    """Produces embeddings. Indexing and querying must use the same model."""

    @property
    def identity(self) -> str:
        """Stable string identifying the model and any prefixes; part of the index identity."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed passages for indexing. Returns one vector per input, in order."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a search question."""
        ...


class ChatClient(Protocol):
    """Generates text from a conversation. Swappable without touching RAG logic."""

    @property
    def model(self) -> str: ...

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant's reply text."""
        ...
