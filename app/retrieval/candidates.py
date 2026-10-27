"""Candidate generators: the seam where hybrid search will plug in later.

``Retriever`` only depends on this protocol. A future lexical or hybrid generator implements
the same method (fusing its own sources internally), so ``Retriever.search`` does not change.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

from app.core.interfaces import EmbeddingClient
from app.core.models import SearchHit
from app.storage.base import VectorStore


class CandidateGenerator(Protocol):
    def generate(
        self, query: str, *, limit: int, document_ids: Optional[Sequence[str]]
    ) -> List[SearchHit]:
        """Return up to ``limit`` hits, best first. ``document_ids=None`` means no restriction."""
        ...


class SemanticCandidates:
    """Nearest neighbours of the question embedding (same model as indexing)."""

    def __init__(self, embedder: EmbeddingClient, vectors: VectorStore) -> None:
        self._embedder = embedder
        self._vectors = vectors

    def generate(
        self, query: str, *, limit: int, document_ids: Optional[Sequence[str]]
    ) -> List[SearchHit]:
        vector = self._embedder.embed_query(query)
        return self._vectors.query(vector, limit, document_ids)
