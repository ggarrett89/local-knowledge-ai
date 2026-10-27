"""Semantic retrieval: filters, relevance threshold, de-duplication. Never generates text."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Set

from app.core.errors import InvalidQuery
from app.core.models import SearchFilters, SearchHit
from app.retrieval.candidates import CandidateGenerator
from app.storage.base import MetadataStore

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 200


class Retriever:
    def __init__(
        self,
        metadata: MetadataStore,
        generator: CandidateGenerator,
        *,
        default_k: int = 5,
        max_k: int = 50,
        overfetch: int = 4,
        max_distance: float = 2.0,
        dedupe_adjacent: bool = True,
        max_query_chars: int = 1000,
    ) -> None:
        self._metadata = metadata
        self._generator = generator
        self._default_k = default_k
        self._max_k = max_k
        self._overfetch = overfetch
        self._max_distance = max_distance
        self._dedupe_adjacent = dedupe_adjacent
        self._max_query_chars = max_query_chars

    def search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        max_distance: Optional[float] = None,
    ) -> List[SearchHit]:
        """Best-first hits for ``query``. Returns an empty list (not an error) when the index is
        empty, the filters match nothing, or nothing is close enough."""
        started = time.perf_counter()
        query = self._validate_query(query)
        k = self._default_k if k is None else k
        if not 1 <= k <= self._max_k:
            raise InvalidQuery(f"k must be between 1 and {self._max_k}")
        threshold = self._max_distance if max_distance is None else max_distance

        document_ids = self._resolve_documents(filters)
        if document_ids is not None and not document_ids:
            return []  # nothing matches the filters: no need to embed the question

        limit = min(k * self._overfetch, _MAX_CANDIDATES)
        candidates = self._generator.generate(query, limit=limit, document_ids=document_ids)
        close = sorted((h for h in candidates if h.distance <= threshold), key=lambda h: h.distance)
        hits = self._deduplicate(close, k)
        # Log counts and timing only: never the question or any document text.
        logger.info(
            "search candidates=%d kept=%d returned=%d seconds=%.3f",
            len(candidates), len(close), len(hits), time.perf_counter() - started,
        )
        return hits

    # -- internals -----------------------------------------------------------------

    def _validate_query(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            raise InvalidQuery("the question is empty")
        if len(query) > self._max_query_chars:
            raise InvalidQuery(f"the question is too long ({len(query)} > {self._max_query_chars} characters)")
        return query

    def _resolve_documents(self, filters: Optional[SearchFilters]) -> Optional[List[str]]:
        """Turn metadata filters into document ids. None means "no restriction"."""
        if filters is None or filters.is_empty():
            return None
        ids: Set[str] = {
            d.document_id
            for d in self._metadata.list_documents(
                category=filters.category,
                topics=filters.topics,
                source_type=filters.source_type,
                status="indexed",
            )
        }
        if filters.document_ids is not None:
            ids &= set(filters.document_ids)
        return sorted(ids)

    def _deduplicate(self, hits: List[SearchHit], k: int) -> List[SearchHit]:
        """Drop chunks with identical text and, optionally, chunks next to an already chosen one
        (neighbouring chunks overlap by design and usually cover the same passage)."""
        chosen: List[SearchHit] = []
        seen_hashes: Set[str] = set()
        positions: Set[tuple] = set()
        for hit in hits:
            chunk = hit.chunk
            if chunk.content_hash in seen_hashes:
                continue
            doc = chunk.provenance.document_id
            if self._dedupe_adjacent and (
                (doc, chunk.chunk_index - 1) in positions or (doc, chunk.chunk_index + 1) in positions
            ):
                continue
            chosen.append(hit)
            seen_hashes.add(chunk.content_hash)
            positions.add((doc, chunk.chunk_index))
            if len(chosen) == k:
                break
        return chosen
