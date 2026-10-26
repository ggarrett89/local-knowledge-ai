"""ChromaDB vector store (persistent, local). Vectors come from our own embeddings client.

One fixed collection holds all chunks. The embeddings identity (model + prefixes) is recorded
in the collection metadata, and opening the store with a different identity is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InvalidArgumentError
from pydantic import BaseModel

from app.core.errors import EmbeddingModelMismatch
from app.core.models import Chunk

COLLECTION_NAME = "lka_chunks"
IDENTITY_KEY = "lka_embedding_identity"
_DELETE_SLICE = 1000


class StoredChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class ChromaVectorStore:
    def __init__(self, path: Path, embedding_identity: str, *, enforce_identity: bool = True) -> None:
        """Open (or create) the index.

        ``enforce_identity=False`` is only for identity-agnostic operations (deleting a
        document, reading counts); indexing and querying must always enforce it.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Telemetry off: no data may leave the machine.
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", IDENTITY_KEY: embedding_identity},
            embedding_function=None,
        )
        stored = (self._collection.metadata or {}).get(IDENTITY_KEY)
        if enforce_identity and stored != embedding_identity:
            raise EmbeddingModelMismatch(
                "the index was built with a different embeddings configuration "
                f"(index: {stored!r}, configured: {embedding_identity!r}); indexing and querying are "
                "refused. Restore the original LKA_EMBEDDING_MODEL / prefixes, or build a new index "
                "in a fresh LKA_DATA_DIR and add your documents again."
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        try:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=[list(v) for v in vectors],
                documents=[c.text for c in chunks],
                metadatas=[_chunk_metadata(c) for c in chunks],
            )
        except InvalidArgumentError as exc:
            if "dimension" in str(exc).lower():
                raise EmbeddingModelMismatch(
                    "embedding dimension differs from the index; the embeddings model changed"
                ) from exc
            raise

    def delete_document(self, document_id: str) -> int:
        ids = self._collection.get(where={"document_id": document_id}, include=[])["ids"]
        self._delete_ids(ids)
        return len(ids)

    def prune_document(self, document_id: str, keep_count: int) -> int:
        """Delete chunks of a document with chunk_index >= keep_count (stale tail after reindex)."""
        where = {"$and": [{"document_id": document_id}, {"chunk_index": {"$gte": keep_count}}]}
        ids = self._collection.get(where=where, include=[])["ids"]
        self._delete_ids(ids)
        return len(ids)

    def count(self) -> int:
        return self._collection.count()

    def count_document(self, document_id: str) -> int:
        return len(self._collection.get(where={"document_id": document_id}, include=[])["ids"])

    def get_chunks(self, document_id: str) -> List[StoredChunk]:
        """Stored chunks of a document ordered by chunk_index (for verification and Stage 2)."""
        res = self._collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        items = [
            StoredChunk(chunk_id=i, text=d, metadata=dict(m))
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        return sorted(items, key=lambda c: c.metadata.get("chunk_index", 0))

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _delete_ids(self, ids: List[str]) -> None:
        for start in range(0, len(ids), _DELETE_SLICE):
            self._collection.delete(ids=ids[start:start + _DELETE_SLICE])


def _chunk_metadata(chunk: Chunk) -> Dict[str, Any]:
    p = chunk.provenance
    meta: Dict[str, Any] = {
        "document_id": p.document_id,
        "source_type": p.source_type,
        "title": p.title,
        "chunk_index": chunk.chunk_index,
        "content_hash": chunk.content_hash,
        "page_start": p.page_start,
        "page_end": p.page_end,
        "url": p.url,
        "section": p.section,
        "repo": p.repo,
        "path": p.path,
        "line_start": p.line_start,
        "line_end": p.line_end,
    }
    # Chroma rejects None; unknown provenance fields are simply absent.
    return {k: v for k, v in meta.items() if v is not None}
