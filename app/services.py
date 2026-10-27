"""Composition root: builds the object graph once, for the API and the CLIs.

This is the only module that knows every package. Business logic stays in the services;
callers (CLI, API) stay thin and receive dependencies instead of creating them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from app.clients.ollama import OllamaCatalog, OllamaChatClient, OllamaEmbeddingClient
from app.core.config import Settings
from app.core.interfaces import ChatClient, EmbeddingClient
from app.ingestion.service import IngestionService
from app.rag.service import RagService
from app.retrieval.candidates import SemanticCandidates
from app.retrieval.service import Retriever
from app.storage.metadata_store import SqliteMetadataStore
from app.storage.vector_store import ChromaVectorStore


@dataclass
class Services:
    settings: Settings
    metadata: SqliteMetadataStore
    vectors: ChromaVectorStore
    embedder: EmbeddingClient
    chat: ChatClient
    catalog: object  # anything with installed_models() and matches(); None-safe in tests
    retriever: Retriever
    rag: RagService
    ingestion: IngestionService
    ingest_lock: threading.Lock = field(default_factory=threading.Lock)
    _owned: list = field(default_factory=list)

    def close(self) -> None:
        self.vectors.close()
        self.metadata.close()
        for obj in self._owned:
            close = getattr(obj, "close", None)
            if callable(close):
                close()


def build_services(
    settings: Settings,
    *,
    embedder: Optional[EmbeddingClient] = None,
    chat: Optional[ChatClient] = None,
    catalog: Optional[object] = None,
    enforce_identity: bool = True,
) -> Services:
    """Open the stores and clients. Raises EmbeddingModelMismatch if the index uses another model.

    Clients passed in (tests) are not closed by ``Services.close``; the ones created here are.
    """
    owned: list = []
    if embedder is None:
        embedder = OllamaEmbeddingClient.from_settings(settings)
        owned.append(embedder)
    if chat is None:
        chat = OllamaChatClient.from_settings(settings)
        owned.append(chat)
    if catalog is None:
        catalog = OllamaCatalog.from_settings(settings)
        owned.append(catalog)

    metadata = SqliteMetadataStore(settings.metadata_db_path)
    try:
        metadata.seed_categories_from_file(settings.categories_file)
        vectors = ChromaVectorStore(settings.vector_db_dir, embedder.identity, enforce_identity=enforce_identity)
    except BaseException:
        metadata.close()
        for obj in owned:
            obj.close()
        raise

    retriever = Retriever(
        metadata,
        SemanticCandidates(embedder, vectors),
        default_k=settings.retrieval_k,
        max_k=settings.retrieval_max_k,
        overfetch=settings.retrieval_overfetch,
        max_distance=settings.relevance_max_distance,
        dedupe_adjacent=settings.dedupe_adjacent,
        max_query_chars=settings.max_query_chars,
    )
    return Services(
        settings=settings,
        metadata=metadata,
        vectors=vectors,
        embedder=embedder,
        chat=chat,
        catalog=catalog,
        retriever=retriever,
        rag=RagService(retriever, chat, context_chars=settings.rag_context_chars),
        ingestion=IngestionService(
            metadata, vectors, embedder,
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, batch_size=settings.embed_batch_size,
        ),
        _owned=owned,
    )
