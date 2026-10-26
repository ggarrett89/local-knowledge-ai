"""PDF ingestion service: incremental, idempotent, one document at a time.

Guarantees:
- Nothing is read or indexed unless a method here is called explicitly.
- Documents are processed as streams (page -> chunk -> small embedding batches).
- A failed ingestion leaves no partial chunks behind and records the error.
- Original files are only read, never modified or deleted.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from app.core.clock import utc_now_iso
from app.core.errors import (
    DocumentNotFound,
    EmbeddingError,
    EmbeddingModelMismatch,
    InvalidDocument,
    LkaError,
    OllamaError,
)
from app.core.interfaces import EmbeddingClient
from app.core.models import (
    Chunk,
    DocumentRecord,
    Provenance,
    make_chunk_id,
    normalize_slug,
)
from app.ingestion.base import Extractor
from app.ingestion.chunking import chunk_sections
from app.ingestion.hashing import sha256_file, sha256_text
from app.ingestion.pdf.extractor import PdfExtractor
from app.storage.base import MetadataStore, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY = "general"

Outcome = Literal["indexed", "updated", "reindexed", "unchanged", "duplicate", "no_text", "error"]


class IngestResult(BaseModel):
    path: str
    outcome: Outcome
    document_id: Optional[str] = None
    title: Optional[str] = None
    pages: Optional[int] = None
    chunks: int = 0
    seconds: float = 0.0
    message: Optional[str] = None


class BatchSummary(BaseModel):
    results: List[IngestResult] = Field(default_factory=list)
    aborted: Optional[str] = None
    skipped_after_abort: int = 0

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for result in self.results:
            counts[result.outcome] = counts.get(result.outcome, 0) + 1
        return counts

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.results)

    @property
    def total_seconds(self) -> float:
        return round(sum(r.seconds for r in self.results), 2)

    @property
    def has_errors(self) -> bool:
        return self.aborted is not None or any(r.outcome == "error" for r in self.results)


class IngestionService:
    def __init__(
        self,
        metadata: MetadataStore,
        vectors: VectorStore,
        embedder: EmbeddingClient,
        *,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        extractors: Optional[Mapping[str, Extractor]] = None,
    ) -> None:
        self._metadata = metadata
        self._vectors = vectors
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size
        self._extractors: Mapping[str, Extractor] = extractors or {".pdf": PdfExtractor()}

    # -- public API ----------------------------------------------------------------

    def ingest_file(
        self,
        path: Path,
        category: Optional[str] = None,
        topics: Optional[Sequence[str]] = None,
        *,
        force: bool = False,
    ) -> IngestResult:
        """Index one file. ``category``/``topics`` of None inherit from a known version of it."""
        started = time.perf_counter()
        category = normalize_slug(category) if category is not None else None
        topic_list = sorted({normalize_slug(t) for t in topics}) if topics is not None else None
        path = Path(path).expanduser().resolve()

        extractor = self._extractors.get(path.suffix.lower())
        if not path.is_file():
            return self._reject(path, "file not found or not a regular file")
        if extractor is None:
            return self._reject(path, f"unsupported file type {path.suffix or '(none)'!r}; Stage 1 supports PDF")
        try:
            file_hash = sha256_file(path)
        except OSError as exc:
            return self._reject(path, f"cannot read file: {type(exc).__name__}")

        document_id = f"{extractor.source_type}-{file_hash[:16]}"
        existing = self._metadata.get_document(document_id)
        if existing is not None and not force and existing.status in ("indexed", "no_text"):
            same_path = existing.file_path == str(path)
            return IngestResult(
                path=str(path),
                outcome="unchanged" if same_path else "duplicate",
                document_id=document_id,
                title=existing.title,
                pages=existing.page_count,
                chunks=existing.chunk_count,
                message=(
                    f"already indexed (status={existing.status})"
                    if same_path
                    else f"same content already indexed from {existing.file_path}"
                ),
            )

        previous = [d for d in self._metadata.find_by_path(str(path)) if d.document_id != document_id]
        base = existing or (previous[0] if previous else None)
        final_category = category or (base.category if base else DEFAULT_CATEGORY)
        final_topics = topic_list if topic_list is not None else (list(base.topics) if base else [])
        self._metadata.ensure_category(final_category)

        try:
            record = self._index(extractor, path, document_id, file_hash, final_category, final_topics, started)
        except BaseException as exc:
            self._cleanup_failed(document_id)
            if not isinstance(exc, Exception):
                raise
            return self._record_failure(exc, path, document_id, file_hash, final_category, final_topics, started)

        self._metadata.clear_errors(document_id)
        if record.status == "no_text":
            self._metadata.record_error(
                "extract",
                "no extractable text (scanned PDF? OCR is not supported yet)",
                document_id=document_id,
                source_uri=record.source_uri,
            )
        self._metadata.upsert_document(record)
        for old in previous:
            self._vectors.delete_document(old.document_id)
            self._metadata.remove_document(old.document_id)

        if record.status == "no_text":
            outcome: Outcome = "no_text"
        elif previous:
            outcome = "updated"
        elif existing is not None:
            outcome = "reindexed"
        else:
            outcome = "indexed"
        logger.info("ingested document_id=%s outcome=%s chunks=%d", document_id, outcome, record.chunk_count)
        return IngestResult(
            path=str(path),
            outcome=outcome,
            document_id=document_id,
            title=record.title,
            pages=record.page_count,
            chunks=record.chunk_count,
            seconds=record.ingest_seconds or 0.0,
        )

    def ingest_folder(
        self,
        root: Path,
        category: Optional[str] = None,
        topics: Optional[Sequence[str]] = None,
        *,
        force: bool = False,
    ) -> BatchSummary:
        """Index every supported file below ``root`` (recursive), one at a time."""
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise InvalidDocument("not a directory")
        files = self._discover(root)
        summary = BatchSummary()
        for position, file in enumerate(files):
            try:
                summary.results.append(self.ingest_file(file, category, topics, force=force))
            except LkaError as exc:
                if not exc.systemic:
                    raise
                summary.results.append(
                    IngestResult(path=str(file), outcome="error", message=str(exc))
                )
                summary.aborted = str(exc)
                summary.skipped_after_abort = len(files) - position - 1
                break
        return summary

    def reindex(self, document_id: str) -> IngestResult:
        record = self._metadata.get_document(document_id)
        if record is None:
            raise DocumentNotFound(f"no document with id {document_id!r}")
        if not record.file_path or not Path(record.file_path).is_file():
            raise InvalidDocument("the original file is no longer available at its recorded path")
        return self.ingest_file(Path(record.file_path), record.category, record.topics, force=True)

    def remove(self, document_id: str) -> bool:
        """Remove a document from the index. The original file is never touched."""
        if self._metadata.get_document(document_id) is None:
            return False
        self._vectors.delete_document(document_id)
        self._metadata.remove_document(document_id)
        return True

    def list_documents(
        self,
        *,
        category: Optional[str] = None,
        topics: Optional[Sequence[str]] = None,
        status: Optional[str] = None,
    ) -> List[DocumentRecord]:
        return self._metadata.list_documents(category=category, topics=topics, status=status)

    def stats(self) -> Dict[str, object]:
        data: Dict[str, object] = dict(self._metadata.stats())
        data["vector_chunks"] = self._vectors.count()
        return data

    # -- internals -----------------------------------------------------------------

    def _discover(self, root: Path) -> List[Path]:
        found: List[Path] = []
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                if Path(name).suffix.lower() in self._extractors:
                    found.append(Path(directory) / name)
        return found

    def _index(
        self,
        extractor: Extractor,
        path: Path,
        document_id: str,
        file_hash: str,
        category: str,
        topics: List[str],
        started: float,
    ) -> DocumentRecord:
        info = extractor.describe(path)
        chunk_count = 0
        batch: List[Chunk] = []
        sections = extractor.extract(path)
        for piece in chunk_sections(sections, chunk_size=self._chunk_size, overlap=self._chunk_overlap):
            batch.append(
                Chunk(
                    chunk_id=make_chunk_id(document_id, piece.index),
                    chunk_index=piece.index,
                    text=piece.text,
                    content_hash=sha256_text(piece.text),
                    provenance=Provenance(
                        source_type=extractor.source_type,
                        document_id=document_id,
                        title=info.title,
                        page_start=piece.page_start,
                        page_end=piece.page_end,
                    ),
                )
            )
            if len(batch) >= self._batch_size:
                self._flush(batch)
                chunk_count += len(batch)
                batch = []
        if batch:
            self._flush(batch)
            chunk_count += len(batch)

        if chunk_count:
            self._vectors.prune_document(document_id, chunk_count)
        else:
            self._vectors.delete_document(document_id)

        return DocumentRecord(
            document_id=document_id,
            title=info.title,
            source_type=extractor.source_type,
            category=category,
            topics=topics,
            source_uri=path.as_uri(),
            file_path=str(path),
            file_hash=file_hash,
            status="indexed" if chunk_count else "no_text",
            page_count=info.page_count,
            chunk_count=chunk_count,
            updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            indexed_at=utc_now_iso(),
            ingest_seconds=round(time.perf_counter() - started, 3),
            embedding_model=self._embedder.identity,
        )

    def _flush(self, batch: List[Chunk]) -> None:
        vectors = self._embedder.embed_documents([c.text for c in batch])
        if len(vectors) != len(batch):
            raise EmbeddingError("embeddings client returned a different number of vectors than inputs")
        self._vectors.upsert(batch, vectors)

    def _cleanup_failed(self, document_id: str) -> None:
        try:
            self._vectors.delete_document(document_id)
        except Exception:  # noqa: BLE001 - cleanup must not mask the original error
            logger.warning("could not clean partial chunks for document_id=%s", document_id)

    def _record_failure(
        self,
        exc: Exception,
        path: Path,
        document_id: str,
        file_hash: str,
        category: str,
        topics: List[str],
        started: float,
    ) -> IngestResult:
        # Only our own messages are stored: they never contain document text.
        message = str(exc) if isinstance(exc, LkaError) else f"unexpected {type(exc).__name__}"
        if isinstance(exc, InvalidDocument):
            stage = "extract"
        elif isinstance(exc, (OllamaError, EmbeddingModelMismatch)):
            stage = "embed"
        else:
            stage = "index"
        self._metadata.upsert_document(
            DocumentRecord(
                document_id=document_id,
                title=path.stem,
                source_type="pdf",
                category=category,
                topics=topics,
                source_uri=path.as_uri(),
                file_path=str(path),
                file_hash=file_hash,
                status="error",
                chunk_count=0,
                indexed_at=utc_now_iso(),
                ingest_seconds=round(time.perf_counter() - started, 3),
                embedding_model=self._embedder.identity,
            )
        )
        self._metadata.record_error(stage, message, document_id=document_id, source_uri=path.as_uri())
        logger.warning("ingestion failed document_id=%s stage=%s", document_id, stage)
        if isinstance(exc, LkaError) and exc.systemic:
            raise exc
        return IngestResult(
            path=str(path),
            outcome="error",
            document_id=document_id,
            title=path.stem,
            seconds=round(time.perf_counter() - started, 3),
            message=message,
        )

    def _reject(self, path: Path, message: str) -> IngestResult:
        self._metadata.record_error("validate", message, source_uri=path.as_uri())
        return IngestResult(path=str(path), outcome="error", message=message)
