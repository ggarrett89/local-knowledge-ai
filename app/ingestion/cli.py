"""Command line interface for PDF ingestion.

Thin layer over ``IngestionService``. Every command is an explicit user action;
nothing is indexed, collected or downloaded automatically.

    python -m app.ingestion.cli add BOOK.pdf --category devops-sre --topic kubernetes
    python -m app.ingestion.cli add-all ~/books --category general
    python -m app.ingestion.cli list
    python -m app.ingestion.cli reindex DOCUMENT_ID
    python -m app.ingestion.cli remove DOCUMENT_ID
    python -m app.ingestion.cli stats
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

from app.clients.ollama import OllamaEmbeddingClient
from app.core.config import Settings, load_settings
from app.core.errors import LkaError
from app.core.interfaces import EmbeddingClient
from app.ingestion.service import BatchSummary, IngestionService, IngestResult
from app.storage.metadata_store import SqliteMetadataStore
from app.storage.vector_store import ChromaVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion.cli", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--category", help="primary category slug (default: inherit, else 'general')")
        p.add_argument("--topic", action="append", default=None, metavar="TOPIC",
                       help="additional topic slug; repeat for several")
        p.add_argument("--force", action="store_true", help="reindex even if the content is already indexed")

    p_add = sub.add_parser("add", help="index one PDF")
    p_add.add_argument("path", type=Path)
    add_source_args(p_add)

    p_all = sub.add_parser("add-all", help="index every PDF below a folder (recursive)")
    p_all.add_argument("directory", type=Path)
    add_source_args(p_all)

    p_list = sub.add_parser("list", help="list indexed documents")
    p_list.add_argument("--category")
    p_list.add_argument("--topic", action="append", default=None, metavar="TOPIC")
    p_list.add_argument("--status", choices=["indexed", "no_text", "error"])

    p_re = sub.add_parser("reindex", help="reindex one document from its original file")
    p_re.add_argument("document_id")

    p_rm = sub.add_parser("remove", help="remove a document from the index (the file is kept)")
    p_rm.add_argument("document_id")
    p_rm.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    sub.add_parser("stats", help="show index statistics")
    return parser


@contextmanager
def _runtime(settings: Settings, embedder: Optional[EmbeddingClient], *, enforce_identity: bool = True) -> Iterator[IngestionService]:
    own_embedder = embedder is None
    if embedder is None:
        embedder = OllamaEmbeddingClient.from_settings(settings)
    metadata = SqliteMetadataStore(settings.metadata_db_path)
    try:
        metadata.seed_categories_from_file(settings.categories_file)
        vectors = ChromaVectorStore(settings.vector_db_dir, embedder.identity, enforce_identity=enforce_identity)
        try:
            yield IngestionService(
                metadata,
                vectors,
                embedder,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                batch_size=settings.embed_batch_size,
            )
        finally:
            vectors.close()
    finally:
        metadata.close()
        if own_embedder and hasattr(embedder, "close"):
            embedder.close()  # type: ignore[attr-defined]


def _print_result(result: IngestResult) -> None:
    parts = [f"[{result.outcome}]", result.document_id or "-"]
    if result.title:
        parts.append(f'"{result.title}"')
    if result.pages is not None:
        parts.append(f"pages={result.pages}")
    if result.outcome in ("indexed", "updated", "reindexed", "no_text"):
        parts.append(f"chunks={result.chunks}")
        parts.append(f"{result.seconds:.1f}s")
    print(" ".join(parts))
    if result.message:
        print(f"    {result.message}")
    if result.outcome == "error" and not result.document_id:
        print(f"    file: {result.path}")


def _print_summary(summary: BatchSummary) -> None:
    for result in summary.results:
        _print_result(result)
    counts = ", ".join(f"{k}={v}" for k, v in sorted(summary.counts().items())) or "no PDF files found"
    print(f"\nSummary: {counts}; chunks={summary.total_chunks}; time={summary.total_seconds}s")
    if summary.aborted:
        print(f"Stopped early: {summary.aborted}")
        print(f"{summary.skipped_after_abort} remaining file(s) were not processed.")


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    settings: Optional[Settings] = None,
    embedder: Optional[EmbeddingClient] = None,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = settings or load_settings()
    try:
        return _dispatch(args, settings, embedder)
    except LkaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace, settings: Settings, embedder: Optional[EmbeddingClient]) -> int:
    cmd = args.command
    if cmd == "add":
        with _runtime(settings, embedder) as service:
            result = service.ingest_file(args.path, args.category, args.topic, force=args.force)
        _print_result(result)
        return 1 if result.outcome == "error" else 0

    if cmd == "add-all":
        with _runtime(settings, embedder) as service:
            summary = service.ingest_folder(args.directory, args.category, args.topic, force=args.force)
        _print_summary(summary)
        return 1 if summary.has_errors else 0

    if cmd == "reindex":
        with _runtime(settings, embedder) as service:
            result = service.reindex(args.document_id)
        _print_result(result)
        return 1 if result.outcome == "error" else 0

    if cmd == "remove":
        with _runtime(settings, embedder, enforce_identity=False) as service:
            documents = {d.document_id: d for d in service.list_documents()}
            doc = documents.get(args.document_id)
            if doc is None:
                print(f"error: no document with id {args.document_id!r}", file=sys.stderr)
                return 1
            print(f'Document {doc.document_id} "{doc.title}" ({doc.chunk_count} chunks)')
            print("This removes it from the index. The original file is not touched.")
            if not args.yes and not _confirm("Remove from the index?"):
                print("Aborted.")
                return 1
            service.remove(args.document_id)
        print(f"Removed {args.document_id} from the index.")
        return 0

    if cmd == "list":
        with _runtime(settings, embedder, enforce_identity=False) as service:
            docs = service.list_documents(category=args.category, topics=args.topic, status=args.status)
        if not docs:
            print("No documents.")
            return 0
        for d in docs:
            topics = ",".join(d.topics) or "-"
            pages = d.page_count if d.page_count is not None else "-"
            print(f'{d.document_id}  {d.status:<8} {d.category:<22} pages={pages} chunks={d.chunk_count} '
                  f'topics={topics}  "{d.title}"')
        return 0

    if cmd == "stats":
        with _runtime(settings, embedder, enforce_identity=False) as service:
            stats = service.stats()
        print(f"Documents : {stats['documents']}  {stats['by_status']}")
        print(f"Pages     : {stats['pages']}")
        print(f"Chunks    : {stats['chunks']} (vectors stored: {stats['vector_chunks']})")
        print(f"Errors    : {stats['errors']}")
        print(f"Ingest    : {stats['ingest_seconds']}s total")
        print(f"Categories: {stats['by_category']}")
        return 0

    raise AssertionError(f"unhandled command {cmd}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
