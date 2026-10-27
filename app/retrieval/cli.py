"""Command line search over the indexed library. Read-only; it never changes the index.

    python -m app.retrieval.cli search "How do Ansible handlers work?"
    python -m app.retrieval.cli search "inventário dinâmico" -k 3 --category iac-automation --json

Distances are cosine distances: 0 is the same direction, 2 is opposite. They rank results;
they are not probabilities of relevance.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence

from app.clients.ollama import OllamaEmbeddingClient
from app.core.config import Settings, load_settings
from app.core.errors import LkaError
from app.core.interfaces import EmbeddingClient
from app.core.models import SearchFilters, SearchHit
from app.core.references import format_reference
from app.retrieval.candidates import SemanticCandidates
from app.retrieval.service import Retriever
from app.storage.metadata_store import SqliteMetadataStore
from app.storage.vector_store import ChromaVectorStore

_SNIPPET_CHARS = 300


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.retrieval.cli", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search", help="semantic search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=None, help="number of results (default from settings)")
    p.add_argument("--category")
    p.add_argument("--topic", action="append", default=None, metavar="TOPIC", help="repeat for several; all must match")
    p.add_argument("--document", action="append", default=None, metavar="DOCUMENT_ID", help="repeat for several")
    p.add_argument("--source-type", choices=["pdf", "web", "code", "note"])
    p.add_argument("--max-distance", type=float, default=None, help="drop hits farther than this (0 < d <= 2)")
    p.add_argument("--full", action="store_true", help="print the whole chunk instead of a snippet")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


@contextmanager
def _runtime(settings: Settings, embedder: Optional[EmbeddingClient]) -> Iterator[tuple]:
    own_embedder = embedder is None
    if embedder is None:
        embedder = OllamaEmbeddingClient.from_settings(settings)
    metadata = SqliteMetadataStore(settings.metadata_db_path)
    try:
        vectors = ChromaVectorStore(settings.vector_db_dir, embedder.identity)  # refuses a different model
        try:
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
            yield retriever, metadata
        finally:
            vectors.close()
    finally:
        metadata.close()
        if own_embedder and hasattr(embedder, "close"):
            embedder.close()  # type: ignore[attr-defined]


def _snippet(text: str, full: bool) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if full or len(flat) <= _SNIPPET_CHARS else flat[:_SNIPPET_CHARS].rstrip() + "..."


def _as_dict(rank: int, hit: SearchHit) -> dict:
    p = hit.chunk.provenance
    return {
        "rank": rank,
        "distance": round(hit.distance, 4),
        "reference": format_reference(p),
        "document_id": p.document_id,
        "chunk_id": hit.chunk.chunk_id,
        "title": p.title,
        "page_start": p.page_start,
        "page_end": p.page_end,
        "text": hit.chunk.text,
    }


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
        filters = SearchFilters(
            document_ids=args.document, category=args.category, topics=args.topic, source_type=args.source_type
        )
        with _runtime(settings, embedder) as (retriever, metadata):
            hits: List[SearchHit] = retriever.search(
                args.query, k=args.k, filters=filters, max_distance=args.max_distance
            )
            indexed = len(metadata.list_documents(status="indexed")) if not hits else 0
    except (LkaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([_as_dict(i, h) for i, h in enumerate(hits, 1)], ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("No results." if indexed else "No results. The index has no documents yet; import a PDF first.")
        return 0
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. [{hit.distance:.3f}] {format_reference(hit.chunk.provenance)}  ({hit.chunk.chunk_id})")
        print(f"   {_snippet(hit.chunk.text, args.full)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
