"""Ask a question and get an answer grounded in the indexed sources.

    python -m app.rag.cli ask "How do Ansible handlers work?"
    python -m app.rag.cli ask "Como funciona o Ansible Vault?" --category iac-automation --json

The list of sources is built from stored metadata, not from the model's text. The model can be
wrong; check the cited passages (--passages prints them).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional, Sequence

from app.core.config import Settings, load_settings
from app.core.errors import LkaError
from app.core.interfaces import ChatClient, EmbeddingClient
from app.core.models import SearchFilters
from app.rag.models import RagAnswer, Source
from app.services import build_services


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.rag.cli", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ask", help="answer a question from the indexed sources")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=None, help="passages to retrieve (default from settings)")
    p.add_argument("--category")
    p.add_argument("--topic", action="append", default=None, metavar="TOPIC")
    p.add_argument("--document", action="append", default=None, metavar="DOCUMENT_ID")
    p.add_argument("--source-type", choices=["pdf", "web", "code", "note"])
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--passages", action="store_true", help="also print every passage given to the model")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def _print_sources(title: str, sources: List[Source]) -> None:
    print(title)
    for s in sources:
        print(f"  [{s.number}] {s.reference}  (distance {s.distance:.3f}, {s.chunk_id})")


def render(result: RagAnswer, *, show_passages: bool) -> None:
    print(f"Status: {result.status}\n")
    print(result.answer)
    if result.inference:
        print("\nInference (written by the model, NOT stated in the sources):")
        print(result.inference)
    if result.sources:
        print()
        _print_sources("Sources cited:", result.sources)
    if result.notice:
        print(f"\nNotice: {result.notice}")
    if show_passages:
        print()
        _print_sources("Passages given to the model:", result.retrieved)
        for s in result.retrieved:
            print(f"\n  [{s.number}] {s.excerpt}")
    elif result.retrieved and not result.sources:
        print()
        _print_sources("Closest passages (not cited):", result.retrieved)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    settings: Optional[Settings] = None,
    embedder: Optional[EmbeddingClient] = None,
    chat: Optional[ChatClient] = None,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = settings or load_settings()
    try:
        filters = SearchFilters(
            document_ids=args.document, category=args.category, topics=args.topic, source_type=args.source_type
        )
        services = build_services(settings, embedder=embedder, chat=chat)
        try:
            result = services.rag.answer(args.question, k=args.k, filters=filters, max_distance=args.max_distance)
        finally:
            services.close()
    except (LkaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    else:
        render(result, show_passages=args.passages)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
