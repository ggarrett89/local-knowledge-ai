"""Measure cosine distances to choose LKA_RELEVANCE_MAX_DISTANCE.

Needs a running Ollama with the embeddings model installed. Read-only for your library.

    # 1) bilingual synthetic corpus, indexed in a temporary directory
    python scripts/calibrate_retrieval.py synthetic

    # 2) your real index (LKA_DATA_DIR), with a questions file
    python scripts/calibrate_retrieval.py index scripts/calibration/book_queries.json

It prints, per question, the distance of the best hit, and then how each candidate threshold
would split answerable from unanswerable questions. Nothing is changed or saved.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clients.ollama import OllamaEmbeddingClient  # noqa: E402
from app.core.config import load_settings  # noqa: E402
from app.core.references import format_reference  # noqa: E402
from app.retrieval.candidates import SemanticCandidates  # noqa: E402
from app.retrieval.service import Retriever  # noqa: E402
from app.storage.metadata_store import SqliteMetadataStore  # noqa: E402
from app.storage.vector_store import ChromaVectorStore  # noqa: E402

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def summarize(name: str, values: List[float]) -> None:
    if not values:
        print(f"  {name:<28} (none)")
        return
    print(f"  {name:<28} n={len(values):<3} min={min(values):.3f}  median={statistics.median(values):.3f}  max={max(values):.3f}")


def threshold_table(answerable: List[float], unanswerable: Dict[str, List[float]]) -> None:
    print("\nThreshold effect (top-1 distance of each question vs. the threshold):")
    header = f"  {'max_distance':>12}  {'answerable kept':>16}"
    for label in unanswerable:
        header += f"  {label + ' rejected':>20}"
    print(header)
    for t in THRESHOLDS:
        kept = sum(d <= t for d in answerable)
        row = f"  {t:>12.2f}  {kept:>6}/{len(answerable):<9}"
        for values in unanswerable.values():
            row += f"  {sum(d > t for d in values):>9}/{len(values):<10}"
        print(row)


def latency(samples: List[float]) -> None:
    if len(samples) < 2:
        return
    warm = samples[1:]
    print(f"\nSearch latency: first (cold) {samples[0]:.2f}s; then n={len(warm)} median={statistics.median(warm):.2f}s max={max(warm):.2f}s")


def run_synthetic(client: OllamaEmbeddingClient) -> None:
    from tests.corpus import OFF_TOPIC, QUERIES
    from tests.retrieval.conftest import Library

    with tempfile.TemporaryDirectory() as tmp:
        lib = Library(Path(tmp), embedder=client).add_corpus()
        try:
            retriever = lib.retriever()
            by_doc = {doc: key for key, doc in lib.doc_ids.items()}
            times: List[float] = []
            on: Dict[str, List[float]] = {"en": [], "pt": []}
            wrong = 0
            print("== synthetic corpus: answerable questions (top hit)")
            for q in QUERIES:
                started = time.perf_counter()
                hits = retriever.search(q.text, k=5)
                times.append(time.perf_counter() - started)
                top = hits[0]
                topic, lang = by_doc[top.chunk.provenance.document_id]
                ok = topic == q.topic
                wrong += not ok
                on[q.lang].append(top.distance)
                print(f"  [{q.lang}] {top.distance:.3f} {'ok ' if ok else 'WRONG'} -> {topic}/{lang}  {q.text[:55]}")
            off: List[float] = []
            print("== synthetic corpus: unanswerable questions (top hit)")
            for text in OFF_TOPIC:
                started = time.perf_counter()
                hits = retriever.search(text, k=5)
                times.append(time.perf_counter() - started)
                off.append(hits[0].distance)
                print(f"  {hits[0].distance:.3f}  {text[:60]}")
            print("\nSummary")
            summarize("answerable EN top-1", on["en"])
            summarize("answerable PT top-1", on["pt"])
            summarize("unanswerable top-1", off)
            print(f"  wrong top-1 topic: {wrong}/{len(QUERIES)}")
            threshold_table(on["en"] + on["pt"], {"unanswerable": off})
            latency(times)
        finally:
            lib.close()


def run_index(client: OllamaEmbeddingClient, queries_file: Path) -> None:
    settings = load_settings()
    spec = json.loads(queries_file.read_text(encoding="utf-8"))
    metadata = SqliteMetadataStore(settings.metadata_db_path)
    vectors = ChromaVectorStore(settings.vector_db_dir, client.identity)
    try:
        retriever = Retriever(metadata, SemanticCandidates(client, vectors))
        docs = metadata.list_documents(status="indexed")
        print(f"index: {len(docs)} document(s), {vectors.count()} chunks")
        times: List[float] = []
        answerable: Dict[str, List[float]] = {"en": [], "pt": []}
        hit1 = hit5 = 0
        print("== answerable questions (best hit; relevant = text contains a keyword)")
        for item in spec["in_scope"]:
            started = time.perf_counter()
            hits = retriever.search(item["q"], k=5)
            times.append(time.perf_counter() - started)
            rel = [any(kw in h.chunk.text.lower() for kw in item["keywords"]) for h in hits]
            hit1 += bool(rel and rel[0])
            hit5 += any(rel)
            answerable[item["lang"]].append(hits[0].distance)
            print(f"  [{item['lang']}] {hits[0].distance:.3f} top1={'rel' if rel[0] else '---'} top5={'rel' if any(rel) else '---'} "
                  f"{format_reference(hits[0].chunk.provenance).split(',', 1)[-1].strip():<10} {item['q'][:50]}")
        groups: Dict[str, List[float]] = {}
        for label in ("unrelated", "other_tech"):
            groups[label] = []
            print(f"== {label} questions (best hit)")
            for text in spec.get(label, []):
                started = time.perf_counter()
                hits = retriever.search(text, k=5)
                times.append(time.perf_counter() - started)
                groups[label].append(hits[0].distance)
                print(f"  {hits[0].distance:.3f}  {text[:70]}")
        n = len(spec["in_scope"])
        print("\nSummary")
        print(f"  keyword-relevant hit at rank 1: {hit1}/{n}; within top 5: {hit5}/{n}  (proxy for recall@k)")
        summarize("answerable EN top-1", answerable["en"])
        summarize("answerable PT top-1", answerable["pt"])
        for label, values in groups.items():
            summarize(f"{label} top-1", values)
        threshold_table(answerable["en"] + answerable["pt"], groups)
        latency(times)
    finally:
        vectors.close()
        metadata.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("synthetic")
    p = sub.add_parser("index")
    p.add_argument("queries", type=Path)
    args = parser.parse_args()
    client = OllamaEmbeddingClient.from_settings(load_settings())
    try:
        if args.mode == "synthetic":
            run_synthetic(client)
        else:
            run_index(client, args.queries)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
