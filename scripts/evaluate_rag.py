"""Measure grounded answering with the real chat model on your own index. Read-only.

    python scripts/evaluate_rag.py scripts/calibration/rag_questions.json [--limit N]

Needs Ollama with the chat and embeddings models. It prints per-question status, citations, how
many references had to be repaired or removed, and timings; never the answer text or passages.
Answers on a CPU take 20-100 s each, so a full run takes several minutes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clients.ollama import OllamaChatClient  # noqa: E402
from app.core.config import load_settings  # noqa: E402
from app.rag.context import select_within_budget  # noqa: E402
from app.services import build_services  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("questions", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions of each group")
    args = parser.parse_args()
    spec = json.loads(args.questions.read_text(encoding="utf-8"))
    settings = load_settings()
    chat = OllamaChatClient.from_settings(settings)
    svc = build_services(settings, chat=chat)
    rows = []
    try:
        for group in ("answerable", "unanswerable"):
            items = spec[group][: args.limit] if args.limit else spec[group]
            print(f"== {group}")
            for item in items:
                started = time.perf_counter()
                result = svc.rag.answer(item["q"])
                took = time.perf_counter() - started
                usage = dict(chat.last_usage) if result.retrieved else {}
                on_topic = None
                if group == "answerable" and result.sources:
                    on_topic = all(any(k in s.excerpt.lower() or k in _full_text(svc, s.chunk_id) for k in item["keywords"])
                                   for s in result.sources)
                row = {"group": group, "lang": item["lang"], "status": result.status, "cited": len(result.sources),
                       "retrieved": len(result.retrieved), "seconds": round(took, 1), "on_topic": on_topic,
                       "repaired": "moved to the passage" in (result.notice or ""),
                       "unsupported": "could not be found in any" in (result.notice or ""),
                       "removed_refs": len(result.removed_references), "inference": bool(result.inference),
                       "invented_ids": len(result.unverified_identifiers),
                       "prompt_tokens": usage.get("prompt_tokens"), "generation_s": usage.get("generation_s"),
                       "prompt_eval_s": usage.get("prompt_eval_s")}
                rows.append(row)
                print(f"  [{item['lang']}] {result.status:<21} cited={row['cited']} retrieved={row['retrieved']} "
                      f"on_topic={on_topic} repaired={row['repaired']} unsupported={row['unsupported']} "
                      f"removed_refs={row['removed_refs']} invented_ids={row['invented_ids']} {took:5.1f}s  {item['q'][:44]}")
    finally:
        svc.close()
    summarize(rows)
    return 0


def _full_text(svc, chunk_id: str) -> str:
    doc_id = chunk_id.rsplit(":", 1)[0]
    for c in svc.vectors.get_chunks(doc_id):
        if c.chunk_id == chunk_id:
            return c.text.lower()
    return ""


def summarize(rows) -> None:
    print("\nSummary")
    for group in ("answerable", "unanswerable"):
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        counts = {}
        for r in g:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"  {group}: {len(g)} questions, statuses {counts}")
        for lang in ("en", "pt"):
            sub = [r for r in g if r["lang"] == lang]
            if sub:
                c = {}
                for r in sub:
                    c[r["status"]] = c.get(r["status"], 0) + 1
                print(f"    {lang}: {c}")
    ans = [r for r in rows if r["group"] == "answerable" and r["status"] == "answered"]
    if ans:
        print(f"  answered with on-topic citations only: {sum(bool(r['on_topic']) for r in ans)}/{len(ans)}")
        print(f"  answers where a citation was repaired: {sum(r['repaired'] for r in ans)}/{len(ans)}; "
              f"with an unsupported statement: {sum(r['unsupported'] for r in ans)}/{len(ans)}; "
              f"with removed model-written references: {sum(bool(r['removed_refs']) for r in ans)}/{len(ans)}; "
              f"with commands/files found in no passage: {sum(bool(r['invented_ids']) for r in ans)}/{len(ans)}")
    times = [r["seconds"] for r in rows if r["retrieved"]]
    if times:
        print(f"  seconds per question that reached the model: median {statistics.median(times):.0f}, "
              f"min {min(times):.0f}, max {max(times):.0f}")
    pe = [r["prompt_eval_s"] for r in rows if r.get("prompt_eval_s") is not None]
    gen = [r["generation_s"] for r in rows if r.get("generation_s") is not None]
    if pe and gen:
        print(f"  prompt processing median {statistics.median(pe):.0f}s, generation median {statistics.median(gen):.0f}s")
    skipped = [r for r in rows if not r["retrieved"]]
    print(f"  questions that never reached the model (no passage close enough): {len(skipped)}/{len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
