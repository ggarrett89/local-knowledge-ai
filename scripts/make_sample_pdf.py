"""Create a synthetic text PDF for trying the ingestion CLI. No real book content is used.

    python scripts/make_sample_pdf.py /tmp/samples/sample.pdf --pages 5 --title "Sample Book"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

SENTENCES = [
    "A Terraform module packages related resources so that they can be reused across environments.",
    "The Kubernetes scheduler assigns each pod to a node based on resource requests and constraints.",
    "Idempotent operations produce the same result no matter how many times they are applied.",
    "A message broker decouples producers from consumers and absorbs bursts of traffic.",
    "Service level objectives turn reliability goals into numbers that teams can track.",
    "Consistent hashing limits how many keys move when a node joins or leaves a cluster.",
    "Retries with exponential backoff and jitter prevent synchronized load spikes after a failure.",
    "A write-ahead log records changes before they are applied so that a crash can be recovered.",
]


def page_text(page: int, chars: int) -> str:
    out = [f"Section {page}."]
    i = page
    while sum(len(s) + 1 for s in out) < chars:
        out.append(f"{SENTENCES[i % len(SENTENCES)]} (note {page}.{i})")
        i += 1
    return " ".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--chars-per-page", type=int, default=2500)
    parser.add_argument("--title", default="Sample Book")
    args = parser.parse_args()

    doc = pymupdf.open()
    for n in range(1, args.pages + 1):
        page = doc.new_page()
        rc = page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), page_text(n, args.chars_per_page), fontsize=10)
        if rc < 0:
            raise SystemExit("--chars-per-page does not fit on a page")
    doc.set_metadata({"title": args.title})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(args.output))
    doc.close()
    print(f"wrote {args.output} ({args.pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
