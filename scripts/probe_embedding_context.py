"""Measure how many characters the embeddings model accepts per input.

Needs a running Ollama with the configured embeddings model installed. It sends inputs of
growing size with truncate=false, so an over-long input fails instead of being cut silently.
Synthetic text only; nothing from your library is read.

    python scripts/probe_embedding_context.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clients.ollama import OllamaEmbeddingClient  # noqa: E402
from app.core.config import load_settings  # noqa: E402
from app.core.errors import EmbeddingError  # noqa: E402

PROSE = (
    "Terraform modules let you package and reuse infrastructure code. A module is a container "
    "for multiple resources that are used together, and the root module is the directory where "
    "you run terraform apply. Kubernetes schedules pods onto nodes, while the kubelet reports "
    "node status to the API server. "
)
# Numeric identifiers tokenize into many short tokens: close to the worst case for a character limit.
DENSE = " ".join(f"p1t{i:04d}" for i in range(20000))
SIZES = (1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000, 32000)


def probe(client: OllamaEmbeddingClient, label: str, source: str) -> int:
    print(f"== {label}")
    largest_ok = 0
    for size in SIZES:
        text = (source * (size // len(source) + 1))[:size]
        started = time.perf_counter()
        try:
            vector = client.embed_documents([text])[0]
        except EmbeddingError as exc:
            print(f"  {size:>6} chars  FAIL  {str(exc)[:110]}")
            break
        print(f"  {size:>6} chars  OK    dim={len(vector)}  {time.perf_counter() - started:5.2f}s")
        largest_ok = size
    print(f"  largest size that worked: {largest_ok} chars")
    return largest_ok


def main() -> int:
    s = load_settings()
    client = OllamaEmbeddingClient(
        s.ollama_base_url, s.embedding_model, timeout_seconds=max(s.ollama_timeout_seconds, 300.0),
        document_prefix=s.embedding_document_prefix, query_prefix=s.embedding_query_prefix,
    )
    try:
        probe(client, "prose (typical book text)", PROSE)
        probe(client, "dense numeric tokens (near worst case)", DENSE)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
