# Resource estimate

Model sizes come from the Ollama pages ([qwen2.5 tags](https://ollama.com/library/qwen2.5/tags), [nomic-embed-text](https://ollama.com/library/nomic-embed-text)). All other figures are **planning estimates, not measurements**. Stage 7 replaces them with measurements.

## Host environment inspected (2026-10-26)
- 12 logical CPUs, 15 GiB RAM (about 10 GiB available at inspection time).
- `/home` partition: 293 GB, 39 GB free (87% used).
- Vagrant 2.4.9, VirtualBox 7.2.16, box `ubuntu/jammy64` already in the local cache.
- Ollama is not installed on the host. It will run inside the VM.

## Disk
| Item | Approximate size | Note |
|------|------------------|------|
| Box `ubuntu/jammy64` | already cached | no new download expected |
| VM disk (dynamic VMDK) | 4–6 GB initially, grows with use | box maximum virtual disk: ~40 GB (to confirm) |
| Ollama binary and runtime | ~1–2 GB | manual install |
| `qwen2.5:3b` (chat) | 1.9 GB | Stage 3 |
| `nomic-embed-text` (embeddings) | 274 MB | Stage 1 |
| Index (Chroma + SQLite) | ~1–3 MB per 300-page book (estimate: ~600 chunks × 768 floats × 4 B ≈ 1.8 MB, plus HNSW overhead) | measure in Stage 1 |
| Original PDFs | depends on the library | kept outside Git |
| **Initial total** | **~10–14 GB** | fits in the 39 GB free, with a reduced margin |

Tip: if host free disk falls below ~15 GB, move the VirtualBox VM folder to another volume before continuing.

## RAM (8 GB VM)
| Component | Estimate |
|-----------|----------|
| Ubuntu Server OS + services | 0.4–0.6 GB |
| Ollama with `qwen2.5:3b` (~1.9 GB weights + KV cache at 4K tokens context + buffers) | ~2.5–3.5 GB |
| `nomic-embed-text` loaded (~0.3–0.6 GB) | 0.5 GB |
| FastAPI API + ChromaDB + PyMuPDF (one PDF ingested at a time) | 0.5–1 GB |
| **Expected peak** | **~4–5.5 GB** |
| Headroom | ~2.5–4 GB |

Notes:
- Ollama keeps models in memory for a while after use (default: a few minutes). Chat and embeddings can coexist in RAM; the headroom above accounts for that.
- The effective context (`num_ctx`) has a large effect on memory. The project default will limit context to about 4K tokens (to be decided in Stage 3).
- If VM RAM is insufficient, alternatives in order of preference: reduce context, use `qwen2.5:1.5b` (986 MB), reduce `LKA_VM_MEMORY_MB`, and keep only one model loaded.

## Host RAM
With the VM at 8 GB, about 7–8 GB remains for the host. The inspection showed ~10 GiB available with applications open. This is feasible but tight. Recommendation: close heavy applications during large ingestion runs, or use `LKA_VM_MEMORY_MB=6144` (sufficient for the 3B model according to the estimate above).

## CPU and time (no GPU)
Orders of magnitude, to be measured in Stage 7:
- Generation with a 3B model on 4 CPUs: a few tokens per second (answers take tens of seconds).
- Embeddings with `nomic-embed-text` on CPU: hundreds of milliseconds per chunk. A 300-page book may take from minutes to tens of minutes to index.
- Ingestion runs in small batches and can be interrupted and resumed (incremental).

## Network
Downloads happen only with confirmation: the VM's APT packages, the Ollama installer and the two models. Stage 4 makes outbound traffic only to the sources you register.
