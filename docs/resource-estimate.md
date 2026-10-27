# Resource estimate

Model sizes come from the Ollama pages ([qwen2.5 tags](https://ollama.com/library/qwen2.5/tags), [nomic-embed-text](https://ollama.com/library/nomic-embed-text)). Figures in the "Measured in Stage 1" section were measured in the VM with synthetic text; all other figures are **planning estimates**. Stage 7 replaces the estimates with systematic measurements.

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
| Index (Chroma + SQLite) | ~17 KB per chunk measured, so ~17 MB for a book of ~1,000 chunks (the earlier 1–3 MB estimate ignored stored text, metadata and index overhead) | see measured section |
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
- Embeddings with `nomic-embed-text` on CPU: about 1 s per chunk measured (the earlier "hundreds of milliseconds" estimate was optimistic). A 300-page book (~1,000 chunks) is roughly 15–20 minutes by extrapolation.
- Ingestion runs in small batches and can be interrupted and resumed (incremental).

## Network
Downloads happen only with confirmation: the VM's APT packages, the Ollama installer and the two models. Stage 4 makes outbound traffic only to the sources you register.

## Measured in Stage 1
VM: Ubuntu 22.04, Python 3.10.12, 4 CPUs, 8 GB, no GPU, Ollama 0.40.11, `nomic-embed-text`. Synthetic PDFs.

| Measurement | Result |
|-------------|--------|
| VM RAM, idle with the embeddings model installed | 520 MB used |
| VM RAM, peak during a 20-page ingestion | 587 MB (Ollama embeddings runner ~305 MB, Ollama server ~37 MB, Python ~139 MB) |
| Python peak RSS, 100-page ingestion | 151 MB |
| Embedding time | 297 s for 300 chunks from 100 pages (~1 s per chunk) |
| Index size | 5.2 MB for 315 chunks, plus 64 KB SQLite |
| Ollama runtime on disk | 2.3 GB |
| Python virtualenv in the VM | 574 MB |
| Bare VM on the host | ~2.5 GB right after `vagrant up` |

### One real book (473 pages, 15 MB PDF)
| Measurement | Result |
|-------------|--------|
| Chunks | 944 (median 1,025 characters) |
| Embedding time | 818 s (~0.87 s per chunk, ~1.7 s per page) |
| Python peak RSS | 191 MB |
| VM RAM in use during the run | ~600 MB |
| Index | 14 MB (~15 KB per chunk) + 64 KB SQLite |

The 4–5.5 GB RAM peak in the table above includes the 3B chat model, which is not installed yet; ingestion alone used well under 1 GB. The chat model's real footprint will be measured in Stage 3.
