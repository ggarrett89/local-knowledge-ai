# local-knowledge-ai

Personal technical-knowledge assistant that answers questions through RAG using free, local LLM and embeddings (Ollama), ChromaDB and FastAPI. Standalone project: it depends on no other repository and sends no data to external services.

> **Current status: Stage 1 (PDF ingestion) implemented and validated in the VM, awaiting approval.** You can import PDFs from the command line; search, RAG and the API do not exist yet. The default test run uses a fake embeddings client; the import path was also exercised against a real Ollama in the VM with synthetic PDFs (see [Measured results](#measured-results-stage-1)). Sections that describe later stages are marked *planned*.

## Goal
Query books (PDF), official documentation (web), code and notes, and get grounded answers with verifiable references (document and page, URL and section, repository, file and lines). When the sources are not enough, the system says so.

Out of scope: training models, fine-tuning, paid APIs, cloud.

## Architecture
```
CLI ──► ingestion ──► storage (SQLite: metadata │ ChromaDB: vectors)
            │
            └──► clients/ollama (embeddings)          [Stage 1: implemented]

retrieval, rag, api                                    [Stages 2-3: planned]
```
Details in [`.kiro/specs/local-knowledge-ai-foundation/design.md`](.kiro/specs/local-knowledge-ai-foundation/design.md).

## Project layout
| Path | Content |
|------|---------|
| `.kiro/steering/` | Durable project rules: `product.md`, `tech.md`, `structure.md` |
| `.kiro/specs/local-knowledge-ai-foundation/` | Requirements, design and task plan for Stages 0–7 |
| `app/core/` | Settings, domain models, errors, interfaces (no I/O) |
| `app/clients/` | Ollama embeddings client |
| `app/storage/` | SQLite metadata store and ChromaDB vector store |
| `app/ingestion/` | PDF extractor, chunking, ingestion service and CLI |
| `config/categories.json` | Initial categories (suggestions; new categories need no code change) |
| `docs/` | Test plan, VM plan, resource estimate |
| `tests/` | Automated tests (mirror `app/`) |
| `Vagrantfile` | Ubuntu 22.04 VM, 4 CPUs, 8 GB (adjustable) |
| `data/` | Metadata database and vector index. **Outside Git**; created on demand |

## Requirements
- Host: 16 GB RAM, VirtualBox and Vagrant. No GPU. See [`docs/resource-estimate.md`](docs/resource-estimate.md).
- Python 3.10+ (the VM uses 3.10; development on the host used 3.12).
- Disk: plan for ~10–14 GB for the VM, Ollama and the two models (the bare VM measured ~2.5 GB on the host).

## Set up the development environment (host)
Commands run successfully (Ubuntu, Python 3.12.3):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```
Result of the last full run: `141 passed, 2 deselected` (the 2 deselected are the opt-in integration tests).

If `python3 -m venv .venv` fails with "ensurepip is not available" (missing `python3-venv` package), the alternative used on this host, without `sudo`, was:
```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install -r requirements-dev.txt
```
On the host the suite ran on Python 3.12. It also ran inside the VM on Python 3.10.12 (see [VM](#vm)). `pymupdf` is kept below 2.0 because 2.x requires Python 3.11.

## Configuration
Copy `.env.example` to `.env` and adjust. All variables use the `LKA_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `LKA_DATA_DIR` | `./data` | Data root |
| `LKA_DOCUMENTS_DIR`, `LKA_VECTOR_DB_DIR`, `LKA_CACHE_DIR`, `LKA_METADATA_DB_PATH` | derived from `LKA_DATA_DIR` | Individual paths |
| `LKA_CATEGORIES_FILE` | `config/categories.json` | Categories registered on first use |
| `LKA_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama address (keep it local: document text is sent to it) |
| `LKA_CHAT_MODEL` | `qwen2.5:3b` | Chat model (Stage 3) |
| `LKA_EMBEDDING_MODEL` | `nomic-embed-text` | Embeddings model (see warning below) |
| `LKA_EMBEDDING_DOCUMENT_PREFIX` / `LKA_EMBEDDING_QUERY_PREFIX` | `search_document: ` / `search_query: ` | Task prefixes required by `nomic-embed-text`; set both to empty for models that do not use them |
| `LKA_OLLAMA_TIMEOUT_SECONDS` | `120` | Call timeout |
| `LKA_CHUNK_SIZE` / `LKA_CHUNK_OVERLAP` | `1200` / `200` | Chunk size and overlap, **in characters** (overlap must be below half the size) |
| `LKA_EMBED_BATCH_SIZE` | `16` | Chunks sent to Ollama per request |
| `LKA_API_HOST` / `LKA_API_PORT` | `127.0.0.1` / `8000` | API address (Stage 3) |
| `LKA_ADMIN_TOKEN` | empty | Token for endpoints that change data (Stage 3) |

> **Warning:** the embeddings model **and its prefixes** define the index. The index records them, and the application refuses to index or query when they differ. To switch models, build a new index in a fresh `LKA_DATA_DIR` and add your documents again.

## VM
Run successfully:
```bash
vagrant validate        # "Vagrantfile validated successfully."
vagrant up              # first run created the VM and installed basic APT packages
```
Observed inside the VM after `vagrant up`: Ubuntu 22.04, Python 3.10.12, 4 CPUs, ~7.9 GB RAM visible, 39 GB virtual disk (1.9 GB used), `/vagrant` mounted with the project, `/home/vagrant/lka-data` created, no Ollama. On the host the VM uses about 2.5 GB of disk in VirtualBox's default machine folder, and the API port is forwarded only to `127.0.0.1:8000`.

Then, inside the VM (all run successfully):
```bash
python3 -m venv ~/venv && . ~/venv/bin/activate
cd /vagrant && pip install -r requirements-dev.txt     # ~574 MB venv
pytest                                                  # 141 passed, 2 deselected (Python 3.10.12)
pytest -m integration                                   # 2 passed against the real Ollama
```
Ollama 0.40.11 was installed in the VM with the official installer (reviewed first; it also needs the `zstd` APT package, installed with `sudo apt-get install zstd`). It runs as a systemd service, CPU-only, and listens on `127.0.0.1:11434` inside the VM only; the port is not reachable from the host.

Full plan, decisions, the exported appliance (OVA) and expected problems in [`docs/vm-plan.md`](docs/vm-plan.md). Stop the VM with `vagrant halt` when you are not using it; it reserves 8 GB of the host's RAM while running.

Downloads are never automatic. Install the embeddings model yourself, inside the VM, after checking size and disk space:

| Model | Size | Purpose | Stage |
|-------|------|---------|-------|
| `nomic-embed-text` | ~274 MB | embeddings | 1 |
| `qwen2.5:3b` | ~1.9 GB | answers | 3 |

```bash
ollama pull nomic-embed-text      # run manually in the VM; on the first attempt the download failed at 99% with a network timeout and succeeded on retry
```
If the model is missing, the CLI says so and shows this command; it never pulls anything.

## Importing PDFs (Stage 1)
Every command is an explicit action; nothing is indexed automatically. Original files are only read, never modified or deleted.

```bash
python -m app.ingestion.cli add /path/book.pdf --category containers-kubernetes --topic networking --topic security
python -m app.ingestion.cli add-all /path/to/folder --category general     # recursive, one file at a time
python -m app.ingestion.cli list [--category C] [--topic T] [--status indexed|no_text|error]
python -m app.ingestion.cli reindex DOCUMENT_ID
python -m app.ingestion.cli remove DOCUMENT_ID [--yes]                     # asks for confirmation; keeps the file
python -m app.ingestion.cli stats
```
- `--category` takes any slug (`a-z`, `0-9`, `-`). A new slug is registered automatically; no code change is needed. Without it, a known document keeps its category, and new ones get `general`.
- `--topic` can be repeated. `--force` reindexes even if the content is already indexed.
- A `document_id` looks like `pdf-` plus 16 hex characters of the file's SHA-256, so it is stable for the same content.

Outcomes printed per file:

| Outcome | Meaning |
|---------|---------|
| `indexed` | New document indexed |
| `unchanged` | Same file, same content, already indexed (nothing is re-embedded) |
| `duplicate` | Same content already indexed from another path |
| `updated` | The file at this path changed; the new version replaced the old one |
| `reindexed` | Forced reindex, or a retry after an earlier error |
| `no_text` | No extractable text (for example a scanned PDF); recorded, not indexed. OCR is not supported |
| `error` | See the message; recorded in the database. Run `add` again to retry |

Exit code is `0` on success and `1` when any file failed or the command could not run.

Behavior worth knowing:
- Incremental means per document: a changed file is reindexed as a whole.
- A failed document leaves no partial chunks. If Ollama is down or the model is missing, `add-all` stops at the first file instead of failing every file.
- `list`, `stats` and `remove` work even when Ollama is down or the configured embeddings model differs from the index.

Observed in the VM with the real Ollama and synthetic PDFs, all executed:
```text
$ python -m app.ingestion.cli add ~/samples/small.pdf --category containers-kubernetes --topic networking --topic security
[indexed] pdf-cefa75d00aace08f "Sample Book" pages=5 chunks=15 14.6s
$ python -m app.ingestion.cli add ~/samples/small.pdf
[unchanged] pdf-cefa75d00aace08f "Sample Book" pages=5
    already indexed (status=indexed)
$ LKA_EMBEDDING_DOCUMENT_PREFIX="" python -m app.ingestion.cli add ~/samples/other.pdf
error: the index was built with a different embeddings configuration (...); indexing and querying are refused. ...   # exit code 1
$ LKA_EMBEDDING_MODEL=does-not-exist python -m app.ingestion.cli add ~/samples/other.pdf     # on a fresh data dir
error: model 'does-not-exist' is not installed in Ollama; install it explicitly with: ollama pull does-not-exist   # exit code 1
$ sudo systemctl stop ollama; python -m app.ingestion.cli add ~/samples/other.pdf
error: cannot reach Ollama at http://127.0.0.1:11434                                          # exit code 1; `list` and `stats` still work
$ sudo systemctl start ollama; python -m app.ingestion.cli add ~/samples/other.pdf          # retries the errored document
[reindexed] pdf-ca94a3dd1a3db6d5 "Other Book" pages=2 chunks=6 6.6s
```
`reindex` kept the category and topics, `remove` asked for confirmation (answering `n` aborted, `--yes` removed it), and the original PDF stayed in place.

### Measured results (Stage 1)
Measured in the VM (Ubuntu 22.04, Python 3.10.12, 4 CPUs, no GPU, Ollama 0.40.11, `nomic-embed-text`) with **synthetic** text. Real books will differ; treat these as orders of magnitude.

| Measurement | Result |
|-------------|--------|
| Context limit (`ollama show`) | 2048 tokens. Embedding 8,000 characters of prose worked and 12,000 failed. Dense numeric text worked at 2,000 characters and failed at 3,000 |
| Default chunk size | 1,200 characters (200 overlap), kept: it is below the worst case measured above |
| Embedding speed | ~1 s per chunk (~1,000 characters); 100 synthetic pages (250,000 characters) became 300 chunks in 297 s |
| Python memory during ingestion | peak RSS 151 MB for 100 pages (documents are streamed) |
| VM RAM during ingestion | 520 MB idle, 587 MB peak (embeddings runner ~305 MB, Ollama server ~37 MB, Python ~139 MB) |
| Index size | 5.2 MB of vectors plus 64 KB of metadata for 315 chunks |
| Disk used by Ollama | 2.3 GB (`/usr/local/lib/ollama`), plus 274 MB for the model |

The measurement scripts are in `scripts/` (`probe_embedding_context.py`, `make_sample_pdf.py`) so the numbers can be reproduced.

#### One real book
One real, text-based technical book (473 pages, 15 MB, not encrypted) was then imported in the VM with `add` (category `iac-automation`, one topic), run in the background:

| Measurement | Result |
|-------------|--------|
| Result | `[indexed]`, 944 chunks, exit code 0, `list`/`stats` consistent (944 chunks stored, 0 errors) |
| Time | 818 s (13.6 min): ~0.87 s per chunk, ~1.7 s per page |
| Memory | Python peak RSS 191 MB; VM RAM about 600 MB in use |
| Index size | 14 MB of vectors plus 64 KB of metadata (~15 KB per chunk) |
| Chunk sizes | 451 to 1,199 characters, median 1,025; none above the 1,200 limit |

Checked against the source PDF afterwards (read-only script, not kept in the repository):
- Chunk indexes are contiguous, every chunk has `page_start`/`page_end` within 1..473, pages never go backwards, and all 473 pages are covered by at least one chunk.
- No replacement characters (U+FFFD) in the text; 0.3% of characters are non-ASCII.
- Page references: for 60 randomly sampled chunks, a 50-character excerpt from the middle of the chunk was searched on the claimed pages. 58 were found on a single page. The other 2 span a page boundary, and for both the whole chunk text lies inside the two claimed pages. So the page references were correct in all 60 samples.

This is one book of one kind (clean text layer). Books with complex layouts, multi-column text, code listings or scanned pages may behave differently. Search quality is still unmeasured because search does not exist yet.

## Backup and restore (planned)
Not yet exercised. The state lives entirely in `LKA_DATA_DIR` (SQLite file plus the Chroma directory), so a backup is a copy of that directory while no ingestion is running. A script and a tested restore procedure are planned for a later stage.

## Usage of later stages (planned)
| Topic | Stage |
|-------|-------|
| Semantic search via CLI | 2 |
| Questions with RAG and the HTTP API | 3 |
| Register official documentation (web) | 4 |
| Multi-source query and filters | 5 |
| Index code | 6 |
| Evaluation and optimization | 7 |

## Tests
See [`docs/test-plan.md`](docs/test-plan.md). Default: `pytest` (no network, no Ollama, no models). Tests with a real Ollama: `pytest -m integration`.

## Privacy and Git
`data/`, `.env`, databases and virtual environments are in `.gitignore`. Never version books, indexes or secrets. Logs and stored error messages contain ids, counts and durations, never document text. ChromaDB telemetry is disabled.

## Known limitations
- No search, RAG or API yet (Stages 2–3).
- Measurements used synthetic text plus one real book, one document at a time, on CPU only. Retrieval quality is not measured yet (no search exists).
- Embedding is slow on CPU (~0.9 s per chunk): the real 473-page book took 13.6 minutes.
- Chunk sizes are in characters; numeric- or code-heavy text uses more tokens per character, so do not raise `LKA_CHUNK_SIZE` above ~2,000 without re-running `scripts/probe_embedding_context.py`.
- PDFs: password-protected files are rejected; scanned PDFs are `no_text` (no OCR). A truncated PDF may be repaired by PyMuPDF and indexed with part of its text missing, without an error.
- Removing documents or switching the embeddings model requires the steps above; there is no bulk "reindex all" yet.
- `PyMuPDF` is licensed under AGPL/commercial; fine for local personal use, reassess if the project is distributed.

## Next steps
Stage 2 (semantic search), after explicit approval. Plan in [`tasks.md`](.kiro/specs/local-knowledge-ai-foundation/tasks.md).
