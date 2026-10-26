# local-knowledge-ai

Personal technical-knowledge assistant that answers questions through RAG using free, local LLM and embeddings (Ollama), ChromaDB and FastAPI. Standalone project: it depends on no other repository and sends no data to external services.

> **Current status: Stage 1 (PDF ingestion) implemented, pending a real-Ollama run.** You can import PDFs from the command line; search, RAG and the API do not exist yet. All automated tests use a fake embeddings client. The path against a real Ollama has **not** been exercised yet (see [Known limitations](#known-limitations)). Sections that describe later stages are marked *planned*.

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
- Disk: plan for ~10–14 GB for the VM, Ollama and the two models.

## Set up the development environment (host)
Commands run successfully (Ubuntu, Python 3.12.3):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```
Result of the last full run: `139 passed, 2 deselected` (the 2 deselected are the opt-in integration tests).

If `python3 -m venv .venv` fails with "ensurepip is not available" (missing `python3-venv` package), the alternative used on this host, without `sudo`, was:
```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install -r requirements-dev.txt
```
The suite ran only on Python 3.12. Compatibility with Python 3.10 (the VM) has **not been tested** yet; the pinned dependencies declare support for it (`pymupdf` is kept below 2.0 because 2.x requires Python 3.11).

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
Configuration validation, run successfully:
```bash
vagrant validate        # "Vagrantfile validated successfully."
```
**Not run so far:** `vagrant up`, the Ollama installation and model downloads. Full plan, decisions and expected problems in [`docs/vm-plan.md`](docs/vm-plan.md).

Downloads are never automatic. Install the embeddings model yourself, inside the VM, after checking size and disk space:

| Model | Size | Purpose | Stage |
|-------|------|---------|-------|
| `nomic-embed-text` | ~274 MB | embeddings | 1 |
| `qwen2.5:3b` | ~1.9 GB | answers | 3 |

```bash
ollama pull nomic-embed-text      # manual, not yet run by the project
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

Observed on this host (no Ollama installed), all executed:
```text
$ python -m app.ingestion.cli list
No documents.
$ python -m app.ingestion.cli add /tmp/lka-smoke-pdf/sample.pdf
error: cannot reach Ollama at http://127.0.0.1:11434            # exit code 1
$ python -m app.ingestion.cli add /tmp/nope.pdf
[error] -
    file not found or not a regular file                          # exit code 1
```
A successful import against a real Ollama has not been run yet; it is covered by the opt-in integration tests (`pytest -m integration`), which need the VM, Ollama and `nomic-embed-text`.

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
- Not yet run against a real Ollama; the effective context window of `nomic-embed-text` (Ollama pages disagree: 2K or 8K tokens) and the best chunk size are unmeasured. The default of 1200 characters is conservative, and requests fail loudly instead of truncating silently.
- Ingestion time and memory use are unmeasured (Stage 7); sizes in `docs/resource-estimate.md` are estimates.
- Python 3.10 compatibility not tested.
- PDFs: password-protected files are rejected; scanned PDFs are `no_text` (no OCR). A truncated PDF may be repaired by PyMuPDF and indexed with part of its text missing, without an error.
- Chunk sizes are in characters, not tokens.
- Removing documents or switching the embeddings model requires the steps above; there is no bulk "reindex all" yet.
- `PyMuPDF` is licensed under AGPL/commercial; fine for local personal use, reassess if the project is distributed.

## Next steps
Stage 2 (semantic search), after explicit approval. Plan in [`tasks.md`](.kiro/specs/local-knowledge-ai-foundation/tasks.md).
