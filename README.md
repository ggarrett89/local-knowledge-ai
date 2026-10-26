# local-knowledge-ai

Personal technical-knowledge assistant that answers questions through RAG using free, local LLM and embeddings (Ollama), ChromaDB and FastAPI. Standalone project: it depends on no other repository and sends no data to external services.

> **Current status: Stage 0 (planning and scaffold).** Ingestion, search, RAG and the API do not exist yet. There is only configuration, Steering, Spec, documentation and a `Vagrantfile`. Sections that describe what will be built are marked *planned*.

## Goal
Query books (PDF), official documentation (web), code and notes, and get grounded answers with verifiable references (document and page, URL and section, repository, file and lines). When the sources are not enough, the system says so.

Out of scope: training models, fine-tuning, paid APIs, cloud.

## Architecture (planned)
```
CLI / FastAPI ──► ingestion ──► storage (SQLite: metadata │ ChromaDB: vectors)
                                   ▲
                 retrieval ────────┘
                     ▲
                    rag ──► Ollama (chat: qwen2.5:3b · embeddings: nomic-embed-text)
```
Details in [`.kiro/specs/local-knowledge-ai-foundation/design.md`](.kiro/specs/local-knowledge-ai-foundation/design.md).

## Project layout
| Path | Content |
|------|---------|
| `.kiro/steering/` | Durable project rules: `product.md`, `tech.md`, `structure.md` |
| `.kiro/specs/local-knowledge-ai-foundation/` | Requirements, design and task plan for Stages 0–7 |
| `app/core/` | Configuration (`config.py`). Other modules arrive in approved stages |
| `config/categories.json` | Initial categories (suggestions; new categories need no code change) |
| `docs/` | Test plan, VM plan, resource estimate |
| `tests/` | Automated tests |
| `Vagrantfile` | Ubuntu 22.04 VM, 4 CPUs, 8 GB (adjustable) |
| `data/` | Documents, index and cache. **Outside Git**; created on demand |

## Requirements
- Host: 16 GB RAM, VirtualBox and Vagrant. No GPU. See [`docs/resource-estimate.md`](docs/resource-estimate.md).
- Python 3.10+ (the VM uses 3.10; development on the host used 3.12).
- Disk: plan for ~10–14 GB for the VM, Ollama and the two models.

## Set up the development environment (host)
Commands run successfully in Stage 0 (Ubuntu, Python 3.12.3):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```
Result obtained: `11 passed`.

If `python3 -m venv .venv` fails with "ensurepip is not available" (missing `python3-venv` package), the alternative used on this host, without `sudo`, was:
```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install -r requirements-dev.txt
```
The Stage 0 suite ran only on Python 3.12. Python 3.10 compatibility has **not been tested** yet; it will be checked inside the VM.

## Configuration
Copy `.env.example` to `.env` and adjust. All variables use the `LKA_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `LKA_DATA_DIR` | `./data` | Data root |
| `LKA_DOCUMENTS_DIR`, `LKA_VECTOR_DB_DIR`, `LKA_CACHE_DIR`, `LKA_METADATA_DB_PATH` | derived from `LKA_DATA_DIR` | Individual paths |
| `LKA_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama address |
| `LKA_CHAT_MODEL` | `qwen2.5:3b` | Chat model |
| `LKA_EMBEDDING_MODEL` | `nomic-embed-text` | Embeddings model (see warning below) |
| `LKA_OLLAMA_TIMEOUT_SECONDS` | `120` | Call timeout |
| `LKA_API_HOST` / `LKA_API_PORT` | `127.0.0.1` / `8000` | API address |
| `LKA_ADMIN_TOKEN` | empty | Token for endpoints that change data (Stage 3) |

> **Warning:** the embeddings model defines the index. Changing it requires reindexing everything; the application will refuse queries with an incompatible model.

## VM
Configuration validation, run successfully:
```bash
vagrant validate        # "Vagrantfile validated successfully."
```
**Not run in Stage 0:** `vagrant up`, the Ollama installation and model downloads. Full plan, decisions and expected problems in [`docs/vm-plan.md`](docs/vm-plan.md).

Planned downloads (always with prior confirmation):

| Model | Size | Purpose | Stage |
|-------|------|---------|-------|
| `nomic-embed-text` | ~274 MB | embeddings | 1 |
| `qwen2.5:3b` | ~1.9 GB | answers | 3 |

## Usage (planned)
These sections will be filled in, with commands actually run, at the end of each stage:

| Topic | Stage |
|-------|-------|
| Import PDFs and view statistics | 1 |
| Semantic search via CLI | 2 |
| Questions with RAG and the HTTP API | 3 |
| Register official documentation (web) | 4 |
| Multi-source query and filters | 5 |
| Index code | 6 |
| Evaluation and optimization | 7 |
| Backup and restore | 1 (script) and 3 |

## Tests
See [`docs/test-plan.md`](docs/test-plan.md). Default: `pytest` (no network, no Ollama). Tests with a real Ollama will use `pytest -m integration`.

## Privacy and Git
`data/`, `.env`, databases and virtual environments are in `.gitignore`. Never version books, indexes or secrets. The Git repository is local; nothing is published without authorization.

## Known limitations (Stage 0)
- No RAG functionality exists yet.
- Model sizes come from the Ollama pages; RAM use, index disk and timings are estimates, not measurements.
- The Vagrantfile was validated syntactically, but the VM was not started.
- Python 3.10 compatibility not tested.
- `PyMuPDF` (Stage 1) is licensed under AGPL/commercial; fine for local personal use, reassess if the project is distributed.

## Next steps
Stage 1 (PDF ingestion), after explicit approval. Plan in [`tasks.md`](.kiro/specs/local-knowledge-ai-foundation/tasks.md).
