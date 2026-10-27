# Structure

Modules are created when the stage that uses them is approved. Do not create empty packages.

```
app/
  core/        configuration, logging, common errors, domain models (no external I/O)
  clients/     external service clients (Ollama) implementing the interfaces in core
  storage/     SQLite (metadata) and ChromaDB (vectors) behind interfaces
  ingestion/   pdf/, web/, code/ + shared chunking and CLI
  retrieval/   semantic search, filters, deduplication, CLI
  rag/         context assembly, prompt, answer verification, citations (uses the chat client from clients/)
  api/         FastAPI: routes, Pydantic schemas, authentication, error mapping
  services.py  composition root: builds stores, clients and services for the API and the CLIs
config/        versionable configuration files without secrets (e.g. categories.json)
data/          documents/, vector_db/, cache/ (outside Git)
docs/          test plan, VM, resource estimate
scripts/       operations utilities (backup, restore, setup helpers)
tests/         mirrors the structure of app/
.kiro/         steering and specs
```

## Responsibilities and boundaries
- `core` does not import from any other package in the project.
- `storage` knows nothing about FastAPI, Ollama or file formats.
- `ingestion` produces chunks with metadata and uses `storage`; it knows nothing about `rag` or `api`.
- `retrieval` queries `storage` and the embeddings client; it does not generate text.
- `rag` uses `retrieval` and the chat client; it does not access Chroma directly.
- `api` and the CLIs are thin layers over services; no business logic in routes.
- External clients (Ollama) sit behind a small interface (protocol in `core`, implementation in `clients`) so they can be mocked and models swapped without touching RAG logic. `clients` imports only from `core`.
- Sources (PDF, web, code) implement a common "extractor" contract that returns text + provenance; chunking and persistence are shared.

## Conventions
- Code, identifiers, comments and all documentation (README, docs, Steering, Specs) in English.
- Type hints on public code. Data models with Pydantic; configuration only through `app.core.config.Settings`.
- No implicit global state: dependencies are injected (Settings, repositories, clients).
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, environment variables `LKA_*`.
- `document_id` is stable and derived from content/origin, not from a timestamp. `chunk_id = document_id + chunk_index`.
- Every chunk persists provenance (page, URL/section, path/lines). An unknown field is absent, never invented.
- Explicit domain errors (e.g. `EmbeddingModelMismatch`, `OllamaUnavailable`), converted to HTTP responses only in the `api` layer.
- Small, incremental changes; each stage updates the README and tests.
