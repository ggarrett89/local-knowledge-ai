# Implementation plan: local-knowledge-ai-foundation

Rules: one stage at a time. At the end of each stage, run the tests, update the README and wait for explicit user approval before starting the next. No task downloads models without confirmation.

## Stage 0: Planning and scaffold

- [x] 0.1 Inspect destination and environment (empty directory, no Git, Python, Vagrant, VirtualBox versions) _R0.2_
- [x] 0.2 Create Steering `product.md`, `tech.md`, `structure.md` _R0.1_
- [x] 0.3 Create Spec `requirements.md`, `design.md`, `tasks.md` _R0.1_
- [x] 0.4 `git init`, `.gitignore`, `.env.example`, `requirements*.txt` _R0.2_
- [x] 0.5 `app/core/config.py` (`LKA_*` Settings) with tests _R0.3_
- [x] 0.6 Initial `config/categories.json`
- [x] 0.7 `Vagrantfile` (4 CPUs, 8 GB, port only on 127.0.0.1) and validation with `vagrant validate`
- [x] 0.8 `docs/test-plan.md`, `docs/vm-plan.md`, `docs/resource-estimate.md`
- [x] 0.9 Initial README
- [x] 0.10 Translate all project documentation to English (user request)
- [ ] 0.11 **User approval** (control point)

## Stage 1: PDF ingestion

Depends on: Stage 0 approved. Requirements: R1.1–R1.5, R2.3 (storage part), NFR2.

- [ ] 1.1 Confirm environment with the user: install Ollama in the VM and download `nomic-embed-text` (~274 MB). State size and disk before downloading.
- [ ] 1.2 Domain and errors: `app/core/models.py`, `app/core/errors.py`
- [ ] 1.3 SQLite `MetadataStore`: schema, initial migration, category seeding, CRUD, errors, statistics
- [ ] 1.4 Chroma `VectorStore`: collection per model, model/dimension validation, upsert, delete by document
- [ ] 1.5 Ollama `EmbeddingClient` (httpx, batches, timeout) and a fake one for tests
- [ ] 1.6 PDF extractor (PyMuPDF, page streaming, `no_text`/invalid detection)
- [ ] 1.7 Chunking with overlap and page provenance
- [ ] 1.8 Ingestion service: duplicate, change, incremental, reindex, remove (without deleting the file)
- [ ] 1.9 CLI `python -m app.ingestion.cli` (`add`, `add-all`, `list`, `remove`, `reindex`, `stats`)
- [ ] 1.10 Tests: valid/invalid/no-text synthetic PDFs, duplicate, change, removal, persistence after reopening, embeddings failure, recursive `add-all`
- [ ] 1.11 Measure and document the effective `nomic-embed-text` window and the default chunk size
- [ ] 1.12 Update README, run tests, report results
- [ ] 1.13 **User approval**

## Stage 2: Semantic search

Depends on: Stage 1. Requirements: R2.1–R2.4.

- [ ] 2.1 `Retriever` with filters (document, category, topics), `k`, overfetch
- [ ] 2.2 Configurable relevance threshold and deduplication
- [ ] 2.3 Reference formatting from `Provenance`
- [ ] 2.4 CLI `python -m app.retrieval.cli search`
- [ ] 2.5 Document cosine distance and calibrate the threshold with a synthetic PT/EN corpus
- [ ] 2.6 Tests: empty database, no results, filters, incompatible model, embeddings unavailable, page references
- [ ] 2.7 Update README, run tests, report
- [ ] 2.8 **User approval**

## Stage 3: RAG and FastAPI

Depends on: Stage 2. Requirements: R3.1–R3.3.

- [ ] 3.1 Confirm the `qwen2.5:3b` download (~1.9 GB) with the user
- [ ] 3.2 Ollama `ChatClient` and a fake one for tests
- [ ] 3.3 Context building with budget and a prompt with anti-hallucination rules
- [ ] 3.4 `RagService`: sources from metadata, insufficient-context fallback, citation validation
- [ ] 3.5 API: `GET /health`, `POST /search`, `POST /chat`, `POST /documents/ingest`, `GET /documents`, `GET /documents/{id}`, `DELETE /documents/{id}`
- [ ] 3.6 Administrative token, path restriction, input limits, timeouts, error mapping, logs without content
- [ ] 3.7 Tests: Ollama unavailable, missing model, empty context, real sources, missing token, path traversal, input above the limit
- [ ] 3.8 Update README, run tests, report
- [ ] 3.9 **User approval**

## Stage 4: Official documentation from the web

Depends on: Stages 1–3. Requirements: R4.1–R4.3.

- [ ] 4.1 Choose and justify the HTML extraction library (approval before installing)
- [ ] 4.2 Source registration (schema, CLI, endpoint)
- [ ] 4.3 `UrlGuard`: scheme, allowlist, DNS, blocked IPs, redirect revalidation
- [ ] 4.4 Fetcher with size, time, rate limits and `robots.txt`
- [ ] 4.5 Main-content extraction, sections, version when identifiable
- [ ] 4.6 Duplicate/change detection, recollection and source removal
- [ ] 4.7 Tests with a fake local HTTP server: SSRF, redirect to a disallowed host, large response, malformed HTML
- [ ] 4.8 Update README, run tests, report
- [ ] 4.9 **User approval**

## Stage 5: Multi-source library

Depends on: Stage 3 and at least one of 4 or 6. Requirements: R5.1–R5.2.

- [ ] 5.1 Filters by source type, technology, version, category and document in `/search` and `/chat`
- [ ] 5.2 Per-request configurable priorities and the official-documentation priority rule by version
- [ ] 5.3 Context grouped by source and a prompt for comparison and divergence
- [ ] 5.4 Tests for book vs documentation conflict, different versions, combined filters
- [ ] 5.5 Update README, run tests, report
- [ ] 5.6 **User approval**

## Stage 6: Code indexing

Depends on: Stages 1–3. Requirements: R6.1–R6.2.

- [ ] 6.1 Directory registration and explicit scope
- [ ] 6.2 `.gitignore` filter, default and configurable exclusions, binary detection
- [ ] 6.3 Secret scanner (file skipped, secret never recorded)
- [ ] 6.4 Chunking with source lines
- [ ] 6.5 Incremental reindexing and removal of files that leave the scope
- [ ] 6.6 Tests with a synthetic repository containing `.env`, a private key, `node_modules`, `terraform.tfstate`
- [ ] 6.7 Update README, run tests, report
- [ ] 6.8 **User approval**

## Stage 7: Evaluation and optimization

Depends on: Stages 2 and 3. Requirements: R7.1–R7.2.

- [ ] 7.1 Synthetic corpus and questions with expected results (PT and EN, including unanswerable)
- [ ] 7.2 Retrieval metrics: Recall@k, reference precision, latency
- [ ] 7.3 Generation metrics: adherence to context, refusal without evidence, latency
- [ ] 7.4 System metrics: ingestion time, memory, disk
- [ ] 7.5 Parameter sweep (chunk, overlap, k, threshold, context)
- [ ] 7.6 Evaluate hybrid search and reranking only if the data justify them
- [ ] 7.7 Report with measurement-based recommendations
- [ ] 7.8 Update README, run tests, report
- [ ] 7.9 **User approval**

## Future stage (outside this Spec)

Local web UI, after the API is validated.
