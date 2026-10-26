# Requirements: local-knowledge-ai-foundation

## Introduction

A local, standalone personal technical-knowledge assistant that answers questions through RAG over books (PDF), official documentation (web), code and notes, using Ollama, ChromaDB and FastAPI on an Ubuntu VM. This Spec covers Stages 0 to 7. The Web UI is a future stage and is outside this Spec.

Implementation is incremental. Each stage starts only after explicit user approval. The requirements below are grouped by stage.

Glossary: **Chunk** is an indexed text fragment with metadata. **Source** is the origin of a document (PDF, web, code, note). **Collection** is the set of vectors produced by a single embeddings model.

---

## Stage 0: Planning and scaffold

### R0.1 Steering and Spec
**Story:** As a user, I want an initial Steering and Spec so that Kiro keeps durable project context.
1. THE project SHALL contain `.kiro/steering/product.md`, `tech.md` and `structure.md`.
2. THE Spec SHALL contain `requirements.md`, `design.md` and `tasks.md`.

### R0.2 Minimal scaffold and independence
1. THE project SHALL have its own directory, Git repository, dependencies, tests, configuration and README.
2. THE project SHALL NOT read, modify or depend on the `llm_lab` repository.
3. THE scaffold SHALL NOT install models, start the VM, or implement ingestion, vector storage, RAG or an API.
4. THE `.gitignore` SHALL exclude `data/`, `.env`, virtual environments, caches and `.vagrant/`.

### R0.3 Configuration
1. THE application SHALL load configuration from `LKA_*` environment variables and from `.env`.
2. WHEN no value is provided, THE application SHALL use safe defaults (API on `127.0.0.1`, data in `./data`).
3. THE document, vector database and cache paths SHALL be configurable.

---

## Stage 1: PDF ingestion

### R1.1 Import
**Story:** As a user, I want to import PDFs so I can query them later.
1. WHEN the user runs `add` with a PDF, THE system SHALL extract the text preserving each page number.
2. WHEN the user runs `add-all` with a folder, THE system SHALL import PDFs recursively, processing one file at a time.
3. THE system SHALL accept a primary category and additional topics at import time.
4. THE system SHALL NOT index any file without an explicit command.

### R1.2 Problematic PDFs
1. IF a PDF has no extractable text, THEN THE system SHALL mark it as "no text" and record the reason, without indexing it.
2. IF a PDF is invalid or corrupted, THEN THE system SHALL record the error, continue with the remaining files and include the error in the statistics.

### R1.3 Chunking and embeddings
1. THE system SHALL split text into chunks with configurable size and overlap, recording the source pages whenever a chunk crosses a page boundary.
2. THE system SHALL generate embeddings locally in batches of configurable size.
3. THE system SHALL use the same embeddings model for indexing and querying.

### R1.4 Incrementality
1. THE `document_id` SHALL be stable for the same content.
2. WHEN a file with the same `file_hash` is already indexed, THE system SHALL treat it as a duplicate and not reindex it.
3. WHEN a known file has a changed hash, THE system SHALL detect the change and reindex only what changed.
4. THE system SHALL allow a document to be reindexed explicitly.
5. WHEN the user removes a document, THE system SHALL remove its vectors and metadata and SHALL NOT delete the original file.

### R1.5 Observability
1. THE system SHALL list documents and show statistics (documents, pages, chunks, errors, duration).
2. THE system SHALL persist processing errors per document.

---

## Stage 2: Semantic search

### R2.1 Retrieval
**Story:** As a user, I want to search for relevant passages so I can validate index quality before RAG.
1. WHEN it receives a question, THE retrieval layer SHALL generate its embedding with the collection's model and query ChromaDB.
2. THE retrieval layer SHALL return chunks with text, metadata, source reference and distance.
3. THE retrieval layer SHALL accept filters by document, category and topics, and a result limit.
4. WHEN the index is empty or nothing passes the relevance threshold, THE retrieval layer SHALL return an empty list without error.
5. THE retrieval layer SHALL reduce redundant results (near-identical or adjacent chunks of the same document) when possible.
6. THE retrieval layer SHALL work for Portuguese and English questions, subject to the quality of the embeddings model, which SHALL be measured in Stage 7.

### R2.2 Distance semantics
1. THE documentation SHALL describe the distance metric used and SHALL NOT treat it as a probability of relevance.
2. THE relevance threshold SHALL be configurable and calibrated by tests.

### R2.3 Embeddings compatibility
1. IF the configured embeddings model differs from the one recorded in the collection, THEN THE system SHALL refuse to index and to query, with a message stating that reindexing is required.

### R2.4 Extensibility
1. THE architecture SHALL allow adding text/hybrid search later without changing the retrieval layer's public interface.

---

## Stage 3: RAG and FastAPI

### R3.1 Grounded generation
**Story:** As a user, I want answers based on my sources, with verifiable references.
1. WHEN it receives a valid question, THE system SHALL retrieve chunks, assemble a context bounded by a configurable budget and call the local Ollama.
2. THE prompt SHALL instruct the model to answer only from the context, to distinguish facts from the sources from inferences, and not to invent references, pages or versions.
3. THE returned sources SHALL be generated from the metadata of the chunks actually retrieved, never from LLM text.
4. IF the context is insufficient, THEN THE answer SHALL state that limitation explicitly.

### R3.2 Endpoints
1. THE API SHALL expose `GET /health`, `POST /chat`, `POST /search`, `POST /documents/ingest`, `GET /documents`, `GET /documents/{document_id}` and `DELETE /documents/{document_id}`.
2. THE contracts SHALL be defined with Pydantic.
3. THE endpoints that change data SHALL require an administrative token configured through the environment.
4. THE API SHALL listen on `127.0.0.1` by default.

### R3.3 Robustness and privacy
1. THE system SHALL apply configurable input-size limits and timeouts to Ollama calls.
2. IF Ollama or the model is unavailable, THEN THE API SHALL respond with a clear error (503) without exposing internal details.
3. THE logs SHALL NOT contain full book content or question text.

---

## Stage 4: Official documentation from the web

### R4.1 Source registration
1. THE system SHALL allow registering sources with: name, start URL, allowed domains and paths, page limit, maximum depth, delay between requests, category, product/technology and version (optional).
2. THE system SHALL NOT contain a fixed list of technologies.

### R4.2 Collection
1. WHEN the user starts collection, THE collector SHALL extract the main content, removing navigation, footers and repetitive content.
2. THE collector SHALL preserve title, original URL, section, capture date and version when identifiable.
3. THE collector SHALL detect duplicates and changed pages by `content_hash`, and SHALL allow recollecting and removing a source.
4. THE system SHALL show collection statistics and errors.
5. THE system SHALL report the date of the last capture and SHALL NOT claim the content is up to date in real time.

### R4.3 Collector security
1. THE collector SHALL restrict navigation to approved domains and paths.
2. THE collector SHALL validate the URL scheme and the resolved IP, blocking loopback, private networks, link-local and cloud metadata addresses.
3. THE collector SHALL revalidate every redirect and every resolved address.
4. THE collector SHALL apply response-size, time and request-rate limits.
5. THE collector SHALL NOT bypass authentication or paywalls, execute code from pages, or send content to external APIs.
6. THE collection SHALL start only through an explicit user action.

---

## Stage 5: Multi-source library

### R5.1 Unified query
1. THE library SHALL keep `source_type` and provenance per chunk (book, web, code, note).
2. THE queries SHALL accept combinable filters by source type, category, technology, version and document.
3. THE user SHALL be able to change filters and priorities per request.

### R5.2 Conflicts between sources
1. WHEN sources diverge, THE answer SHALL present the relevant content from each source separately, with available dates and versions.
2. THE system SHALL NOT assume the most recent source holds for all versions, and SHALL NOT invent a resolution without evidence.
3. WHEN the question concerns a specific version and the documented version matches, THE official documentation SHALL take priority.

---

## Stage 6: Code indexing

### R6.1 Scope and security
1. THE system SHALL index only explicitly provided directories and SHALL treat them as read-only.
2. THE system SHALL respect `.gitignore` and configurable exclusions.
3. THE default exclusions SHALL include `.git`, virtual environments, `node_modules`, build directories, binaries, Terraform state, private keys, `.env` and files containing secrets.
4. THE system SHALL detect possible secrets before indexing and SHALL skip the file, recording the reason without recording the secret.
5. THE system SHALL NOT execute indexed files.

### R6.2 Indexing
1. THE system SHALL support Python, JavaScript, TypeScript, Shell, YAML, JSON, HCL/Terraform and Markdown.
2. THE chunks SHALL preserve repository, file path and line range.
3. THE reindexing SHALL be incremental, and files that leave the scope SHALL be removed from the index.

---

## Stage 7: Evaluation and optimization

### R7.1 Evaluation suite
1. THE system SHALL provide a reproducible evaluation with a synthetic corpus and questions, without paid APIs.
2. THE evaluation SHALL measure Recall@k, reference precision, adequacy to the context, behavior on unanswerable questions, retrieval and generation latency, ingestion time, approximate memory and disk usage.
3. THE evaluation SHALL NOT treat fluency as evidence of accuracy.

### R7.2 Configurability and recommendations
1. THE chat model, embeddings model, chunk size and overlap, k, relevance threshold, context budget, timeouts and batch size SHALL be configurable.
2. THE hybrid search and reranking SHALL be added only if the results justify them.
3. THE system SHALL NOT change installed models automatically, and SHALL present recommendations based on measurements.

---

## Non-functional requirements (all stages)

- **NFR1 Resources:** SHALL operate on a VM with ~8 GB RAM and 4 CPUs, no GPU, with the default chat model.
- **NFR2 Memory:** ingestion SHALL process one document at a time, streaming/batched, without loading the whole library into RAM.
- **NFR3 Privacy:** no data SHALL leave the machine, except explicit web collection (Stage 4).
- **NFR4 Models:** no model download SHALL happen automatically; before downloading, state size, required disk and purpose.
- **NFR5 Persistence:** data SHALL survive restarts; backup and restore SHALL be documented.
- **NFR6 Testability:** tests SHALL run without network, Ollama or models; integration tests SHALL be marked and optional.
- **NFR7 Documentation:** the README SHALL be updated each stage; no command SHALL be documented as validated without a successful run.
- **NFR8 Portability:** code SHALL be compatible with Python 3.10+.
- **NFR9 Modularity:** ingestion, retrieval, generation, storage, API and configuration SHALL be separated as defined in `structure.md`.
