# Design: local-knowledge-ai-foundation

## 1. Overview

```
        CLI (add, search)        FastAPI (Stage 3)
               \                     /
                \                   /
   ingestion ──► storage ◄── retrieval ◄── rag ──► Ollama (chat)
      │            │  │          │
      │            │  └─ ChromaDB (vectors + scalar metadata)
      │            └──── SQLite   (documents, categories, topics, errors)
      └────────────────────────► Ollama (embeddings)
```

Everything runs inside the Ubuntu VM: one Python process (API/CLI) and the Ollama service. No other services.

## 2. Architecture decisions

| # | Decision | Alternatives | Rationale |
|---|----------|--------------|-----------|
| D1 | SQLite holds document metadata; Chroma holds vectors and scalar per-chunk metadata | Chroma only | Chroma accepts only scalar metadata and has limited filters. Lists (topics), statistics, errors and status fit SQL better. Meets "metadata in the database, independent of the physical directory". |
| D2 | Category/topic/version filters are resolved in SQLite into a set of `document_id`s; Chroma receives `where document_id $in [...]` | Duplicate topics as strings in Chroma | Avoids stale duplicated metadata when the user corrects a category. For very large sets, filter directly by `source_type`/`document_id`. |
| D3 | One Chroma collection per embeddings model, named `lka_<model-slug>`; model and dimension stored in collection metadata | Collection per category | Vectors from different models are not comparable. Categories and source types are filters, not collections. |
| D4 | `cosine` metric set at collection creation (`hnsw:space`) | Default L2 | More stable for text embeddings. Distance is a similarity measure, not a probability. The relevance threshold is calibrated in Stage 7. |
| D5 | Categories are free slugs (`^[a-z0-9][a-z0-9-]{0,63}$`). `config/categories.json` only seeds the catalog in SQLite. | Enum in code | New categories without code changes. |
| D6 | Ollama clients behind `EmbeddingClient` and `ChatClient` protocols | Call `httpx` directly | Enables mocks in tests and swapping model/runtime without touching RAG. |
| D7 | Ollama HTTP client via `httpx` (added in Stage 1) | `ollama` library | One small dependency, timeout control. To be re-evaluated in Stage 1. |
| D8 | Character-based chunking with overlap, respecting page boundaries; unit configurable | Model tokenizer | Avoids a heavy dependency. Conservative default size, validated in Stage 7. |
| D9 | Python 3.10+ | 3.12 | VM `ubuntu/jammy64` ships 3.10. |
| D10 | Index data on the VM's local disk (`/home/vagrant/lka-data`), not a synced folder | Synced `./data` | Chroma/SQLite over `vboxsf` have locking and performance problems. Original books may stay in the synced folder (read-only use). |
| D11 | Ollama installed manually inside the VM | Automatic provisioning | Avoids download and install without confirmation. The Vagrantfile installs only basic packages. |

## 3. Data model

### 3.1 SQLite (`metadata.db`)

- `documents`: `document_id` (PK), `title`, `source_type` (`pdf`, `web`, `code`, `note`), `category`, `source_uri`, `file_path`, `file_hash`, `version`, `language`, `status` (`indexed`, `no_text`, `error`, `removed`), `page_count`, `chunk_count`, `captured_at`, `updated_at`, `indexed_at`, `embedding_model`.
- `document_topics`: (`document_id`, `topic`).
- `categories`: `slug` (PK), `description`.
- `ingestion_errors`: `id`, `document_id` or `source_uri`, `stage`, `message` (no document content), `created_at`.
- `schema_meta`: schema version, for simple migrations.

Stages 4 and 6 add their own tables (`web_sources`, `code_sources`) through additive migrations.

### 3.2 Per-chunk metadata (Chroma)

`chunk_id` (= `document_id:chunk_index`), `document_id`, `source_type`, `chunk_index`, `content_hash`, `page_start`, `page_end` (PDF), `section`, `url` (web), `repo`, `path`, `line_start`, `line_end` (code), `language`. Unknown fields are omitted, never filled with invented values.

### 3.3 Identifiers

- `document_id`: PDF = `pdf-` + first 16 hex chars of the content SHA-256. Web = hash of the canonical URL. Code = hash of `repo + path`.
- `file_hash`/`content_hash`: SHA-256.
- Reindexing replaces a document's chunks as atomically as possible: write the new ones, then remove the old ones.

## 4. Interfaces between modules (contracts)

```python
# app/core/models.py
class Provenance(BaseModel):
    source_type: Literal["pdf", "web", "code", "note"]
    document_id: str
    title: str
    page_start: int | None = None
    page_end: int | None = None
    url: str | None = None
    section: str | None = None
    repo: str | None = None
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None

class Chunk(BaseModel):
    chunk_id: str
    text: str
    content_hash: str
    provenance: Provenance

class SearchHit(BaseModel):
    chunk: Chunk
    distance: float          # cosine distance; NOT a probability

# app/ingestion/base.py
class Extractor(Protocol):
    def extract(self, source: Path | str) -> Iterator[Section]: ...   # streaming, with provenance

# app/ingestion/embeddings.py (client interface, Ollama implementation)
class EmbeddingClient(Protocol):
    model: str
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

# app/storage/vector_store.py
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def query(self, vector: Sequence[float], k: int, document_ids: Sequence[str] | None) -> list[SearchHit]: ...
    def delete_document(self, document_id: str) -> int: ...

# app/storage/metadata_store.py
class MetadataStore(Protocol):
    def upsert_document(self, doc: DocumentRecord) -> None: ...
    def find_by_hash(self, file_hash: str) -> DocumentRecord | None: ...
    def list_documents(self, *, category=None, topics=None, source_type=None, version=None) -> list[DocumentRecord]: ...
    def remove_document(self, document_id: str) -> None: ...
    def record_error(self, ...) -> None: ...

# app/retrieval/service.py
class Retriever:
    def search(self, query: str, *, k: int, filters: SearchFilters) -> list[SearchHit]: ...

# app/rag/chat.py
class ChatClient(Protocol):
    def complete(self, messages: Sequence[Message]) -> str: ...
class RagService:
    def answer(self, question: str, filters: SearchFilters) -> RagAnswer: ...
# RagAnswer.sources is built from SearchHit.chunk.provenance
```

The sketches above define the boundaries. Final signatures are fixed in each stage.

## 5. Flows

### 5.1 PDF ingestion (Stage 1)
1. Compute `file_hash` in streaming mode. If it exists in SQLite with status `indexed`, report a duplicate and stop. If the path exists with a different hash, treat it as a change.
2. Open with PyMuPDF and iterate pages one at a time. If no page has text, status `no_text`.
3. Chunk by page/window with overlap, preserving `page_start`/`page_end`.
4. Embeddings in small batches (`LKA_EMBED_BATCH_SIZE`). Write to Chroma after each batch.
5. Update SQLite at the end (`indexed`), or record the error and clean up partial chunks.
6. Remove = delete chunks and record. The original file is never touched.

### 5.2 Search (Stage 2)
1. Check embeddings-model compatibility with the collection (R2.3).
2. Resolve filters in SQLite into `document_id`s.
3. Embed the question, query Chroma with `k * overfetch`.
4. Apply the relevance threshold and deduplication (by `content_hash` and adjacency), cut to `k`.
5. Preparation for hybrid search: `Retriever` composes a list of "candidate generators"; today only the semantic one exists.

### 5.3 RAG (Stage 3)
1. Validate the question (maximum length).
2. `Retriever.search`. No results: answer "insufficient evidence" without calling the LLM.
3. Build a numbered context `[1]..[n]` up to the character/token budget.
4. System prompt: use only the context, separate "Found in the sources" from "Inference", cite by number, do not invent references.
5. Call Ollama with a timeout. The answer is validated: cited numbers that do not exist in the context are discarded.
6. `sources` is generated from the metadata of the chunks used.

### 5.4 Web collector (Stage 4)
Pipeline: registered source → bounded queue (depth/pages) → `UrlGuard` (http/https scheme, domain/path allowlist, DNS resolution, non-public IP blocking) → fetch with `httpx` without automatic redirects (each hop passes through `UrlGuard` again) → byte limit and `Content-Type` → main-content extraction → `content_hash` → chunking by section → embeddings. The validated IP is the one used for the connection, to reduce DNS-rebinding risk. Respect `robots.txt` and terms of use.
The HTML extraction library will be chosen in Stage 4 (lightweight candidates: `beautifulsoup4` + `lxml`, `trafilatura`), with justification.

### 5.5 Code (Stage 6)
Walk only the provided directories. Filters: `.gitignore` (via `git ls-files` when a repository exists, or an internal parser), default exclusions, binary detection, size limit, secret scanner (known key patterns + entropy) before reading content for embeddings. Chunk by function/block boundaries when simple (Python via `ast`; others by line windows), keeping `line_start`/`line_end`.

## 6. Configuration (`LKA_*`)

Paths: `LKA_DATA_DIR`, `LKA_DOCUMENTS_DIR`, `LKA_VECTOR_DB_DIR`, `LKA_CACHE_DIR`, `LKA_METADATA_DB_PATH`.
Ollama: `LKA_OLLAMA_BASE_URL`, `LKA_CHAT_MODEL`, `LKA_EMBEDDING_MODEL`, `LKA_OLLAMA_TIMEOUT_SECONDS`.
API: `LKA_API_HOST`, `LKA_API_PORT`, `LKA_ADMIN_TOKEN` (Stage 3).
Later stages add: chunk size/overlap, batch size, k, relevance threshold, context budget, collector limits.
The scaffold implements only the paths, Ollama and API groups.

## 7. Error handling

Domain errors in `app/core/errors.py`: `EmbeddingModelMismatch`, `OllamaUnavailable`, `ModelNotInstalled`, `InvalidDocument`, `NoExtractableText`, `UrlNotAllowed`. The API maps them: unavailable → 503, invalid input → 422, not found → 404, missing token → 401/403. Messages never include document text.

## 8. Security

- Listens on `127.0.0.1`. Administrative token for `POST /documents/ingest` and `DELETE /documents/{id}`. Constant-time comparison. The API refuses to start with ingestion enabled and no token.
- `documents/ingest` accepts only paths inside `LKA_DOCUMENTS_DIR` (path-traversal prevention).
- Logging without content (only ids, counts, durations).
- Prompt-injection threat via documents: retrieved content is treated as data in the prompt, delimited, and the system instruction forbids following instructions found in the context. The risk remains (small LLM) and will be documented.

## 9. Test strategy

See `docs/test-plan.md`. Summary: unit tests with synthetic PDFs generated by PyMuPDF inside the test, fake Ollama clients returning deterministic vectors, Chroma in a temporary directory, `httpx.MockTransport`/a local fake server for the collector, `integration` markers for real Ollama.

## 10. Resource estimate

See `docs/resource-estimate.md`.

## 11. Risks and limitations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `qwen2.5:3b` hallucinates or ignores instructions in Portuguese | Inaccurate answers | Citations generated from metadata, "no evidence" fallback, Stage 7 evaluation, larger model by configuration. |
| `nomic-embed-text` was trained mostly on English | Lower recall in Portuguese | Measure in Stage 7. A multilingual alternative only with approval and reindexing. |
| `nomic-embed-text` context window (effective size to be confirmed in Stage 1) | Silent truncation | Keep chunk size below the window and test it. |
| Generation latency on CPU (3B) | Answers take tens of seconds | Bounded context, timeouts, streaming possible later. |
| VM RAM (8 GB) on a 16 GB host | Swap/freezes | See estimate; reduce to a smaller model or smaller context. |
| Scanned PDF without text | Book not queryable | Marked `no_text`. OCR is a later stage. |
| PDFs with poor extraction (columns, tables) | Poor chunks | Record; evaluate in Stage 7. |
| SSRF/DNS rebinding in the collector | Access to internal network | `UrlGuard`, validated IP used for the connection, dedicated tests. |
| Web content under terms of use | Legal concerns | Collect only user-registered sources, respect robots.txt and limits. |
| Prompt injection in documents | Manipulated answer | Delimiting, system instruction, no tool execution by the LLM. |
| Chroma/PyMuPDF version change | Broken index | Pinned versions, backup before upgrading. |
| Low host disk space (39 GB free, 87% used) | VM and models do not fit | See estimate; confirm before `vagrant up`. |
| PyMuPDF license (AGPL or commercial) | Restriction if distributed | Personal local use. Record the license and reassess if the project is distributed. |

## 12. Dependencies between stages

Stage 0 → 1 (storage, chunking, embeddings client, PDF) → 2 (retrieval) → 3 (rag + api) → 4 (web) and 6 (code), both depend on 1–3 and are independent of each other → 5 (multi-source) requires at least 4 or 6 → 7 (evaluation) requires 2 and 3 and improves with 4–6.
