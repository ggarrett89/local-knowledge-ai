# local-knowledge-ai

Personal technical-knowledge assistant that answers questions through RAG using free, local LLM and embeddings (Ollama), ChromaDB and FastAPI. Standalone project: it depends on no other repository and sends no data to external services.

> **Current status: Stage 3 (grounded answers and HTTP API) implemented and validated in the VM, awaiting approval.** You can import PDFs, search them, ask questions answered by a local model that cites checked sources, and call all of it over a local HTTP API. Answers are slow on this CPU-only setup (about 70 s per fresh question) and the model can still be wrong; see [Asking questions](#asking-questions-stage-3). The default test run uses fake clients; the real models were exercised in the VM with one real book and a synthetic bilingual corpus. Sections that describe later stages are marked *planned*.

## Goal
Query books (PDF), official documentation (web), code and notes, and get grounded answers with verifiable references (document and page, URL and section, repository, file and lines). When the sources are not enough, the system says so.

Out of scope: training models, fine-tuning, paid APIs, cloud.

## Architecture
```
CLI / HTTP API ──► ingestion ──► storage (SQLite: metadata │ ChromaDB: vectors)
        │                │                 ▲
        │                └──► clients/ollama (embeddings) ◄── retrieval     [Stages 1-2]
        └──► rag (context, prompt, verification) ──► clients/ollama (chat)  [Stage 3]
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
| `app/retrieval/` | Semantic search service, candidate generators and CLI |
| `app/rag/` | Context assembly, prompt, answer verification, `RagService` and the `ask` CLI |
| `app/api/` | FastAPI app, schemas, administrative authentication, error mapping |
| `app/services.py` | Composition root: builds the stores, clients and services for the API and CLIs |
| `scripts/` | Measurement helpers: context probe, sample PDF, retrieval calibration |
| `config/categories.json` | Initial categories (suggestions; new categories need no code change) |
| `docs/` | Test plan, VM plan, resource estimate, and `status.md` (where work stopped and how to resume) |
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
Result of the last full run: `440 passed, 19 deselected` (the 19 deselected are the opt-in integration tests).

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
| `LKA_CHAT_MODEL` | `qwen2.5:3b` | Chat model that writes answers |
| `LKA_CHAT_TIMEOUT_SECONDS` | `300` | Timeout for one answer |
| `LKA_CHAT_NUM_CTX` | `4096` | Context window in tokens (memory grows with it) |
| `LKA_CHAT_MAX_TOKENS` / `LKA_CHAT_TEMPERATURE` | `600` / `0.1` | Reply length limit and randomness |
| `LKA_RAG_CONTEXT_CHARS` | `6000` | Budget for the passages shown to the model; keep the prompt well inside `LKA_CHAT_NUM_CTX` |
| `LKA_EMBEDDING_MODEL` | `nomic-embed-text` | Embeddings model (see warning below) |
| `LKA_EMBEDDING_DOCUMENT_PREFIX` / `LKA_EMBEDDING_QUERY_PREFIX` | `search_document: ` / `search_query: ` | Task prefixes required by `nomic-embed-text`; set both to empty for models that do not use them |
| `LKA_OLLAMA_TIMEOUT_SECONDS` | `120` | Call timeout |
| `LKA_CHUNK_SIZE` / `LKA_CHUNK_OVERLAP` | `1200` / `200` | Chunk size and overlap, **in characters** (overlap must be below half the size) |
| `LKA_EMBED_BATCH_SIZE` | `16` | Chunks sent to Ollama per request |
| `LKA_RETRIEVAL_K` / `LKA_RETRIEVAL_MAX_K` | `5` / `50` | Default and maximum number of search results |
| `LKA_RELEVANCE_MAX_DISTANCE` | `0.45` | Search hits farther than this cosine distance are dropped (`2.0` disables it); see [Distances](#distances-and-the-relevance-threshold) |
| `LKA_RETRIEVAL_OVERFETCH` | `4` | Candidates fetched per result before de-duplication |
| `LKA_DEDUPE_ADJACENT` | `true` | Drop chunks next to an already selected chunk of the same document |
| `LKA_MAX_QUERY_CHARS` | `1000` | Longest accepted question (`nomic-embed-text` accepts about 2,000 characters of dense text) |
| `LKA_API_HOST` / `LKA_API_PORT` | `127.0.0.1` / `8000` | API address. In the VM use `0.0.0.0` (Vagrant forwards the port only to the host's `127.0.0.1`) |
| `LKA_ADMIN_TOKEN` | empty | Token for ingest and delete, at least 16 characters. Empty means those endpoints are disabled (`403`) |
| `LKA_API_MAX_BODY_BYTES` | `65536` | Largest accepted request body |

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
pytest                                                  # 440 passed, 19 deselected (Python 3.10.12)
pytest -m integration                                   # 19 passed against the real Ollama (slow: real answers take tens of seconds)
```
Ollama 0.40.11 was installed in the VM with the official installer (reviewed first; it also needs the `zstd` APT package, installed with `sudo apt-get install zstd`). It runs as a systemd service, CPU-only, and listens on `127.0.0.1:11434` inside the VM only; the port is not reachable from the host.

Full plan, decisions, the exported appliance (OVA) and expected problems in [`docs/vm-plan.md`](docs/vm-plan.md). Stop the VM with `vagrant halt` when you are not using it; it reserves 8 GB of the host's RAM while running.

Downloads are never automatic. Install the embeddings model yourself, inside the VM, after checking size and disk space:

| Model | Size | Purpose | Stage |
|-------|------|---------|-------|
| `nomic-embed-text` | ~274 MB | embeddings | 1 |
| `qwen2.5:3b` | ~1.9 GB | answers | 3 (installed in the VM) |

```bash
ollama pull nomic-embed-text      # run manually in the VM; on the first attempt the download failed at 99% with a network timeout and succeeded on retry
ollama pull qwen2.5:3b            # same; the first attempt failed when the VM lost DNS and resumed on retry
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

## Searching (Stage 2)
Search is read-only: it never changes the index. It embeds your question with the same model used for indexing and returns the closest chunks with their real source references.

```bash
python -m app.retrieval.cli search "How do I encrypt secrets with Ansible Vault?" -k 3
python -m app.retrieval.cli search "inventário dinâmico" --category iac-automation --topic ansible
python -m app.retrieval.cli search "handlers" --document DOCUMENT_ID --json
```
Options: `-k N` (default 5, max 50), `--category`, `--topic` (repeat; all must match), `--document` (repeat), `--source-type pdf`, `--max-distance X` (override the threshold for one query), `--full` (whole chunk instead of a 300-character snippet), `--json` (fields: `rank`, `distance`, `reference`, `document_id`, `chunk_id`, `title`, `page_start`, `page_end`, `text`).

Output, from the real index of the 473-page book (snippets omitted here):
```text
1. [0.174] Ansible: Up and Running, p. 227  (pdf-2d07e2588734a305:448)
2. [0.176] Ansible: Up and Running, p. 228  (pdf-2d07e2588734a305:451)
3. [0.213] Ansible: Up and Running, p. 321  (pdf-2d07e2588734a305:632)
```
- **References come from stored metadata only** (title and the pages recorded at indexing), never from generated text. A chunk that continues across a page break is cited as a range (`pp. 95-96`). The question's matching text can sit on either page of that range.
- With no match, filters that select nothing, or everything beyond the threshold, the command prints `No results.` and exits `0`. If the index has no documents it says to import a PDF first.
- Neighbouring chunks overlap by design, so a chunk right next to an already selected one (same document) is dropped, as are chunks with identical text. You may therefore get fewer than `-k` results. Set `LKA_DEDUPE_ADJACENT=false` to keep them.
- Errors exit `1` with a one-line message: empty or too long question (`LKA_MAX_QUERY_CHARS`, default 1000), `k` out of range, invalid category or topic slug, Ollama unreachable or model not installed, or an index built with a different embeddings model.
- Logs record counts and timings only, never the question or any text.
- Portuguese and English both work. A Portuguese question can retrieve an English book, but its distances are larger (see below).

### Distances and the relevance threshold
The distance printed in brackets is a **cosine distance**: `1 - cosine similarity` between the question and the chunk embeddings. `0` means the same direction, `1` unrelated (orthogonal) and `2` opposite. Lower is closer.

It ranks results. It is **not a probability** and not a percentage of relevance, and its scale depends on the embeddings model, the language pair and the corpus. A threshold chosen for one model must be re-measured for another.

`LKA_RELEVANCE_MAX_DISTANCE` (default **0.45**) drops hits farther than that. It was chosen from measurements with `nomic-embed-text` on one real book and a small synthetic corpus, in the VM:

| Questions (top-1 distance) | Count | Min | Median | Max |
|----------------------------|------:|----:|-------:|----:|
| Answerable, English, real book | 12 | 0.174 | 0.223 | 0.293 |
| Answerable, Portuguese, real book | 6 | 0.296 | 0.381 | 0.449 |
| Other technologies, not in the book | 4 | 0.311 | 0.376 | 0.418 |
| Unrelated (cake, football, geography) | 8 | 0.390 | 0.465 | 0.500 |

| Threshold | Answerable kept (of 18) | Unrelated cut (of 8) | Other technologies cut (of 4) |
|----------:|------------------------:|---------------------:|------------------------------:|
| 0.30 | 13 | 8 | 4 |
| 0.35 | 14 | 8 | 2 |
| 0.40 | 17 | 7 | 2 |
| **0.45** | **18** | **5** | **0** |
| 0.50 | 18 | 0 | 0 |

Reading the table honestly:
- **0.45 keeps every answerable question** in both data sets and cuts clearly unrelated ones, but it lets 3 of 8 unrelated questions through. Use **0.40** to cut more, at the cost of losing 1 of 18 answerable questions here (a Portuguese one). Setting `2.0` disables the cutoff.
- **Distance alone cannot say "the book does not contain the answer".** Portuguese questions about an English book (0.30–0.45) overlap the unrelated range (0.39–0.50), and questions about *other* technologies land close to the book's range. The final "not enough evidence" decision therefore has to look at the retrieved text too, which is Stage 3.
- On the real book, a chunk containing a keyword of the question was the top hit for 17 of 18 questions and appeared within the top 5 for all 18. This keyword check is a rough proxy, not ground truth; the systematic evaluation (Recall@k with labelled data) is Stage 7.
- All 40 hits returned for 8 questions had their text on exactly the pages they cite (checked against the PDF).
- On the synthetic bilingual corpus (5 topics, EN and PT books), the top hit had the right topic for 10 of 10 questions, answerable distances were 0.14–0.33 and unanswerable ones 0.36–0.53 (a gap of only 0.03).

To re-measure after changing the model or adding books:
```bash
python scripts/calibrate_retrieval.py synthetic                                    # bilingual synthetic corpus, temporary index
python scripts/calibrate_retrieval.py index scripts/calibration/book_queries.json  # your index (LKA_DATA_DIR); edit the questions for your books
```

### Search performance (measured in the VM, CPU only)
| Measurement | Result |
|-------------|--------|
| Search call (embed question + query + filter), 944-chunk index | median 0.10 s, max 0.13 s over 29 questions |
| Whole CLI invocation, including starting Python and loading ChromaDB | about 2.8 s |
| CLI peak memory | about 110 MB |

## Asking questions (Stage 3)
`ask` retrieves the closest passages (Stage 2) and has the local chat model (`qwen2.5:3b`) answer **only from them**. The model writes prose; everything else is checked or built by the system.

```bash
python -m app.rag.cli ask "How do I encrypt secrets with Ansible Vault?"
python -m app.rag.cli ask "Quando os handlers executam?" --category iac-automation --passages
python -m app.rag.cli ask "What is an inventory file?" --json
```
Options: `-k`, `--category`, `--topic`, `--document`, `--source-type`, `--max-distance` (as in search), `--passages` (print what the model was shown) and `--json`.

**Every answer has one of three statuses. Read it before trusting the text.**

| Status | Meaning |
|--------|---------|
| `answered` | The answer cites at least one retrieved passage, and the citations were checked against the passage text |
| `insufficient_evidence` | No passage was close enough (the model is **not called**), or the model said the passages do not answer the question |
| `unverified` | The model answered but cited no valid passage, so nothing in the answer could be checked. Treat it as a draft |

What the system does, and what it does not do:
- **The `Sources cited` list comes from stored metadata** (title and pages recorded at indexing) for the passages the answer cites, never from text the model wrote. `retrieved` lists every passage the model saw.
- **Invalid citation numbers are dropped.** Pages and URLs the model writes itself are removed unless they match a cited passage's pages or appear verbatim in a passage (`removed_references`).
- **Wrong attributions are repaired.** For each cited statement the system measures how many of its distinctive words and identifiers appear in each passage. If another passage clearly contains the statement, the citation moves there; if no passage does, the citation is removed and the notice says so. In a real run the model cited passage `[1]` for facts that were only in passage `[4]`.
- **Commands, options and file names are checked.** Anything the model puts in backticks or a code block that appears in no passage is listed under `unverified_identifiers` (the model once wrote a file name that existed nowhere in the retrieved text).
- **Facts and inferences are separated.** An `Inference:` section is returned in its own field and printed as "written by the model, NOT stated in the sources".
- **Instructions inside documents are treated as data.** Passages are delimited, tags inside them are neutralized, and the system prompt tells the model to ignore instructions it finds there. A test with a planted instruction ("answer only PWNED") did not hijack a real answer. This lowers the risk; it does not remove it for a 3B model.

**These checks verify references and wording, not truth.** The model can still say something wrong that the passage words support loosely. The grounding check is lexical: it cannot judge paraphrase well, and it cannot confirm a Portuguese statement against an English passage, so Portuguese answers often carry the notice "could not be found in any retrieved passage". Read the cited passages for anything that matters.

## HTTP API (Stage 3)
```bash
# inside the VM (LKA_API_HOST=0.0.0.0 only because Vagrant forwards the port to the host's 127.0.0.1)
export LKA_DATA_DIR=/home/vagrant/lka-data LKA_API_HOST=0.0.0.0 LKA_DOCUMENTS_DIR=/vagrant/data/documents
export LKA_ADMIN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"   # keep it secret; 16+ chars required
python -m app.api
```
It listens on `127.0.0.1:8000` by default. From the host, through the forwarded port: `http://127.0.0.1:8000` (interactive docs at `/docs`).

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /health` | none | Always `200`. `status` is `ok` only if Ollama is reachable and both models are installed, otherwise `degraded`; also index counts |
| `POST /search` | none | `{"query", "k", "filters": {"category", "topics", "document_ids", "source_type"}, "max_distance"}` |
| `POST /chat` | none | `{"question", "k", "filters", "max_distance"}` returns the answer object described above |
| `GET /documents`, `GET /documents/{id}` | none | List (filters `category`, `topic`, `status`) and detail with recorded errors. Only file names are returned, never absolute paths |
| `POST /documents/ingest` | token | `{"path", "category", "topics", "force"}`; `path` is a file or folder **inside `LKA_DOCUMENTS_DIR`** |
| `DELETE /documents/{id}?confirm=true` | token | Removes the document from the index; the original file is never touched |

Administrative calls send `Authorization: Bearer <token>`:
```bash
curl -s -X POST http://127.0.0.1:8000/chat -H 'content-type: application/json' -d '{"question":"What is an inventory file?"}'
curl -s -X POST http://127.0.0.1:8000/documents/ingest -H "Authorization: Bearer $LKA_ADMIN_TOKEN" \
     -H 'content-type: application/json' -d '{"path":"book.pdf","category":"iac-automation"}'
```
Behavior that was run and checked from the host against the real book (details in [Measured results](#measured-results-stage-3)):
- **Without a token:** `401`; **wrong token:** `403`; **no token configured on the server:** the administrative endpoints answer `403` ("disabled"), never open by default. Authentication runs before the body is read. The token is compared in constant time and never appears in responses or logs.
- **Paths:** `../../etc/passwd`, absolute paths and symlinks that leave the documents directory get `403`; a missing file `404`. In folder mode, symlinked files pointing outside the folder are skipped.
- **One ingestion at a time** (`409` otherwise). Ingestion through the API is **synchronous**: a 473-page book takes about 14 minutes, so use the CLI for large books.
- **`DELETE` without `confirm=true`** is a `400`, with the token missing `401`.
- **Limits:** request bodies above 64 KiB (`LKA_API_MAX_BODY_BYTES`) get `413`; empty or over-long questions and invalid category or topic names get `422`.
- **Failures** return fixed text without URLs, model names or tracebacks: `503` when Ollama or a model is unavailable, `502` when it answers with an error, `409` when the embeddings model differs from the index, `500` for anything unexpected. Logs hold counts, timings and exception class names, never questions, answers or document text.
- There is no TLS and no user login: the API is meant for the local machine only (the forward is bound to the host's `127.0.0.1`).

### Measured results (Stage 3)
Measured in the VM (4 CPUs, no GPU, Ollama 0.40.11, `qwen2.5:3b`, `nomic-embed-text`) on the 473-page book (944 chunks), with the settings in this README. **Small samples: 10 answerable and 6 unanswerable questions, run through `scripts/evaluate_rag.py`.** Individual questions flipped between runs with identical settings even at temperature 0.1, so read the counts as indications, not guarantees.

| Result (final prompt, cold start) | Value |
|-----------------------------------|-------|
| Answerable questions answered with checked citations | 8 of 10 (English 6 of 6, Portuguese 2 of 4) |
| The other 2 answerable | 1 never reached the model (no passage under the 0.45 distance threshold), 1 `unverified` |
| Unanswerable questions (cake, football, Kubernetes, Kafka, PostgreSQL) refused | 6 of 6 (2 stopped before the model, 4 refused by the model) |
| Answers whose cited passages were all on topic (keyword proxy) | 8 of 8 |
| Answers where a citation had to be moved | 4 of 8 |
| Answers containing a statement found in no passage (its citation was removed) | 4 of 8, including both Portuguese answers |
| Answers with a command or file name found in no passage | 1 of 8 |
| Pages or URLs written by the model and removed | 0 of 8 |

| Latency, per question that reached the model | Value |
|----------------------------------------------|-------|
| Fresh question, models loaded | median **71 s** (31 s to 128 s) |
| of which reading the passages (1,300 to 1,500 tokens) | median 52 s |
| of which generating the reply | median 21 s (about 5 tokens per second) |
| Same question again | about 20 s (Ollama reuses the prompt it has already read) |
| Question with no close passage | about 1 s |
| Memory | VM 2.6 GB with both models loaded (about 3 GB while answering); Python about 140 MB |

How the prompt was chosen (same 10 answerable questions, real model; in every run one Portuguese question never reached the model because no passage was close enough, so at most 9 could be answered):

| Prompt | Answered (EN / PT) | `unverified` | Refused by the model though answerable |
|--------|--------------------|--------------|----------------------------------------|
| Rules only (1 run) | 5 (5 / 0) | 4 | 0 |
| + worked example and a reminder that repeats the refusal word (2 runs) | 6 and 6 | 1 and 0 | 2 and 3 |
| + worked example and a reminder **without** the refusal word (kept, 2 runs) | 8 and 8 (6 / 2) | 1 and 1 | 0 and 0 |

Other things measured:
- Without the verification layers, the first real answer cited the wrong passage for several facts and named a file that is in no passage. That is why they exist.
- **Ollama silently cuts a prompt that is larger than the context window and reports a token count below the window** (a 3,529-token prompt in a 512-token window was reported as 258). Comparing the count with the window therefore detects nothing. The client now refuses a prompt that cannot fit before sending it and compares the reported count with a size estimate afterwards. The estimate undercounts token-dense text (hashes, base64), so truncation of that kind of text can still go unnoticed. The default context budget uses about 2,800 of 4,096 tokens on ordinary prose.
- Model download: the first attempt of `qwen2.5:3b` failed when the VM lost DNS and resumed from the saved data on retry.

## Backup and restore (planned)
Not yet exercised. The state lives entirely in `LKA_DATA_DIR` (SQLite file plus the Chroma directory), so a backup is a copy of that directory while no ingestion is running. A script and a tested restore procedure are planned for a later stage.

## Usage of later stages (planned)
| Topic | Stage |
|-------|-------|
| Register official documentation (web) | 4 |
| Multi-source query and filters | 5 |
| Index code | 6 |
| Evaluation and optimization | 7 |

## Tests
See [`docs/test-plan.md`](docs/test-plan.md). Default: `pytest` (no network, no Ollama, no models). Tests with a real Ollama: `pytest -m integration`.

## Privacy and Git
`data/`, `.env`, databases and virtual environments are in `.gitignore`. Never version books, indexes or secrets. Logs and stored error messages contain ids, counts and durations, never document text. ChromaDB telemetry is disabled.

## Known limitations
- **Answers are slow on CPU:** about 70 s for a fresh question (31 to 128 s), mostly reading the passages. A GPU or a smaller context budget would help; neither was tried.
- **The 3B model is not reliable.** It can misattribute, paraphrase loosely, over-cite (once all five passages for one answer), refuse an answerable question or answer one it should not, and its behavior differs between runs. The checks catch wrong references and invented identifiers; they cannot prove a statement is true.
- **Portuguese is weaker:** 2 of 4 Portuguese questions got cited answers (6 of 6 in English), and Portuguese statements cannot be confirmed lexically against English passages.
- **Verification is lexical.** It judges overlap of distinctive words and identifiers, not meaning. Plain-word commands (`terraform plan` in backticks) are not checked as identifiers.
- **Refusal is only measured on 6 unanswerable questions** (all refused). Related-but-absent topics (other technologies) were refused by the model, not by the distance threshold.
- The context-window guard can miss truncation of token-dense text (hashes, base64) because it estimates size from characters.
- The API has no TLS or user login, ingestion through it is synchronous, and starlette differs between the host (1.10.0) and the VM (1.7.0), both with the same `fastapi` 0.162.0 and the same test results.
- The relevance threshold is model- and corpus-specific and was calibrated on one English book, a Portuguese-question set and 10 synthetic questions. Distance alone cannot tell "not in the book" from "asked in another language"; Stage 3 has to look at the text too.
- Search quality was measured with a keyword proxy and small question sets, not with labelled ground truth (Stage 7).
- Search is purely semantic: exact terms, option names and error messages are not matched lexically (hybrid search is only added if Stage 7 data justify it).
- Measurements used synthetic text plus one real book, one document at a time, on CPU only. Retrieval quality is not measured yet (no search exists).
- Embedding is slow on CPU (~0.9 s per chunk): the real 473-page book took 13.6 minutes.
- Chunk sizes are in characters; numeric- or code-heavy text uses more tokens per character, so do not raise `LKA_CHUNK_SIZE` above ~2,000 without re-running `scripts/probe_embedding_context.py`.
- PDFs: password-protected files are rejected; scanned PDFs are `no_text` (no OCR). A truncated PDF may be repaired by PyMuPDF and indexed with part of its text missing, without an error.
- Removing documents or switching the embeddings model requires the steps above; there is no bulk "reindex all" yet.
- `PyMuPDF` is licensed under AGPL/commercial; fine for local personal use, reassess if the project is distributed.

## Next steps
Stage 4 (official documentation from the web), after explicit approval. It adds a web collector with its own security rules (allowlists, address validation, redirects) and needs an HTML extraction library, which will be proposed and justified before it is installed. Plan in [`tasks.md`](.kiro/specs/local-knowledge-ai-foundation/tasks.md).
