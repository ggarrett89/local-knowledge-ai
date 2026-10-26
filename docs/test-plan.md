# Test plan

## Principles
- Default tests use no network, Ollama or models. They run with `pytest` at the repository root.
- Tests that depend on a real Ollama use `@pytest.mark.integration` and run only with `pytest -m integration`.
- Test data is synthetic and generated inside the test (PDFs created with PyMuPDF in a temporary directory). No real book enters the repository.
- A result is valid only if the command was actually run. The README records only commands that ran successfully.

## Layers
| Layer | What it covers | Tools |
|-------|----------------|-------|
| Unit | config, chunking, hashing, filters, context assembly, reference formatting | pytest |
| Storage | SQLite and Chroma in a temporary directory, persistence after reopening | pytest + `tmp_path` |
| Ingestion | valid, invalid and no-text PDFs, duplicate, change, removal | synthetic PDFs, fake embeddings client |
| Retrieval | filters, empty database, no results, incompatible model | fake deterministic vectors |
| RAG | Ollama unavailable, empty context, sources taken from metadata | fake chat client |
| API | contracts, token, limits, paths, 4xx/5xx errors | FastAPI `TestClient` |
| Web collector | SSRF, redirects, limits, malformed HTML | fake local HTTP server / `httpx.MockTransport` |
| Code | `.gitignore`, exclusions, secrets, source lines | temporary synthetic repository |
| Integration | real Ollama in the VM | `-m integration` |
| Evaluation | Recall@k, citations, latency, memory (Stage 7) | synthetic corpus |

## Requirements matrix
| Topic | Stage | Main cases |
|-------|-------|------------|
| Configuration | 0 | safe defaults, environment override, validation, secret not exposed, no filesystem side effects |
| PDFs | 1 | valid, corrupted, no text, several files, recursive |
| Incremental | 1 | duplicate, change, reindex, remove without deleting the file |
| Persistence | 1 | reopen database and collection and get the same data |
| Embeddings | 1–2 | incompatible model refused, service unavailable |
| Search | 2 | filters by document/category/topic, result limit, empty, redundancy |
| References | 2–3 | correct page, real URL and section, no invented reference |
| RAG | 3 | insufficient context, Ollama down, timeout, nonexistent citations discarded |
| API | 3 | missing/invalid token, path traversal, input above the limit |
| Web | 4 | private/loopback/metadata IPs blocked, redirect outside the allowlist, large response |
| Code | 6 | `.env`, keys, `node_modules`, `terraform.tfstate`, detected secret, source lines |
| Limits | 1, 3 | memory (document processed as a stream), context budget |

## Exit criteria per stage
1. `pytest` at the root passes, with the output attached to the report.
2. Integration tests not run are listed with the reason and the command to run them.
3. README updated and consistent with the commands actually run.

## Stage 0 (current)
`pytest` validates the configuration and the categories file. The `Vagrantfile` is validated with `vagrant validate`. There are no ingestion, search, RAG or API tests because those features do not exist yet.
