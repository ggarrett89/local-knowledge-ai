# Technology

## Approved stack
- Python >= 3.10 (VM `ubuntu/jammy64` ships 3.10; the development host uses 3.12). Do not use 3.11+-only features without a recorded decision.
- FastAPI + Pydantic v2 + pydantic-settings (API, contracts, configuration).
- Ollama as the LLM and embeddings runtime. Initial chat model: `qwen2.5:3b` (~1.9 GB). Initial embeddings: `nomic-embed-text` (~274 MB). Both swappable by configuration.
- ChromaDB persisted on disk for vectors.
- SQLite (stdlib) as the source of truth for document, category, topic and ingestion-error metadata.
- PyMuPDF for PDFs (with page numbers).
- pytest for tests.
- Additional dependencies only when the current stage requires them, with justification. Avoid LangChain, LlamaIndex and heavy orchestrators.

## Hardware constraints
- Host: i7, 16 GB RAM, no GPU required. Target VM: 4 CPUs and ~8 GB RAM (confirm before starting).
- Never assume the VM's entire RAM belongs to the LLM. Reserve memory for the OS, API, Chroma and embeddings.
- One VM at a time. No Kubernetes, distributed databases or extra services.
- Process PDFs and embedding batches as streams in small batches; never load all documents into RAM.

## Models
- Never download models automatically. Before any download, state size, disk space and purpose, and wait for confirmation.
- The embeddings model is part of the collection's identity. Changing it requires a planned reindex; the application must refuse queries with an incompatible model.
- Same embeddings model for indexing and querying.

## Local execution
- Configuration through environment variables with the `LKA_` prefix (see `.env.example`). Data paths are configurable.
- Data (`data/`) stays out of Git. In the VM, prefer the VM's local disk for the index, not a VirtualBox-synced folder.
- The API listens on `127.0.0.1` by default. In the VM it listens on `0.0.0.0` only because Vagrant forwards the port solely to the host's `127.0.0.1`.

## Security
- No data sent to external services. The only allowed outbound traffic is explicit collection of web documentation approved by the user.
- Secrets stay out of code and logs. `.env` is never versioned.
- Logs contain neither full book text nor sensitive questions; record IDs, counts and durations.
- Web collector: domain/path allowlist, URL and resolved-IP validation (block loopback, private network, link-local, cloud metadata), revalidate redirects, size/time/count limits, never execute collected content.
- Code indexing: read-only, explicitly chosen directories, respect `.gitignore`, detect secrets before indexing.
- Administrative endpoints require a token and confirmation for destructive operations.

## Tests and quality
- All new code has pytest tests. Use small synthetic documents and Ollama mocks; tests do not depend on network, Ollama or models.
- Tests that need a real Ollama are marked `@pytest.mark.integration` and excluded from the default run.
- Never claim tests passed without running them; never document a command as validated if it was not run.
- Default command: `pytest` at the repository root, inside the virtualenv.

## Dependencies
Runtime in `requirements.txt`, development in `requirements-dev.txt`. Versions pinned with an upper bound once the stage that introduces them is validated.
