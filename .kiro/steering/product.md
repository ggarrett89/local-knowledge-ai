# Product: local-knowledge-ai

Personal technical-knowledge assistant that answers questions with RAG over the user's own sources, using free, local LLM and embeddings.

## Independence
- Standalone project. It does not depend on, modify or reuse files from the `llm_lab` repository.
- Other projects may consume it in the future only through its HTTP API.
- No remote repository is created or published without explicit user authorization.

## User
A single technical user, on a personal laptop (Ubuntu VM), studying development, architecture, DevOps/SRE, cloud, containers, IaC, databases/messaging, security and AI/ML.

## Knowledge sources
PDF books, official documentation (web), local documentation (Markdown, TXT, HTML), code repositories and notes. Each source keeps its type and provenance.

## Product principles
- Answers are based only on the retrieved sources. Without sufficient evidence, the system says so.
- References (document + page, URL + section, repository + file + lines) come from the system's real metadata, never from LLM-generated text.
- Never invent pages, URLs, versions or line ranges.
- Distinguish facts found in the sources from model inferences.
- Categories are configurable and extensible without code changes. Automatic classification (future) can always be corrected by the user.
- Every indexing action is explicit and user-initiated. Nothing is indexed, collected or downloaded automatically.

## Out of scope
- Training an LLM or fine-tuning.
- Paid APIs, cloud services, sending content to third parties.
- Multi-user, end-user authentication, public network exposure.
- Modifying indexed repositories or files (read-only).
- OCR and web UI: only after RAG and the API are validated.

## Success criteria
- Runs on a 16 GB RAM laptop with a ~8 GB VM, no GPU.
- Every answer lists verifiable sources.
- Questions with no answer in the sources are recognized as such.
- Reindexing is incremental and idempotent.
- Private data never enters Git.

## Process
Stage-by-stage development (see Spec `local-knowledge-ai-foundation`). Each stage ends with tests executed, README updated and explicit user approval before the next stage.
