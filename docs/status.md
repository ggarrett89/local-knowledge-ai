# Where we stopped

Written 2026-10-27 so work can resume in a new session. Facts here were checked on that date; re-check the live state with the commands in "Resume".

## Stage status

| Stage | State |
|-------|-------|
| 0 Scaffold, steering, spec | Done, approved, merged |
| 1 PDF ingestion | Done, approved, merged (also validated with a real Ollama and one real book) |
| 2 Semantic search | Done, approved, merged |
| 3 Grounded answers (RAG) and HTTP API | **Implemented and validated. Waiting for your review of PR #6 and your approval** |
| 4 Official documentation from the web | Not started |
| 5 Multi-source library, 6 Code indexing, 7 Evaluation | Not started |

Pull request: **#6 "Stage 3: grounded answers (RAG) and the HTTP API"** is open against `main` (branch `stage-3/rag-api`). Everything else is merged. Task 3.9 (user approval) in `.kiro/specs/local-knowledge-ai-foundation/tasks.md` is still unchecked; tick it once you approve.

## What exists

- `python -m app.ingestion.cli` imports PDFs; `python -m app.retrieval.cli search` finds passages; `python -m app.rag.cli ask` answers with checked sources; `python -m app.api` serves the HTTP API.
- Tests: 440 passed, 19 deselected (the 19 need real Ollama models). Same result on the host (Python 3.12) and in the VM (Python 3.10.12).
- The README documents commands, settings, measured results and limitations. `docs/` has the test plan, VM plan and resource estimate. The spec is in `.kiro/specs/local-knowledge-ai-foundation/` (design decisions D1 to D24).

## Environment state

| Item | State |
|------|-------|
| VM `local-knowledge-ai` | Running (4 CPUs, 8 GB). Halt it with `vagrant halt` when not in use; it reserves 8 GB of host RAM |
| Ollama in the VM | Active, with `nomic-embed-text` (274 MB) and `qwen2.5:3b` (1.9 GB) installed |
| API process | Stopped (`lka-api` unit inactive). Start it as described in the README |
| Index | In the VM at `/home/vagrant/lka-data`: 1 document (the 473-page book), 944 chunks, 0 errors |
| The book | `data/documents/` on the host (git-ignored; never commit it). Visible in the VM as `/vagrant/data/documents/` |
| VM appliance (OVA) | `~/vm-appliances/local-knowledge-ai-stage1-20261026.ova` (3.2 GB). It is **out of date**: it was exported after Stage 1 and has no Stage 2/3 code, no `qwen2.5:3b` and no API dependencies. It was never imported or booted |
| Disk | Host `/home`: 40 GB free (86% used). VM disk: 32 GB free |
| Git | Local branch `stage-3/rag-api` is pushed. `main` does not yet contain Stage 3 |

## Decisions waiting for you

1. **Review and merge PR #6, then approve Stage 3.**
2. **Chat model choice.** The 3B model takes about 70 s per answer on this CPU and is weaker in Portuguese (2 of 4 questions cited, 6 of 6 in English). Options, none tried: a larger model (slower), a different family (needs a download and your confirmation), fewer or shorter passages (faster, may lose context). I would decide this with Stage 7 evaluation data, not before.
3. **Relevance threshold** (`LKA_RELEVANCE_MAX_DISTANCE`, default 0.45). Lowering it to 0.40 cuts more unrelated questions but lost one Portuguese question in the measurements.
4. **Next stage.** Stage 4 (web documentation) adds an HTML extraction library. I will propose candidates and justify one before installing anything.

## Known issues and limits (details in the README)

- Answers are slow on CPU, the 3B model can be wrong, and results vary between runs.
- The verification layers check references and wording, not truth. The grounding check is lexical and cannot confirm Portuguese statements against English passages.
- The context-window guard can miss truncation of token-dense text.
- API ingestion is synchronous (about 14 minutes for the 473-page book); use the CLI for large books. The API has no TLS or user login.
- The host has `starlette` 1.10.0 and the VM has 1.7.0 (same `fastapi`); both pass all tests.
- Evaluation so far is 16 questions on one book. Nothing has been measured with labelled data.
- Backup and restore of `LKA_DATA_DIR` is described but not yet scripted or tested.

## Resume

```bash
cd /home/garrett/projetos/local-knowledge-ai
git fetch && git status -sb && gh pr list                  # is PR #6 merged yet?
git checkout main && git pull                              # then branch for the next work
vagrant status                                             # start with `vagrant up` if it is not running
vagrant ssh -c 'ollama list; systemctl is-active ollama'   # both models should be listed
.venv/bin/python -m pytest -q                              # expect 440 passed, 19 deselected
```
Checks inside the VM (activate `~/venv`, `cd /vagrant`, set `LKA_DATA_DIR=/home/vagrant/lka-data`):
`python -m app.ingestion.cli stats` should show 1 document and 944 chunks, and `python -m app.retrieval.cli search "inventory file" -k 3` should return pages from the book.

Run long jobs in the VM with `sudo systemd-run --unit=NAME --uid=vagrant --gid=vagrant /bin/bash -c '...'`. Plain `nohup` and `setsid` jobs die when the SSH session ends.

## Working agreements (so they carry over)

- Every stage works on its own branch and ends in a pull request to `main`. Do not stack PRs: a stacked PR was once merged into a feature branch and never reached `main`.
- No model or tool is downloaded without stating size, disk and purpose and getting your confirmation.
- The repository is public: no books, indexes, tokens or private paths in Git.
- Documentation is written in English. Nothing is documented as validated unless it was run.
- Each stage ends with tests run, README updated and your explicit approval before the next stage.

## Suggested next steps

1. Merge PR #6 and approve Stage 3.
2. Optionally import more of your books (`add` or `add-all`) and ask real questions to see where the system helps and where it fails; those notes are the best input for Stage 7.
3. Start Stage 4 on a new branch from `main`, beginning with the choice of HTML extraction library.
