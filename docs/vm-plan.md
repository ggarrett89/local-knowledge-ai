# VM execution plan

> Status: steps 1 to 6 have been run in the VM (see the README for the observed results). Step 7 (operation: API, backup, restore) is not available yet.

## Overview
An Ubuntu 22.04 VM (`ubuntu/jammy64`), 4 CPUs and 8 GB RAM, runs Ollama, the API and the CLIs. Original PDFs may stay in a host folder synced with the VM (the application only reads them); the index lives on the VM's local disk.

```
Host (16 GB) ── VirtualBox ── VM local-knowledge-ai
  ./ (project)   ──synced──►  /vagrant            (code)
  127.0.0.1:8000 ◄─forward──  :8000 (API)
                              /home/vagrant/lka-data   (index, SQLite, cache)
                              Ollama :11434 (inside the VM only)
```

## Decisions
- **Index outside the synced folder.** `LKA_DATA_DIR=/home/vagrant/lka-data`. Chroma and SQLite over `vboxsf` can suffer locking problems and slowness.
- **Virtualenv on the VM disk** (`~/venv`), not in `/vagrant`.
- **Ollama listens only on the VM's `127.0.0.1`** (default behavior). Do not forward port 11434 to the host.
- **API in the VM on `0.0.0.0:8000`**, forwarded only to the host's `127.0.0.1:8000`.
- **No model provisioning.** Download only by manual command, after stating size and purpose.
- **One VM at a time.**

## Steps

### 1. Pre-check on the host
```bash
vagrant --version && VBoxManage --version
free -h && df -h .
vagrant box list          # confirm ubuntu/jammy64
```

### 2. Validate the configuration (without starting the VM)
```bash
vagrant validate
```

### 3. Start the VM (first run installs basic APT packages) — done
```bash
vagrant up
vagrant ssh
```
To use less memory: `LKA_VM_MEMORY_MB=6144 vagrant up`. Stop: `vagrant halt`. Remove the VM (destructive; deletes the VM's index): `vagrant destroy`.

### 4. Prepare Python (inside the VM) — done
```bash
python3 -m venv ~/venv
. ~/venv/bin/activate
cd /vagrant
pip install -r requirements-dev.txt
cp .env.example .env     # set LKA_DATA_DIR=/home/vagrant/lka-data and LKA_API_HOST=0.0.0.0
pytest
```
Note: the VM's Python is 3.10, which is why the code stays compatible with 3.10+.

### 5. Install Ollama (manual, Stage 1) — done
Official installer: see [ollama.com/download](https://ollama.com/download/linux) and review the script before running it. It downloads the runtime (2.3 GB installed) and needs the `zstd` package (`sudo apt-get install zstd`). It creates an `ollama` system user and a systemd service that listens on `127.0.0.1:11434` only.

### 6. Download models (manual, with confirmation) — `nomic-embed-text` done
| When | Command | Size | Purpose |
|------|---------|------|---------|
| Stage 1 | `ollama pull nomic-embed-text` | ~274 MB | embeddings for indexing and querying |
| Stage 3 | `ollama pull qwen2.5:3b` | ~1.9 GB | answer generation |

### 7. Operation
- Start the API (from Stage 3) and open `http://127.0.0.1:8000/health` on the host.
- Backup (planned, not yet exercised): no ingestion running, copy `LKA_DATA_DIR` (SQLite + Chroma) and note the embeddings model. A script is planned for a later stage.
- Restore: copy back with the API stopped and check `stats`.

## Snapshot and appliance export (OVA)

### VirtualBox snapshot
`stage3-two-books-20261027` was taken with the VM powered off, after Stage 3 was merged. It captures Ollama with `nomic-embed-text` and `qwen2.5:3b`, the Python environment with the API dependencies, and the index of two books (Ansible 944 chunks, CKA guide 412 chunks, 0 errors). List it with `vagrant snapshot list` or `VBoxManage snapshot local-knowledge-ai list`; roll back with `vagrant snapshot restore default stage3-two-books-20261027` (this discards changes made after the snapshot).

### Appliance (OVA)
Two exports exist in `~/vm-appliances/`, each with a `.sha256` file:

| File | Size | State |
|------|------|-------|
| `local-knowledge-ai-stage1-20261026.ova` | 3.2 GB | Superseded. After Stage 1: Ollama and `nomic-embed-text` only. No index, no Stage 2/3 code dependencies, no `qwen2.5:3b`. Can be deleted |
| `local-knowledge-ai-stage3-20261027.ova` | 4.9 GB | Current. Ubuntu 22.04, the Python 3.10 environment with the project dependencies (`~/venv`), Ollama 0.40.11, `nomic-embed-text`, `qwen2.5:3b`, **and the search index of your books** |

**The Stage 3 appliance contains text chunks of your books. Keep it private; never share or publish it.** The first appliance deliberately had no data. If you need a shareable environment without the index, ask for a data-free export (it requires deleting the index inside the VM first, which is why it was not done here).

Verified for the current file: all three SHA-1 entries of the OVA manifest match the archive contents; the OVF records disk capacities of 40 GB and 10 MB; `VBoxManage import --dry-run` lists 4 CPUs and both disks. **Not verified:** an actual import and boot of either appliance.

How it was made, and a pitfall that was hit:
1. APT lists, the APT and pip caches and the shell history were removed inside the VM, then it was halted.
2. The snapshot was taken.
3. **Exporting the VM directly gave an invalid OVA.** Once a snapshot exists, the VM runs on differencing disks, and `VBoxManage export` wrote a disk capacity of `0` into the OVF. That file was discarded. The working method is to clone the current state (`VBoxManage clonevm local-knowledge-ai --name lka-export-tmp --mode machine --register`), export the clone, then remove it (`VBoxManage unregistervm lka-export-tmp --delete`). A side effect: the disk files inside the archive are named `lka-export-tmp-disk001.vmdk`; the VM name inside is `local-knowledge-ai`.
4. Free space was not zero-filled, on purpose: it would grow the dynamic disk by tens of GB on a host with limited space.

Caveats:
- The appliance contains the default Vagrant SSH key of the `ubuntu/jammy64` box.
- The OVF records the project's shared-folder path on the host and the two port forwards. On another machine, remove the shared folder from the imported VM settings and get the code with `git clone` inside the VM.
- The code in `/vagrant` is not inside the appliance; only the environment is.

Import (not yet exercised): `VBoxManage import local-knowledge-ai-stage3-20261027.ova`.

## Expected problems
| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `vagrant up` fails with a VirtualBox error | kernel modules / KVM conflict | check `VBoxManage` and the `vboxdrv` modules |
| VM slow or frozen | insufficient host RAM | `LKA_VM_MEMORY_MB=6144`, close applications |
| API unreachable from the host | API listening on `127.0.0.1` inside the VM | use `LKA_API_HOST=0.0.0.0` in the VM |
| Locking error in SQLite/Chroma | data in a synced folder | use `LKA_DATA_DIR` on the VM disk |
| Ollama answers slowly | CPU only, large context | reduce context and results; smaller model |
