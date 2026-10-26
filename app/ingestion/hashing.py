from __future__ import annotations

import hashlib
from pathlib import Path

_BLOCK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks (never loads the whole file)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
