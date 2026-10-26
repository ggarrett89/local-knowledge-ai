"""Shared test helpers: synthetic PDFs and a deterministic fake embeddings client."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence

import pymupdf


def words(tag: str, count: int) -> str:
    """Distinct, parseable tokens: 'p1t0000 p1t0001 ...' (tag identifies the page)."""
    return " ".join(f"{tag}{i:04d}" for i in range(count))


def make_pdf(
    path: Path,
    pages: Sequence[str],
    *,
    title: Optional[str] = None,
    password: Optional[str] = None,
) -> Path:
    """Create a small synthetic PDF. An empty string produces a blank page (no text)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            rc = page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=10)
            assert rc >= 0, "test page text does not fit on the page"
    if title:
        doc.set_metadata({"title": title})
    path.parent.mkdir(parents=True, exist_ok=True)
    if password:
        doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password + "-owner")
    else:
        doc.save(str(path))
    doc.close()
    return path


class FakeEmbedder:
    """Deterministic embeddings without Ollama. Records the size of every batch."""

    def __init__(
        self,
        identity: str = "fake-embed|v1",
        dim: int = 8,
        *,
        error: Optional[Exception] = None,
        fail_on_call: Optional[int] = None,
    ) -> None:
        self._identity = identity
        self._dim = dim
        self.error = error
        self.fail_on_call = fail_on_call
        self.calls: List[int] = []
        self.on_call = None

    @property
    def identity(self) -> str:
        return self._identity

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b + 1) / 256.0 for b in digest[: self._dim]]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls.append(len(texts))
        if self.on_call is not None:
            self.on_call(len(self.calls))
        if self.error is not None and (self.fail_on_call is None or len(self.calls) == self.fail_on_call):
            raise self.error
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    def close(self) -> None:
        pass
