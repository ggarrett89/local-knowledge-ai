"""PDF text extraction with PyMuPDF, one page at a time. Read-only; never modifies the file."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pymupdf

from app.core.errors import InvalidDocument
from app.core.models import Section, SourceInfo

# Keep the terminal clean: MuPDF prints low-level warnings for slightly broken PDFs.
pymupdf.TOOLS.mupdf_display_errors(False)

_MAX_TITLE_LEN = 300


class PdfExtractor:
    source_type = "pdf"

    def describe(self, path: Path) -> SourceInfo:
        doc = self._open(path)
        try:
            title = ((doc.metadata or {}).get("title") or "").strip()[:_MAX_TITLE_LEN]
            return SourceInfo(title=title or Path(path).stem, page_count=doc.page_count)
        finally:
            doc.close()

    def extract(self, path: Path) -> Iterator[Section]:
        doc = self._open(path)
        try:
            for index in range(doc.page_count):
                try:
                    text = doc.load_page(index).get_text("text")
                except Exception as exc:  # noqa: BLE001 - library raises many types
                    raise InvalidDocument(f"cannot read page {index + 1}: {type(exc).__name__}") from exc
                yield Section(text=text, page_number=index + 1)
        finally:
            doc.close()

    @staticmethod
    def _open(path: Path) -> "pymupdf.Document":
        try:
            doc = pymupdf.open(str(path), filetype="pdf")
        except Exception as exc:  # noqa: BLE001 - library raises many types
            raise InvalidDocument(f"cannot open PDF: {type(exc).__name__}") from exc
        if doc.needs_pass:
            doc.close()
            raise InvalidDocument("PDF is password protected")
        return doc
