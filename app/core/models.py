"""Domain models shared across packages. No I/O here."""

from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.core.errors import InvalidSlug

SourceType = Literal["pdf", "web", "code", "note"]
DocumentStatus = Literal["indexed", "no_text", "error"]

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def normalize_slug(value: str) -> str:
    """Lower-case and validate a category/topic slug."""
    slug = value.strip().lower()
    if not SLUG_RE.fullmatch(slug):
        raise InvalidSlug(
            f"invalid slug {value!r}: use 1-64 chars of a-z, 0-9 and '-', starting with a letter or digit"
        )
    return slug


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


class Provenance(BaseModel):
    """Where a chunk came from. Unknown fields stay None (never invented)."""

    source_type: SourceType
    document_id: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    url: Optional[str] = None
    section: Optional[str] = None
    repo: Optional[str] = None
    path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class Chunk(BaseModel):
    chunk_id: str
    chunk_index: int = Field(ge=0)
    text: str
    content_hash: str
    provenance: Provenance


class Section(BaseModel):
    """A unit of extracted text (one PDF page) with its provenance."""

    text: str
    page_number: Optional[int] = None


class SourceInfo(BaseModel):
    title: str
    page_count: Optional[int] = None


class DocumentRecord(BaseModel):
    document_id: str
    title: str
    source_type: SourceType
    category: str
    topics: List[str] = Field(default_factory=list)
    source_uri: Optional[str] = None
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    version: Optional[str] = None
    language: Optional[str] = None
    status: DocumentStatus
    page_count: Optional[int] = None
    chunk_count: int = 0
    captured_at: Optional[str] = None
    updated_at: Optional[str] = None
    indexed_at: Optional[str] = None
    ingest_seconds: Optional[float] = None
    embedding_model: Optional[str] = None
