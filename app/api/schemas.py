"""HTTP contracts (Pydantic). Responses expose file names, never absolute host paths."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.models import SearchFilters


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)  # the effective limit is LKA_MAX_QUERY_CHARS
    k: Optional[int] = Field(default=None, ge=1, le=100)
    filters: Optional[SearchFilters] = None
    max_distance: Optional[float] = Field(default=None, gt=0, le=2)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: Optional[int] = Field(default=None, ge=1, le=100)
    filters: Optional[SearchFilters] = None
    max_distance: Optional[float] = Field(default=None, gt=0, le=2)


class SearchHitOut(BaseModel):
    rank: int
    distance: float
    reference: str
    document_id: str
    chunk_id: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str


class SearchResponse(BaseModel):
    count: int
    results: List[SearchHitOut]


class DocumentOut(BaseModel):
    document_id: str
    title: str
    source_type: str
    category: str
    topics: List[str]
    status: str
    page_count: Optional[int] = None
    chunk_count: int
    file_name: Optional[str] = None
    version: Optional[str] = None
    language: Optional[str] = None
    indexed_at: Optional[str] = None
    updated_at: Optional[str] = None
    embedding_model: Optional[str] = None


class ErrorOut(BaseModel):
    stage: str
    message: str
    created_at: str


class DocumentDetail(DocumentOut):
    errors: List[ErrorOut] = Field(default_factory=list)


class DocumentList(BaseModel):
    count: int
    documents: List[DocumentOut]


class IngestRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096, description="File or folder inside the documents directory")
    category: Optional[str] = Field(default=None, max_length=64)
    topics: Optional[List[str]] = Field(default=None, max_length=20)
    force: bool = False


class IngestResultOut(BaseModel):
    path: str
    outcome: str
    document_id: Optional[str] = None
    title: Optional[str] = None
    pages: Optional[int] = None
    chunks: int = 0
    seconds: float = 0.0
    message: Optional[str] = None


class IngestResponse(BaseModel):
    counts: Dict[str, int]
    results: List[IngestResultOut]
    aborted: Optional[str] = None
    skipped_after_abort: int = 0


class DeleteResponse(BaseModel):
    document_id: str
    removed: bool


class OllamaHealth(BaseModel):
    reachable: bool
    embedding_model: str
    embedding_model_installed: bool
    chat_model: str
    chat_model_installed: bool


class IndexHealth(BaseModel):
    documents: int
    chunks: int


class HealthResponse(BaseModel):
    status: str  # "ok" when Ollama and both models are available, otherwise "degraded"
    ollama: OllamaHealth
    index: IndexHealth
    admin_endpoints_enabled: bool
