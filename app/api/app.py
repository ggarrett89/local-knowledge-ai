"""FastAPI application. Routes are thin: validation, authentication, then one service call."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.api.errors import install_error_handlers
from app.api.schemas import (
    ChatRequest,
    DeleteResponse,
    DocumentDetail,
    DocumentList,
    DocumentOut,
    ErrorOut,
    HealthResponse,
    IndexHealth,
    IngestRequest,
    IngestResponse,
    IngestResultOut,
    OllamaHealth,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
)
from app.api.security import AdminRoute
from app.clients.ollama import OllamaCatalog
from app.core.config import Settings, load_settings
from app.core.errors import DocumentNotFound, InvalidDocument, OllamaUnavailable
from app.core.models import DocumentRecord, normalize_slug
from app.core.references import format_reference
from app.rag.models import RagAnswer
from app.services import Services, build_services

logger = logging.getLogger(__name__)


class BodyLimitMiddleware:
    """Reject request bodies above ``max_bytes`` (declared length or actually streamed)."""

    def __init__(self, app, max_bytes: int) -> None:  # noqa: ANN001 - ASGI app
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return
        received = 0

        async def limited_receive():  # noqa: ANN202
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI lets HTTPException through while reading the body (other errors become 400)
                    raise HTTPException(status_code=413, detail=f"Request body exceeds {self.max_bytes} bytes.")
            return message

        await self.app(scope, limited_receive, send)

    async def _reject(self, scope, receive, send) -> None:  # noqa: ANN001
        response = JSONResponse(status_code=413, content={"detail": f"Request body exceeds {self.max_bytes} bytes."})
        await response(scope, receive, send)


def get_services(request: Request) -> Services:
    return request.app.state.services


def resolve_inside(root: Path, raw: str) -> Path:
    """Resolve ``raw`` (relative to ``root`` or absolute) and require it to stay inside ``root``.

    Symlinks are resolved first, so a link that points outside the directory is rejected too.
    """
    root = root.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=403, detail="The path is outside the documents directory.")
    return resolved


def _document_out(doc: DocumentRecord) -> DocumentOut:
    return DocumentOut(
        document_id=doc.document_id, title=doc.title, source_type=doc.source_type, category=doc.category,
        topics=doc.topics, status=doc.status, page_count=doc.page_count, chunk_count=doc.chunk_count,
        file_name=Path(doc.file_path).name if doc.file_path else None, version=doc.version,
        language=doc.language, indexed_at=doc.indexed_at, updated_at=doc.updated_at,
        embedding_model=doc.embedding_model,
    )


def create_app(settings: Optional[Settings] = None, services: Optional[Services] = None) -> FastAPI:
    """Build the application. Pass ``services`` to inject fakes (tests); otherwise they are opened
    at startup (failing fast if the index uses another embeddings model) and closed at shutdown."""
    settings = settings or (services.settings if services else load_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        owned = services is None
        app.state.services = services if services is not None else build_services(settings)
        if settings.api_host not in ("127.0.0.1", "localhost", "::1") and settings.admin_token is None:
            logger.warning("API is reachable beyond loopback; administrative endpoints stay disabled (no token)")
        try:
            yield
        finally:
            if owned:
                app.state.services.close()

    app = FastAPI(title="local-knowledge-ai", lifespan=lifespan)
    if services is not None:  # available even when the lifespan is not run
        app.state.services = services
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.api_max_body_bytes)
    install_error_handlers(app)

    # -- health ----------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health(svc: Services = Depends(get_services)) -> HealthResponse:
        s = svc.settings
        reachable, embed_ok, chat_ok = True, False, False
        try:
            installed = svc.catalog.installed_models()  # type: ignore[attr-defined]
            embed_ok = OllamaCatalog.matches(installed, s.embedding_model)
            chat_ok = OllamaCatalog.matches(installed, s.chat_model)
        except OllamaUnavailable:
            reachable = False
        stats = svc.metadata.stats()
        return HealthResponse(
            status="ok" if (reachable and embed_ok and chat_ok) else "degraded",
            ollama=OllamaHealth(
                reachable=reachable, embedding_model=s.embedding_model, embedding_model_installed=embed_ok,
                chat_model=s.chat_model, chat_model_installed=chat_ok,
            ),
            index=IndexHealth(documents=stats["documents"], chunks=stats["chunks"]),
            admin_endpoints_enabled=s.admin_token is not None,
        )

    # -- search and chat ---------------------------------------------------------
    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, svc: Services = Depends(get_services)) -> SearchResponse:
        hits = svc.retriever.search(body.query, k=body.k, filters=body.filters, max_distance=body.max_distance)
        results = [
            SearchHitOut(
                rank=i, distance=round(h.distance, 4), reference=format_reference(h.chunk.provenance),
                document_id=h.chunk.provenance.document_id, chunk_id=h.chunk.chunk_id,
                title=h.chunk.provenance.title, page_start=h.chunk.provenance.page_start,
                page_end=h.chunk.provenance.page_end, text=h.chunk.text,
            )
            for i, h in enumerate(hits, 1)
        ]
        return SearchResponse(count=len(results), results=results)

    @app.post("/chat", response_model=RagAnswer)
    def chat(body: ChatRequest, svc: Services = Depends(get_services)) -> RagAnswer:
        return svc.rag.answer(body.question, k=body.k, filters=body.filters, max_distance=body.max_distance)

    # -- documents (read) -------------------------------------------------------
    @app.get("/documents", response_model=DocumentList)
    def list_documents(
        category: Optional[str] = Query(None, max_length=64),
        topic: Optional[List[str]] = Query(None, max_length=20),
        status: Optional[str] = Query(None, pattern="^(indexed|no_text|error)$"),
        svc: Services = Depends(get_services),
    ) -> DocumentList:
        # normalize_slug raises InvalidSlug, which the error handlers turn into a 422
        docs = svc.metadata.list_documents(
            category=normalize_slug(category) if category else None,
            topics=[normalize_slug(t) for t in topic] if topic else None,
            status=status,
        )
        return DocumentList(count=len(docs), documents=[_document_out(d) for d in docs])

    @app.get("/documents/{document_id}", response_model=DocumentDetail)
    def get_document(document_id: str, svc: Services = Depends(get_services)) -> DocumentDetail:
        doc = svc.metadata.get_document(document_id)
        if doc is None:
            raise DocumentNotFound(document_id)
        errors = [ErrorOut(stage=e["stage"], message=e["message"], created_at=e["created_at"])
                  for e in svc.metadata.list_errors(document_id)]
        return DocumentDetail(**_document_out(doc).model_dump(), errors=errors)

    # -- documents (administrative) ------------------------------------------------
    admin = APIRouter(route_class=AdminRoute)

    @admin.post("/documents/ingest", response_model=IngestResponse)
    def ingest(body: IngestRequest, svc: Services = Depends(get_services)) -> IngestResponse:
        """Index a file or folder inside LKA_DOCUMENTS_DIR. Runs synchronously: large books take minutes
        (about 1 s per chunk on CPU), so prefer the CLI for those."""
        documents_dir = svc.settings.documents_dir
        target = resolve_inside(documents_dir, body.path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="File or folder not found inside the documents directory.")
        if not svc.ingest_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another ingestion is already running.")
        try:
            root = documents_dir.resolve()
            if target.is_dir():
                summary = svc.ingestion.ingest_folder(
                    target, body.category, body.topics, force=body.force, confine_to_root=True
                )
                raw = summary.results
                counts, aborted, skipped = summary.counts(), summary.aborted, summary.skipped_after_abort
            else:
                raw = [svc.ingestion.ingest_file(target, body.category, body.topics, force=body.force)]
                counts, aborted, skipped = {raw[0].outcome: 1}, None, 0
        finally:
            svc.ingest_lock.release()
        results = [
            IngestResultOut(**{**r.model_dump(), "path": _relative(Path(r.path), root)}) for r in raw
        ]
        return IngestResponse(counts=counts, results=results, aborted=aborted, skipped_after_abort=skipped)

    @admin.delete("/documents/{document_id}", response_model=DeleteResponse)
    def delete_document(
        document_id: str,
        confirm: bool = Query(False, description="Must be true: removing a document from the index is destructive"),
        svc: Services = Depends(get_services),
    ) -> DeleteResponse:
        if not confirm:
            raise HTTPException(status_code=400, detail="Destructive operation: repeat the request with confirm=true.")
        if not svc.ingest_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="An ingestion is running; try again when it finishes.")
        try:
            removed = svc.ingestion.remove(document_id)  # the original file is never touched
        finally:
            svc.ingest_lock.release()
        if not removed:
            raise DocumentNotFound(document_id)
        return DeleteResponse(document_id=document_id, removed=True)

    app.include_router(admin)
    return app


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name
