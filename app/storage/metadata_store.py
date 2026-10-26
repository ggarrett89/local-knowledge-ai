"""SQLite metadata store: source of truth for documents, categories, topics and errors."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.clock import utc_now_iso
from app.core.errors import LkaError
from app.core.models import DocumentRecord, normalize_slug

SCHEMA_VERSION = 1
_MAX_ERROR_LEN = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS categories (
    slug TEXT PRIMARY KEY,
    description TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    category TEXT NOT NULL,
    source_uri TEXT,
    file_path TEXT,
    file_hash TEXT,
    version TEXT,
    language TEXT,
    status TEXT NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    captured_at TEXT,
    updated_at TEXT,
    indexed_at TEXT,
    ingest_seconds REAL,
    embedding_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE TABLE IF NOT EXISTS document_topics (
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    PRIMARY KEY (document_id, topic)
);
CREATE TABLE IF NOT EXISTS ingestion_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    source_uri TEXT,
    stage TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_errors_document ON ingestion_errors(document_id);
"""

_DOC_COLUMNS = (
    "document_id", "title", "source_type", "category", "source_uri", "file_path", "file_hash",
    "version", "language", "status", "page_count", "chunk_count", "captured_at", "updated_at",
    "indexed_at", "ingest_seconds", "embedding_model",
)

_SELECT_DOCS = (
    "SELECT d.*, "
    "(SELECT group_concat(topic, ',') FROM "
    "(SELECT topic FROM document_topics t WHERE t.document_id = d.document_id ORDER BY topic)) "
    "AS topics_csv FROM documents d"
)


class SqliteMetadataStore:
    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    # -- lifecycle -----------------------------------------------------------------

    def _migrate(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('version', ?)", (str(SCHEMA_VERSION),)
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise LkaError(
                    f"metadata database schema v{row['value']} is newer than this application "
                    f"(v{SCHEMA_VERSION}); upgrade the application"
                )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteMetadataStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- categories ----------------------------------------------------------------

    def ensure_category(self, slug: str, description: Optional[str] = None) -> bool:
        """Register a category if missing. Returns True when it was created."""
        slug = normalize_slug(slug)
        with self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO categories (slug, description) VALUES (?, ?)", (slug, description)
            )
        return cur.rowcount > 0

    def seed_categories_from_file(self, path: Path) -> int:
        """Register categories listed in a JSON file; existing ones are untouched."""
        path = Path(path)
        if not path.is_file():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        created = 0
        for item in data.get("categories", []):
            if self.ensure_category(item["slug"], item.get("description")):
                created += 1
        return created

    def list_categories(self) -> List[Dict[str, Optional[str]]]:
        rows = self._conn.execute("SELECT slug, description FROM categories ORDER BY slug").fetchall()
        return [{"slug": r["slug"], "description": r["description"]} for r in rows]

    # -- documents -----------------------------------------------------------------

    def upsert_document(self, doc: DocumentRecord) -> None:
        values = {col: getattr(doc, col) for col in _DOC_COLUMNS}
        placeholders = ", ".join(f":{c}" for c in _DOC_COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in _DOC_COLUMNS if c != "document_id")
        with self._conn:
            self._conn.execute(
                f"INSERT INTO documents ({', '.join(_DOC_COLUMNS)}) VALUES ({placeholders}) "
                f"ON CONFLICT(document_id) DO UPDATE SET {updates}",
                values,
            )
            self._conn.execute("DELETE FROM document_topics WHERE document_id = ?", (doc.document_id,))
            self._conn.executemany(
                "INSERT INTO document_topics (document_id, topic) VALUES (?, ?)",
                [(doc.document_id, t) for t in sorted(set(doc.topics))],
            )

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        row = self._conn.execute(f"{_SELECT_DOCS} WHERE d.document_id = ?", (document_id,)).fetchone()
        return _to_record(row) if row else None

    def find_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        row = self._conn.execute(
            f"{_SELECT_DOCS} WHERE d.file_hash = ? ORDER BY d.document_id LIMIT 1", (file_hash,)
        ).fetchone()
        return _to_record(row) if row else None

    def find_by_path(self, file_path: str) -> List[DocumentRecord]:
        rows = self._conn.execute(
            f"{_SELECT_DOCS} WHERE d.file_path = ? ORDER BY d.document_id", (file_path,)
        ).fetchall()
        return [_to_record(r) for r in rows]

    def list_documents(
        self,
        *,
        category: Optional[str] = None,
        topics: Optional[Sequence[str]] = None,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DocumentRecord]:
        where: List[str] = []
        params: List[Any] = []
        if category is not None:
            where.append("d.category = ?")
            params.append(category)
        if source_type is not None:
            where.append("d.source_type = ?")
            params.append(source_type)
        if status is not None:
            where.append("d.status = ?")
            params.append(status)
        for topic in topics or []:
            where.append(
                "EXISTS (SELECT 1 FROM document_topics t WHERE t.document_id = d.document_id AND t.topic = ?)"
            )
            params.append(topic)
        sql = _SELECT_DOCS + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY d.title COLLATE NOCASE, d.document_id"
        return [_to_record(r) for r in self._conn.execute(sql, params).fetchall()]

    def remove_document(self, document_id: str) -> bool:
        with self._conn:
            self._conn.execute("DELETE FROM ingestion_errors WHERE document_id = ?", (document_id,))
            cur = self._conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cur.rowcount > 0

    # -- errors --------------------------------------------------------------------

    def record_error(
        self,
        stage: str,
        message: str,
        *,
        document_id: Optional[str] = None,
        source_uri: Optional[str] = None,
    ) -> None:
        """Record a processing error. ``message`` must never contain document text."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO ingestion_errors (document_id, source_uri, stage, message, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (document_id, source_uri, stage, message[:_MAX_ERROR_LEN], utc_now_iso()),
            )

    def list_errors(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if document_id is None:
            rows = self._conn.execute("SELECT * FROM ingestion_errors ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM ingestion_errors WHERE document_id = ? ORDER BY id", (document_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_errors(self, document_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM ingestion_errors WHERE document_id = ?", (document_id,))

    # -- statistics ----------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        c = self._conn
        totals = c.execute(
            "SELECT COUNT(*) AS documents, COALESCE(SUM(page_count), 0) AS pages, "
            "COALESCE(SUM(chunk_count), 0) AS chunks, COALESCE(SUM(ingest_seconds), 0) AS seconds "
            "FROM documents"
        ).fetchone()
        by_status = {r["status"]: r["n"] for r in c.execute(
            "SELECT status, COUNT(*) AS n FROM documents GROUP BY status ORDER BY status")}
        by_category = {r["category"]: r["n"] for r in c.execute(
            "SELECT category, COUNT(*) AS n FROM documents GROUP BY category ORDER BY category")}
        errors = c.execute("SELECT COUNT(*) AS n FROM ingestion_errors").fetchone()["n"]
        return {
            "documents": totals["documents"],
            "pages": totals["pages"],
            "chunks": totals["chunks"],
            "ingest_seconds": round(totals["seconds"], 2),
            "by_status": by_status,
            "by_category": by_category,
            "errors": errors,
        }


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    data = {col: row[col] for col in _DOC_COLUMNS}
    csv = row["topics_csv"]
    data["topics"] = csv.split(",") if csv else []
    return DocumentRecord(**data)
