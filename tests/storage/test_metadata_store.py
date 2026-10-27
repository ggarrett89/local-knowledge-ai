from pathlib import Path

import pytest

from app.core.errors import InvalidSlug
from app.core.models import DocumentRecord
from app.storage.metadata_store import SqliteMetadataStore

REPO_ROOT = Path(__file__).resolve().parents[2]


def doc(document_id="pdf-1", **overrides) -> DocumentRecord:
    data = dict(
        document_id=document_id,
        title=f"Title {document_id}",
        source_type="pdf",
        category="general",
        topics=[],
        file_path=f"/books/{document_id}.pdf",
        file_hash=f"hash-{document_id}",
        status="indexed",
        page_count=10,
        chunk_count=5,
    )
    data.update(overrides)
    return DocumentRecord(**data)


@pytest.fixture
def store(tmp_path):
    s = SqliteMetadataStore(tmp_path / "meta" / "metadata.db")
    yield s
    s.close()


def test_creates_parent_directory_and_roundtrips_a_document(tmp_path):
    with SqliteMetadataStore(tmp_path / "deep" / "dir" / "m.db") as s:
        record = doc(topics=["networking", "security"], version="1.2", embedding_model="m|x", ingest_seconds=1.5)
        s.upsert_document(record)
        assert s.get_document("pdf-1") == record


def test_get_missing_document_returns_none(store):
    assert store.get_document("nope") is None


def test_upsert_updates_and_replaces_topics(store):
    store.upsert_document(doc(topics=["a", "b"]))
    store.upsert_document(doc(topics=["c"], chunk_count=9))
    got = store.get_document("pdf-1")
    assert got.topics == ["c"] and got.chunk_count == 9
    assert len(store.list_documents()) == 1


def test_find_by_hash_and_path(store):
    store.upsert_document(doc("pdf-1"))
    store.upsert_document(doc("pdf-2", file_path="/books/pdf-1.pdf", file_hash="other"))
    assert store.find_by_hash("hash-pdf-1").document_id == "pdf-1"
    assert store.find_by_hash("missing") is None
    assert [d.document_id for d in store.find_by_path("/books/pdf-1.pdf")] == ["pdf-1", "pdf-2"]


def test_list_filters(store):
    store.upsert_document(doc("pdf-1", category="cloud", topics=["aws", "iam"]))
    store.upsert_document(doc("pdf-2", category="cloud", topics=["aws"]))
    store.upsert_document(doc("pdf-3", category="security", topics=["iam"], status="error"))
    ids = lambda **kw: [d.document_id for d in store.list_documents(**kw)]  # noqa: E731
    assert ids() == ["pdf-1", "pdf-2", "pdf-3"]
    assert ids(category="cloud") == ["pdf-1", "pdf-2"]
    assert ids(topics=["aws"]) == ["pdf-1", "pdf-2"]
    assert ids(topics=["aws", "iam"]) == ["pdf-1"]  # all topics must match
    assert ids(status="error") == ["pdf-3"]
    assert ids(source_type="web") == []


def test_filters_are_parameterized_against_sql_injection(store):
    store.upsert_document(doc())
    assert store.list_documents(category="x'; DROP TABLE documents; --") == []
    assert store.list_documents(topics=["' OR '1'='1"]) == []
    assert len(store.list_documents()) == 1


def test_remove_document_deletes_topics_and_errors(store):
    store.upsert_document(doc(topics=["a"]))
    store.record_error("extract", "boom", document_id="pdf-1")
    assert store.remove_document("pdf-1") is True
    assert store.get_document("pdf-1") is None
    assert store.list_errors("pdf-1") == []
    assert store._conn.execute("SELECT COUNT(*) FROM document_topics").fetchone()[0] == 0
    assert store.remove_document("pdf-1") is False


def test_errors_are_recorded_listed_cleared_and_truncated(store):
    store.record_error("extract", "x" * 2000, document_id="pdf-1", source_uri="file:///a.pdf")
    store.record_error("validate", "not found", source_uri="file:///b.pdf")
    all_errors = store.list_errors()
    assert [e["stage"] for e in all_errors] == ["extract", "validate"]
    assert len(all_errors[0]["message"]) == 500
    assert all_errors[1]["document_id"] is None
    assert len(store.list_errors("pdf-1")) == 1
    store.clear_errors("pdf-1")
    assert store.list_errors("pdf-1") == []
    assert len(store.list_errors()) == 1


def test_ensure_category_is_idempotent_and_validates(store):
    assert store.ensure_category("My-New-Cat", "desc") is True
    assert store.ensure_category("my-new-cat") is False
    assert {"slug": "my-new-cat", "description": "desc"} in store.list_categories()
    with pytest.raises(InvalidSlug):
        store.ensure_category("not valid!")


def test_seed_categories_from_repo_file_is_idempotent(store):
    path = REPO_ROOT / "config" / "categories.json"
    assert store.seed_categories_from_file(path) == 11
    assert store.seed_categories_from_file(path) == 0
    assert "iac-automation" in [c["slug"] for c in store.list_categories()]


def test_seed_with_missing_file_is_a_noop(store, tmp_path):
    assert store.seed_categories_from_file(tmp_path / "missing.json") == 0
    assert store.list_categories() == []


def test_data_persists_after_reopening(tmp_path):
    path = tmp_path / "m.db"
    s = SqliteMetadataStore(path)
    s.upsert_document(doc(topics=["t"]))
    s.ensure_category("custom")
    s.record_error("x", "msg", document_id="pdf-1")
    s.close()
    s2 = SqliteMetadataStore(path)
    try:
        assert s2.get_document("pdf-1").topics == ["t"]
        assert "custom" in [c["slug"] for c in s2.list_categories()]
        assert len(s2.list_errors()) == 1
    finally:
        s2.close()


def test_stats(store):
    store.upsert_document(doc("pdf-1", category="cloud", page_count=10, chunk_count=5, ingest_seconds=1.0))
    store.upsert_document(doc("pdf-2", category="cloud", page_count=4, chunk_count=2, ingest_seconds=0.5))
    store.upsert_document(doc("pdf-3", status="no_text", page_count=2, chunk_count=0))
    store.record_error("extract", "e", document_id="pdf-3")
    stats = store.stats()
    assert stats["documents"] == 3
    assert stats["pages"] == 16 and stats["chunks"] == 7
    assert stats["ingest_seconds"] == 1.5
    assert stats["by_status"] == {"indexed": 2, "no_text": 1}
    assert stats["by_category"] == {"cloud": 2, "general": 1}
    assert stats["errors"] == 1


def test_stats_on_empty_store(store):
    stats = store.stats()
    assert stats["documents"] == 0 and stats["chunks"] == 0 and stats["by_status"] == {}


def test_store_can_be_used_from_several_threads(tmp_path):
    """The API serves requests from a thread pool: one connection, guarded by a lock."""
    import threading

    store = SqliteMetadataStore(tmp_path / "threads.db")
    errors = []

    def worker(n):
        try:
            for i in range(25):
                store.upsert_document(doc(f"pdf-{n}-{i}", topics=["t"]))
                assert store.get_document(f"pdf-{n}-{i}") is not None
                store.list_documents(topics=["t"])
                store.record_error("x", "m", document_id=f"pdf-{n}-{i}")
        except BaseException as exc:  # noqa: BLE001 - report any failure from the thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert errors == []
        assert len(store.list_documents()) == 150
        assert len(store.list_errors()) == 150
    finally:
        store.close()
