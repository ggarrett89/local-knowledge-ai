import hashlib
import logging
import shutil
from typing import Iterator

import pytest

from app.core.errors import (
    DocumentNotFound,
    EmbeddingError,
    InvalidDocument,
    InvalidSlug,
    OllamaUnavailable,
)
from app.core.models import Section, SourceInfo
from app.ingestion.service import IngestionService
from app.storage.metadata_store import SqliteMetadataStore
from app.storage.vector_store import ChromaVectorStore
from tests.helpers import FakeEmbedder, make_pdf, words

CHUNK_SIZE = 400
OVERLAP = 50
BATCH = 3


class Env:
    """Real SQLite + Chroma in a temp dir, fake embeddings."""

    def __init__(self, root, embedder=None, **service_kwargs):
        self.root = root
        self.embedder = embedder or FakeEmbedder()
        self.metadata = SqliteMetadataStore(root / "data" / "metadata.db")
        self.vectors = ChromaVectorStore(root / "data" / "vdb", self.embedder.identity)
        kwargs = dict(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, batch_size=BATCH)
        kwargs.update(service_kwargs)
        self.service = IngestionService(self.metadata, self.vectors, self.embedder, **kwargs)

    def close(self):
        self.vectors.close()
        self.metadata.close()


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


def book(tmp_path, name="book.pdf", tag="a", n_pages=3, words_per_page=150, **kw):
    pages = [words(f"{tag}{p}t", words_per_page) for p in range(1, n_pages + 1)]
    return make_pdf(tmp_path / "library" / name, pages, **kw)


# --- happy path -------------------------------------------------------------------


def test_indexes_a_pdf_with_metadata_and_provenance(env, tmp_path):
    path = book(tmp_path, title="Terraform Up and Running")
    result = env.service.ingest_file(path, "iac-automation", ["terraform", "Modules"])

    assert result.outcome == "indexed"
    doc = env.metadata.get_document(result.document_id)
    assert doc.document_id.startswith("pdf-") and len(doc.document_id) == len("pdf-") + 16
    assert doc.title == "Terraform Up and Running"
    assert (doc.category, doc.topics) == ("iac-automation", ["modules", "terraform"])
    assert doc.status == "indexed" and doc.source_type == "pdf" and doc.page_count == 3
    assert doc.file_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc.file_path == str(path.resolve()) and doc.source_uri == path.resolve().as_uri()
    assert doc.embedding_model == env.embedder.identity
    assert doc.chunk_count == env.vectors.count_document(doc.document_id) > 3
    assert doc.version is None and doc.language is None and doc.captured_at is None  # never invented

    stored = env.vectors.get_chunks(doc.document_id)
    assert [c.chunk_id for c in stored] == [f"{doc.document_id}:{i}" for i in range(doc.chunk_count)]
    for chunk in stored:
        meta = chunk.metadata
        assert meta["content_hash"] == hashlib.sha256(chunk.text.encode()).hexdigest()
        assert meta["title"] == "Terraform Up and Running" and meta["source_type"] == "pdf"
        assert 1 <= meta["page_start"] <= meta["page_end"] <= 3
        text_pages = {int(tok[1]) for tok in chunk.text.split()}  # tokens look like "a2t0007"
        assert (meta["page_start"], meta["page_end"]) == (min(text_pages), max(text_pages))
    assert stored[0].metadata["page_start"] == 1 and stored[-1].metadata["page_end"] == 3


def test_category_defaults_to_general_and_is_registered(env, tmp_path):
    result = env.service.ingest_file(book(tmp_path))
    assert env.metadata.get_document(result.document_id).category == "general"
    env.service.ingest_file(book(tmp_path, "b.pdf", tag="b"), "brand-new-category")
    assert "brand-new-category" in [c["slug"] for c in env.metadata.list_categories()]


def test_embeddings_are_requested_in_small_batches(env, tmp_path):
    result = env.service.ingest_file(book(tmp_path))
    assert max(env.embedder.calls) <= BATCH
    assert len(env.embedder.calls) > 1
    assert sum(env.embedder.calls) == result.chunks


def test_documents_are_streamed_not_loaded_whole(tmp_path):
    class CountingExtractor:
        source_type = "pdf"

        def __init__(self):
            self.yielded = 0

        def describe(self, path):
            return SourceInfo(title="Big", page_count=60)

        def extract(self, path) -> Iterator[Section]:
            for n in range(1, 61):
                self.yielded += 1
                yield Section(text=words(f"p{n}t", 60), page_number=n)

    extractor = CountingExtractor()
    embedder = FakeEmbedder()
    seen_at_first_call = []
    embedder.on_call = lambda call_no: call_no == 1 and seen_at_first_call.append(extractor.yielded)
    e = Env(tmp_path, embedder, extractors={".pdf": extractor})
    try:
        path = tmp_path / "x.pdf"
        path.write_bytes(b"placeholder bytes, the fake extractor ignores the content")
        assert e.service.ingest_file(path).outcome == "indexed"
    finally:
        e.close()
    assert seen_at_first_call and seen_at_first_call[0] < 10, "embedding must start before reading all pages"
    assert extractor.yielded == 60


# --- incremental behaviour --------------------------------------------------------


def test_adding_the_same_file_again_is_a_noop(env, tmp_path):
    path = book(tmp_path)
    first = env.service.ingest_file(path)
    calls_after_first = list(env.embedder.calls)
    second = env.service.ingest_file(path)
    assert second.outcome == "unchanged" and second.document_id == first.document_id
    assert env.embedder.calls == calls_after_first
    assert len(env.metadata.list_documents()) == 1
    assert env.vectors.count() == first.chunks


def test_same_content_at_another_path_is_a_duplicate(env, tmp_path):
    path = book(tmp_path)
    copy = tmp_path / "library" / "copy-of-book.pdf"
    shutil.copy(path, copy)
    first = env.service.ingest_file(path)
    second = env.service.ingest_file(copy)
    assert second.outcome == "duplicate" and second.document_id == first.document_id
    assert str(path.resolve()) in second.message
    assert len(env.metadata.list_documents()) == 1


def test_changed_file_replaces_the_old_version_and_inherits_metadata(env, tmp_path):
    path = book(tmp_path, tag="old")
    old = env.service.ingest_file(path, "cloud", ["aws"])
    make_pdf(path, [words("newp1t", 150), words("newp2t", 150)])  # same path, new content
    new = env.service.ingest_file(path)

    assert new.outcome == "updated" and new.document_id != old.document_id
    assert env.metadata.get_document(old.document_id) is None
    assert env.vectors.count_document(old.document_id) == 0
    doc = env.metadata.get_document(new.document_id)
    assert (doc.category, doc.topics, doc.page_count) == ("cloud", ["aws"], 2)
    assert env.vectors.count() == doc.chunk_count


def test_forced_reindex_removes_stale_chunks(tmp_path):
    path = book(tmp_path, n_pages=4)
    small = Env(tmp_path, chunk_size=300)
    try:
        first = small.service.ingest_file(path)
        many = first.chunks
        big = IngestionService(
            small.metadata, small.vectors, small.embedder, chunk_size=1500, chunk_overlap=50, batch_size=BATCH
        )
        again = big.ingest_file(path, force=True)
        assert again.outcome == "reindexed" and again.document_id == first.document_id
        assert again.chunks < many
        assert small.vectors.count_document(first.document_id) == again.chunks
        assert small.metadata.get_document(first.document_id).chunk_count == again.chunks
    finally:
        small.close()


def test_reindex_by_id_keeps_category_and_topics(env, tmp_path):
    result = env.service.ingest_file(book(tmp_path), "security", ["iam"])
    again = env.service.reindex(result.document_id)
    assert again.outcome == "reindexed"
    doc = env.metadata.get_document(result.document_id)
    assert (doc.category, doc.topics) == ("security", ["iam"])


def test_reindex_unknown_id_and_missing_file(env, tmp_path):
    with pytest.raises(DocumentNotFound):
        env.service.reindex("pdf-doesnotexist")
    path = book(tmp_path)
    result = env.service.ingest_file(path)
    path.unlink()
    with pytest.raises(InvalidDocument):
        env.service.reindex(result.document_id)


# --- problematic input ------------------------------------------------------------


def test_pdf_without_text_is_marked_and_not_indexed(env, tmp_path):
    path = make_pdf(tmp_path / "library" / "scan.pdf", ["", ""])
    result = env.service.ingest_file(path)
    assert result.outcome == "no_text" and result.chunks == 0
    doc = env.metadata.get_document(result.document_id)
    assert doc.status == "no_text" and doc.page_count == 2
    assert env.vectors.count() == 0 and env.embedder.calls == []
    errors = env.metadata.list_errors(result.document_id)
    assert len(errors) == 1 and "no extractable text" in errors[0]["message"]
    assert env.service.ingest_file(path).outcome == "unchanged"


def test_corrupted_pdf_is_recorded_and_other_files_continue(env, tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "a-broken.pdf").write_bytes(b"not really a pdf")
    book(tmp_path, "z-good.pdf")
    summary = env.service.ingest_folder(lib, "general")

    assert summary.counts() == {"error": 1, "indexed": 1}
    assert summary.has_errors and summary.aborted is None
    broken = next(r for r in summary.results if r.outcome == "error")
    doc = env.metadata.get_document(broken.document_id)
    assert doc.status == "error" and doc.chunk_count == 0
    assert env.vectors.count_document(broken.document_id) == 0
    assert env.metadata.list_errors(broken.document_id)[0]["stage"] == "extract"
    assert env.vectors.count() == next(r for r in summary.results if r.outcome == "indexed").chunks


def test_failure_in_the_middle_leaves_no_partial_chunks_and_can_be_retried(tmp_path):
    embedder = FakeEmbedder(error=EmbeddingError("input too long"), fail_on_call=2)
    e = Env(tmp_path, embedder)
    try:
        path = book(tmp_path)
        failed = e.service.ingest_file(path)
        assert failed.outcome == "error" and "input too long" in failed.message
        assert len(embedder.calls) == 2
        assert e.vectors.count() == 0, "chunks written before the failure must be removed"
        assert e.metadata.get_document(failed.document_id).status == "error"
        assert e.metadata.list_errors(failed.document_id)[0]["stage"] == "embed"

        embedder.error = None  # the problem is fixed; `add` again retries error documents
        retried = e.service.ingest_file(path)
        assert retried.outcome == "reindexed"
        assert e.metadata.get_document(retried.document_id).status == "indexed"
        assert e.metadata.list_errors(retried.document_id) == []
        assert e.vectors.count() == retried.chunks
    finally:
        e.close()


def test_ollama_down_raises_records_the_error_and_leaves_nothing(tmp_path):
    e = Env(tmp_path, FakeEmbedder(error=OllamaUnavailable("cannot reach Ollama")))
    try:
        with pytest.raises(OllamaUnavailable):
            e.service.ingest_file(book(tmp_path))
        assert e.vectors.count() == 0
        (doc,) = e.metadata.list_documents()
        assert doc.status == "error"
    finally:
        e.close()


def test_batch_stops_early_when_the_service_is_down(tmp_path):
    e = Env(tmp_path, FakeEmbedder(error=OllamaUnavailable("cannot reach Ollama")))
    try:
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            book(tmp_path, name, tag=name[0])
        summary = e.service.ingest_folder(tmp_path / "library")
        assert summary.aborted and "Ollama" in summary.aborted
        assert len(summary.results) == 1 and summary.skipped_after_abort == 2
        assert len(e.embedder.calls) == 1, "must not hammer a service that is down"
        assert summary.has_errors
    finally:
        e.close()


def test_missing_and_unsupported_files_are_rejected_without_creating_documents(env, tmp_path):
    missing = env.service.ingest_file(tmp_path / "nope.pdf")
    note = tmp_path / "notes.txt"
    note.write_text("plain text")
    unsupported = env.service.ingest_file(note)
    assert missing.outcome == unsupported.outcome == "error"
    assert "unsupported" in unsupported.message
    assert env.metadata.list_documents() == []
    assert [e["stage"] for e in env.metadata.list_errors()] == ["validate", "validate"]


def test_invalid_category_or_topic_is_rejected_before_any_work(env, tmp_path):
    path = book(tmp_path)
    with pytest.raises(InvalidSlug):
        env.service.ingest_file(path, "Bad Category!")
    with pytest.raises(InvalidSlug):
        env.service.ingest_file(path, "ok", ["../etc"])
    assert env.embedder.calls == [] and env.metadata.list_documents() == []


# --- folders ----------------------------------------------------------------------


def test_add_all_is_recursive_filters_by_type_and_is_deterministic(env, tmp_path):
    lib = tmp_path / "library"
    book(tmp_path, "one.pdf", tag="a")
    make_pdf(lib / "sub" / "deeper" / "two.PDF", [words("bt", 60)])
    (lib / "readme.txt").write_text("ignore me")
    (lib / "sub" / "image.png").write_bytes(b"\x89PNG")
    summary = env.service.ingest_folder(lib, "devops-sre", ["sre"])
    assert summary.counts() == {"indexed": 2}
    assert [r.path.rsplit("/", 1)[-1] for r in summary.results] == ["one.pdf", "two.PDF"]
    assert {d.category for d in env.metadata.list_documents()} == {"devops-sre"}
    calls_before = len(env.embedder.calls)
    again = env.service.ingest_folder(lib)
    assert again.counts() == {"unchanged": 2}
    assert len(env.embedder.calls) == calls_before, "re-running add-all must not re-embed anything"


def test_add_all_on_a_missing_directory_fails_clearly(env, tmp_path):
    with pytest.raises(InvalidDocument):
        env.service.ingest_folder(tmp_path / "nope")


def test_add_all_on_an_empty_directory_is_not_an_error(env, tmp_path):
    (tmp_path / "empty").mkdir()
    summary = env.service.ingest_folder(tmp_path / "empty")
    assert summary.results == [] and not summary.has_errors


# --- removal, persistence, safety -------------------------------------------------


def test_remove_deletes_the_index_entry_but_never_the_file(env, tmp_path):
    path = book(tmp_path)
    other = env.service.ingest_file(book(tmp_path, "keep.pdf", tag="k"))
    result = env.service.ingest_file(path)
    assert env.service.remove(result.document_id) is True
    assert env.metadata.get_document(result.document_id) is None
    assert env.vectors.count_document(result.document_id) == 0
    assert path.is_file(), "the original file must be kept"
    assert env.vectors.count_document(other.document_id) == other.chunks
    assert env.service.remove(result.document_id) is False
    # the file can be indexed again afterwards
    assert env.service.ingest_file(path).outcome == "indexed"


def test_index_survives_a_restart_without_reprocessing(tmp_path):
    path = book(tmp_path)
    first_env = Env(tmp_path)
    first = first_env.service.ingest_file(path, "cloud", ["aws"])
    first_env.close()

    second_env = Env(tmp_path)
    try:
        (doc,) = second_env.service.list_documents()
        assert doc.document_id == first.document_id and doc.topics == ["aws"]
        assert second_env.vectors.count() == first.chunks
        again = second_env.service.ingest_file(path)
        assert again.outcome == "unchanged" and second_env.embedder.calls == []
    finally:
        second_env.close()


def test_ingestion_does_not_touch_the_source_directory(env, tmp_path):
    path = book(tmp_path)
    snapshot = lambda: sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in path.parent.iterdir())  # noqa: E731
    before = snapshot()
    result = env.service.ingest_file(path)
    env.service.reindex(result.document_id)
    env.service.remove(result.document_id)
    assert snapshot() == before


def test_stats_combine_metadata_and_vector_counts(env, tmp_path):
    result = env.service.ingest_file(book(tmp_path), "cloud")
    stats = env.service.stats()
    assert stats["documents"] == 1 and stats["pages"] == 3
    assert stats["chunks"] == stats["vector_chunks"] == result.chunks
    assert stats["by_category"] == {"cloud": 1}


def test_logs_never_contain_document_text(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    secret = "ZEBRA-CONFIDENTIAL-PARAGRAPH"
    path = make_pdf(tmp_path / "library" / "private.pdf", [f"{secret} " + words("t", 50)])
    ok_env = Env(tmp_path)
    try:
        ok_env.service.ingest_file(path)
    finally:
        ok_env.close()
    failing = Env(tmp_path / "other", FakeEmbedder(error=EmbeddingError("boom")))
    try:
        failing.service.ingest_file(path)
        stored_errors = " ".join(
            row[0] for row in failing.metadata._conn.execute("SELECT message FROM ingestion_errors")
        )
    finally:
        failing.close()
    assert stored_errors, "the failure must have been recorded"
    assert secret not in caplog.text
    assert secret not in stored_errors
