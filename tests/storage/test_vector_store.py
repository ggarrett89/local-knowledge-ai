import pytest

from app.core.errors import EmbeddingModelMismatch
from app.core.models import Chunk, Provenance, make_chunk_id
from app.storage.vector_store import ChromaVectorStore

IDENTITY = "fake-embed|v1"


def make_chunks(document_id, n, *, start=0):
    return [
        Chunk(
            chunk_id=make_chunk_id(document_id, i),
            chunk_index=i,
            text=f"text of {document_id} chunk {i}",
            content_hash=f"hash-{document_id}-{i}",
            provenance=Provenance(
                source_type="pdf", document_id=document_id, title=f"Title {document_id}",
                page_start=i + 1, page_end=i + 2,
            ),
        )
        for i in range(start, start + n)
    ]


def vectors_for(chunks, dim=4):
    return [[(i + 1) / 10.0] * dim for i, _ in enumerate(chunks)]


@pytest.fixture
def store(tmp_path):
    s = ChromaVectorStore(tmp_path / "vdb", IDENTITY)
    yield s
    s.close()


def test_upsert_stores_text_and_provenance_without_unknown_fields(store):
    chunks = make_chunks("pdf-a", 3)
    store.upsert(chunks, vectors_for(chunks))
    assert store.count() == 3 and store.count_document("pdf-a") == 3
    stored = store.get_chunks("pdf-a")
    assert [c.chunk_id for c in stored] == ["pdf-a:0", "pdf-a:1", "pdf-a:2"]
    assert stored[1].text == "text of pdf-a chunk 1"
    meta = stored[1].metadata
    assert meta["document_id"] == "pdf-a" and meta["source_type"] == "pdf"
    assert meta["page_start"] == 2 and meta["page_end"] == 3 and meta["chunk_index"] == 1
    assert meta["content_hash"] == "hash-pdf-a-1" and meta["title"] == "Title pdf-a"
    for absent in ("url", "section", "repo", "path", "line_start", "line_end"):
        assert absent not in meta


def test_upsert_is_idempotent(store):
    chunks = make_chunks("pdf-a", 2)
    store.upsert(chunks, vectors_for(chunks))
    store.upsert(chunks, vectors_for(chunks))
    assert store.count() == 2


def test_upsert_empty_is_a_noop_and_length_mismatch_fails(store):
    store.upsert([], [])
    assert store.count() == 0
    with pytest.raises(ValueError):
        store.upsert(make_chunks("pdf-a", 2), [[0.1] * 4])


def test_delete_document_only_removes_that_document(store):
    a, b = make_chunks("pdf-a", 3), make_chunks("pdf-b", 2)
    store.upsert(a + b, vectors_for(a + b))
    assert store.delete_document("pdf-a") == 3
    assert store.count_document("pdf-a") == 0 and store.count_document("pdf-b") == 2
    assert store.delete_document("pdf-a") == 0


def test_prune_removes_only_the_stale_tail(store):
    chunks = make_chunks("pdf-a", 5)
    store.upsert(chunks, vectors_for(chunks))
    assert store.prune_document("pdf-a", 2) == 3
    assert [c.chunk_id for c in store.get_chunks("pdf-a")] == ["pdf-a:0", "pdf-a:1"]
    assert store.prune_document("pdf-a", 2) == 0


def test_data_persists_after_reopening(tmp_path):
    path = tmp_path / "vdb"
    s = ChromaVectorStore(path, IDENTITY)
    chunks = make_chunks("pdf-a", 3)
    s.upsert(chunks, vectors_for(chunks))
    s.close()
    s2 = ChromaVectorStore(path, IDENTITY)
    try:
        assert s2.count() == 3
        assert s2.get_chunks("pdf-a")[2].text == "text of pdf-a chunk 2"
    finally:
        s2.close()


def test_opening_with_a_different_embedding_identity_is_refused(tmp_path):
    path = tmp_path / "vdb"
    ChromaVectorStore(path, IDENTITY).close()
    with pytest.raises(EmbeddingModelMismatch) as info:
        ChromaVectorStore(path, "another-model|v1")
    assert "fresh LKA_DATA_DIR" in str(info.value)


def test_identity_check_can_be_skipped_for_identity_agnostic_operations(tmp_path):
    path = tmp_path / "vdb"
    s = ChromaVectorStore(path, IDENTITY)
    chunks = make_chunks("pdf-a", 2)
    s.upsert(chunks, vectors_for(chunks))
    s.close()
    other = ChromaVectorStore(path, "another-model|v1", enforce_identity=False)
    try:
        assert other.count() == 2
        assert other.delete_document("pdf-a") == 2
    finally:
        other.close()
    # The index identity is unchanged: the original configuration still opens it.
    ChromaVectorStore(path, IDENTITY).close()


def test_embedding_dimension_change_is_reported_as_model_mismatch(store):
    first = make_chunks("pdf-a", 1)
    store.upsert(first, vectors_for(first, dim=4))
    second = make_chunks("pdf-b", 1)
    with pytest.raises(EmbeddingModelMismatch):
        store.upsert(second, vectors_for(second, dim=6))
    assert store.count() == 1


# --- query ---------------------------------------------------------------------------


def seed_axes(store):
    """Three chunks on the unit axes plus one opposite, in two documents."""
    chunks = make_chunks("pdf-a", 2) + make_chunks("pdf-b", 2)
    vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]]
    store.upsert(chunks, vectors)
    return chunks


def test_query_returns_cosine_distances_best_first(store):
    seed_axes(store)
    hits = store.query([1.0, 0.0], 10)
    assert [h.chunk.chunk_id for h in hits] == ["pdf-a:0", "pdf-b:1", "pdf-a:1", "pdf-b:0"]
    assert [round(h.distance, 3) for h in hits] == [0.0, 0.4, 1.0, 2.0]


def test_query_limits_results_and_tolerates_k_above_count(store):
    seed_axes(store)
    assert len(store.query([1.0, 0.0], 2)) == 2
    assert len(store.query([1.0, 0.0], 100)) == 4


def test_query_filters_by_document_ids(store):
    seed_axes(store)
    hits = store.query([1.0, 0.0], 10, ["pdf-b"])
    assert {h.chunk.provenance.document_id for h in hits} == {"pdf-b"}
    assert store.query([1.0, 0.0], 10, ["pdf-a", "pdf-b"]) == store.query([1.0, 0.0], 10)
    assert store.query([1.0, 0.0], 10, ["pdf-unknown"]) == []
    assert store.query([1.0, 0.0], 10, []) == [], "an empty id list matches nothing"


def test_query_on_an_empty_store_returns_nothing(store):
    assert store.query([1.0, 0.0], 5) == []


def test_query_rebuilds_provenance_from_stored_metadata(store):
    chunks = make_chunks("pdf-a", 1)
    store.upsert(chunks, [[1.0, 0.0]])
    (hit,) = store.query([1.0, 0.0], 1)
    c = hit.chunk
    assert (c.chunk_id, c.chunk_index, c.text) == ("pdf-a:0", 0, "text of pdf-a chunk 0")
    assert c.content_hash == "hash-pdf-a-0"
    p = c.provenance
    assert (p.source_type, p.document_id, p.title, p.page_start, p.page_end) == ("pdf", "pdf-a", "Title pdf-a", 1, 2)
    assert p.url is None and p.section is None and p.repo is None and p.path is None and p.line_start is None


def test_query_with_a_different_dimension_is_a_model_mismatch(store):
    seed_axes(store)
    with pytest.raises(EmbeddingModelMismatch):
        store.query([1.0, 0.0, 0.0], 3)


def test_query_rejects_k_below_one(store):
    with pytest.raises(ValueError):
        store.query([1.0, 0.0], 0)
