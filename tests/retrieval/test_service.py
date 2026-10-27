import logging

import pytest

from app.core.errors import InvalidQuery, OllamaUnavailable
from app.core.models import Chunk, DocumentRecord, Provenance, SearchFilters, SearchHit
from app.retrieval.candidates import SemanticCandidates
from app.retrieval.service import Retriever
from app.storage.metadata_store import SqliteMetadataStore
from tests.helpers import BagOfWordsEmbedder


def hit(doc="pdf-a", index=0, distance=0.3, text=None, content_hash=None, page=1):
    return SearchHit(
        distance=distance,
        chunk=Chunk(
            chunk_id=f"{doc}:{index}", chunk_index=index, text=text or f"text {doc} {index}",
            content_hash=content_hash or f"hash-{doc}-{index}",
            provenance=Provenance(source_type="pdf", document_id=doc, title=f"Title {doc}",
                                  page_start=page, page_end=page),
        ),
    )


class StubGenerator:
    def __init__(self, hits=()):
        self.hits = list(hits)
        self.calls = []

    def generate(self, query, *, limit, document_ids):
        self.calls.append({"query": query, "limit": limit, "document_ids": document_ids})
        return list(self.hits)


@pytest.fixture
def metadata(tmp_path):
    m = SqliteMetadataStore(tmp_path / "m.db")
    for doc_id, category, topics, status, source in [
        ("pdf-a", "cloud", ["aws", "iam"], "indexed", "pdf"),
        ("pdf-b", "cloud", ["aws"], "indexed", "pdf"),
        ("pdf-c", "security", ["iam"], "indexed", "pdf"),
        ("pdf-err", "cloud", ["aws"], "error", "pdf"),
        ("web-d", "cloud", ["aws"], "indexed", "web"),
    ]:
        m.upsert_document(DocumentRecord(document_id=doc_id, title=doc_id, source_type=source,
                                         category=category, topics=topics, status=status))
    yield m
    m.close()


def make(metadata, hits=(), **kwargs):
    gen = StubGenerator(hits)
    return Retriever(metadata, gen, **kwargs), gen


# --- validation -------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_questions_are_rejected(metadata, query):
    r, gen = make(metadata)
    with pytest.raises(InvalidQuery):
        r.search(query)
    assert gen.calls == []


def test_overlong_questions_are_rejected(metadata):
    r, gen = make(metadata, max_query_chars=50)
    with pytest.raises(InvalidQuery, match="too long"):
        r.search("x" * 51)
    r.search("x" * 50)
    assert len(gen.calls) == 1


@pytest.mark.parametrize("k", [0, -1, 51])
def test_k_must_be_within_bounds(metadata, k):
    r, _ = make(metadata, max_k=50)
    with pytest.raises(InvalidQuery):
        r.search("question", k=k)


def test_question_is_stripped_and_k_defaults_from_settings(metadata):
    r, gen = make(metadata, default_k=3, overfetch=4)
    r.search("  what is a pod?  ")
    assert gen.calls[0] == {"query": "what is a pod?", "limit": 12, "document_ids": None}


def test_candidate_limit_is_capped(metadata):
    r, gen = make(metadata, max_k=50, overfetch=20)
    r.search("q", k=50)
    assert gen.calls[0]["limit"] == 200


# --- empty results ----------------------------------------------------------------


def test_no_candidates_is_an_empty_list_not_an_error(metadata):
    r, _ = make(metadata, hits=[])
    assert r.search("anything") == []


# --- filters ----------------------------------------------------------------------


def ids_for(metadata, **filters):
    r, gen = make(metadata, hits=[hit()])
    r.search("q", filters=SearchFilters(**filters))
    return gen.calls[0]["document_ids"], gen


def test_no_filters_means_no_restriction(metadata):
    assert ids_for(metadata)[0] is None


def test_filter_by_category_only_includes_indexed_documents(metadata):
    ids, _ = ids_for(metadata, category="cloud")
    assert ids == ["pdf-a", "pdf-b", "web-d"], "documents in error status have no usable chunks"


def test_filter_by_topics_requires_all_of_them(metadata):
    assert ids_for(metadata, topics=["aws"])[0] == ["pdf-a", "pdf-b", "web-d"]
    assert ids_for(metadata, topics=["aws", "iam"])[0] == ["pdf-a"]


def test_filter_by_source_type(metadata):
    assert ids_for(metadata, source_type="web")[0] == ["web-d"]


def test_explicit_document_ids_are_intersected_with_other_filters(metadata):
    assert ids_for(metadata, document_ids=["pdf-a", "pdf-c"], category="cloud")[0] == ["pdf-a"]
    assert ids_for(metadata, document_ids=["pdf-c"])[0] == ["pdf-c"]


def test_filters_that_match_nothing_return_empty_without_searching(metadata):
    for filters in (dict(category="nonexistent"), dict(topics=["nope"]), dict(document_ids=[]),
                    dict(document_ids=["pdf-unknown"]), dict(category="security", topics=["aws"])):
        r, gen = make(metadata, hits=[hit()])
        assert r.search("q", filters=SearchFilters(**filters)) == []
        assert gen.calls == [], f"{filters} should not even embed the question"


def test_an_empty_filters_object_is_the_same_as_none(metadata):
    r, gen = make(metadata, hits=[hit()])
    assert len(r.search("q", filters=SearchFilters())) == 1
    assert gen.calls[0]["document_ids"] is None


# --- relevance threshold ----------------------------------------------------------


def test_hits_beyond_the_threshold_are_dropped(metadata):
    hits = [hit(index=0, distance=0.2), hit(index=5, distance=0.5), hit(index=9, distance=0.9)]
    r, _ = make(metadata, hits=hits, max_distance=0.5)
    assert [h.distance for h in r.search("q")] == [0.2, 0.5], "a hit exactly at the threshold is kept"


def test_threshold_can_be_overridden_per_call(metadata):
    hits = [hit(index=0, distance=0.2), hit(index=5, distance=0.9)]
    r, _ = make(metadata, hits=hits, max_distance=0.5)
    assert len(r.search("q", max_distance=1.0)) == 2
    assert r.search("q", max_distance=0.1) == []


def test_results_are_sorted_by_distance_even_if_the_generator_is_not(metadata):
    hits = [hit(index=9, distance=0.7), hit(index=0, distance=0.1), hit(index=5, distance=0.4)]
    r, _ = make(metadata, hits=hits)
    assert [h.distance for h in r.search("q")] == [0.1, 0.4, 0.7]


# --- de-duplication ---------------------------------------------------------------


def test_identical_text_is_returned_once_keeping_the_closest(metadata):
    hits = [hit("pdf-a", 1, 0.3, content_hash="same"), hit("pdf-b", 7, 0.2, content_hash="same"),
            hit("pdf-a", 9, 0.4)]
    r, _ = make(metadata, hits=hits)
    result = r.search("q")
    assert [(h.chunk.chunk_id, h.distance) for h in result] == [("pdf-b:7", 0.2), ("pdf-a:9", 0.4)]


def test_adjacent_chunks_of_the_same_document_are_collapsed(metadata):
    hits = [hit("pdf-a", 5, 0.1), hit("pdf-a", 6, 0.2), hit("pdf-a", 7, 0.3), hit("pdf-a", 20, 0.4)]
    r, _ = make(metadata, hits=hits)
    assert [h.chunk.chunk_index for h in r.search("q")] == [5, 7, 20], "6 touches the best hit 5; 7 is kept"


def test_adjacency_only_applies_within_one_document(metadata):
    r, _ = make(metadata, hits=[hit("pdf-a", 5, 0.1), hit("pdf-b", 6, 0.2)])
    assert len(r.search("q")) == 2


def test_adjacent_dedup_can_be_disabled(metadata):
    r, _ = make(metadata, hits=[hit("pdf-a", 5, 0.1), hit("pdf-a", 6, 0.2)], dedupe_adjacent=False)
    assert len(r.search("q")) == 2


def test_k_applies_after_deduplication_and_fewer_results_are_fine(metadata):
    hits = [hit("pdf-a", i, 0.1 + i * 0.01) for i in range(0, 20, 2)]  # none adjacent
    r, _ = make(metadata, hits=hits)
    assert len(r.search("q", k=3)) == 3
    r2, _ = make(metadata, hits=[hit("pdf-a", 1, 0.1), hit("pdf-a", 2, 0.2)])
    assert len(r2.search("q", k=5)) == 1


# --- semantic generator -----------------------------------------------------------


def test_embedding_service_errors_propagate(metadata, tmp_path):
    from app.storage.vector_store import ChromaVectorStore
    embedder = BagOfWordsEmbedder(error=OllamaUnavailable("cannot reach Ollama"))
    vectors = ChromaVectorStore(tmp_path / "v", embedder.identity)
    try:
        r = Retriever(metadata, SemanticCandidates(embedder, vectors))
        with pytest.raises(OllamaUnavailable):
            r.search("anything")
    finally:
        vectors.close()


# --- privacy ----------------------------------------------------------------------


def test_logs_never_contain_the_question_or_the_text(metadata, caplog):
    caplog.set_level(logging.DEBUG)
    r, _ = make(metadata, hits=[hit(text="CONFIDENTIAL-PASSAGE")])
    r.search("my very private question about salaries")
    assert "private" not in caplog.text and "CONFIDENTIAL" not in caplog.text
    assert "returned=1" in caplog.text
