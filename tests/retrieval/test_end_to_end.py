"""PDF -> ingestion -> real SQLite + ChromaDB -> search, using a word-overlap fake embedder.

The fake does not understand meaning or other languages; it proves the plumbing, filters,
references and thresholds. Semantic quality (PT/EN) is covered by the opt-in integration test.
"""

import pytest

from app.core.errors import EmbeddingModelMismatch, OllamaUnavailable
from app.core.models import SearchFilters
from app.core.references import format_reference
from app.storage.vector_store import ChromaVectorStore
from tests.corpus import CORPUS
from tests.helpers import BagOfWordsEmbedder
from tests.retrieval.conftest import Library


def test_english_question_finds_the_right_book_and_page(library):
    r = library.retriever()
    top = r.search("Terraform remote state S3 bucket locking prevents two engineers applying changes", k=3)[0]
    p = top.chunk.provenance
    assert p.document_id == library.doc_ids[("terraform", "en")]
    assert "S3 bucket" in top.chunk.text
    # The phrase is on page 2. The chunk may also start with the overlap tail of page 1, so the
    # reference is allowed to be pp. 1-2, but it must contain page 2 and never leave the book.
    assert p.page_start <= 2 <= p.page_end <= 2
    assert format_reference(p) in ("Terraform (en), p. 2", "Terraform (en), pp. 1-2")


def test_portuguese_question_finds_the_portuguese_book(library):
    top = library.retriever().search("estado remoto bucket S3 bloqueio impede aplicar mudanças ao mesmo tempo", k=3)[0]
    assert top.chunk.provenance.document_id == library.doc_ids[("terraform", "pt")]
    assert top.chunk.provenance.page_start <= 2 <= top.chunk.provenance.page_end


@pytest.mark.parametrize("topic,question,page", [
    ("kubernetes", "scheduler assigns each pod to a node and the kubelet starts the containers", 1),
    ("kafka", "consumers in the same consumer group share partitions and committed offsets", 2),
    ("postgres", "EXPLAIN ANALYZE shows an index or a sequential scan", 1),
    ("sre", "burn rate of the error budget alerts and blameless postmortems", 2),
])
def test_each_topic_is_found_with_its_page(library, topic, question, page):
    top = library.retriever().search(question, k=3)[0]
    p = top.chunk.provenance
    assert p.document_id == library.doc_ids[(topic, "en")]
    assert p.page_start <= page <= p.page_end


def test_results_are_ranked_and_limited(library):
    hits = library.retriever().search("partitions brokers consumers topic", k=4)
    assert len(hits) == 4
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_filter_by_category(library):
    hits = library.retriever().search("partitions topic brokers query index", k=10,
                                      filters=SearchFilters(category="databases-messaging"))
    docs = {h.chunk.provenance.document_id for h in hits}
    assert docs <= {library.doc_ids[("kafka", "en")], library.doc_ids[("kafka", "pt")],
                    library.doc_ids[("postgres", "en")], library.doc_ids[("postgres", "pt")]}
    assert docs, "expected hits inside the category"


def test_filter_by_topic_and_by_document(library):
    r = library.retriever()
    kafka_only = r.search("partitions", k=10, filters=SearchFilters(topics=["kafka"]))
    assert {h.chunk.provenance.title for h in kafka_only} <= {"Kafka (en)", "Kafka (pt)"}
    one_doc = r.search("partitions", k=10, filters=SearchFilters(document_ids=[library.doc_ids[("kafka", "pt")]]))
    assert {h.chunk.provenance.document_id for h in one_doc} == {library.doc_ids[("kafka", "pt")]}


def test_filters_matching_nothing_return_no_results(library):
    r = library.retriever()
    assert r.search("partitions", filters=SearchFilters(category="no-such-category")) == []
    assert r.search("partitions", filters=SearchFilters(topics=["kafka", "terraform"])) == []


def test_off_topic_question_is_cut_by_the_threshold(library):
    on_topic = library.retriever().search("Kafka partitions consumers offsets", k=1)[0].distance
    off_topic = library.retriever().search("chocolate cake sugar butter", k=1)[0].distance
    assert 0 < on_topic < off_topic <= 1.0, "no shared words means no similarity at all"
    threshold = (on_topic + off_topic) / 2  # derived from the data, not hand-picked
    r = library.retriever(max_distance=threshold)
    assert r.search("chocolate cake sugar butter") == []
    assert r.search("Kafka partitions consumers offsets") != []


def test_threshold_default_keeps_everything(library):
    assert len(library.retriever().search("chocolate cake sugar butter", k=5)) == 5


def test_empty_index_returns_no_results(tmp_path):
    lib = Library(tmp_path)
    try:
        assert lib.retriever().search("anything at all") == []
    finally:
        lib.close()


def test_removed_documents_are_no_longer_returned(library):
    doc = library.doc_ids[("kafka", "en")]
    library.ingestion.remove(doc)
    hits = library.retriever().search("Kafka partitions consumers offsets", k=20)
    assert doc not in {h.chunk.provenance.document_id for h in hits}


def test_search_survives_a_restart(tmp_path):
    lib = Library(tmp_path).add_corpus(languages=("en",))
    expected = [h.chunk.chunk_id for h in lib.retriever().search("Kafka partitions consumers", k=3)]
    lib.close()
    again = Library(tmp_path)  # reopens the same SQLite file and Chroma directory
    try:
        assert [h.chunk.chunk_id for h in again.retriever().search("Kafka partitions consumers", k=3)] == expected
    finally:
        again.close()


def test_a_different_embeddings_model_is_refused(library, tmp_path):
    library.vectors.close()
    with pytest.raises(EmbeddingModelMismatch):
        ChromaVectorStore(tmp_path / "data" / "vdb", "another-model|v1")


def test_embeddings_service_down_is_reported(tmp_path):
    lib = Library(tmp_path).add_corpus(languages=("en",))
    try:
        lib.embedder.error = OllamaUnavailable("cannot reach Ollama")
        with pytest.raises(OllamaUnavailable):
            lib.retriever().search("Kafka partitions")
    finally:
        lib.close()


def test_every_hit_has_real_provenance(library):
    for hit in library.retriever().search("pods nodes partitions indexes", k=10):
        p = hit.chunk.provenance
        assert p.source_type == "pdf" and p.title and p.page_start is not None
        assert p.url is None and p.section is None and p.path is None
        assert hit.chunk.text and hit.chunk.content_hash
