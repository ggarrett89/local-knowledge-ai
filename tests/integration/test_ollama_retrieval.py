"""Semantic quality with the REAL embeddings model (English and Portuguese).

Requires a running Ollama with nomic-embed-text installed. Run explicitly in the VM:

    pytest -m integration

Uses only the synthetic corpus in tests/corpus.py. Nothing is pulled automatically.
"""

import pytest

from app.clients.ollama import OllamaEmbeddingClient
from app.core.config import Settings
from app.core.errors import ModelNotInstalled, OllamaUnavailable
from tests.corpus import CORPUS, OFF_TOPIC, QUERIES
from tests.retrieval.conftest import Library

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    s = Settings(_env_file=None)
    client = OllamaEmbeddingClient.from_settings(s)
    try:
        client.embed_query("ping")
    except OllamaUnavailable:
        pytest.skip("Ollama is not reachable")
    except ModelNotInstalled as exc:
        pytest.skip(str(exc))
    lib = Library(tmp_path_factory.mktemp("real"), embedder=client).add_corpus()
    yield lib
    lib.close()
    client.close()


def doc_topic(library, document_id):
    for (name, lang), doc in library.doc_ids.items():
        if doc == document_id:
            return name, lang
    raise AssertionError(document_id)


@pytest.mark.parametrize("query", QUERIES, ids=[f"{q.lang}-{q.topic}" for q in QUERIES])
def test_the_right_topic_is_in_the_top_three(library, query):
    hits = library.retriever().search(query.text, k=3)
    topics = [doc_topic(library, h.chunk.provenance.document_id)[0] for h in hits]
    assert query.topic in topics, f"{query.text!r} -> {topics}"


def test_top_hit_is_correct_for_most_questions_in_both_languages(library):
    correct = {"en": 0, "pt": 0}
    total = {"en": 0, "pt": 0}
    for q in QUERIES:
        top = library.retriever().search(q.text, k=1)[0]
        total[q.lang] += 1
        correct[q.lang] += doc_topic(library, top.chunk.provenance.document_id)[0] == q.topic
    assert correct["en"] / total["en"] >= 0.8 and correct["pt"] / total["pt"] >= 0.8, (correct, total)


def test_cross_language_retrieval_works(library):
    """A Portuguese question should find the English book too (the model is multilingual enough)."""
    q = next(q for q in QUERIES if q.lang == "pt" and q.topic == "kafka")
    hits = library.retriever().search(q.text, k=4)
    langs = {doc_topic(library, h.chunk.provenance.document_id) for h in hits}
    assert ("kafka", "pt") in langs


def test_off_topic_questions_are_farther_than_on_topic_ones(library):
    retriever = library.retriever()
    on = max(retriever.search(q.text, k=1)[0].distance for q in QUERIES)
    off = min(retriever.search(text, k=1)[0].distance for text in OFF_TOPIC)
    print(f"worst on-topic top distance={on:.3f}, best off-topic top distance={off:.3f}")
    assert on < off, "the threshold cannot separate the two groups on this corpus"
