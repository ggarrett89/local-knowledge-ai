"""Requires a running Ollama with the embeddings model already installed.

Not part of the default run. Execute explicitly, e.g. inside the VM:

    pytest -m integration

The test never pulls a model: if it is missing it reports it and skips.
"""

import pytest

from app.clients.ollama import OllamaEmbeddingClient
from app.core.config import Settings
from app.core.errors import ModelNotInstalled, OllamaUnavailable
from app.ingestion.service import IngestionService
from app.storage.metadata_store import SqliteMetadataStore
from app.storage.vector_store import ChromaVectorStore
from tests.helpers import make_pdf, words

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    s = Settings(_env_file=None)
    c = OllamaEmbeddingClient(
        s.ollama_base_url, s.embedding_model, timeout_seconds=s.ollama_timeout_seconds,
        document_prefix=s.embedding_document_prefix, query_prefix=s.embedding_query_prefix,
    )
    try:
        c.embed_query("ping")
    except OllamaUnavailable:
        pytest.skip("Ollama is not reachable")
    except ModelNotInstalled as exc:
        pytest.skip(str(exc))
    yield c
    c.close()


def test_real_embeddings_have_a_consistent_dimension(client):
    docs = client.embed_documents(["Terraform modules", "Kubernetes pods"])
    query = client.embed_query("What is a Terraform module?")
    assert len(docs) == 2 and len(docs[0]) == len(docs[1]) == len(query) > 100


def test_real_end_to_end_ingestion(client, tmp_path):
    pdf = make_pdf(tmp_path / "book.pdf", [words("p1t", 150), words("p2t", 150)])
    with SqliteMetadataStore(tmp_path / "m.db") as metadata:
        vectors = ChromaVectorStore(tmp_path / "vdb", client.identity)
        try:
            service = IngestionService(metadata, vectors, client, chunk_size=1200, chunk_overlap=200, batch_size=8)
            result = service.ingest_file(pdf, "general")
            assert result.outcome == "indexed" and vectors.count() == result.chunks > 0
        finally:
            vectors.close()
