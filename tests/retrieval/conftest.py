import pytest

from app.ingestion.service import IngestionService
from app.retrieval.candidates import SemanticCandidates
from app.retrieval.service import Retriever
from app.storage.metadata_store import SqliteMetadataStore
from app.storage.vector_store import ChromaVectorStore
from tests.corpus import CORPUS
from tests.helpers import BagOfWordsEmbedder, make_pdf

CHUNK_SIZE = 400
OVERLAP = 50


class Library:
    """Indexes the synthetic corpus (one PDF per topic and language, two pages each)."""

    def __init__(self, root, embedder=None):
        self.root = root
        self.embedder = embedder or BagOfWordsEmbedder()
        self.metadata = SqliteMetadataStore(root / "data" / "metadata.db")
        self.vectors = ChromaVectorStore(root / "data" / "vdb", self.embedder.identity)
        self.ingestion = IngestionService(self.metadata, self.vectors, self.embedder,
                                          chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, batch_size=8)
        self.doc_ids = {}

    def add_corpus(self, languages=("en", "pt")):
        for name, topic in CORPUS.items():
            for lang in languages:
                pages = getattr(topic, lang)
                path = make_pdf(self.root / "pdfs" / f"{name}-{lang}.pdf", pages, title=f"{name.title()} ({lang})")
                result = self.ingestion.ingest_file(path, topic.category, topic.topics)
                assert result.outcome == "indexed", result
                self.doc_ids[(name, lang)] = result.document_id
        return self

    def retriever(self, **kwargs):
        return Retriever(self.metadata, SemanticCandidates(self.embedder, self.vectors), **kwargs)

    def close(self):
        self.vectors.close()
        self.metadata.close()


@pytest.fixture
def library(tmp_path):
    lib = Library(tmp_path).add_corpus()
    yield lib
    lib.close()
