import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.core.config import Settings
from app.services import build_services
from tests.corpus import CORPUS
from tests.helpers import BagOfWordsEmbedder, FakeCatalog, FakeChat, make_pdf

TOKEN = "test-admin-token-0123456789"


class Env:
    """An API over real SQLite + ChromaDB in a temp dir, with fake embeddings, chat and catalog."""

    def __init__(self, tmp_path, *, token=TOKEN, **overrides):
        self.root = tmp_path
        self.settings = Settings(
            _env_file=None, data_dir=tmp_path / "data", admin_token=token, chunk_size=400, chunk_overlap=50,
            embed_batch_size=8, retrieval_k=3, relevance_max_distance=2.0, chat_model="fake-chat", **overrides,
        )
        self.docs = self.settings.documents_dir
        self.docs.mkdir(parents=True, exist_ok=True)
        self.embedder = BagOfWordsEmbedder()
        self.chat = FakeChat("Locking stops concurrent applies [1].")
        self.catalog = FakeCatalog()
        self.services = build_services(self.settings, embedder=self.embedder, chat=self.chat, catalog=self.catalog)
        self.client = TestClient(create_app(self.settings, self.services), raise_server_exceptions=False)
        self.auth = {"Authorization": f"Bearer {TOKEN}"}

    def add_pdf(self, name="terraform.pdf", topic="terraform", title=None):
        t = CORPUS[topic]
        return make_pdf(self.docs / name, t.en, title=title or f"{topic.title()} Handbook")

    def ingest(self, path="terraform.pdf", **body):
        return self.client.post("/documents/ingest", json={"path": path, **body}, headers=self.auth)

    def close(self):
        self.client.close()
        self.services.close()


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


@pytest.fixture
def loaded(env):
    """An env with a Terraform and a Kafka book already indexed."""
    env.add_pdf("terraform.pdf", "terraform")
    env.add_pdf("kafka.pdf", "kafka")
    assert env.ingest("terraform.pdf", category="iac-automation", topics=["terraform"]).status_code == 200
    assert env.ingest("kafka.pdf", category="databases-messaging", topics=["kafka"]).status_code == 200
    return env
