import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import OllamaUnavailable
from app.ingestion import cli as ingest_cli
from app.retrieval import cli
from tests.corpus import CORPUS
from tests.helpers import BagOfWordsEmbedder, make_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Kafka partitions consumers committed offsets"


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", chunk_size=400, chunk_overlap=50,
                    embed_batch_size=8, retrieval_k=3, relevance_max_distance=2.0)


@pytest.fixture
def embedder():
    return BagOfWordsEmbedder()


@pytest.fixture
def indexed(settings, embedder, tmp_path):
    for name in ("kafka", "terraform"):
        topic = CORPUS[name]
        pdf = make_pdf(tmp_path / "pdfs" / f"{name}.pdf", topic.en, title=f"{name.title()} Handbook")
        code = ingest_cli.main(["add", str(pdf), "--category", topic.category, "--topic", topic.topics[0]],
                               settings=settings, embedder=embedder)
        assert code == 0
    return settings


@pytest.fixture
def search(indexed, embedder, capsys):
    capsys.readouterr()

    def _search(*argv, embedder_override=None):
        code = cli.main(["search", *argv], settings=indexed, embedder=embedder_override or embedder)
        out = capsys.readouterr()
        return code, out.out, out.err

    return _search


def test_prints_ranked_results_with_real_references(search):
    code, out, _ = search(QUESTION)
    assert code == 0
    lines = out.strip().splitlines()
    header, snippet = lines[0], lines[1]
    assert header.startswith("1. [") and "Kafka Handbook, " in header and "(pdf-" in header
    assert any(ref in header for ref in ("p. 1", "p. 2", "pp. 1-2")), header
    assert snippet.startswith("   ") and "partition" in snippet.lower()
    ranks = [ln.split(".")[0] for ln in lines if ln[:1].isdigit()]
    assert ranks == [str(i) for i in range(1, len(ranks) + 1)]


def test_k_limits_results(search):
    _, out, _ = search(QUESTION, "-k", "1")
    assert sum(1 for ln in out.splitlines() if ln[:1].isdigit()) == 1


def test_snippet_is_shortened_and_whitespace_collapsed_unless_full():
    text = "word   one\n\ntwo " * 60
    short = cli._snippet(text, full=False)
    assert short.endswith("...") and len(short) <= 303 and "  " not in short and "\n" not in short
    full = cli._snippet(text, full=True)
    assert not full.endswith("...") and len(full) > len(short)
    assert cli._snippet("tiny text", full=False) == "tiny text"


def test_json_output_is_machine_readable(search):
    code, out, _ = search(QUESTION, "-k", "2", "--json")
    data = json.loads(out)
    assert code == 0 and len(data) == 2
    first = data[0]
    assert set(first) == {"rank", "distance", "reference", "document_id", "chunk_id", "title",
                          "page_start", "page_end", "text"}
    assert first["rank"] == 1 and first["title"] == "Kafka Handbook"
    assert first["reference"].startswith("Kafka Handbook, ")
    assert data[0]["distance"] <= data[1]["distance"]


def test_filters_restrict_results(search):
    _, out, _ = search("partitions modules", "--category", "iac-automation", "--json")
    assert {r["title"] for r in json.loads(out)} == {"Terraform Handbook"}
    _, out, _ = search("partitions modules", "--topic", "kafka", "--json")
    assert {r["title"] for r in json.loads(out)} == {"Kafka Handbook"}
    doc_id = json.loads(out)[0]["document_id"]
    _, out, _ = search("partitions modules", "--document", doc_id, "--json")
    assert {r["document_id"] for r in json.loads(out)} == {doc_id}
    _, out, _ = search("partitions modules", "--source-type", "pdf", "--json")
    assert len(json.loads(out)) > 0


def test_no_match_prints_a_clear_message(search):
    code, out, _ = search(QUESTION, "--category", "no-such-category")
    assert code == 0 and out.strip() == "No results."


def test_max_distance_drops_far_hits(search):
    code, out, _ = search("chocolate cake sugar butter", "--max-distance", "0.5")
    assert code == 0 and out.strip() == "No results."


def test_empty_index_explains_what_to_do(tmp_path, embedder, capsys):
    s = Settings(_env_file=None, data_dir=tmp_path / "empty")
    code = cli.main(["search", "anything"], settings=s, embedder=embedder)
    assert code == 0 and "import a PDF first" in capsys.readouterr().out


@pytest.mark.parametrize("argv,message", [
    (["   "], "empty"),
    (["x" * 1001], "too long"),
    (["q", "-k", "0"], "k must be"),
    (["q", "-k", "500"], "k must be"),
    (["q", "--category", "Not Valid!"], "invalid slug"),
])
def test_invalid_input_is_a_clear_error(search, argv, message):
    code, out, err = search(*argv)
    assert code == 1 and message in err and out == "" and "Traceback" not in err


def test_incompatible_embeddings_model_is_refused(search):
    code, _, err = search(QUESTION, embedder_override=BagOfWordsEmbedder(identity="another-model|v1"))
    assert code == 1 and "different embeddings configuration" in err


def test_ollama_down_is_reported_without_a_traceback(search):
    down = BagOfWordsEmbedder(error=OllamaUnavailable("cannot reach Ollama at http://127.0.0.1:11434"))
    code, _, err = search(QUESTION, embedder_override=down)
    assert code == 1 and "cannot reach Ollama" in err and "Traceback" not in err


def test_search_never_modifies_the_index(search, indexed):
    before = sorted((p.name, p.stat().st_size) for p in indexed.metadata_db_path.parent.glob("*.db"))
    stats_before = ingest_cli.main(["stats"], settings=indexed, embedder=BagOfWordsEmbedder())
    search(QUESTION)
    search(QUESTION, "--json")
    after = sorted((p.name, p.stat().st_size) for p in indexed.metadata_db_path.parent.glob("*.db"))
    assert stats_before == 0 and before == after


def test_module_entry_point_runs_as_documented():
    proc = subprocess.run([sys.executable, "-m", "app.retrieval.cli", "search", "--help"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "--max-distance" in proc.stdout and "--json" in proc.stdout
