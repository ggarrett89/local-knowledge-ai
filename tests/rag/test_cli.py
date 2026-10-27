import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import ModelNotInstalled, OllamaUnavailable
from app.ingestion import cli as ingest_cli
from app.rag import cli
from tests.corpus import CORPUS
from tests.helpers import BagOfWordsEmbedder, FakeChat, make_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Terraform remote state S3 bucket locking prevents two engineers applying changes"


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", chunk_size=400, chunk_overlap=50, embed_batch_size=8,
                    retrieval_k=3, relevance_max_distance=2.0)


@pytest.fixture
def indexed(settings, tmp_path):
    pdf = make_pdf(tmp_path / "pdfs" / "tf.pdf", CORPUS["terraform"].en, title="Terraform Handbook")
    assert ingest_cli.main(["add", str(pdf), "--category", "iac-automation"], settings=settings,
                           embedder=BagOfWordsEmbedder()) == 0
    return settings


@pytest.fixture
def ask(indexed, capsys):
    capsys.readouterr()

    def _ask(*argv, chat=None):
        code = cli.main(["ask", *argv], settings=indexed, embedder=BagOfWordsEmbedder(), chat=chat or FakeChat("Locking stops concurrent applies [1]."))
        out = capsys.readouterr()
        return code, out.out, out.err

    return _ask


def test_prints_status_answer_and_sources(ask):
    code, out, _ = ask(QUESTION)
    assert code == 0
    assert "Status: answered" in out and "Locking stops concurrent applies [1]." in out
    assert "Sources cited:" in out and "[1] Terraform Handbook, " in out


def test_inference_is_labelled_as_not_from_the_sources(ask):
    _, out, _ = ask(QUESTION, chat=FakeChat("Fact [1].\n\nInference: a guess."))
    assert "Inference (written by the model, NOT stated in the sources):" in out and "a guess." in out


def test_refusal_shows_the_closest_passages_without_claiming_them_as_sources(ask):
    code, out, _ = ask(QUESTION, chat=FakeChat("INSUFFICIENT_EVIDENCE"))
    assert code == 0 and "Status: insufficient_evidence" in out
    assert "Sources cited:" not in out and "Closest passages (not cited):" in out


def test_passages_flag_prints_what_the_model_saw(ask):
    _, out, _ = ask(QUESTION, "--passages")
    assert "Passages given to the model:" in out and "S3 bucket" in out


def test_json_output_matches_the_answer_model(ask):
    code, out, _ = ask(QUESTION, "--json")
    data = json.loads(out)
    assert code == 0 and data["status"] == "answered" and data["sources"][0]["title"] == "Terraform Handbook"
    assert set(data) == {"status", "answer", "inference", "sources", "retrieved", "notice", "removed_references", "model"}


def test_filters_that_match_nothing_do_not_call_the_model(ask):
    chat = FakeChat("never")
    code, out, _ = ask(QUESTION, "--category", "no-such-category", chat=chat)
    assert code == 0 and "insufficient_evidence" in out and chat.calls == []


@pytest.mark.parametrize("argv,message", [
    (["   "], "empty"),
    (["q", "--category", "Not Valid!"], "invalid slug"),
    (["q", "-k", "0"], "k must be"),
])
def test_invalid_input_is_a_clear_error(ask, argv, message):
    code, out, err = ask(*argv)
    assert code == 1 and message in err and out == "" and "Traceback" not in err


@pytest.mark.parametrize("error,text", [(OllamaUnavailable("cannot reach Ollama"), "cannot reach Ollama"),
                                        (ModelNotInstalled("model 'x' is not installed; ollama pull x"), "ollama pull")])
def test_model_problems_are_reported_without_a_traceback(ask, error, text):
    code, _, err = ask(QUESTION, chat=FakeChat(error=error))
    assert code == 1 and text in err and "Traceback" not in err


def test_module_entry_point_runs_as_documented():
    proc = subprocess.run([sys.executable, "-m", "app.rag.cli", "ask", "--help"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "--passages" in proc.stdout and "--json" in proc.stdout
