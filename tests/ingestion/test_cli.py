import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import OllamaUnavailable
from app.ingestion import cli
from tests.helpers import FakeEmbedder, make_pdf, words

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", chunk_size=400, chunk_overlap=50, embed_batch_size=3)


@pytest.fixture
def run(settings, capsys):
    embedder = FakeEmbedder()

    def _run(*argv, embedder_override=None):
        code = cli.main(list(argv), settings=settings, embedder=embedder_override or embedder)
        out = capsys.readouterr()
        return code, out.out, out.err

    _run.embedder = embedder
    return _run


@pytest.fixture
def pdf(tmp_path):
    return make_pdf(tmp_path / "books" / "k8s.pdf", [words("a1t", 120), words("a2t", 120)], title="Kubernetes in Action")


def test_add_then_list_then_stats(run, pdf):
    code, out, _ = run("add", str(pdf), "--category", "containers-kubernetes", "--topic", "networking", "--topic", "security")
    assert code == 0 and out.startswith("[indexed] pdf-") and '"Kubernetes in Action"' in out and "pages=2" in out

    code, out, _ = run("list")
    assert code == 0
    assert "containers-kubernetes" in out and "networking,security" in out and '"Kubernetes in Action"' in out

    code, out, _ = run("stats")
    assert code == 0
    assert "Documents : 1" in out and "Pages     : 2" in out and "Errors    : 0" in out
    assert "vectors stored" in out


def test_add_twice_reports_unchanged(run, pdf):
    run("add", str(pdf))
    calls = list(run.embedder.calls)
    code, out, _ = run("add", str(pdf))
    assert code == 0 and out.startswith("[unchanged]")
    assert run.embedder.calls == calls


def test_add_all_prints_a_summary(run, tmp_path):
    make_pdf(tmp_path / "lib" / "a.pdf", [words("a1t", 60)])
    make_pdf(tmp_path / "lib" / "nested" / "b.pdf", [words("b1t", 60)])
    code, out, _ = run("add-all", str(tmp_path / "lib"), "--category", "cloud")
    assert code == 0
    assert out.count("[indexed]") == 2 and "Summary: indexed=2" in out


def test_add_all_exit_code_is_1_when_a_file_fails_but_others_are_indexed(run, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.pdf").write_bytes(b"nope")
    make_pdf(lib / "good.pdf", [words("g1t", 60)])
    code, out, _ = run("add-all", str(lib))
    assert code == 1 and "[error]" in out and "[indexed]" in out and "error=1" in out


def test_new_categories_need_no_code_change(run, pdf):
    assert run("add", str(pdf), "--category", "quantum-computing")[0] == 0
    code, out, _ = run("list", "--category", "quantum-computing")
    assert "quantum-computing" in out
    assert run("list", "--category", "cloud")[1].strip() == "No documents."


def test_list_filters_by_topic_and_status(run, pdf, tmp_path):
    run("add", str(pdf), "--topic", "kubernetes")
    assert "Kubernetes in Action" in run("list", "--topic", "kubernetes")[1]
    assert run("list", "--topic", "terraform")[1].strip() == "No documents."
    assert run("list", "--status", "error")[1].strip() == "No documents."


def test_invalid_category_is_a_clear_error(run, pdf):
    code, out, err = run("add", str(pdf), "--category", "Not Valid!")
    assert code == 1 and "invalid slug" in err and out == ""
    assert run.embedder.calls == []


def test_missing_file_is_reported_with_exit_code_1(run, tmp_path):
    code, out, _ = run("add", str(tmp_path / "missing.pdf"))
    assert code == 1 and "[error]" in out and "not found" in out


def test_blank_pdf_is_reported_as_no_text(run, tmp_path):
    scan = make_pdf(tmp_path / "scan.pdf", ["", ""])
    code, out, _ = run("add", str(scan))
    assert code == 0 and out.startswith("[no_text]")
    assert "[no_text]" in run("list")[1] or "no_text" in run("list")[1]


def test_ollama_unavailable_is_reported_without_a_traceback(run, pdf):
    down = FakeEmbedder(error=OllamaUnavailable("cannot reach Ollama at http://127.0.0.1:11434"))
    code, out, err = run("add", str(pdf), embedder_override=down)
    assert code == 1 and "cannot reach Ollama" in err and "Traceback" not in err


def test_embedding_model_change_is_refused(run, pdf):
    run("add", str(pdf))
    other = FakeEmbedder(identity="a-different-model|v1")
    code, _, err = run("add", str(pdf), "--force", embedder_override=other)
    assert code == 1 and "different embeddings configuration" in err


def test_reindex_by_id(run, pdf):
    document_id = run("add", str(pdf), "--category", "cloud")[1].split()[1]
    code, out, _ = run("reindex", document_id)
    assert code == 0 and out.startswith("[reindexed]")
    assert "cloud" in run("list")[1]
    code, _, err = run("reindex", "pdf-unknown")
    assert code == 1 and "no document" in err


def test_remove_requires_confirmation_and_keeps_the_file(run, pdf, monkeypatch):
    document_id = run("add", str(pdf))[1].split()[1]

    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    code, out, _ = run("remove", document_id)
    assert code == 1 and "Aborted" in out
    assert document_id in run("list")[1]

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    code, out, _ = run("remove", document_id)
    assert code == 0 and "Removed" in out
    assert run("list")[1].strip() == "No documents."
    assert pdf.is_file(), "the original file must never be deleted"


def test_remove_with_yes_flag_and_unknown_id(run, pdf):
    document_id = run("add", str(pdf))[1].split()[1]
    assert run("remove", document_id, "--yes")[0] == 0
    code, _, err = run("remove", "pdf-unknown", "--yes")
    assert code == 1 and "no document" in err


def test_remove_works_even_if_the_embedding_config_changed(run, pdf):
    document_id = run("add", str(pdf))[1].split()[1]
    other = FakeEmbedder(identity="a-different-model|v1")
    assert run("remove", document_id, "--yes", embedder_override=other)[0] == 0


def test_list_and_stats_do_not_require_a_working_embeddings_service(run, pdf):
    run("add", str(pdf))
    down = FakeEmbedder(error=OllamaUnavailable("down"))
    assert run("list", embedder_override=down)[0] == 0
    assert run("stats", embedder_override=down)[0] == 0


def test_cli_module_runs_as_documented():
    proc = subprocess.run(
        [sys.executable, "-m", "app.ingestion.cli", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    for command in ("add", "add-all", "list", "remove", "reindex", "stats"):
        assert command in proc.stdout
