import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = REPO_ROOT / "scripts" / "calibration" / "rag_questions.json"


def test_script_loads_and_documents_its_arguments():
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "evaluate_rag.py"), "--help"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "questions" in proc.stdout and "--limit" in proc.stdout


def test_question_file_is_well_formed():
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    assert len(data["answerable"]) >= 6 and len(data["unanswerable"]) >= 4
    assert {i["lang"] for i in data["answerable"]} == {"en", "pt"}
    for item in data["answerable"]:
        assert item["q"].endswith("?") and item["keywords"] and all(k == k.lower() for k in item["keywords"])
    for item in data["unanswerable"]:
        assert item["q"].endswith("?") and item["lang"] in ("en", "pt")
    questions = [i["q"] for g in ("answerable", "unanswerable") for i in data[g]]
    assert len(questions) == len(set(questions)) and max(map(len, questions)) < 1000


def test_summary_counts_statuses_without_printing_answers(capsys):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import evaluate_rag as ev
    finally:
        sys.path.pop(0)
    rows = [
        dict(group="answerable", lang="en", status="answered", cited=1, retrieved=5, seconds=40.0, on_topic=True,
             repaired=True, unsupported=False, removed_refs=0, invented_ids=1, inference=False, prompt_tokens=1300,
             generation_s=20.0, prompt_eval_s=15.0),
        dict(group="answerable", lang="pt", status="unverified", cited=0, retrieved=5, seconds=60.0, on_topic=None,
             repaired=False, unsupported=False, removed_refs=1, invented_ids=0, inference=True, prompt_tokens=1400,
             generation_s=30.0, prompt_eval_s=25.0),
        dict(group="unanswerable", lang="en", status="insufficient_evidence", cited=0, retrieved=0, seconds=0.4,
             on_topic=None, repaired=False, unsupported=False, removed_refs=0, invented_ids=0, inference=False, prompt_tokens=None,
             generation_s=None, prompt_eval_s=None),
    ]
    ev.summarize(rows)
    out = capsys.readouterr().out
    assert "answerable: 2 questions" in out and "'answered': 1" in out and "'unverified': 1" in out
    assert "answered with on-topic citations only: 1/1" in out and "never reached the model" in out
    assert "median 50" in out and "found in no passage: 1/1" in out
