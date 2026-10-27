import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = REPO_ROOT / "scripts" / "calibration" / "book_queries.json"


def test_script_loads_and_documents_its_modes():
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "calibrate_retrieval.py"), "--help"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "synthetic" in proc.stdout and "index" in proc.stdout


def test_question_file_is_well_formed():
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    assert len(data["in_scope"]) >= 10
    langs = {item["lang"] for item in data["in_scope"]}
    assert langs == {"en", "pt"}
    for item in data["in_scope"]:
        assert item["q"].strip().endswith("?")
        assert item["keywords"] and all(k == k.lower() and k.strip() for k in item["keywords"])
    assert len(data["unrelated"]) >= 4 and len(data["other_tech"]) >= 2
    all_questions = [i["q"] for i in data["in_scope"]] + data["unrelated"] + data["other_tech"]
    assert len(all_questions) == len(set(all_questions)), "duplicate questions skew the statistics"
    assert max(map(len, all_questions)) < 1000, "must fit the default LKA_MAX_QUERY_CHARS"


def test_threshold_table_helper_counts_correctly(capsys):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import calibrate_retrieval as c
    finally:
        sys.path.pop(0)
    c.threshold_table([0.2, 0.3, 0.5], {"unrelated": [0.4, 0.6]})
    rows = {line.split()[0]: line.split() for line in capsys.readouterr().out.splitlines()[2:]}
    assert rows["0.30"][1].startswith("2/3") and rows["0.30"][2].startswith("2/2")
    assert rows["0.50"][1].startswith("3/3") and rows["0.50"][2].startswith("1/2")
