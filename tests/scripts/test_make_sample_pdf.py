import subprocess
import sys
from pathlib import Path

from app.ingestion.pdf.extractor import PdfExtractor

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_sample_pdf.py"), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )


def test_creates_a_readable_pdf_with_the_requested_pages_and_title(tmp_path):
    out = tmp_path / "nested" / "sample.pdf"
    proc = run_script(str(out), "--pages", "3", "--title", "Demo Book")
    assert proc.returncode == 0, proc.stderr
    extractor = PdfExtractor()
    info = extractor.describe(out)
    assert (info.title, info.page_count) == ("Demo Book", 3)
    sections = list(extractor.extract(out))
    assert [s.page_number for s in sections] == [1, 2, 3]
    assert all(len(s.text) > 1000 for s in sections)


def test_rejects_page_text_that_does_not_fit(tmp_path):
    proc = run_script(str(tmp_path / "x.pdf"), "--chars-per-page", "200000")
    assert proc.returncode != 0
    assert not (tmp_path / "x.pdf").exists()
