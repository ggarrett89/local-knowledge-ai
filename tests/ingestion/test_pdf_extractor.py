import hashlib

import pytest

from app.core.errors import InvalidDocument
from app.ingestion.pdf.extractor import PdfExtractor
from tests.helpers import make_pdf


def test_extracts_text_per_page_with_page_numbers(tmp_path):
    path = make_pdf(tmp_path / "book.pdf", ["First page about Terraform modules.", "", "Third page about Kafka."])
    sections = list(PdfExtractor().extract(path))
    assert [s.page_number for s in sections] == [1, 2, 3]
    assert "Terraform modules" in sections[0].text
    assert sections[1].text.strip() == ""
    assert "Kafka" in sections[2].text


def test_describe_uses_pdf_title_and_page_count(tmp_path):
    path = make_pdf(tmp_path / "book.pdf", ["a", "b"], title="Designing Systems")
    info = PdfExtractor().describe(path)
    assert (info.title, info.page_count) == ("Designing Systems", 2)


def test_describe_falls_back_to_file_name_without_title(tmp_path):
    path = make_pdf(tmp_path / "my-notes.pdf", ["a"])
    assert PdfExtractor().describe(path).title == "my-notes"


def test_extraction_never_modifies_the_file(tmp_path):
    path = make_pdf(tmp_path / "book.pdf", ["some text"])
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    extractor = PdfExtractor()
    extractor.describe(path)
    list(extractor.extract(path))
    after = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    assert before == after


@pytest.mark.parametrize("name,content", [
    ("empty.pdf", b""),
    ("garbage.pdf", b"this is not a pdf"),
    ("header-only.pdf", b"%PDF-1.4\n garbage garbage"),
])
def test_invalid_files_raise_invalid_document(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    extractor = PdfExtractor()
    with pytest.raises(InvalidDocument):
        extractor.describe(path)
    with pytest.raises(InvalidDocument):
        list(extractor.extract(path))


def test_missing_file_raises_invalid_document(tmp_path):
    with pytest.raises(InvalidDocument):
        PdfExtractor().describe(tmp_path / "missing.pdf")


def test_password_protected_pdf_is_rejected(tmp_path):
    path = make_pdf(tmp_path / "locked.pdf", ["secret"], password="hunter2")
    with pytest.raises(InvalidDocument, match="password"):
        PdfExtractor().describe(path)
    with pytest.raises(InvalidDocument, match="password"):
        list(PdfExtractor().extract(path))


def test_error_messages_do_not_leak_file_content(tmp_path):
    path = tmp_path / "x.pdf"
    path.write_bytes(b"SECRET-CONTENT-123 not a pdf")
    with pytest.raises(InvalidDocument) as info:
        PdfExtractor().describe(path)
    assert "SECRET-CONTENT-123" not in str(info.value)
