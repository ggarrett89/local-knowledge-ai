import pytest

from app.core.models import Provenance
from app.core.references import format_pages, format_reference


def prov(**kw):
    data = dict(source_type="pdf", document_id="pdf-1", title="Book Title")
    data.update(kw)
    return Provenance(**data)


@pytest.mark.parametrize("kwargs,expected", [
    (dict(page_start=95, page_end=95), "Book Title, p. 95"),
    (dict(page_start=95), "Book Title, p. 95"),
    (dict(page_start=95, page_end=96), "Book Title, pp. 95-96"),
    (dict(), "Book Title"),
])
def test_pdf_references(kwargs, expected):
    assert format_reference(prov(**kwargs)) == expected


def test_web_reference_uses_url_and_section_only_when_known():
    p = prov(source_type="web", title="Install guide", url="https://docs.example.org/install", section="Linux")
    assert format_reference(p) == "Install guide, \u00a7 Linux, https://docs.example.org/install"
    assert format_reference(prov(source_type="web", title="T", url="https://x.org")) == "T, https://x.org"


def test_code_reference_with_and_without_lines_and_repo():
    base = dict(source_type="code", title="main.tf")
    assert format_reference(prov(**base, repo="infra", path="envs/prod/main.tf", line_start=10, line_end=42)) == \
        "main.tf, infra:envs/prod/main.tf:10-42"
    assert format_reference(prov(**base, path="main.tf", line_start=7, line_end=7)) == "main.tf, main.tf:7"
    assert format_reference(prov(**base, path="main.tf")) == "main.tf, main.tf"


def test_unknown_fields_are_never_invented():
    ref = format_reference(prov())
    assert "p." not in ref and "http" not in ref and "\u00a7" not in ref


def test_format_pages():
    assert format_pages(prov()) == ""
    assert format_pages(prov(page_start=3, page_end=5)) == "pp. 3-5"
