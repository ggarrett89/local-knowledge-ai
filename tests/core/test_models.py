import pytest

from app.core.errors import (
    EmbeddingError,
    EmbeddingModelMismatch,
    InvalidDocument,
    InvalidSlug,
    LkaError,
    ModelNotInstalled,
    OllamaUnavailable,
)
from app.core.models import DocumentRecord, Provenance, make_chunk_id, normalize_slug


@pytest.mark.parametrize("raw,expected", [
    ("devops-sre", "devops-sre"),
    ("  Kubernetes ", "kubernetes"),
    ("a", "a"),
    ("0day", "0day"),
])
def test_normalize_slug_accepts_valid(raw, expected):
    assert normalize_slug(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", "-bad", "has space", "under_score", "ação", "a" * 65, "x/y", "../etc"])
def test_normalize_slug_rejects_invalid(raw):
    with pytest.raises(InvalidSlug):
        normalize_slug(raw)


def test_invalid_slug_is_a_value_error():
    assert issubclass(InvalidSlug, ValueError)


def test_chunk_id_is_document_id_plus_index():
    assert make_chunk_id("pdf-abc", 7) == "pdf-abc:7"


def test_provenance_leaves_unknown_fields_unset():
    p = Provenance(source_type="pdf", document_id="pdf-1", title="Book", page_start=3, page_end=4)
    assert p.url is None and p.section is None and p.line_start is None and p.repo is None


def test_document_record_defaults():
    d = DocumentRecord(document_id="pdf-1", title="T", source_type="pdf", category="general", status="indexed")
    assert d.topics == [] and d.chunk_count == 0 and d.version is None and d.language is None


def test_only_service_wide_errors_are_systemic():
    assert all(e.systemic for e in (OllamaUnavailable, ModelNotInstalled, EmbeddingModelMismatch))
    assert not any(e.systemic for e in (InvalidDocument, EmbeddingError, LkaError))


def test_search_filters_normalize_and_detect_emptiness():
    from app.core.models import SearchFilters
    f = SearchFilters(category=" Cloud ", topics=["AWS", "IAM"])
    assert (f.category, f.topics) == ("cloud", ["aws", "iam"])
    assert SearchFilters().is_empty() and SearchFilters(topics=[]).is_empty()
    assert not SearchFilters(document_ids=[]).is_empty(), "an explicit empty id list restricts to nothing"
    assert not SearchFilters(source_type="pdf").is_empty()


def test_search_filters_reject_invalid_slugs():
    from pydantic import ValidationError
    from app.core.models import SearchFilters
    with pytest.raises(ValidationError):
        SearchFilters(category="Not Valid!")
    with pytest.raises(ValidationError):
        SearchFilters(topics=["ok", "../etc"])
    with pytest.raises(ValidationError):
        SearchFilters(source_type="video")


def test_invalid_query_is_a_value_error():
    from app.core.errors import InvalidQuery
    assert issubclass(InvalidQuery, ValueError) and not InvalidQuery.systemic
