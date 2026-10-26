import pytest

from app.core.models import Section
from app.ingestion.chunking import chunk_sections, normalize_text
from tests.helpers import words


def pages(*texts):
    return [Section(text=t, page_number=i + 1) for i, t in enumerate(texts)]


def run(sections, size=200, overlap=40):
    return list(chunk_sections(sections, chunk_size=size, overlap=overlap))


def test_short_text_is_a_single_chunk_with_its_page():
    chunks = run(pages("Hello world."))
    assert len(chunks) == 1
    assert (chunks[0].index, chunks[0].text) == (0, "Hello world.")
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)


def test_chunks_respect_size_and_do_not_lose_text():
    all_words = words("w", 300).split()
    chunks = run(pages(" ".join(all_words)), size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(all_words)


def test_consecutive_chunks_overlap():
    chunks = run(pages(words("w", 300)), size=200, overlap=40)
    for prev, nxt in zip(chunks, chunks[1:]):
        first_word = nxt.text.split()[0]
        assert first_word in prev.text.split(), "next chunk must start inside the previous one"
        assert first_word != prev.text.split()[0]  # but it still moves forward


def test_zero_overlap_partitions_the_text():
    all_words = words("w", 200).split()
    chunks = run(pages(" ".join(all_words)), size=200, overlap=0)
    assert [w for c in chunks for w in c.text.split()] == all_words


def test_page_provenance_matches_the_text_of_each_chunk():
    texts = [words(f"p{n}t", 40) for n in (1, 2, 3)]  # ~280 chars per page
    chunks = run(pages(*texts), size=500, overlap=60)
    assert len(chunks) >= 3
    for c in chunks:
        pages_in_text = {int(tok[1]) for tok in c.text.split()}
        assert (min(pages_in_text), max(pages_in_text)) == (c.page_start, c.page_end)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    assert any(c.page_start != c.page_end for c in chunks), "some chunk must span a page boundary"


def test_blank_sections_are_skipped_but_page_numbers_are_kept():
    chunks = run([Section(text="   \n ", page_number=1), Section(text="real content", page_number=2)])
    assert len(chunks) == 1 and (chunks[0].page_start, chunks[0].page_end) == (2, 2)


def test_all_blank_input_yields_nothing():
    assert run(pages("", "  ", "\n\n")) == []
    assert run([]) == []


def test_sections_are_consumed_lazily():
    consumed = []

    def source():
        for n in range(1, 51):
            consumed.append(n)
            yield Section(text=words(f"p{n}t", 20), page_number=n)

    first = next(iter(chunk_sections(source(), chunk_size=200, overlap=20)))
    assert first.index == 0
    assert len(consumed) <= 3, "the first chunk must not require reading the whole document"


def test_text_without_any_whitespace_still_terminates_and_is_covered():
    chunks = run(pages("x" * 5000), size=200, overlap=20)
    assert all(len(c.text) <= 200 for c in chunks)
    assert "".join(c.text for c in chunks) == "x" * 5000


def test_page_numbers_may_be_unknown():
    chunks = run([Section(text="no page info here")])
    assert (chunks[0].page_start, chunks[0].page_end) == (None, None)


@pytest.mark.parametrize("size,overlap", [(200, 100), (200, 150), (200, -1), (1, 0)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        run(pages("text"), size=size, overlap=overlap)


def test_normalize_text():
    assert normalize_text("a\x00b  \t c\r\n\r\n\r\n\r\nd \n e") == "ab c\n\nd\ne"
    assert normalize_text("   ") == ""
