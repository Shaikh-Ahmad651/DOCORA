import pytest

from utils.helpers import format_sources, truncate_text, validate_query


def test_validate_query_strips():
    assert validate_query("  hello  ") == "hello"


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_validate_query_rejects_empty(bad):
    with pytest.raises(ValueError):
        validate_query(bad)


def test_format_sources_dedupes_and_keeps_first_appearance_order():
    chunks = [
        {"filename": "a.pdf", "page_num": 2},
        {"filename": "b.pdf", "page_num": 1},
        {"filename": "a.pdf", "page_num": 2},
        {"filename": "a.pdf", "page_num": 1},
    ]
    assert format_sources(chunks).splitlines() == [
        "- **a.pdf**, Page 2",
        "- **b.pdf**, Page 1",
        "- **a.pdf**, Page 1",
    ]


def test_format_sources_empty():
    assert format_sources([]) == ""


def test_truncate_text():
    assert truncate_text("  hello world  ", 5) == "hello…"
    assert truncate_text("short", 50) == "short"
