import logging
import random

import pytest

from core.chunker import chunk_pages

WORDS = "revenue growth margin quarter board policy contract clause renewal supplier audit".split()


def prose(paragraphs, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(paragraphs):
        sentences = [
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14))).capitalize() + "."
            for _ in range(rng.randint(2, 6))
        ]
        out.append(" ".join(sentences))
    return "\n\n".join(out)


def page(text, num=1, filename="a.pdf"):
    return {"text": text, "filename": filename, "page_num": num, "ocr_used": False}


def positions(text, chunks):
    pos, found = 0, []
    for chunk in chunks:
        p = text.find(chunk)
        assert p != -1, "chunk is not a substring of the page"
        p = text.find(chunk, pos)
        found.append(p)
        pos = p + 1
    return found


def test_size_and_emptiness_invariants():
    chunks = chunk_pages([page(prose(14))])
    assert len(chunks) > 3
    assert all(0 < len(c["text"]) <= 1000 for c in chunks)
    assert all(c["text"] == c["text"].strip() for c in chunks)


def test_chunks_are_substrings_and_never_span_pages():
    p1, p2 = prose(12, seed=1), prose(9, seed=2)
    chunks = chunk_pages([page(p1, 1), page(p2, 2)])
    texts = {1: p1, 2: p2}
    assert {c["page_num"] for c in chunks} == {1, 2}
    assert all(c["text"] in texts[c["page_num"]] for c in chunks)


def test_consecutive_chunks_overlap_without_gaps():
    text = prose(14)
    chunks = [c["text"] for c in chunk_pages([page(text)])]
    pos = positions(text, chunks)
    ends = [p + len(c) for p, c in zip(pos, chunks)]
    for i in range(len(chunks) - 1):
        assert pos[i + 1] < ends[i], "chunks should overlap"
        assert ends[i] - pos[i + 1] <= 200, "overlap should not exceed 200"
    assert ends[-1] >= len(text.rstrip())


def test_chunk_ids_restart_on_each_page():
    chunks = chunk_pages([page(prose(10), 1), page(prose(10, seed=3), 2)])
    ids = [c["chunk_id"] for c in chunks]
    assert ids[0] == "a.pdf_p1_c0"
    assert "a.pdf_p2_c0" in ids
    assert len(set(ids)) == len(ids)


def test_short_page_gives_one_chunk():
    chunks = chunk_pages([page("A short page.")])
    assert [c["text"] for c in chunks] == ["A short page."]


def test_empty_input_returns_empty_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="core.chunker"):
        assert chunk_pages([]) == []
    assert "no pages" in caplog.text


def test_whitespace_only_page_gives_no_chunks():
    assert chunk_pages([page("  \n\n \t ")]) == []


def test_unicode_text():
    text = "Café résumé naïve — 売上は増加しました。 ☃ 😀 " * 60
    chunks = chunk_pages([page(text)])
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 1000 and c["text"] in text for c in chunks)


def test_very_long_unbroken_string():
    rng = random.Random(1)
    text = "".join(rng.choice("abcdefghij") for _ in range(5000))
    chunks = chunk_pages([page(text)])
    assert len(chunks) >= 5
    assert all(len(c["text"]) <= 1000 and c["text"] in text for c in chunks)


def test_repeated_blank_lines_keep_exact_substrings():
    text = "a\n\n\n\nb " * 400
    assert all(c["text"] in text for c in chunk_pages([page(text)]))


def test_custom_sizes_and_validation():
    small = chunk_pages([page(prose(3))], chunk_size=60, overlap=15)
    assert all(0 < len(c["text"]) <= 60 for c in small)
    with pytest.raises(ValueError):
        chunk_pages([page("x")], chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_pages([page("x")], chunk_size=0)
