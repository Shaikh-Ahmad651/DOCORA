import numpy as np
import pytest

from core.index import SessionIndex, tokenize


def mk(filename, page, i, text):
    return {"text": text, "filename": filename, "page_num": page, "chunk_id": f"{filename}_p{page}_c{i}"}


# ---- tokenizer
@pytest.mark.parametrize(
    "text, expected",
    [
        ("revenue,", ["revenue"]),
        ("grew 4.2 percent", ["grew", "4.2", "percent"]),
        ("Q3-2023 results", ["q3-2023", "results"]),
        ("don't panic", ["don't", "panic"]),
        ("", []),
        ("!!! ---", []),
    ],
)
def test_tokenizer(text, expected):
    assert tokenize(text) == expected


# ---- add / remove / replace
def test_add_stats_and_lookup():
    s = SessionIndex()
    assert s.is_empty()
    s.add_file("a.pdf", "h1", [mk("a.pdf", 1, 0, "alpha"), mk("a.pdf", 2, 0, "beta")], [[1, 0, 0], [0, 1, 0]])
    s.add_file("b.pdf", "h2", [mk("b.pdf", 1, 0, "gamma")], [[0, 0, 5]], pages=7)
    stats = s.stats()
    assert stats["files"] == 2 and stats["chunks"] == 3
    assert stats["per_file"]["a.pdf"] == {"pages": 2, "chunks": 2}
    assert stats["per_file"]["b.pdf"]["pages"] == 7
    assert s.has_file("a.pdf", "h1") and not s.has_file("a.pdf", "other")
    assert s.file_hash("b.pdf") == "h2" and s.file_hash("nope") is None


def test_replace_and_remove():
    s = SessionIndex()
    s.add_file("a.pdf", "h1", [mk("a.pdf", 1, 0, "old one"), mk("a.pdf", 2, 0, "old two")], [[1, 0], [0, 1]])
    s.add_file("a.pdf", "h2", [mk("a.pdf", 1, 0, "new")], [[1, 1]])
    assert s.stats()["chunks"] == 1 and s.file_hash("a.pdf") == "h2"
    s.remove_file("a.pdf")
    s.remove_file("never-there")  # ignored
    assert s.is_empty()


def test_empty_index_accepts_a_new_vector_size():
    s = SessionIndex()
    s.add_file("a.pdf", "h", [mk("a.pdf", 1, 0, "x")], [[1, 2, 3, 4]])
    s.add_file("a.pdf", "h2", [mk("a.pdf", 1, 0, "x")], [[1, 2]])  # replacing the only file
    assert s.vector_search([1, 2], 1)


# ---- vector search
def test_vector_search_with_handmade_vectors():
    s = SessionIndex()
    s.add_file("a.pdf", "h", [mk("a.pdf", 1, 0, "alpha"), mk("a.pdf", 2, 0, "beta")], [[1, 0, 0], [0, 1, 0]])
    s.add_file("b.pdf", "h", [mk("b.pdf", 1, 0, "gamma")], [[0, 0, 5]])
    result = s.vector_search([0.9, 0.1, 0], 2)
    assert [r["chunk_id"] for r in result] == ["a.pdf_p1_c0", "a.pdf_p2_c0"]
    assert result[0]["score"] == pytest.approx(0.9 / np.hypot(0.9, 0.1), abs=1e-5)
    everything = s.vector_search([0, 0, 100], 10)  # k larger than the index
    assert len(everything) == 3
    assert everything[0]["chunk_id"] == "b.pdf_p1_c0" and everything[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_vector_search_edge_cases():
    s = SessionIndex()
    assert s.vector_search([1, 0, 0], 3) == []
    s.add_file("a.pdf", "h", [mk("a.pdf", 1, 0, "alpha")], [[1, 0, 0]])
    assert s.vector_search([0, 0, 0], 3) == []
    with pytest.raises(ValueError):
        s.vector_search([1, 0], 3)


def test_zero_vector_is_stored_safely():
    s = SessionIndex()
    s.add_file("z.pdf", "h", [mk("z.pdf", 1, 0, "zero")], [[0, 0, 0]])
    assert s.vector_search([1, 0, 0], 1)[0]["score"] == 0.0


def test_dimension_mismatch_raises_and_changes_nothing():
    s = SessionIndex()
    s.add_file("a.pdf", "h", [mk("a.pdf", 1, 0, "alpha")], [[1, 0, 0]])
    before = s.stats()
    with pytest.raises(ValueError, match="mismatch"):
        s.add_file("c.pdf", "h3", [mk("c.pdf", 1, 0, "x")], [[1, 2]])
    assert s.stats() == before


@pytest.mark.parametrize(
    "chunks, embeddings",
    [([], []), ([mk("c.pdf", 1, 0, "x")], []), ([mk("c.pdf", 1, 0, "x"), mk("c.pdf", 1, 1, "y")], [[1, 2, 3], [1, 2]])],
)
def test_bad_add_input_raises(chunks, embeddings):
    with pytest.raises(ValueError):
        SessionIndex().add_file("c.pdf", "h", chunks, embeddings)


# ---- BM25
UNIQUE = ["zebra", "quartz", "lantern", "harbor", "violin", "cactus", "meadow", "falcon", "glacier", "tundra", "ember", "walnut"]


@pytest.fixture
def bm25_index():
    s = SessionIndex()
    chunks = [mk("k.pdf", i + 1, 0, f"This page talks about the {w} and ordinary filler words.") for i, w in enumerate(UNIQUE)]
    s.add_file("k.pdf", "h", chunks, [[float(i == j) for j in range(12)] for i in range(12)])
    return s


def test_bm25_returns_only_positive_scores(bm25_index):
    hits = bm25_index.bm25_search("zebra", 5)
    assert [h["chunk_id"] for h in hits] == ["k.pdf_p1_c0"]
    assert hits[0]["score"] > 0


def test_bm25_no_match_and_empty_queries(bm25_index):
    assert bm25_index.bm25_search("qqqqq", 5) == []
    assert bm25_index.bm25_search("", 5) == []
    assert bm25_index.bm25_search("!!!", 5) == []


def test_bm25_k_and_punctuation(bm25_index):
    assert len(bm25_index.bm25_search("zebra harbor falcon", 2)) == 2
    assert bm25_index.bm25_search("ZEBRA,", 3)[0]["chunk_id"] == "k.pdf_p1_c0"


def test_bm25_on_empty_index_and_wordless_chunk():
    assert SessionIndex().bm25_search("x", 3) == []
    s = SessionIndex()
    s.add_file("p.pdf", "h", [mk("p.pdf", 1, 0, "!!! --- ???")], [[1, 0]])
    assert s.bm25_search("anything", 3) == []
