import pytest

from core.retriever import reciprocal_rank_fusion


def chunk(cid, score=0.5):
    return {"text": f"text {cid}", "filename": "f.pdf", "page_num": 1, "chunk_id": cid, "score": score}


A, B, C = chunk("A", 0.9), chunk("B", 0.8), chunk("C", 0.7)


def test_merge_order_and_scores():
    fused = reciprocal_rank_fusion([A, B, C], [dict(C, score=11.0), dict(A, score=7.0)])
    assert [f["chunk_id"] for f in fused] == ["A", "C", "B"]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1]["rrf_score"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2]["rrf_score"] == pytest.approx(1 / 62)


def test_ranks_and_scores_filled_where_present():
    a, _, b = reciprocal_rank_fusion([A, B, C], [dict(C, score=11.0), dict(A, score=7.0)])
    assert (a["vector_rank"], a["bm25_rank"], a["vector_score"], a["bm25_score"]) == (1, 2, 0.9, 7.0)
    assert (b["vector_rank"], b["bm25_rank"], b["bm25_score"]) == (2, None, None)


def test_dedupe_and_clean_fields():
    fused = reciprocal_rank_fusion([A, B], [A, B])
    assert len(fused) == 2
    assert all(f["rerank_score"] is None and "score" not in f for f in fused)


def test_empty_lists():
    assert reciprocal_rank_fusion([], []) == []
    only_bm25 = reciprocal_rank_fusion([], [A, B])
    assert [f["chunk_id"] for f in only_bm25] == ["A", "B"]
    assert only_bm25[0]["vector_rank"] is None


def test_overlap_ranks_above_single_list_hits():
    # D is rank 3 in both lists; E is rank 1 in only one list.
    # 2/63 (two lists) beats 1/61 (one list, rank 1).
    fused = reciprocal_rank_fusion([chunk("E"), chunk("X"), chunk("D")], [chunk("Y"), chunk("Z"), chunk("D")])
    assert fused[0]["chunk_id"] == "D"


def test_tie_keeps_first_seen_order():
    fused = reciprocal_rank_fusion([A], [B])
    assert [f["chunk_id"] for f in fused] == ["A", "B"]
