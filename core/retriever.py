"""Hybrid retrieval: vector search + BM25, merged with Reciprocal Rank Fusion."""

from __future__ import annotations

import time

from config import settings
from core import index
from providers.registry import get_embedder
from utils.helpers import validate_query


def reciprocal_rank_fusion(
    vector_results: list[dict], bm25_results: list[dict]
) -> list[dict]:
    """Merge two ranked lists with RRF (pure function, no I/O).

    Input: the vector and BM25 results, each best-first, each chunk carrying
        a "score". For a chunk at 1-indexed rank r in a list, RRF adds
        1 / (RRF_K + r). Chunks are matched on chunk_id.
    Output: candidates sorted by rrf_score, best first:
        {text, filename, page_num, chunk_id, vector_rank, vector_score,
         bm25_rank, bm25_score, rrf_score, rerank_score (None)}.
        A chunk missing from one list has None for that list's rank/score.
    Errors: none.
    """
    candidates: dict[str, dict] = {}

    def entry(chunk: dict) -> dict:
        found = candidates.get(chunk["chunk_id"])
        if found is None:
            found = {
                "text": chunk["text"],
                "filename": chunk["filename"],
                "page_num": chunk["page_num"],
                "chunk_id": chunk["chunk_id"],
                "vector_rank": None,
                "vector_score": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
                "rerank_score": None,
            }
            candidates[chunk["chunk_id"]] = found
        return found

    for rank, chunk in enumerate(vector_results, start=1):
        item = entry(chunk)
        item["vector_rank"] = rank
        item["vector_score"] = chunk["score"]
        item["rrf_score"] += 1.0 / (settings.RRF_K + rank)
    for rank, chunk in enumerate(bm25_results, start=1):
        item = entry(chunk)
        item["bm25_rank"] = rank
        item["bm25_score"] = chunk["score"]
        item["rrf_score"] += 1.0 / (settings.RRF_K + rank)

    # sorted() is stable, so ties keep their first-seen order.
    return sorted(candidates.values(), key=lambda c: c["rrf_score"], reverse=True)


def retrieve(query: str, timings: dict | None = None) -> list[dict]:
    """Find the chunks most relevant to a question.

    Input: the question. Optional ``timings`` dict; if given, the time spent
        embedding the query is stored in timings["embed_query"] (seconds).
    Output: candidates (see reciprocal_rank_fusion), best first; [] if the
        index is empty or nothing matched.
    Errors: ValueError for an empty query, a missing API key, or an embedding
        size that does not match the index; ProviderError if the embedding
        call fails after its retries.
    """
    query = validate_query(query)

    start = time.perf_counter()
    query_vec = get_embedder().embed_query(query)
    if timings is not None:
        timings["embed_query"] = time.perf_counter() - start

    vector_results = index.vector_search(query_vec, settings.TOP_K_VECTOR)
    bm25_results = index.bm25_search(query, settings.TOP_K_BM25)
    if not vector_results and not bm25_results:
        return []
    return reciprocal_rank_fusion(vector_results, bm25_results)
