"""Reranking with a hosted model. Never breaks a query: any failure falls
back to the RRF order, with a warning the UI can show."""

from __future__ import annotations

import logging

from config import settings
from providers.registry import get_reranker
from utils.helpers import truncate_text

logger = logging.getLogger(__name__)


def rerank(
    query: str, candidates: list[dict], top_k: int | None = None
) -> tuple[list[dict], str | None]:
    """Reorder candidates by relevance to the query and keep the best ones.

    Input: the question, the candidates (best RRF first), and how many to keep
        (default TOP_K_FINAL).
    Output: (chunks, warning).
        On success: the top_k chunks in the reranker's order, each with
        "rerank_score" set, and warning None.
        On any failure: the first top_k candidates in RRF order, rerank_score
        None, and a warning string. Empty candidates give ([], None).
    Errors: none raised.
    """
    top_k = settings.TOP_K_FINAL if top_k is None else top_k
    if not candidates:
        return [], None

    try:
        ranked = get_reranker().rerank(
            query, [c["text"] for c in candidates], top_n=top_k
        )
        if not ranked:
            raise RuntimeError("the reranker returned no results")
        chunks = [{**candidates[index], "rerank_score": score} for index, score in ranked]
        return chunks[:top_k], None
    except Exception as exc:  # deliberately broad: a query must never crash here
        logger.warning("Reranker failed: %s", exc)
        fallback = [{**c, "rerank_score": None} for c in candidates[:top_k]]
        reason = truncate_text(str(exc), 150)
        return fallback, f"Reranker unavailable ({reason}); showing results in RRF order."
