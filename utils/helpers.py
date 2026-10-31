"""Small helpers: query validation, source formatting, timing, truncation."""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def validate_query(q: str) -> str:
    """Clean and check a user question.

    Input: the raw question.
    Output: the question with surrounding whitespace removed.
    Errors: ValueError if it is None, not a string, or empty after stripping.
    """
    if not isinstance(q, str) or not q.strip():
        raise ValueError("Please enter a question.")
    return q.strip()


def format_sources(chunks: list[dict]) -> str:
    """Build a Markdown list of the sources behind an answer.

    Input: chunks, each with "filename" and "page_num".
    Output: a Markdown bullet list, one line per distinct (filename, page),
        in order of first appearance. Empty input gives "".
    Errors: none.
    """
    seen: set[tuple[str, int]] = set()
    lines: list[str] = []
    for chunk in chunks:
        key = (chunk["filename"], chunk["page_num"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **{key[0]}**, Page {key[1]}")
    return "\n".join(lines)


def timer(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator: log how long the wrapped function took, at INFO level.

    Input: any function.
    Output: the wrapped function with the same behavior.
    Errors: whatever the wrapped function raises (the time is still logged).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info("%s took %.2fs", func.__name__, elapsed)

    return wrapper


def truncate_text(text: str, max_chars: int) -> str:
    """Shorten text for previews.

    Input: text and the maximum number of characters to keep.
    Output: stripped text; if it was cut, "…" is appended after the cut.
    Errors: none.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
