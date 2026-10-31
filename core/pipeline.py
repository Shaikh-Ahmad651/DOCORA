"""Orchestration: index files, answer questions. Used by app.py and by eval.

This is the only module that changes the session index. It never imports
Gradio, so the eval script can use it directly.

``progress`` arguments are optional callables used as ``progress(fraction,
desc=text)`` (a ``gr.Progress`` object works as-is).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from config import settings
from config.prompts import DEFAULT_FORMAT, NO_ANSWER_FALLBACK
from core import generator, index, reranker, retriever
from core.chunker import chunk_pages
from core.loader import LoadReport, load_pdf
from providers.base import ProviderError
from providers.registry import get_embedder
from utils.helpers import format_sources, truncate_text, validate_query

logger = logging.getLogger(__name__)

NO_DOCUMENTS_MESSAGE = (
    "No documents indexed yet. Upload PDFs in the Documents tab and click Index."
)

# Only one indexing run at a time, so two clicks cannot interleave.
_index_lock = threading.Lock()


@dataclass
class IndexResult:
    """Outcome of one index_files call."""

    reports: list[LoadReport] = field(default_factory=list)
    files_added: list[str] = field(default_factory=list)
    files_replaced: list[str] = field(default_factory=list)
    files_skipped_duplicate: list[str] = field(default_factory=list)
    files_rejected_limit: list[str] = field(default_factory=list)  # beyond MAX_FILES
    files_failed: dict[str, str] = field(default_factory=dict)  # filename -> reason
    chunks_added: int = 0


@dataclass
class AnswerResult:
    """Outcome of one answer_question call."""

    answer: str
    chunks: list[dict] = field(default_factory=list)  # chunks passed to the LLM, in order
    sources_md: str = ""
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # seconds
    reranked: bool = False  # True only if the reranker actually ran successfully


# --------------------------------------------------------------------------- indexing
def index_files(
    paths: list[str],
    ocr_enabled: bool,
    progress: Callable[..., None] | None = None,
) -> IndexResult:
    """Read, chunk, embed and index PDF files.

    Input: file paths, whether to run OCR on pages without text, and an
        optional progress callable.
    Output: an IndexResult describing what was added, replaced, skipped,
        rejected or failed, plus one LoadReport per loaded file.
    Rules: at most MAX_FILES files in the session in total; a file with the
        same name and content is skipped; the same name with new content
        replaces the old one; a file is added only after ALL its embeddings
        succeeded, and one failing file does not stop the others.
    Errors: none raised; problems are reported in the result.
    """
    with _index_lock:
        return _index_files_locked(list(paths or []), ocr_enabled, progress)


def _index_files_locked(
    paths: list[str], ocr_enabled: bool, progress: Callable[..., None] | None
) -> IndexResult:
    result = IndexResult()
    total = len(paths)

    def report(fraction: float, text: str) -> None:
        if progress is not None:
            progress(min(max(fraction, 0.0), 1.0), desc=text)

    for n, path in enumerate(paths):
        filename = os.path.basename(path)

        def file_progress(share: float, text: str, _n: int = n) -> None:
            report((_n + share) / total, text)

        try:
            _index_one(path, filename, ocr_enabled, result, file_progress)
        except Exception as exc:  # one bad file must not stop the others
            logger.exception("Unexpected error while indexing %s", filename)
            result.files_failed[filename] = f"Unexpected error: {truncate_text(str(exc), 150)}"

    report(1.0, "Done")
    return result


def _index_one(path, filename, ocr_enabled, result: IndexResult, file_progress) -> None:
    """Index one file and record the outcome in ``result``."""
    try:
        file_hash = _file_sha256(path)
    except OSError as exc:
        result.files_failed[filename] = f"Could not read the file ({exc})."
        return

    existing_hash = index.file_hash(filename)
    if existing_hash == file_hash:
        result.files_skipped_duplicate.append(filename)
        return
    is_replacement = existing_hash is not None
    if not is_replacement and index.stats()["files"] >= settings.MAX_FILES:
        result.files_rejected_limit.append(filename)
        return

    # 1. Read the pages (first 70% of this file's progress).
    def page_progress(page_num: int, total_pages: int) -> None:
        file_progress(0.7 * page_num / total_pages, f"Reading {filename} (page {page_num}/{total_pages})")

    file_progress(0.0, f"Reading {filename}")
    pages, load_report = load_pdf(path, filename, ocr_enabled, page_progress)
    result.reports.append(load_report)
    if load_report.error:
        result.files_failed[filename] = load_report.error
        return

    # 2. Split into chunks.
    chunks = chunk_pages(pages)
    if not chunks:
        result.files_failed[filename] = (
            "No text could be extracted (see the page notes for this file)."
        )
        return

    # 3. Embed everything before touching the index.
    file_progress(0.7, f"Embedding {filename}")
    try:
        embeddings = _embed_chunks(chunks)
        index.add_file(
            filename, file_hash, chunks, embeddings, pages=load_report.total_pages
        )
    except (ProviderError, ValueError) as exc:
        result.files_failed[filename] = str(exc)
        return

    (result.files_replaced if is_replacement else result.files_added).append(filename)
    result.chunks_added += len(chunks)
    file_progress(1.0, f"Indexed {filename}")


def _embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Embed all chunk texts in batches. Retries happen inside the provider.

    Errors: ProviderError, or ValueError for a missing key.
    """
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), settings.EMBED_BATCH_SIZE):
        vectors.extend(embedder.embed_documents(texts[start : start + settings.EMBED_BATCH_SIZE]))
    return vectors


def _file_sha256(path: str) -> str:
    """SHA-256 of a file's bytes. Errors: OSError if it cannot be read."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_session() -> None:
    """Forget every indexed file. Errors: none."""
    with _index_lock:
        index.clear()


def list_indexed() -> list[dict]:
    """The indexed files as [{"filename", "pages", "chunks"}], in index order.
    Errors: none."""
    per_file = index.stats()["per_file"]
    return [
        {"filename": name, "pages": info["pages"], "chunks": info["chunks"]}
        for name, info in per_file.items()
    ]


# --------------------------------------------------------------------------- answering
def answer_question(
    query: str,
    format_name: str = DEFAULT_FORMAT,
    use_reranker: bool = True,
) -> AnswerResult:
    """Answer a question from the indexed documents.

    Input: the question, a key of FORMAT_OPTIONS, and whether to rerank.
    Output: an AnswerResult. ``timings`` always has embed_query, retrieve,
        rerank, generate and total (seconds; 0.0 for a stage that did not run).
        ``sources_md`` is empty when the answer is the "not enough
        information" fallback, since nothing was actually cited.
    Errors: none raised. Provider problems, a missing key or an empty model
        response become a readable message in ``answer``.
    """
    started = time.perf_counter()
    timings = {"embed_query": 0.0, "retrieve": 0.0, "rerank": 0.0, "generate": 0.0, "total": 0.0}
    warnings: list[str] = []
    final_chunks: list[dict] = []
    reranked = False

    def result(answer: str, sources_md: str = "") -> AnswerResult:
        timings["total"] = time.perf_counter() - started
        return AnswerResult(
            answer=answer,
            chunks=final_chunks,
            sources_md=sources_md,
            warnings=warnings,
            timings=timings,
            reranked=reranked,
        )

    if index.is_empty():
        return result(NO_DOCUMENTS_MESSAGE)
    try:
        query = validate_query(query)
    except ValueError as exc:
        return result(str(exc))

    try:
        t = time.perf_counter()
        candidates = retriever.retrieve(query, timings)
        timings["retrieve"] = time.perf_counter() - t

        if use_reranker and len(candidates) >= 2:
            t = time.perf_counter()
            final_chunks, warning = reranker.rerank(query, candidates, settings.TOP_K_FINAL)
            timings["rerank"] = time.perf_counter() - t
            if warning:
                warnings.append(warning)
            else:
                reranked = True
        else:
            final_chunks = candidates[: settings.TOP_K_FINAL]

        t = time.perf_counter()
        answer = generator.generate(query, final_chunks, format_name)
        timings["generate"] = time.perf_counter() - t
    except ProviderError as exc:
        logger.warning("Provider error while answering: %s", exc)
        return result(f"Sorry, the request failed: {exc}")
    except (ValueError, RuntimeError) as exc:
        logger.warning("Could not answer: %s", exc)
        return result(f"Sorry, I could not answer: {exc}")
    except Exception as exc:  # last resort: never show a stack trace
        logger.exception("Unexpected error while answering")
        return result(f"Sorry, something went wrong: {truncate_text(str(exc), 150)}")

    sources_md = "" if answer == NO_ANSWER_FALLBACK else format_sources(final_chunks)
    return result(answer, sources_md)
