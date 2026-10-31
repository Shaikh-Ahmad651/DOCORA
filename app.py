"""Docora: Gradio UI. Presentation only; all logic lives in core/pipeline.py.

Run with:  python app.py
"""

from __future__ import annotations

import logging

import gradio as gr

from config import settings
from config.prompts import DEFAULT_FORMAT, FORMAT_OPTIONS
from core import pipeline
from core.pipeline import AnswerResult, IndexResult
from utils.helpers import truncate_text

logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)

OCR_AVAILABLE = settings.OCR_PROVIDER != "none"
TABLE_HEADERS = ["#", "File", "Page", "Vector", "BM25", "RRF", "Rerank", "Preview"]
FILES_HEADERS = ["File", "Pages", "Chunks"]


# ----------------------------------------------------------------- text builders
def _page_list(pages: list[int]) -> str:
    """[1, 2, 3, 7] -> "1-3, 7"."""
    parts, start, prev = [], None, None
    for page in sorted(pages) + [None]:
        if start is None:
            start = prev = page
        elif page is not None and page == prev + 1:
            prev = page
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = page
    return ", ".join(parts)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _where(pages: list[int]) -> str:
    """"page 4" or "pages 1-3, 7"."""
    return ("page " if len(pages) == 1 else "pages ") + _page_list(pages)


def build_status(result: IndexResult, ocr_enabled: bool) -> str:
    """Plain-language Markdown summary of an index run.

    Input: the IndexResult and whether OCR was ticked. Output: Markdown.
    """
    lines: list[str] = []
    if result.files_added:
        lines.append("✅ **Added:** " + ", ".join(f"`{n}`" for n in result.files_added))
    if result.files_replaced:
        lines.append("🔁 **Replaced (new version):** " + ", ".join(f"`{n}`" for n in result.files_replaced))
    if result.files_skipped_duplicate:
        lines.append("⏭️ **Skipped (already indexed, unchanged):** " + ", ".join(f"`{n}`" for n in result.files_skipped_duplicate))
    if result.files_rejected_limit:
        lines.append(
            f"🚫 **Not added (limit of {settings.MAX_FILES} files per session):** "
            + ", ".join(f"`{n}`" for n in result.files_rejected_limit)
        )
    for name, reason in result.files_failed.items():
        lines.append(f"❌ **Failed:** `{name}`: {reason}")

    for report in result.reports:
        if report.skipped_pages:
            count = _plural(len(report.skipped_pages), "page")
            where = _where(report.skipped_pages)
            if ocr_enabled:
                lines.append(f"⚠️ `{report.filename}`: {count} had no readable text even with OCR ({where}).")
            elif OCR_AVAILABLE:
                lines.append(
                    f"⚠️ `{report.filename}`: {count} had no extractable text and "
                    f"{'was' if len(report.skipped_pages) == 1 else 'were'} skipped ({where}). "
                    "Tick the scan option and re-index."
                )
            else:
                lines.append(
                    f"⚠️ `{report.filename}`: {count} had no extractable text and were skipped "
                    f"({where}). OCR is turned off in this app's settings."
                )
        if report.failed_pages:
            lines.append(
                f"⚠️ `{report.filename}`: {_plural(len(report.failed_pages), 'page')} could not be read "
                f"({_where(report.failed_pages)})."
            )

    if not lines:
        return "Nothing to do."
    if result.chunks_added:
        lines.append(f"\n{_plural(result.chunks_added, 'chunk')} added to the index.")
    return "\n\n".join(lines)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def chunk_rows(chunks: list[dict]) -> list[list]:
    """Rows for the 'Retrieved chunks and scores' table."""
    rows = []
    for i, c in enumerate(chunks, start=1):
        preview = truncate_text(" ".join(c["text"].split()), settings.PREVIEW_CHARS)
        rows.append(
            [
                i,
                c["filename"],
                c["page_num"],
                _fmt(c.get("vector_score")),
                _fmt(c.get("bm25_score"), 2),
                _fmt(c.get("rrf_score")),
                _fmt(c.get("rerank_score")),
                preview,
            ]
        )
    return rows


def timings_line(timings: dict[str, float]) -> str:
    """One-line summary of the stage timings (stages that did not run are left out)."""
    if not timings:
        return ""
    labels = [
        ("embed_query", "embed query"),
        ("retrieve", "retrieve (incl. embed)"),
        ("rerank", "rerank"),
        ("generate", "generate"),
    ]
    parts = [f"{label} {timings[key]:.2f}s" for key, label in labels if timings.get(key, 0) > 0]
    parts.append(f"**total {timings.get('total', 0):.2f}s**")
    return "⏱️ " + " · ".join(parts)


# -------------------------------------------------------------------- handlers
def _files_table() -> list[list]:
    return [[f["filename"], f["pages"], f["chunks"]] for f in pipeline.list_indexed()]


def on_index(files, ocr_enabled, progress=gr.Progress()):
    """Index button: returns (status Markdown, files table)."""
    if not files:
        return "Please choose at least one PDF first.", _files_table()
    paths = [f if isinstance(f, str) else f.name for f in files]
    ocr_on = bool(ocr_enabled) and OCR_AVAILABLE
    result = pipeline.index_files(paths, ocr_on, progress=progress)
    return build_status(result, ocr_on), _files_table()


def on_reset():
    """Reset button: forget everything."""
    pipeline.reset_session()
    return "Session reset. All indexed documents were removed.", _files_table()


def on_ask(query, format_name, use_reranker):
    """Ask button / Enter: returns (answer, sources, warnings, timings, table rows)."""
    result: AnswerResult = pipeline.answer_question(query, format_name, bool(use_reranker))
    sources = f"**Sources**\n\n{result.sources_md}" if result.sources_md else ""
    warnings = "\n\n".join(f"⚠️ {w}" for w in result.warnings)
    return result.answer, sources, warnings, timings_line(result.timings), chunk_rows(result.chunks)


# ------------------------------------------------------------------------- layout
def build_ui() -> gr.Blocks:
    """Build the Gradio interface. Errors: none."""
    problems = settings.check_config()

    with gr.Blocks(title="Docora") as demo:
        gr.Markdown("# 📄 Docora\nAsk questions about your PDFs. Every answer cites its source pages.")
        if problems:
            gr.Markdown("**Setup needed:**\n\n" + "\n".join(f"- {p}" for p in problems))

        with gr.Tabs():
            with gr.Tab("Documents"):
                files = gr.File(
                    label=f"Upload PDFs (up to {settings.MAX_FILES} per session)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                ocr = gr.Checkbox(
                    label="My files contain scans or images (run OCR)",
                    value=False,
                    visible=OCR_AVAILABLE,
                )
                with gr.Row():
                    index_btn = gr.Button("Index", variant="primary")
                    reset_btn = gr.Button("Reset session")
                status = gr.Markdown()
                files_table = gr.Dataframe(
                    headers=FILES_HEADERS,
                    value=_files_table,
                    datatype=["str", "number", "number"],
                    interactive=False,
                    label="Indexed files",
                )

            with gr.Tab("Ask"):
                query = gr.Textbox(
                    label="Your question",
                    placeholder="Ask something about your documents…",
                    lines=1,
                    max_lines=1,
                )
                with gr.Row():
                    format_dd = gr.Dropdown(
                        choices=list(FORMAT_OPTIONS),
                        value=DEFAULT_FORMAT,
                        label="Answer format",
                    )
                    rerank_cb = gr.Checkbox(label="Use reranker", value=True)
                ask_btn = gr.Button("Ask", variant="primary")

                answer = gr.Markdown()
                sources = gr.Markdown()
                warnings = gr.Markdown()
                timings = gr.Markdown()
                with gr.Accordion("Retrieved chunks and scores", open=False):
                    scores = gr.Dataframe(
                        headers=TABLE_HEADERS,
                        datatype=["number", "str", "number", "str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )

        index_btn.click(on_index, [files, ocr], [status, files_table])
        reset_btn.click(on_reset, None, [status, files_table])
        ask_inputs = [query, format_dd, rerank_cb]
        ask_outputs = [answer, sources, warnings, timings, scores]
        ask_btn.click(on_ask, ask_inputs, ask_outputs)
        query.submit(on_ask, ask_inputs, ask_outputs)
        demo.load(_files_table, None, files_table)

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
