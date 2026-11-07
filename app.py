"""Docora: Gradio 6 UI. Presentation only; all logic lives in core/pipeline.py.

One page: a blue header with live stats, a documents panel (upload, OCR option,
index, file list) and a chat area (quick questions, options, answers, scores).
On laptops the panel sits beside the chat; on phones everything stacks and the
header gets jump links. Each question is still answered independently (no
conversation memory is sent to the model); the chat thread is only a display.

Run with:  python app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from config import settings
from config.prompts import DEFAULT_FORMAT, FORMAT_OPTIONS
from core import pipeline
from core.pipeline import AnswerResult, IndexResult
from utils.helpers import truncate_text

logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)

OCR_AVAILABLE = settings.OCR_PROVIDER != "none"
TABLE_HEADERS = ["#", "File", "Page", "Vector", "BM25", "RRF", "Rerank", "Preview"]
QUICK_QUESTIONS = [
]

WELCOME = (
    "👋 **Welcome to Docora.** Add your PDFs in the documents panel and press "
    "**Index**, then ask anything about them. Every answer cites its source pages."
)

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    text_size=gr.themes.sizes.text_lg,
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#eef4ff",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)

CSS = """
:root { --blue: #1d4ed8; --blue-mid: #2563eb; --blue-soft: #dbeafe; --blue-wash: #eaf2ff;
        --line: #bfdbfe; --surface: #ffffff; --ink: #0f172a; }
.dark { --blue-soft: #1e3a8a; --blue-wash: #0f1b36; --line: #1e40af; --surface: #0b1220; --ink: #e2e8f0; }
html { font-size: 17px; }
footer { display: none !important; }
.gradio-container { max-width: 1600px !important; margin: 0 auto !important; padding: 0 20px 24px !important; }

/* header */
#hero { background: var(--blue); color: #fff; border-radius: 0 0 24px 24px; padding: 18px 28px;
        display: flex; flex-wrap: wrap; align-items: center; gap: 14px 20px; margin-bottom: 18px; }
#hero .logo { font-size: 2rem; width: 58px; height: 58px; display: grid; place-items: center;
              background: #fff; border-radius: 16px; }
#hero h1 { margin: 0; font-size: 1.8rem; font-weight: 800; color: #fff; line-height: 1.1; }
#hero p { margin: 4px 0 0; font-size: 1.05rem; color: #dbeafe; }
#hero .stats { margin-left: auto; display: flex; gap: 10px; }
#hero .stat { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.35);
              border-radius: 14px; padding: 8px 18px; text-align: center; min-width: 92px; }
#hero .stat b { display: block; font-size: 1.5rem; color: #fff; }
#hero .stat span { font-size: .9rem; color: #dbeafe; }
#hero .jump { display: none; gap: 10px; width: 100%; }
#hero .jump a { flex: 1; text-align: center; padding: 12px; border-radius: 12px; font-weight: 600;
                font-size: 1.05rem; background: #fff; color: var(--blue); text-decoration: none; }

/* layout */
#shell { gap: 20px; align-items: flex-start; }
#panel { flex: 0 0 380px !important; max-width: 380px; background: var(--blue-wash);
         border: 1px solid var(--line); border-radius: 20px; padding: 18px !important; }
#chat-col { flex: 1 1 0 !important; min-width: 0; background: var(--surface);
            border: 1px solid var(--line); border-radius: 20px; padding: 18px !important; }
.panel-title { font-size: 1.15rem; font-weight: 700; color: var(--blue); margin: 4px 0 2px; }
.panel-hint { font-size: .95rem; opacity: .75; margin-bottom: 6px; }

/* controls: larger text and touch targets */
.big { min-height: 52px !important; font-size: 1.1rem !important; font-weight: 600 !important; }
textarea, input[type=text], .gradio-container label span { font-size: 1.05rem !important; }
input[type=checkbox] { width: 20px; height: 20px; }
.qchip { border-radius: 999px !important; font-size: .98rem !important; min-height: 44px;
         background: var(--blue-soft) !important; color: var(--blue) !important;
         border: 1px solid var(--line) !important; }
#quick-row { flex-wrap: wrap !important; gap: 8px; }
#quick-row > * { flex: 0 1 auto !important; min-width: 0 !important; }
#ask-row { flex-wrap: nowrap !important; gap: 10px; align-items: stretch; }
#ask-row textarea { min-height: 52px; }
#send-btn { min-width: 110px; flex: 0 0 110px !important; }
#options-row { gap: 12px; align-items: end; }

/* file list */
#files-list { max-height: 300px; overflow-y: auto; }
.doc { display: flex; gap: 12px; align-items: center; padding: 12px 14px; margin-bottom: 8px;
       background: var(--surface); border: 1px solid var(--line); border-left: 5px solid var(--blue-mid);
       border-radius: 12px; }
.doc .ico { font-size: 1.7rem; }
.doc b { display: block; font-size: 1.02rem; overflow-wrap: anywhere; color: var(--ink); }
.doc small { font-size: .92rem; opacity: .7; }
.empty { padding: 18px; text-align: center; border: 2px dashed var(--line); border-radius: 12px;
         font-size: 1rem; opacity: .8; }
.pending small { display: block; margin: 4px 0; opacity: .7; font-size: .92rem; }
.pill { display: inline-block; margin: 3px 4px 3px 0; padding: 6px 12px; border-radius: 999px;
        background: var(--blue-soft); color: var(--blue); font-size: .95rem; overflow-wrap: anywhere; }

/* chat */
#chat { height: calc(100dvh - 430px) !important; min-height: 340px; border-radius: 16px; }
#chat .message, #chat .prose { font-size: 1.08rem !important; line-height: 1.65; }
#chat-col table { font-size: .95rem; }

@media (max-width: 900px) {
  html { font-size: 16px; }
  .gradio-container { padding: 0 10px 20px !important; }
  #hero { padding: 16px; border-radius: 0 0 18px 18px; }
  #hero h1 { font-size: 1.5rem; }
  #hero .stats { margin-left: 0; width: 100%; }
  #hero .stat { flex: 1; min-width: 0; padding: 8px; }
  #hero .jump { display: flex; }
  #shell { flex-direction: column !important; }
  #panel, #chat-col { flex: none !important; width: 100% !important; max-width: 100% !important;
                      min-width: 0 !important; padding: 14px !important; }
  #chat { height: 62dvh !important; min-height: 320px; }
  #options-row { flex-direction: column; align-items: stretch; }
}
@media (min-width: 901px) {
  #panel { position: sticky; top: 12px; max-height: calc(100dvh - 24px); overflow-y: auto; }
}
"""


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


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _header_html() -> str:
    """Blue top bar with live counts and (on phones) jump links."""
    files = pipeline.list_indexed()
    stats = [
        ("Documents", len(files)),
        ("Pages", sum(f["pages"] for f in files)),
        ("Chunks", sum(f["chunks"] for f in files)),
    ]
    boxes = "".join(f'<div class="stat"><b>{n}</b><span>{label}</span></div>' for label, n in stats)
    return (
        '<div id="hero"><div class="logo">📄</div>'
        "<div><h1>Docora</h1><p>Ask questions about your PDFs and get answers with page citations.</p></div>"
        f'<div class="stats">{boxes}</div>'
        '<div class="jump"><a href="#panel">📁 Documents</a><a href="#chat-col">💬 Chat</a></div></div>'
    )


def _files_md() -> str:
    """Indexed-file cards (HTML inside the sidebar Markdown)."""
    files = pipeline.list_indexed()
    if not files:
        return '<div class="empty">No documents yet.<br>Upload a PDF above and press Index.</div>'
    return "".join(
        f'<div class="doc"><span class="ico">📄</span><div><b>{_escape(f["filename"])}</b>'
        f'<small>{_plural(f["pages"], "page")}, {_plural(f["chunks"], "chunk")}</small></div></div>'
        for f in files
    )


def _pending_html(files) -> str:
    """Names of files picked in the upload box but not indexed yet."""
    if not files:
        return ""
    names = [Path(f if isinstance(f, str) else f.name).name for f in files]
    pills = "".join(f'<span class="pill">📎 {_escape(n)}</span>' for n in names)
    return f'<div class="pending"><small>Selected, press Index to add them</small>{pills}</div>'


def _assistant_message(result: AnswerResult) -> str:
    """Answer text plus sources and warnings, as one chat message."""
    parts = [result.answer]
    if result.sources_md:
        parts.append(f"**Sources**\n\n{result.sources_md}")
    for w in result.warnings:
        parts.append(f"⚠️ {w}")
    return "\n\n".join(parts)


# -------------------------------------------------------------------- handlers
def on_index(files, ocr_enabled, progress=gr.Progress()):
    """Index button: returns (status Markdown, files list, header)."""
    if not files:
        return "Please choose at least one PDF first.", _files_md(), _header_html()
    paths = [f if isinstance(f, str) else f.name for f in files]
    ocr_on = bool(ocr_enabled) and OCR_AVAILABLE
    result = pipeline.index_files(paths, ocr_on, progress=progress)
    return build_status(result, ocr_on), _files_md(), _header_html()


def on_reset():
    """Reset button: forget everything, clear the upload box and the chat."""
    pipeline.reset_session()
    return (
        "Session reset. All indexed documents were removed.",
        _files_md(),
        _header_html(),
        [{"role": "assistant", "content": WELCOME}],
        [],
        "",
        None,
        "",
    )


def on_clear_chat():
    """Clear chat button: keeps the indexed documents."""
    return [{"role": "assistant", "content": WELCOME}], [], ""


def on_ask(query, format_name, use_reranker, history):
    """Send button / Enter. Streams two updates: the question first, then the
    answer. Output: (chat, input box, scores rows, timings text).

    Each question is answered on its own; ``history`` is only displayed."""
    history = list(history or [])
    text = (query or "").strip()
    if not text:
        yield history, query, gr.skip(), gr.skip()
        return

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": "⏳ Searching your documents…"})
    yield history, "", gr.skip(), gr.skip()

    result: AnswerResult = pipeline.answer_question(text, format_name, bool(use_reranker))
    history[-1] = {"role": "assistant", "content": _assistant_message(result)}
    yield history, "", chunk_rows(result.chunks), timings_line(result.timings)


# ------------------------------------------------------------------------- layout
def build_ui() -> gr.Blocks:
    """Build the Gradio interface. Errors: none."""
    problems = settings.check_config()

    with gr.Blocks(title="Docora") as demo:
        header = gr.HTML(_header_html)

        with gr.Row(elem_id="shell"):
            # ---------------- documents panel
            with gr.Column(elem_id="panel"):
                gr.HTML('<div class="panel-title">Your documents</div>'
                        f'<div class="panel-hint">Up to {settings.MAX_FILES} PDFs per session.</div>')
                files = gr.File(
                    label="Drop PDFs here or click to browse",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                    height=170,
                )
                pending = gr.HTML()
                ocr = gr.Checkbox(
                    label="My files contain scans or images (run OCR)",
                    value=False,
                    visible=OCR_AVAILABLE,
                )
                index_btn = gr.Button("📥 Index documents", variant="primary", elem_classes="big")
                reset_btn = gr.Button("🗑️ Reset session", elem_classes="big")
                status = gr.Markdown()
                gr.HTML('<div class="panel-title">Indexed documents</div>')
                files_list = gr.Markdown(value=_files_md, elem_id="files-list")

            # ---------------- chat
            with gr.Column(elem_id="chat-col"):
                if problems:
                    gr.Markdown("**Setup needed:**\n\n" + "\n".join(f"- {p}" for p in problems))
                chat = gr.Chatbot(
                    value=[{"role": "assistant", "content": WELCOME}],
                    show_label=False,
                    elem_id="chat",
                    autoscroll=True,
                )
                with gr.Row(elem_id="quick-row"):
                    quick_btns = [gr.Button(q, size="sm", elem_classes="qchip") for q in QUICK_QUESTIONS]
                with gr.Row(elem_id="ask-row"):
                    query = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question about your documents…",
                        lines=1,
                        max_lines=4,
                        scale=6,
                        container=False,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", elem_id="send-btn", elem_classes="big")
                with gr.Row(elem_id="options-row"):
                    format_dd = gr.Dropdown(
                        choices=list(FORMAT_OPTIONS), value=DEFAULT_FORMAT, label="Answer format", scale=3
                    )
                    rerank_cb = gr.Checkbox(label="Use reranker", value=True, scale=1)
                    clear_btn = gr.Button("🧹 Clear chat", scale=1, elem_classes="big")
                timings = gr.Markdown()
                with gr.Accordion("🔎 Retrieved chunks and scores", open=False):
                    scores = gr.Dataframe(
                        headers=TABLE_HEADERS,
                        datatype=["number", "str", "number", "str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )

        # ---------------- events
        ask_inputs = [query, format_dd, rerank_cb, chat]
        ask_outputs = [chat, query, scores, timings]

        files.change(_pending_html, files, pending)
        index_btn.click(on_index, [files, ocr], [status, files_list, header])
        reset_btn.click(on_reset, None, [status, files_list, header, chat, scores, timings, files, pending])
        clear_btn.click(on_clear_chat, None, [chat, scores, timings])
        send_btn.click(on_ask, ask_inputs, ask_outputs)
        query.submit(on_ask, ask_inputs, ask_outputs)
        for btn, text in zip(quick_btns, QUICK_QUESTIONS):
            btn.click(lambda t=text: t, None, query).then(on_ask, ask_inputs, ask_outputs)
        demo.load(_files_md, None, files_list)
        demo.load(_header_html, None, header)

    return demo


demo = build_ui()

FORCE_LIGHT_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.toString());
    }
}
"""

if __name__ == "__main__":
    # Gradio 6 takes the theme, CSS and JS here, not in gr.Blocks().
    # FORCE_LIGHT_JS reloads once with ?__theme=light so dark system settings are ignored.
    demo.launch(theme=THEME, css=CSS, js=FORCE_LIGHT_JS)