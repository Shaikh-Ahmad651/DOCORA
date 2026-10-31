"""PDF loading: pages of text, with optional OCR for pages that have none.

Uses PyMuPDF. A bad file never raises: the problem is reported in the
LoadReport and the caller moves on to the next file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import pymupdf

from config import settings
from providers.base import ProviderError
from providers.registry import get_ocr
from utils.helpers import truncate_text

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    """What happened while loading one file."""

    filename: str
    total_pages: int = 0
    text_pages: int = 0  # pages taken from the text layer
    ocr_pages: int = 0  # pages taken from OCR
    skipped_pages: list[int] = field(default_factory=list)  # no text (OCR off, or OCR found none)
    failed_pages: list[int] = field(default_factory=list)  # reading or OCR raised an error
    error: str | None = None  # the whole file could not be loaded


def load_pdf(
    path: str,
    filename: str,
    ocr_enabled: bool,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], LoadReport]:
    """Read a PDF page by page.

    Input:
        path: where the PDF is on disk.
        filename: the name to record (used for citations).
        ocr_enabled: run OCR on pages with (almost) no text layer.
        progress_cb: optional; called as progress_cb(page_number, total_pages)
            after each page, with page_number starting at 1.
    Output: (pages, report). Each page is
        {"text", "filename", "page_num" (1-indexed), "ocr_used"}.
        Pages with no text are left out and listed in the report.
    Errors: none raised. A file-level problem gives ([], report) with
        report.error set. A failing page goes in report.failed_pages and the
        next page is still read.
    """
    report = LoadReport(filename=filename)

    ocr = None
    if ocr_enabled:
        try:
            ocr = get_ocr()  # None when OCR_PROVIDER is "none"
        except (ProviderError, ValueError) as exc:
            report.error = f"OCR is not available: {exc}"
            return [], report

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        report.error = f"Could not open the PDF ({truncate_text(str(exc), 150)})."
        return [], report

    pages: list[dict] = []
    try:
        if doc.needs_pass:
            report.error = "The PDF is password-protected."
            return [], report
        total = doc.page_count
        report.total_pages = total
        if total == 0:
            report.error = "The PDF has no pages."
            return [], report

        zoom = settings.OCR_DPI / 72  # PDF points are 1/72 inch
        for index in range(total):
            page_num = index + 1
            try:
                _read_page(doc[index], page_num, filename, ocr, zoom, pages, report)
            except Exception as exc:
                logger.warning("%s page %d failed: %s", filename, page_num, exc)
                report.failed_pages.append(page_num)
            if progress_cb is not None:
                progress_cb(page_num, total)
    except Exception as exc:
        logger.exception("Unexpected error while reading %s", filename)
        report.error = f"The PDF could not be read ({truncate_text(str(exc), 150)})."
        return [], report
    finally:
        doc.close()

    return pages, report


def _read_page(page, page_num, filename, ocr, zoom, pages, report) -> None:
    """Extract one page and record the outcome in ``pages`` and ``report``.

    Errors: any exception here is caught by load_pdf and the page is marked
    as failed.
    """

    def add(text: str, ocr_used: bool) -> None:
        pages.append(
            {"text": text, "filename": filename, "page_num": page_num, "ocr_used": ocr_used}
        )
        if ocr_used:
            report.ocr_pages += 1
        else:
            report.text_pages += 1

    text = page.get_text("text") or ""

    # A page with a real text layer: use it.
    if len(text.strip()) >= settings.OCR_MIN_CHARS:
        add(text, False)
        return

    # Little or no text: try OCR if it is on.
    if ocr is not None:
        png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
        ocr_text = ocr.ocr_page(png).strip()
        if ocr_text == settings.OCR_SENTINEL:
            ocr_text = ""
        if ocr_text:
            add(ocr_text, True)
            return
        # OCR found nothing: fall through to whatever the text layer has.

    if text.strip():
        add(text, False)  # short text layer; better than nothing
    else:
        report.skipped_pages.append(page_num)
