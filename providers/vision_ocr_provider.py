"""OCR using the selected LLM's vision ability (the default OCR provider)."""

from __future__ import annotations

from config import settings
from config.prompts import OCR_PROMPT
from providers.base import LLMProvider, OCRProvider


class VisionLLMOCR(OCRProvider):
    """Sends the page image to the LLM provider with OCR_PROMPT.

    The LLM is passed in (by the registry), so this file never names a
    provider and there is no import cycle. Retries happen inside the LLM
    provider, so there is no extra retry layer here.
    """

    name = "vision_llm"

    def __init__(self, llm: LLMProvider) -> None:
        """Input: an LLM provider that accepts images."""
        self._llm = llm

    def ocr_page(self, png_bytes: bytes) -> str:
        """Input: PNG bytes of one page. Output: the page text, or "" if the
        model reports no text. Errors: ProviderError from the LLM."""
        text = self._llm.generate(prompt=OCR_PROMPT, images=[png_bytes]).strip()
        return "" if text == settings.OCR_SENTINEL else text
