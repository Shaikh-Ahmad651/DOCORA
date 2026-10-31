"""Optional local OCR with Tesseract. Needs the Python package AND the
Tesseract program installed on the computer (see the message below)."""

from __future__ import annotations

import io

from config import settings
from providers.base import OCRProvider, ProviderError, short

_INSTALL_HELP = (
    "Local Tesseract OCR is not ready. You need both: (1) the Python package "
    "(pip install -r requirements-local-ocr.txt) and (2) the Tesseract program "
    "itself: on macOS run 'brew install tesseract', on Ubuntu/Debian run "
    "'sudo apt install tesseract-ocr'. If it is installed but not found, set "
    "TESSERACT_CMD_PATH in your .env file. Or use OCR_PROVIDER=vision_llm."
)


class TesseractOCR(OCRProvider):
    """Runs Tesseract on the page image. Local, so no retries and no API key."""

    name = "tesseract"

    def __init__(self) -> None:
        """Errors: ProviderError (not retryable) if pytesseract or the
        Tesseract program is missing."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise ProviderError(_INSTALL_HELP, retryable=False) from exc
        if settings.TESSERACT_CMD_PATH:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD_PATH
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # TesseractNotFoundError and friends
            raise ProviderError(_INSTALL_HELP, retryable=False) from exc
        self._pytesseract = pytesseract
        self._image_module = Image

    def ocr_page(self, png_bytes: bytes) -> str:
        """Input: PNG bytes of one page. Output: the page text, or "".
        Errors: ProviderError (not retryable)."""
        try:
            with self._image_module.open(io.BytesIO(png_bytes)) as image:
                text = self._pytesseract.image_to_string(image)
        except Exception as exc:
            raise ProviderError(f"Tesseract failed: {short(exc)}", retryable=False) from exc
        return text.strip()
