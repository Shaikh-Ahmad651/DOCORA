"""Google Gemini providers: LLM (with image input) and embeddings.

Uses the ``google-genai`` SDK, imported lazily so a missing SDK only matters
when Gemini is actually selected. Call shapes were checked against the
installed SDK (google-genai 2.x); see the VERIFY notes in the final report.
"""

from __future__ import annotations

import logging

from config import settings
from providers.base import (
    EmbeddingProvider,
    LLMProvider,
    ProviderError,
    error_from_status,
    is_transient_error,
    short,
)
from utils.retry import retry

logger = logging.getLogger(__name__)

_INSTALL_HINT = "The Google SDK is not installed. Run: pip install google-genai"


def _import_sdk():
    """Import the SDK lazily. Errors: ProviderError if it is not installed."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ProviderError(_INSTALL_HINT, retryable=False) from exc
    return genai, types


def _make_client(api_key: str):
    genai, types = _import_sdk()
    # The SDK's timeout is in milliseconds.
    options = types.HttpOptions(timeout=int(settings.API_TIMEOUT_SECONDS * 1000))
    return genai.Client(api_key=api_key, http_options=options), types


def _wrap_error(exc: Exception) -> ProviderError:
    """Turn any SDK exception into a ProviderError."""
    if isinstance(exc, ProviderError):
        return exc
    status = getattr(exc, "code", None)
    if isinstance(status, int):
        detail = getattr(exc, "message", None) or str(exc)
        return error_from_status("Gemini", status, detail)
    return ProviderError(
        f"Gemini request failed: {short(exc)}", retryable=is_transient_error(exc)
    )


class GeminiLLM(LLMProvider):
    """Gemini text generation. PNG images are sent as inline image parts."""

    name = "gemini"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Input: API key, optional model override. Errors: ProviderError if
        the SDK is missing."""
        self._model = model or settings.GEMINI_LLM_MODEL
        self._client, self._types = _make_client(api_key)

    @retry()
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        """Generate text. Errors: ProviderError (retryable for 429, 5xx,
        timeouts and connection problems; not for a blocked response)."""
        types = self._types
        contents: list = [
            types.Part.from_bytes(data=png, mime_type="image/png")
            for png in (images or [])
        ]
        contents.append(prompt)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except Exception as exc:
            raise _wrap_error(exc) from exc
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Read the text from a response, mapping blocks to ProviderError."""
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            raise ProviderError(
                f"Gemini blocked the request ({getattr(block_reason, 'name', block_reason)}).",
                retryable=False,
            )
        candidates = response.candidates or []
        if not candidates:
            raise ProviderError(
                "Gemini returned no answer (the response may have been blocked).",
                retryable=False,
            )
        reason = candidates[0].finish_reason
        reason_name = getattr(reason, "name", str(reason))
        text = response.text or ""
        if text.strip():
            if reason_name == "MAX_TOKENS":
                logger.warning(
                    "Gemini output was cut off at MAX_OUTPUT_TOKENS=%d",
                    settings.MAX_OUTPUT_TOKENS,
                )
            return text
        if reason_name == "MAX_TOKENS":
            raise ProviderError(
                "Gemini used up its output budget before writing any text. "
                "Increase MAX_OUTPUT_TOKENS in config/settings.py.",
                retryable=False,
            )
        if reason_name not in ("STOP", "FINISH_REASON_UNSPECIFIED", "None"):
            raise ProviderError(
                f"Gemini returned no text (finish reason: {reason_name}).",
                retryable=False,
            )
        return ""


class GeminiEmbedder(EmbeddingProvider):
    """Gemini embeddings. Documents and queries use different task types.
    The vector size is whatever the model returns (nothing is hard-coded)."""

    name = "gemini"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Input: API key, optional model override. Errors: ProviderError if
        the SDK is missing."""
        self._model = model or settings.GEMINI_EMBEDDING_MODEL
        self._client, self._types = _make_client(api_key)

    @retry()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing (one vector per text, in order).
        The caller batches (EMBED_BATCH_SIZE). Errors: ProviderError."""
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    @retry()
    def embed_query(self, text: str) -> list[float]:
        """Embed one search query. Errors: ProviderError."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors = self._embed_call(texts, task_type)
        if len(vectors) != len(texts) and len(texts) > 1:
            # Some newer models merge a list into ONE vector. Embed one by one.
            vectors = []
            for text in texts:
                vectors.extend(self._embed_call([text], task_type))
        if len(vectors) != len(texts):
            raise ProviderError(
                f"Gemini returned {len(vectors)} embeddings for {len(texts)} texts. "
                "Check GEMINI_EMBEDDING_MODEL (it must return one vector per text).",
                retryable=False,
            )
        return vectors

    def _embed_call(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = self._types.EmbedContentConfig(task_type=task_type)
        try:
            response = self._client.models.embed_content(
                model=self._model, contents=texts, config=config
            )
        except Exception as exc:
            raise _wrap_error(exc) from exc
        embeddings = response.embeddings or []
        if any(not e.values for e in embeddings):
            raise ProviderError("Gemini returned an empty embedding.", retryable=False)
        return [list(e.values) for e in embeddings]