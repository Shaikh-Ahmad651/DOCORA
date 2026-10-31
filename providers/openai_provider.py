"""OpenAI providers: LLM (with image input) and embeddings.

Uses the ``openai`` SDK, imported lazily. The SDK's own retries are turned
off (max_retries=0) so ``utils.retry`` is the only retry layer.

Note: some newer OpenAI models only accept their default temperature. If the
API rejects ``temperature``, pick another model in OPENAI_LLM_MODEL.
"""

from __future__ import annotations

import base64
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


def _make_client(api_key: str):
    """Build the SDK client. Errors: ProviderError if the SDK is missing."""
    try:
        import openai
    except ImportError as exc:
        raise ProviderError(
            "The OpenAI SDK is not installed. Run: pip install openai", retryable=False
        ) from exc
    return openai.OpenAI(
        api_key=api_key, timeout=settings.API_TIMEOUT_SECONDS, max_retries=0
    )


def _wrap_error(exc: Exception) -> ProviderError:
    """Turn any SDK exception into a ProviderError."""
    if isinstance(exc, ProviderError):
        return exc
    status = getattr(exc, "status_code", None)
    detail = getattr(exc, "message", None) or str(exc)
    if status == 429 and getattr(exc, "code", None) == "insufficient_quota":
        # Retrying cannot fix an empty account.
        return ProviderError(
            "OpenAI says the account has no quota left. Check your billing.",
            retryable=False,
        )
    if isinstance(status, int):
        return error_from_status("OpenAI", status, detail)
    return ProviderError(
        f"OpenAI request failed: {short(exc)}", retryable=is_transient_error(exc)
    )


class OpenAILLM(LLMProvider):
    """OpenAI chat model. PNG images are sent as base64 data URLs."""

    name = "openai"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Input: API key, optional model override. Errors: ProviderError if
        the SDK is missing."""
        self._model = model or settings.OPENAI_LLM_MODEL
        self._client = _make_client(api_key)

    @retry()
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        """Generate text. Errors: ProviderError (retryable for 429, 5xx,
        timeouts and connection problems; not for a filtered response)."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            content: list[dict] | str = [{"type": "text", "text": prompt}]
            for png in images:
                encoded = base64.b64encode(png).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    }
                )
        else:
            content = prompt
        messages.append({"role": "user", "content": content})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_completion_tokens=settings.MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            raise _wrap_error(exc) from exc
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Read the text from a response, mapping blocks to ProviderError."""
        if not response.choices:
            raise ProviderError("OpenAI returned no answer.", retryable=False)
        choice = response.choices[0]
        finish = choice.finish_reason
        if finish == "content_filter":
            raise ProviderError("OpenAI blocked the response (content filter).", retryable=False)
        text = choice.message.content or ""
        if text.strip():
            if finish == "length":
                logger.warning(
                    "OpenAI output was cut off at MAX_OUTPUT_TOKENS=%d",
                    settings.MAX_OUTPUT_TOKENS,
                )
            return text
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise ProviderError(f"OpenAI refused: {short(refusal)}", retryable=False)
        if finish == "length":
            raise ProviderError(
                "OpenAI used up its output budget before writing any text. "
                "Increase MAX_OUTPUT_TOKENS in config/settings.py.",
                retryable=False,
            )
        return ""


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embeddings (no document/query distinction)."""

    name = "openai"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Input: API key, optional model override. Errors: ProviderError if
        the SDK is missing."""
        self._model = model or settings.OPENAI_EMBEDDING_MODEL
        self._client = _make_client(api_key)

    @retry()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing (one vector per text, in order).
        The caller batches (EMBED_BATCH_SIZE). Errors: ProviderError."""
        if not texts:
            return []
        return self._embed(texts)

    @retry()
    def embed_query(self, text: str) -> list[float]:
        """Embed one search query. Errors: ProviderError."""
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:
            raise _wrap_error(exc) from exc
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts):
            raise ProviderError(
                f"OpenAI returned {len(items)} embeddings for {len(texts)} texts.",
                retryable=False,
            )
        return [list(item.embedding) for item in items]
