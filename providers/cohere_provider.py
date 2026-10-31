"""Cohere reranker, using the ``cohere`` SDK (v2 client), imported lazily."""

from __future__ import annotations

from config import settings
from providers.base import (
    ProviderError,
    RerankProvider,
    error_from_status,
    is_transient_error,
    short,
)
from utils.retry import retry


def _wrap_error(exc: Exception) -> ProviderError:
    """Turn any SDK exception into a ProviderError."""
    if isinstance(exc, ProviderError):
        return exc
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return error_from_status("Cohere", status, getattr(exc, "body", None) or str(exc))
    return ProviderError(
        f"Cohere request failed: {short(exc)}", retryable=is_transient_error(exc)
    )


class CohereReranker(RerankProvider):
    """Hosted Cohere reranker."""

    name = "cohere"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Input: API key, optional model override. Errors: ProviderError if
        the SDK is missing."""
        try:
            import cohere
        except ImportError as exc:
            raise ProviderError(
                "The Cohere SDK is not installed. Run: pip install cohere", retryable=False
            ) from exc
        self._model = model or settings.COHERE_RERANK_MODEL
        # max_retries=0: utils.retry is the only retry layer.
        self._client = cohere.ClientV2(
            api_key=api_key, timeout=settings.API_TIMEOUT_SECONDS, max_retries=0
        )

    @retry()
    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Rerank documents. Output: (index, score), best first, at most top_n.
        Errors: ProviderError."""
        if not documents:
            return []
        count = max(1, min(top_n, len(documents)))
        try:
            response = self._client.rerank(
                model=self._model, query=query, documents=documents, top_n=count
            )
        except Exception as exc:
            raise _wrap_error(exc) from exc
        ranked = [(int(r.index), float(r.relevance_score)) for r in response.results]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
