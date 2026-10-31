"""Jina reranker, built on plain REST calls with ``requests`` (imported lazily).

VERIFY: the endpoint, header and payload below come from public docs and
examples (POST /v1/rerank with model, query, top_n, documents). The response
shape (results[].index and results[].relevance_score) is assumed from the
same sources. Confirm both against Jina's current docs.
"""

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

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class JinaReranker(RerankProvider):
    """Hosted Jina reranker."""

    name = "jina"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Input: API key, optional model override. Errors: ProviderError if
        ``requests`` is missing."""
        try:
            import requests
        except ImportError as exc:
            raise ProviderError(
                "The 'requests' package is not installed. Run: pip install requests",
                retryable=False,
            ) from exc
        self._requests = requests
        self._api_key = api_key
        self._model = model or settings.JINA_RERANK_MODEL

    @retry()
    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Rerank documents. Output: (index, score), best first, at most top_n.
        Errors: ProviderError."""
        if not documents:
            return []
        count = max(1, min(top_n, len(documents)))
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": self._model,
            "query": query,
            "top_n": count,
            "documents": documents,
        }
        try:
            response = self._requests.post(
                JINA_RERANK_URL,
                headers=headers,
                json=payload,
                timeout=settings.API_TIMEOUT_SECONDS,
            )
        except self._requests.exceptions.RequestException as exc:
            raise ProviderError(
                f"Jina request failed: {short(exc)}", retryable=is_transient_error(exc)
            ) from exc

        if response.status_code >= 400:
            raise error_from_status("Jina", response.status_code, response.text)
        try:
            results = response.json()["results"]
            ranked = [(int(r["index"]), float(r["relevance_score"])) for r in results]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                "Jina returned an unexpected response format.", retryable=False
            ) from exc
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
