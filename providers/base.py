"""Provider interfaces, the shared error type, and two small error helpers.

Nothing here imports a provider SDK. Concrete providers live in the other
files of this folder and are chosen by ``providers/registry.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(Exception):
    """A failed call to a hosted provider.

    ``retryable`` is True for temporary problems (rate limit, timeout,
    connection error, 5xx). It is False for permanent ones (bad key, bad
    request, blocked content). The retry decorator only retries the former.
    """

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(ABC):
    """Text (and optionally image) generation."""

    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        """Generate text.

        Input: the prompt, an optional system instruction, optional PNG images.
        Output: the model's text (may be empty).
        Errors: ProviderError.
        """


class EmbeddingProvider(ABC):
    """Text embeddings. Providers map the document/query distinction to their
    own mechanism (task type, prefix, or nothing)."""

    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing. One vector per text, in order.
        Errors: ProviderError."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed one search query. Errors: ProviderError."""


class RerankProvider(ABC):
    """Reorders documents by relevance to a query."""

    name: str

    @abstractmethod
    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Rerank documents.

        Output: (index_into_documents, relevance_score), best first, at most
        top_n items. Errors: ProviderError.
        """


class OCRProvider(ABC):
    """Reads text from a page image."""

    name: str

    @abstractmethod
    def ocr_page(self, png_bytes: bytes) -> str:
        """Return the page text, or "" if there is none. Errors: ProviderError."""


# ---------------------------------------------------------------------------
# Helpers shared by the concrete providers
# ---------------------------------------------------------------------------
_TRANSIENT_CLASS_NAMES = {
    "Timeout",
    "TimeoutException",
    "ConnectTimeout",
    "ReadTimeout",
    "TimeoutError",
    "APITimeoutError",
    "ConnectError",
    "ConnectionError",
    "APIConnectionError",
    "NetworkError",
    "ReadError",
    "RemoteProtocolError",
}


def short(value: object, limit: int = 300) -> str:
    """Make any error detail a single short line for messages.

    Input: anything printable. Output: text of at most ``limit`` characters.
    Errors: none.
    """
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"


def is_transient_error(exc: BaseException) -> bool:
    """Guess whether a non-HTTP exception is a timeout or connection problem.

    Input: any exception. Output: True if its class (or a parent class) looks
    like a timeout or network error. Errors: none.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any(cls.__name__ in _TRANSIENT_CLASS_NAMES for cls in type(exc).__mro__)


def error_from_status(service: str, status_code: int, detail: object = "") -> ProviderError:
    """Build a ProviderError from an HTTP status code.

    Input: service name, HTTP status, optional detail text.
    Output: ProviderError, retryable for 408, 429 and 5xx; not otherwise.
    Errors: none.
    """
    retryable = status_code in (408, 429) or status_code >= 500
    message = f"{service} request failed (HTTP {status_code})"
    if detail:
        message += f": {short(detail)}"
    if status_code in (401, 403):
        message += " Check that your API key is correct and has access."
    elif status_code == 429:
        message += " (rate limit or quota reached)"
    return ProviderError(message, retryable=retryable)
