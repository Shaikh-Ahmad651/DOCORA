"""Chooses and builds providers from config. The only file core/ talks to.

``get_llm()``, ``get_embedder()``, ``get_reranker()`` and ``get_ocr()`` read
the selection from ``config.settings``, build the provider once, and cache it.

Adding a provider = one new file in this folder + one line in the table below
(+ its key name and model name lines in config/settings.py).
"""

from __future__ import annotations

from config import settings
from providers.base import EmbeddingProvider, LLMProvider, OCRProvider, RerankProvider
from providers.cohere_provider import CohereReranker
from providers.gemini_provider import GeminiEmbedder, GeminiLLM
from providers.jina_provider import JinaReranker
from providers.openai_provider import OpenAIEmbedder, OpenAILLM
from providers.tesseract_provider import TesseractOCR
from providers.vision_ocr_provider import VisionLLMOCR

# One line per provider. Importing these files is safe even when an SDK is
# missing: each provider imports its SDK only when it is built.
LLM_PROVIDERS = {"gemini": GeminiLLM, "openai": OpenAILLM}
EMBEDDING_PROVIDERS = {"gemini": GeminiEmbedder, "openai": OpenAIEmbedder}
RERANK_PROVIDERS = {"cohere": CohereReranker, "jina": JinaReranker}
OCR_NAMES = ("vision_llm", "tesseract", "none")

_cache: dict[str, object] = {}


def reset_registry() -> None:
    """Forget the cached providers (used by tests). Errors: none."""
    _cache.clear()


def _pick(role_var: str, name: str, table: dict):
    """Look up a provider class. Errors: ValueError listing the valid names."""
    cls = table.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown {role_var} '{name}'. Valid values: {', '.join(table)}."
        )
    return cls


def _api_key(role_var: str, name: str) -> str:
    """Read the key for a provider. Errors: ValueError naming the env var."""
    key = settings.get_api_key(name)
    if not key:
        raise ValueError(
            f"{role_var}={name} but {settings.API_KEY_ENV[name]} is not set. "
            "Add it to your .env file (or the Space secrets)."
        )
    return key


def get_llm() -> LLMProvider:
    """Output: the LLM provider chosen by LLM_PROVIDER (built once).
    Errors: ValueError for an unknown name or a missing key."""
    if "llm" not in _cache:
        name = settings.LLM_PROVIDER
        cls = _pick("LLM_PROVIDER", name, LLM_PROVIDERS)
        _cache["llm"] = cls(api_key=_api_key("LLM_PROVIDER", name))
    return _cache["llm"]  # type: ignore[return-value]


def get_embedder() -> EmbeddingProvider:
    """Output: the embedding provider chosen by EMBEDDING_PROVIDER (built once).
    Errors: ValueError for an unknown name or a missing key."""
    if "embedder" not in _cache:
        name = settings.EMBEDDING_PROVIDER
        cls = _pick("EMBEDDING_PROVIDER", name, EMBEDDING_PROVIDERS)
        _cache["embedder"] = cls(api_key=_api_key("EMBEDDING_PROVIDER", name))
    return _cache["embedder"]  # type: ignore[return-value]


def get_reranker() -> RerankProvider:
    """Output: the reranker chosen by RERANK_PROVIDER (built once).
    Errors: ValueError for an unknown name or a missing key."""
    if "reranker" not in _cache:
        name = settings.RERANK_PROVIDER
        cls = _pick("RERANK_PROVIDER", name, RERANK_PROVIDERS)
        _cache["reranker"] = cls(api_key=_api_key("RERANK_PROVIDER", name))
    return _cache["reranker"]  # type: ignore[return-value]


def get_ocr() -> OCRProvider | None:
    """Output: the OCR provider chosen by OCR_PROVIDER, or None when it is
    "none" (built once).
    Errors: ValueError for an unknown name or (vision_llm) a missing LLM key;
    ProviderError if local Tesseract is selected but not installed."""
    name = settings.OCR_PROVIDER
    if name == "none":
        return None
    if "ocr" not in _cache:
        if name == "vision_llm":
            _cache["ocr"] = VisionLLMOCR(get_llm())
        elif name == "tesseract":
            _cache["ocr"] = TesseractOCR()
        else:
            raise ValueError(
                f"Unknown OCR_PROVIDER '{name}'. Valid values: {', '.join(OCR_NAMES)}."
            )
    return _cache["ocr"]  # type: ignore[return-value]
