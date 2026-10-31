"""Docora settings: every constant and environment variable in one place.

Values come from the environment (optionally loaded from a ``.env`` file).
Nothing here raises at import time: missing API keys are reported by
``check_config()`` (shown as a banner in the UI) and by the provider
registry when a provider is actually built.

Items marked "untested" are starting values, not tuned results.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    """Read an env var, treating unset or blank values as the default.

    Input: variable name and default.
    Output: the stripped value, or ``default``.
    Errors: none.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


# ---------------------------------------------------------------------------
# 1. Provider selection (env vars)
# ---------------------------------------------------------------------------
VALID_LLM_PROVIDERS = ("gemini", "openai")
VALID_EMBEDDING_PROVIDERS = ("gemini", "openai")
VALID_RERANK_PROVIDERS = ("cohere", "jina")
VALID_OCR_PROVIDERS = ("vision_llm", "tesseract", "none")

LLM_PROVIDER = _env("LLM_PROVIDER", "gemini").lower()
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "gemini").lower()
RERANK_PROVIDER = _env("RERANK_PROVIDER", "cohere").lower()
OCR_PROVIDER = _env("OCR_PROVIDER", "vision_llm").lower()

# ---------------------------------------------------------------------------
# 2. API keys: which env var holds the key for each provider
# ---------------------------------------------------------------------------
API_KEY_ENV = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "jina": "JINA_API_KEY",
}


def get_api_key(provider: str) -> str:
    """Return the API key for a provider, read from the environment now.

    Input: provider name, e.g. "gemini".
    Output: the key, or "" if it is not set.
    Errors: ValueError if the provider has no key mapping.
    """
    env_name = API_KEY_ENV.get(provider)
    if env_name is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Known: {', '.join(API_KEY_ENV)}"
        )
    return os.getenv(env_name, "").strip()


# ---------------------------------------------------------------------------
# 3. Model names (env-overridable: change .env, not code)
#    VERIFY all of these against current provider docs; they change often.
# ---------------------------------------------------------------------------
GEMINI_LLM_MODEL = _env("GEMINI_LLM_MODEL", "gemini-3.5-flash")
GEMINI_EMBEDDING_MODEL = _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
OPENAI_LLM_MODEL = _env("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
COHERE_RERANK_MODEL = _env("COHERE_RERANK_MODEL", "rerank-v3.5")
JINA_RERANK_MODEL = _env("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")

# Optional: path to the Tesseract binary (local Tesseract OCR only).
TESSERACT_CMD_PATH = _env("TESSERACT_CMD_PATH", "")

# ---------------------------------------------------------------------------
# 4. Constants
# ---------------------------------------------------------------------------
# Chunking (characters)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]  # priority order

# Retrieval (TOP_K_* are untested starting values)
TOP_K_VECTOR = 20
TOP_K_BM25 = 20
TOP_K_FINAL = 5  # chunks sent to the LLM, with or without the reranker
RRF_K = 60

# Limits
MAX_FILES = 20  # session total

# Generation
LLM_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2048

# OCR (untested)
OCR_DPI = 200  # raise for small print, lower for cost/speed
OCR_MIN_CHARS = 25  # text-layer length below which a page needs OCR
OCR_SENTINEL = "[NO TEXT]"

# Embedding batching. VERIFY the per-provider limit; lower it if rejected.
EMBED_BATCH_SIZE = 100

# Hosted API calls
API_TIMEOUT_SECONDS = 60  # vision OCR may need more
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0  # seconds
RETRY_FACTOR = 2.0

# UI
PREVIEW_CHARS = 200  # scores-panel preview length

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ---------------------------------------------------------------------------
# Config check (used for the UI banner; never raises)
# ---------------------------------------------------------------------------
def check_config() -> list[str]:
    """List human-readable configuration problems.

    Input: none (reads the module-level provider settings and the live
        environment for keys).
    Output: list of problem strings; empty if everything looks fine.
    Errors: none. The app still launches when problems exist.
    """
    problems: list[str] = []
    roles = [
        ("LLM_PROVIDER", LLM_PROVIDER, VALID_LLM_PROVIDERS),
        ("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER, VALID_EMBEDDING_PROVIDERS),
        ("RERANK_PROVIDER", RERANK_PROVIDER, VALID_RERANK_PROVIDERS),
        ("OCR_PROVIDER", OCR_PROVIDER, VALID_OCR_PROVIDERS),
    ]
    for var, value, valid in roles:
        if value not in valid:
            problems.append(
                f"{var}={value} is not a valid value. Allowed: {', '.join(valid)}."
            )
            continue
        env_name = API_KEY_ENV.get(value)
        # vision_llm, tesseract and none need no key of their own
        # (vision_llm reuses the LLM provider, which is checked above).
        if env_name and not get_api_key(value):
            message = f"{var}={value} but {env_name} is not set."
            if var == "RERANK_PROVIDER":
                message += " The reranker will fall back to RRF order."
            problems.append(message)
    return problems
