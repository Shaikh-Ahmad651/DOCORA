"""Registry tests: no SDK is ever imported or called; fake providers are used."""

import pytest

from config import settings
from providers import registry
from providers.base import LLMProvider


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, api_key, model=None):
        self.api_key = api_key

    def generate(self, prompt, system=None, images=None):
        return "ok"


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    registry.reset_registry()
    monkeypatch.setitem(registry.LLM_PROVIDERS, "fake", FakeLLM)
    monkeypatch.setitem(settings.API_KEY_ENV, "fake", "FAKE_API_KEY")
    yield
    registry.reset_registry()


def test_unknown_provider_lists_valid_names(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "nope")
    with pytest.raises(ValueError, match="gemini"):
        registry.get_llm()


def test_missing_key_names_the_env_var(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "fake")
    with pytest.raises(ValueError, match="FAKE_API_KEY"):
        registry.get_llm()


def test_builds_and_caches_the_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "fake")
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    llm = registry.get_llm()
    assert isinstance(llm, FakeLLM) and llm.api_key == "secret"
    assert registry.get_llm() is llm


def test_keys_of_unselected_providers_are_not_needed(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "fake")
    monkeypatch.setenv("FAKE_API_KEY", "x")
    registry.get_llm()  # no GOOGLE / OPENAI / COHERE / JINA keys set: no error


def test_unknown_reranker_and_embedder(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "nope")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "nope")
    with pytest.raises(ValueError, match="cohere"):
        registry.get_reranker()
    with pytest.raises(ValueError, match="gemini"):
        registry.get_embedder()


def test_ocr_none_returns_none_and_unknown_raises(monkeypatch):
    monkeypatch.setattr(settings, "OCR_PROVIDER", "none")
    assert registry.get_ocr() is None
    monkeypatch.setattr(settings, "OCR_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="vision_llm"):
        registry.get_ocr()


def test_vision_ocr_wraps_the_llm(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "fake")
    monkeypatch.setattr(settings, "OCR_PROVIDER", "vision_llm")
    monkeypatch.setenv("FAKE_API_KEY", "x")
    ocr = registry.get_ocr()
    assert ocr.name == "vision_llm" and ocr._llm is registry.get_llm()
