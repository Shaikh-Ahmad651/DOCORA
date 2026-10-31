import pytest

from providers.base import ProviderError
from utils import retry as retry_module


@pytest.fixture
def sleeps(monkeypatch):
    """Record the delays instead of sleeping."""
    recorded = []
    monkeypatch.setattr(retry_module, "_sleep", recorded.append)
    return recorded


def test_retries_until_success_with_exponential_delays(sleeps):
    calls = {"n": 0}

    @retry_module.retry(max_attempts=3, base_delay=1.0, factor=2.0, jitter=0.0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("rate limit", retryable=True)
        return "done"

    assert flaky() == "done"
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_attempts_and_reraises_original(sleeps):
    calls = {"n": 0}
    original = ProviderError("down", retryable=True)

    @retry_module.retry(max_attempts=3, jitter=0.0)
    def always_bad():
        calls["n"] += 1
        raise original

    with pytest.raises(ProviderError) as info:
        always_bad()
    assert info.value is original
    assert calls["n"] == 3


def test_non_retryable_error_is_not_retried(sleeps):
    calls = {"n": 0}

    @retry_module.retry(max_attempts=3)
    def bad_key():
        calls["n"] += 1
        raise ProviderError("bad key", retryable=False)

    with pytest.raises(ProviderError):
        bad_key()
    assert calls["n"] == 1 and sleeps == []


def test_other_exceptions_propagate_immediately(sleeps):
    calls = {"n": 0}

    @retry_module.retry(max_attempts=3)
    def other():
        calls["n"] += 1
        raise KeyError("x")

    with pytest.raises(KeyError):
        other()
    assert calls["n"] == 1


def test_jitter_adds_a_bounded_random_delay(sleeps):
    @retry_module.retry(max_attempts=2, base_delay=1.0, jitter=0.25)
    def once_bad(state={"n": 0}):
        state["n"] += 1
        if state["n"] == 1:
            raise ProviderError("x", retryable=True)

    once_bad()
    assert 1.0 <= sleeps[0] <= 1.25
