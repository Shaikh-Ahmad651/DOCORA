"""Retry with exponential backoff for hosted API calls.

Only ``ProviderError`` with ``retryable=True`` is retried. Any other
exception (including a non-retryable ProviderError) propagates immediately.

Note: this module imports ``ProviderError`` from ``providers.base`` (the
exception class only; ``providers/__init__.py`` must stay free of imports).
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, TypeVar

from config.settings import RETRY_BASE_DELAY, RETRY_FACTOR, RETRY_MAX_ATTEMPTS
from providers.base import ProviderError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Looked up at call time so tests can replace it and avoid real sleeping.
_sleep = time.sleep


def retry(
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY,
    factor: float = RETRY_FACTOR,
    jitter: float = 0.25,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory: retry a function on retryable ProviderErrors.

    Input:
        max_attempts: total tries including the first (3 means 2 retries).
        base_delay: seconds before the first retry.
        factor: delay multiplier per retry.
        jitter: random extra delay in seconds, between 0 and this value.
    Output: a decorator. Delay before retry n (0-based) is
        ``base_delay * factor ** n`` plus jitter.
    Errors: after the last attempt the original ProviderError is re-raised.
        Non-retryable ProviderErrors and other exceptions are not caught.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except ProviderError as exc:
                    is_last = attempt >= max_attempts - 1
                    if not exc.retryable or is_last:
                        raise
                    delay = base_delay * factor**attempt + random.uniform(0, jitter)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                        func.__name__,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    _sleep(delay)
            # Only reachable if max_attempts < 1.
            return func(*args, **kwargs)

        return wrapper

    return decorator
