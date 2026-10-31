"""Make the project root importable and keep tests independent of a real .env."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """No API keys or provider choices leak in from the developer's machine."""
    for name in list(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)
