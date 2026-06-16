"""Test configuration — isolate settings from real env/filesystem."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    """Reset the settings singleton before each test and clear key env vars."""
    import voicelab.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_settings", None)
    # Remove real API keys from environment so tests are keyless by default
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    yield
    # Reset again after test
    settings_mod._settings = None


@pytest.fixture(autouse=True)
def isolate_personas():
    """Reset persona cache between tests."""
    import voicelab.personas as personas_mod

    personas_mod._personas = None
    yield
    personas_mod._personas = None
