"""Test configuration — isolate settings from real env/filesystem."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_model_manager(monkeypatch):
    """Reset the heavy-model manager + backend hooks between tests, and make the
    memory preflight pass by default so heavy-load tests aren't flaky on CI."""
    import voicelab.tts._model_manager as mm
    import voicelab.tts.chatterbox as cb
    import voicelab.tts.chatterbox_turbo as cbt

    mm.evict()
    monkeypatch.setattr(mm, "_free_mb", lambda: 999_999.0)  # plenty of RAM by default
    monkeypatch.setattr(cb, "_model", None, raising=False)
    monkeypatch.setattr(cbt, "_model", None, raising=False)
    yield
    mm.evict()


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    """Reset the settings singleton before each test and clear key env vars."""
    import voicelab.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_settings", None)
    # ELEVENLABS_API_KEY lives in the .env FILE, so set "" (an env var overrides
    # the .env file in pydantic-settings) — a developer's local .env can't leak a
    # real key into the suite. AUTH_TOKEN is not in .env, so delenv is correct
    # (setting it "" is fine for keylessness but delenv keeps behavior identical).
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    # Protect telemetry-off env vars from being overwritten by gradio imports
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
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
