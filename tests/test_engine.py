"""Tests for the TTS engine — keyless and mocked-cloud paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from voicelab.tts.local import LocalTts

# ---------------------------------------------------------------------------
# Keyless path — get_tts() must return LocalTts when no key
# ---------------------------------------------------------------------------


def test_get_tts_keyless_returns_local_tts():
    from voicelab.engine import get_tts

    tts = get_tts()
    assert isinstance(tts, LocalTts)


def test_get_tts_with_key_but_no_sdk_falls_back_to_local(monkeypatch):
    """If elevenlabs SDK is not installed, fall back to LocalTts silently."""
    import voicelab.config.settings as settings_mod

    settings_mod._settings = None
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key-xyz")
    settings_mod._settings = None  # force re-read

    from voicelab.engine import get_tts

    # Patch the import inside ElevenLabsTts.__init__ to raise ImportError
    with patch.dict("sys.modules", {"elevenlabs": None, "elevenlabs.client": None}):
        tts = get_tts()
    assert isinstance(tts, LocalTts)


# ---------------------------------------------------------------------------
# LocalTts mock — synthesize returns bytes or None without raising
# ---------------------------------------------------------------------------


def test_local_tts_synthesize_with_pyttsx3_mocked():
    """Mock pyttsx3 engine so synthesize returns bytes without audio hardware."""
    fake_engine = MagicMock()
    fake_engine.getProperty.return_value = 200  # fake rate

    # synthesize writes a file then reads it; mock open/write cycle
    fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "  # minimal WAV-like bytes

    import builtins

    import voicelab.tts.local as local_mod

    local_mod._PYTTSX3_AVAILABLE = None  # reset sentinel

    fake_pyttsx3 = MagicMock()
    fake_pyttsx3.init.return_value = fake_engine

    with patch.dict("sys.modules", {"pyttsx3": fake_pyttsx3}):
        local_mod._PYTTSX3_AVAILABLE = None  # reset so lazy import runs

        # Override open to return fake bytes when reading the tmp file
        real_open = builtins.open
        opened_path: list[str] = []

        def mock_open(path, mode="r", **kwargs):
            if "rb" in mode and path.endswith(".wav"):
                opened_path.append(path)
                m = MagicMock()
                m.__enter__ = lambda s: MagicMock(read=lambda: fake_wav)
                m.__exit__ = MagicMock(return_value=False)
                return m
            if "wb" in mode and path.endswith(".wav"):
                m = MagicMock()
                m.__enter__ = MagicMock(return_value=MagicMock())
                m.__exit__ = MagicMock(return_value=False)
                return m
            return real_open(path, mode, **kwargs)

        with patch("builtins.open", side_effect=mock_open):
            with patch("os.unlink"):
                tts = LocalTts()
                result = tts.synthesize("hello", "default", {})

    # Result is bytes or None — never raises
    assert result is None or isinstance(result, bytes)


def test_local_tts_synthesize_without_pyttsx3_returns_none():
    """If pyttsx3 is unavailable, synthesize returns None and does not raise."""
    import voicelab.tts.local as local_mod

    local_mod._PYTTSX3_AVAILABLE = None  # reset

    with patch.dict("sys.modules", {"pyttsx3": None}):
        local_mod._PYTTSX3_AVAILABLE = None
        tts = LocalTts()
        result = tts.synthesize("hello", "default", {})

    assert result is None


def test_local_tts_list_voices_without_pyttsx3():
    """list_voices() degrades gracefully when pyttsx3 is unavailable."""
    import voicelab.tts.local as local_mod

    local_mod._PYTTSX3_AVAILABLE = None

    with patch.dict("sys.modules", {"pyttsx3": None}):
        local_mod._PYTTSX3_AVAILABLE = None
        tts = LocalTts()
        voices = tts.list_voices()

    assert isinstance(voices, list)
    assert len(voices) >= 1
    assert "id" in voices[0]
    assert "name" in voices[0]


# ---------------------------------------------------------------------------
# ElevenLabs mocked path
# ---------------------------------------------------------------------------


def test_get_tts_returns_elevenlabs_when_key_and_sdk_present(monkeypatch):
    """When key + SDK present, get_tts() returns ElevenLabsTts."""
    import voicelab.config.settings as settings_mod

    settings_mod._settings = None
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    settings_mod._settings = None

    # Mock the ElevenLabs SDK
    fake_elevenlabs_module = MagicMock()
    fake_client_class = MagicMock()
    fake_elevenlabs_module.ElevenLabs = fake_client_class

    fake_client_instance = MagicMock()
    fake_client_class.return_value = fake_client_instance

    import sys

    fake_sdk = MagicMock()
    fake_sdk.client = fake_elevenlabs_module
    fake_sdk.client.ElevenLabs = fake_client_class

    with patch.dict(
        sys.modules,
        {
            "elevenlabs": fake_sdk,
            "elevenlabs.client": fake_elevenlabs_module,
        },
    ):
        from voicelab.engine import get_tts
        from voicelab.tts.elevenlabs import ElevenLabsTts

        tts = get_tts()
        assert isinstance(tts, ElevenLabsTts)


# ---------------------------------------------------------------------------
# compare() returns 2 entries
# ---------------------------------------------------------------------------


def test_compare_returns_two_entries():
    """compare() always returns one entry per voice, even on None audio."""
    import voicelab.tts.local as local_mod

    local_mod._PYTTSX3_AVAILABLE = False  # force None returns
    tts = LocalTts()

    from voicelab.engine import compare

    results = compare("hello world", ["voice_a", "voice_b"], {}, tts=tts)
    assert len(results) == 2
    labels = [r[0] for r in results]
    assert "voice_a" in labels
    assert "voice_b" in labels
    # Each result is (str, bytes|None)
    for label, audio in results:
        assert isinstance(label, str)
        assert audio is None or isinstance(audio, bytes)
