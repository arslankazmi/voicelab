"""Unit tests for ElevenLabs voice-id resolution (no SDK required)."""

from __future__ import annotations

from voicelab.tts.elevenlabs import _resolve_voice_id

_VOICES = [
    {"id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Roger - Laid-Back, Casual, Resonant"},
    {"id": "ZZZ111", "name": "Rachel"},
]


def test_default_resolves_to_first_voice():
    assert _resolve_voice_id("default", _VOICES) == "CwhRBWXzGAHq8TQ4Fs17"


def test_empty_voice_resolves_to_first():
    assert _resolve_voice_id("", _VOICES) == "CwhRBWXzGAHq8TQ4Fs17"


def test_exact_id_passthrough():
    assert _resolve_voice_id("ZZZ111", _VOICES) == "ZZZ111"


def test_exact_name_match():
    assert _resolve_voice_id("Rachel", _VOICES) == "ZZZ111"


def test_name_prefix_match():
    assert _resolve_voice_id("Roger", _VOICES) == "CwhRBWXzGAHq8TQ4Fs17"


def test_no_voices_returns_none_for_default():
    assert _resolve_voice_id("default", []) is None


def test_no_voices_passes_through_explicit_id():
    assert _resolve_voice_id("SOMEID", []) == "SOMEID"
