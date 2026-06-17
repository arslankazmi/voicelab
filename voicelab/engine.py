"""TTS engine — picks backend based on settings, provides high-level helpers."""

from __future__ import annotations

import logging

from voicelab.config.settings import Settings, get_settings
from voicelab.contracts import Tts

logger = logging.getLogger(__name__)


def get_tts(settings: Settings | None = None) -> Tts:
    """Return the best available TTS backend.

    Respects ``settings.tts_engine``:
    - "auto"       → ElevenLabs (if key) → Kokoro → Piper → Local
    - "elevenlabs" → ElevenLabs (fallback to auto chain if unavailable)
    - "kokoro"     → KokoroTts (fallback to auto chain if unavailable)
    - "piper"      → PiperTts  (fallback to auto chain if unavailable)
    - "local"      → LocalTts  (always available)
    """
    if settings is None:
        settings = get_settings()

    from voicelab.tts.registry import get_tts_for_engine  # noqa: PLC0415

    engine = getattr(settings, "tts_engine", "auto") or "auto"
    api_key = settings.elevenlabs_api_key
    return get_tts_for_engine(engine, api_key=api_key)


def synthesize_one(
    text: str,
    voice: str,
    settings: dict,
    tts: Tts | None = None,
) -> bytes | None:
    """Synthesize a single piece of text.

    Args:
        text: Text to synthesize.
        voice: Voice id/name for the backend.
        settings: Dict of voice settings (stability, similarity_boost, style, speed).
        tts: Optional pre-constructed backend; auto-constructed if None.

    Returns:
        Audio bytes or None.
    """
    if tts is None:
        tts = get_tts()
    return tts.synthesize(text, voice, settings)


def audio_media_type(tts: Tts | None = None) -> str:
    """Return the MIME type produced by *tts* (or the default backend).

    LocalTts / KokoroTts / PiperTts produce WAV; ElevenLabsTts produces MP3.
    """
    from voicelab.tts.local import LocalTts  # noqa: PLC0415

    if tts is None:
        tts = get_tts()

    # Import locally to avoid circular deps
    try:
        from voicelab.tts.kokoro import KokoroTts  # noqa: PLC0415

        if isinstance(tts, KokoroTts):
            return "audio/wav"
    except ImportError:
        pass

    try:
        from voicelab.tts.piper import PiperTts  # noqa: PLC0415

        if isinstance(tts, PiperTts):
            return "audio/wav"
    except ImportError:
        pass

    if isinstance(tts, LocalTts):
        return "audio/wav"
    return "audio/mpeg"


def compare(
    text: str,
    voices: list[str],
    settings: dict,
    tts: Tts | None = None,
) -> list[tuple[str, bytes | None]]:
    """Synthesize *text* with multiple voices and return labelled results.

    Args:
        text: Text to synthesize.
        voices: List of voice ids/names.
        settings: Shared voice settings dict.
        tts: Optional pre-constructed backend.

    Returns:
        List of (label, audio_bytes|None) tuples, one per voice.
    """
    if tts is None:
        tts = get_tts()
    return [(voice, tts.synthesize(text, voice, settings)) for voice in voices]


def compare_across_engines(
    text: str,
    engine_voice_pairs: list[tuple[str, str]],
    settings: dict,
    app_settings: Settings | None = None,
) -> list[tuple[str, str, bytes | None]]:
    """Synthesize *text* across multiple (engine, voice) pairs.

    Enables cross-engine comparison: same text via Kokoro vs Piper vs ElevenLabs.

    Args:
        text: Text to synthesize.
        engine_voice_pairs: List of (engine_name, voice_id) tuples.
        settings: Shared settings dict.
        app_settings: Optional Settings instance for API keys.

    Returns:
        List of (engine, voice, audio_bytes|None) tuples.
    """
    if app_settings is None:
        app_settings = get_settings()

    from voicelab.tts.registry import get_tts_for_engine  # noqa: PLC0415

    results = []
    for engine_name, voice_id in engine_voice_pairs:
        tts = get_tts_for_engine(engine_name, api_key=app_settings.elevenlabs_api_key)
        audio = tts.synthesize(text, voice_id, settings)
        results.append((engine_name, voice_id, audio))
    return results
