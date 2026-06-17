"""TTS engine — picks backend based on settings, provides high-level helpers."""

from __future__ import annotations

import logging

from voicelab.config.settings import Settings, get_settings
from voicelab.contracts import Tts

logger = logging.getLogger(__name__)


def get_tts(settings: Settings | None = None) -> Tts:
    """Return the best available TTS backend.

    If ``ELEVENLABS_API_KEY`` is present in settings, returns
    :class:`~voicelab.tts.elevenlabs.ElevenLabsTts`.
    Otherwise falls back to :class:`~voicelab.tts.local.LocalTts`.
    """
    if settings is None:
        settings = get_settings()

    if settings.elevenlabs_api_key:
        try:
            from voicelab.tts.elevenlabs import ElevenLabsTts

            return ElevenLabsTts(api_key=settings.elevenlabs_api_key)
        except ImportError:
            logger.warning(
                "elevenlabs package not installed — falling back to LocalTts. "
                "Install with: uv pip install 'voicelab[cloud]'"
            )

    from voicelab.tts.local import LocalTts

    return LocalTts()


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

    LocalTts produces WAV; ElevenLabsTts produces MP3.
    """
    from voicelab.tts.local import LocalTts  # noqa: PLC0415

    if tts is None:
        tts = get_tts()
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
