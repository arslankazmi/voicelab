"""ElevenLabs TTS backend — gated on ELEVENLABS_API_KEY.

Only constructed when a key is present. The elevenlabs SDK is a soft dep
(in [project.optional-dependencies] cloud); imports are lazy so the module
can be imported without the package installed.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ElevenLabsTts:
    """TTS backend using the ElevenLabs SDK.

    Raises ImportError at construction time if the ``elevenlabs`` package
    is not installed.
    """

    def __init__(self, api_key: str) -> None:
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ElevenLabs SDK not installed. "
                "Run: uv pip install 'voicelab[cloud]'"
            ) from exc

        self._client = ElevenLabs(api_key=api_key)

    # ------------------------------------------------------------------
    # Tts protocol implementation
    # ------------------------------------------------------------------

    def list_voices(self) -> list[dict]:
        """Return all voices available on the account."""
        try:
            resp = self._client.voices.get_all()
            voices = resp.voices if hasattr(resp, "voices") else []
            return [
                {"id": v.voice_id, "name": v.name, "category": getattr(v, "category", "premade")}
                for v in voices
            ]
        except Exception:
            logger.exception("ElevenLabs list_voices failed")
            return []

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Synthesize *text* with *voice*.

        Args:
            text: Input text.
            voice: Voice name or voice_id.
            settings: Dict with optional keys: stability, similarity_boost, style.

        Returns:
            MP3 bytes or None on failure.
        """
        try:
            from elevenlabs import VoiceSettings  # type: ignore[import]

            voice_settings = VoiceSettings(
                stability=float(settings.get("stability", 0.75)),
                similarity_boost=float(settings.get("similarity_boost", 0.75)),
                style=float(settings.get("style", 0.0)),
                use_speaker_boost=True,
            )

            audio_iter = self._client.text_to_speech.convert(
                voice_id=voice,
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=voice_settings,
            )
            buf = io.BytesIO()
            for chunk in audio_iter:
                buf.write(chunk)
            return buf.getvalue() or None
        except Exception:
            logger.exception("ElevenLabs synthesize failed for voice=%r", voice)
            return None

    # ------------------------------------------------------------------
    # Extended / gated features
    # ------------------------------------------------------------------

    def clone_voice(self, sample_path: str, name: str) -> str | None:
        """Clone a voice from an audio sample file.

        **Requires ElevenLabs Professional plan or higher.**
        This is a stub — it documents the capability but does not fully
        implement error recovery or file validation.

        Args:
            sample_path: Absolute path to the audio sample (WAV / MP3).
            name: Display name for the cloned voice.

        Returns:
            New voice_id string, or None on failure.
        """
        try:
            with open(sample_path, "rb") as f:
                resp = self._client.voices.add(
                    name=name,
                    files=[f],
                )
            return getattr(resp, "voice_id", None)
        except Exception:
            logger.exception("ElevenLabs clone_voice failed for sample=%r", sample_path)
            return None
