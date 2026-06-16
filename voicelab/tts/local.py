"""Local TTS fallback backend using pyttsx3.

pyttsx3 is entirely optional — if unavailable (or if audio hardware is
absent, as in CI) every method degrades gracefully and NEVER raises.
"""

from __future__ import annotations

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# Sentinel to avoid repeated failed import attempts.
_PYTTSX3_AVAILABLE: bool | None = None


def _try_import_pyttsx3():
    """Lazy import of pyttsx3; returns the module or None."""
    global _PYTTSX3_AVAILABLE
    if _PYTTSX3_AVAILABLE is True:
        import pyttsx3  # type: ignore[import]
        return pyttsx3
    if _PYTTSX3_AVAILABLE is False:
        return None
    try:
        import pyttsx3  # type: ignore[import]
        _PYTTSX3_AVAILABLE = True
        return pyttsx3
    except Exception:
        _PYTTSX3_AVAILABLE = False
        logger.warning("pyttsx3 not available — local TTS synthesis will return None")
        return None


class LocalTts:
    """System TTS fallback via pyttsx3.

    All methods are safe to call without audio hardware or the pyttsx3
    package installed; they log a warning and return gracefully.
    """

    # ------------------------------------------------------------------
    # Tts protocol implementation
    # ------------------------------------------------------------------

    def list_voices(self) -> list[dict]:
        """Return system voices from pyttsx3, or a minimal stub list."""
        pyttsx3 = _try_import_pyttsx3()
        if pyttsx3 is None:
            return [{"id": "default", "name": "System Default (pyttsx3 unavailable)"}]
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices") or []
            engine.stop()
            return [
                {"id": v.id, "name": v.name or v.id, "category": "local"}
                for v in voices
            ] or [{"id": "default", "name": "System Default"}]
        except Exception:
            logger.warning("LocalTts.list_voices failed — returning stub", exc_info=True)
            return [{"id": "default", "name": "System Default (error)"}]

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Synthesize *text* using pyttsx3, writing to a /tmp WAV then reading bytes.

        Args:
            text: Input text.
            voice: Voice id (pyttsx3 voice id) or "default".
            settings: Ignored for local backend (pyttsx3 has no equivalent).

        Returns:
            WAV bytes or None if synthesis fails.
        """
        pyttsx3 = _try_import_pyttsx3()
        if pyttsx3 is None:
            return None
        try:
            engine = pyttsx3.init()
            # Apply voice if not default
            if voice and voice != "default":
                engine.setProperty("voice", voice)
            # Apply speed from settings if present
            rate = engine.getProperty("rate")
            speed_mult = float(settings.get("speed", 1.0))
            engine.setProperty("rate", int(rate * speed_mult))

            with tempfile.NamedTemporaryFile(
                suffix=".wav", dir="/tmp", delete=False
            ) as tmp:
                tmp_path = tmp.name

            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            engine.stop()

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_path)
            return audio_bytes if audio_bytes else None
        except Exception:
            logger.warning("LocalTts.synthesize failed", exc_info=True)
            return None
