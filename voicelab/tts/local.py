"""Local TTS fallback backend using pyttsx3 or macOS ``say``/``afconvert``.

pyttsx3 is entirely optional — if unavailable (or if audio hardware is
absent, as in CI) every method degrades gracefully and NEVER raises.

Fallback hierarchy (macOS only when pyttsx3 absent):
  1. pyttsx3   (cross-platform)
  2. ``say`` + ``afconvert``   (macOS built-ins, zero extra deps)
  3. None       (Linux CI / pyttsx3 absent / ``say`` absent)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

# Sentinel to avoid repeated failed import attempts.
_PYTTSX3_AVAILABLE: bool | None = None


def _try_import_pyttsx3():
    """Lazy import of pyttsx3; returns the module or None."""
    global _PYTTSX3_AVAILABLE
    if _PYTTSX3_AVAILABLE is True:
        import pyttsx3  # noqa: PLC0415

        return pyttsx3
    if _PYTTSX3_AVAILABLE is False:
        return None
    try:
        import pyttsx3  # noqa: PLC0415

        _PYTTSX3_AVAILABLE = True
        return pyttsx3
    except Exception:
        _PYTTSX3_AVAILABLE = False
        logger.warning("pyttsx3 not available — local TTS synthesis will try macOS say fallback")
        return None


def _say_available() -> bool:
    """Return True only on macOS when both ``say`` and ``afconvert`` are on PATH."""
    return (
        sys.platform == "darwin"
        and shutil.which("say") is not None
        and shutil.which("afconvert") is not None
    )


def _synthesize_via_say(text: str, settings: dict) -> bytes | None:
    """Synthesize *text* using macOS ``say`` + ``afconvert``.

    Produces a 16-bit little-endian WAV file and returns its bytes.

    Args:
        text: Input text to speak.
        settings: Dict that may contain ``speed`` (float multiplier, default 1.0)
                  and ``voice`` (voice name, optional).

    Returns:
        WAV bytes or None on any failure (never raises).
    """
    aiff_path: str | None = None
    wav_path: str | None = None
    try:
        # Build temp file paths in /tmp
        with tempfile.NamedTemporaryFile(
            suffix=".aiff", prefix="voicelab_say_", dir="/tmp", delete=False
        ) as f:
            aiff_path = f.name
        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="voicelab_say_", dir="/tmp", delete=False
        ) as f:
            wav_path = f.name

        # Build say command
        rate = int(175 * float(settings.get("speed", 1.0)))
        voice = settings.get("voice", "") or ""

        say_cmd: list[str] = ["say", "-o", aiff_path, "-r", str(rate)]
        if voice and voice != "default":
            say_cmd += ["-v", voice]
        say_cmd.append(text)

        result = subprocess.run(
            say_cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                "say command failed (rc=%d): %s", result.returncode, result.stderr.decode()
            )
            return None

        # Convert AIFF → WAV (16-bit little-endian PCM)
        afc_cmd = ["afconvert", aiff_path, wav_path, "-d", "LEI16", "-f", "WAVE"]
        result = subprocess.run(
            afc_cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                "afconvert failed (rc=%d): %s", result.returncode, result.stderr.decode()
            )
            return None

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        return wav_bytes if wav_bytes else None

    except Exception:
        logger.warning("_synthesize_via_say failed", exc_info=True)
        return None
    finally:
        for path in (aiff_path, wav_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


class LocalTts:
    """System TTS fallback via pyttsx3 or macOS ``say``/``afconvert``.

    All methods are safe to call without audio hardware or the pyttsx3
    package installed; they log a warning and return gracefully.

    Audio format note
    -----------------
    Both pyttsx3 and the macOS ``say`` fallback produce **WAV** bytes
    (RIFF header).  Callers should use ``audio/wav`` / ``.wav`` for the
    local backend, not ``audio/mpeg``.
    """

    # ------------------------------------------------------------------
    # Tts protocol implementation
    # ------------------------------------------------------------------

    def list_voices(self) -> list[dict]:
        """Return system voices from pyttsx3 (or macOS say stub), or a minimal stub list."""
        pyttsx3 = _try_import_pyttsx3()
        if pyttsx3 is None:
            if _say_available():
                return [
                    {
                        "id": "default",
                        "name": "System Default (macOS say)",
                        "category": "local",
                    }
                ]
            return [{"id": "default", "name": "System Default (pyttsx3 unavailable)"}]
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices") or []
            engine.stop()
            return [{"id": v.id, "name": v.name or v.id, "category": "local"} for v in voices] or [
                {"id": "default", "name": "System Default"}
            ]
        except Exception:
            logger.warning("LocalTts.list_voices failed — returning stub", exc_info=True)
            return [{"id": "default", "name": "System Default (error)"}]

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Synthesize *text* using pyttsx3 (or macOS ``say`` fallback).

        Args:
            text: Input text.
            voice: Voice id (pyttsx3 voice id / say voice name) or "default".
            settings: Dict of voice settings; ``speed`` (float) is honoured.

        Returns:
            WAV bytes or None if synthesis fails / no backend available.
        """
        pyttsx3 = _try_import_pyttsx3()
        if pyttsx3 is None:
            # --- macOS say fallback ---
            if _say_available():
                merged = dict(settings)
                # Thread voice into settings so _synthesize_via_say can read it
                if voice and voice != "default":
                    merged.setdefault("voice", voice)
                return _synthesize_via_say(text, merged)
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

            with tempfile.NamedTemporaryFile(suffix=".wav", dir="/tmp", delete=False) as tmp:
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
