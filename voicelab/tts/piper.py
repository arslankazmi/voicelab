"""Piper TTS backend — ONNX, torch-free, CPU/cross-platform.

Package: piper-tts (MIT)
Install: uv sync --extra piper

Voice models (.onnx + .onnx.json) are downloaded from the Hugging Face
Piper voices repository (rhasspy/piper-voices) on first use and cached under
~/.local/share/piper-voices/ (or PIPER_VOICE_DIR env var).

A default English voice is used when none is specified; other voices can be
requested by their short name (e.g. "en_US-amy-medium") or auto-resolved from
a partial name.

All methods degrade gracefully — never raises.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Silence HuggingFace telemetry
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")

# ------------------------------------------------------------------
# Voice model cache directory
# ------------------------------------------------------------------

_VOICE_DIR = Path(
    os.environ.get("PIPER_VOICE_DIR", Path.home() / ".local" / "share" / "piper-voices")
)

# Default voice — compact English model, good balance of quality/size.
_DEFAULT_VOICE_NAME = "en_US-amy-medium"

# HuggingFace repo hosting piper voice files
_HF_REPO = "rhasspy/piper-voices"

# Module-level voice cache: voice_name -> PiperVoice instance
_voice_cache: dict[str, object] = {}

# Curated built-in voice list for display without downloading
PIPER_VOICES: list[str] = [
    "en_US-amy-medium",
    "en_US-arctic-medium",
    "en_US-danny-low",
    "en_US-hfc_female-medium",
    "en_US-joe-medium",
    "en_US-kathleen-low",
    "en_US-kristin-medium",
    "en_US-lessac-medium",
    "en_US-ryan-medium",
    "en_GB-alan-medium",
    "en_GB-jenny_dioco-medium",
]


def _parse_voice_name(voice_name: str) -> tuple[str, str, str, str]:
    """Parse a voice name like 'en_US-amy-medium' into (lang, lang_region, speaker, quality).

    Examples:
      "en_US-amy-medium"    -> ("en", "en_US", "amy", "medium")
      "en_GB-alan-medium"   -> ("en", "en_GB", "alan", "medium")
    """
    # Split on first '-' to separate lang_region from rest
    if "-" in voice_name:
        lang_region, rest = voice_name.split("-", 1)
        parts = rest.rsplit("-", 1)
        speaker = parts[0] if len(parts) > 1 else rest
        quality = parts[1] if len(parts) > 1 else "medium"
    else:
        lang_region = voice_name
        speaker = voice_name
        quality = "medium"
    lang = lang_region.split("_")[0] if "_" in lang_region else lang_region
    return lang, lang_region, speaker, quality


def _voice_dir_for(voice_name: str) -> Path:
    """Return the local directory where *voice_name* files are stored.

    Mirrors the HF repo layout:
      {lang}/{lang_region}/{speaker}/{quality}/
    e.g. en/en_US/amy/medium/
    """
    lang, lang_region, speaker, quality = _parse_voice_name(voice_name)
    return _VOICE_DIR / lang / lang_region / speaker / quality


def _hf_voice_path(voice_name: str) -> str:
    """Return the HF repo path prefix for a voice file.

    Piper voices repo layout (rhasspy/piper-voices):
      {lang}/{lang_region}/{speaker}/{quality}/{name}
    e.g. en/en_US/amy/medium/en_US-amy-medium
    """
    lang, lang_region, speaker, quality = _parse_voice_name(voice_name)
    return f"{lang}/{lang_region}/{speaker}/{quality}/{voice_name}"


def _download_voice(voice_name: str) -> Path | None:
    """Download *voice_name* .onnx + .json from HF if not cached.

    Returns the path to the .onnx file, or None on failure.
    """
    voice_dir = _voice_dir_for(voice_name)
    onnx_path = voice_dir / f"{voice_name}.onnx"
    json_path = voice_dir / f"{voice_name}.onnx.json"

    if onnx_path.exists() and json_path.exists():
        return onnx_path

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        voice_dir.mkdir(parents=True, exist_ok=True)
        hf_path = _hf_voice_path(voice_name)

        hf_hub_download(
            repo_id=_HF_REPO,
            filename=f"{hf_path}.onnx",
            local_dir=str(_VOICE_DIR),
        )
        hf_hub_download(
            repo_id=_HF_REPO,
            filename=f"{hf_path}.onnx.json",
            local_dir=str(_VOICE_DIR),
        )
        if onnx_path.exists():
            return onnx_path
        logger.warning("Piper voice download completed but .onnx not found at %s", onnx_path)
        return None
    except Exception:
        logger.warning("Failed to download Piper voice %r", voice_name, exc_info=True)
        return None


def _load_voice(voice_name: str):  # noqa: ANN201
    """Load (or retrieve cached) PiperVoice for *voice_name*."""
    if voice_name in _voice_cache:
        return _voice_cache[voice_name]

    onnx_path = _download_voice(voice_name)
    if onnx_path is None:
        return None

    try:
        from piper import PiperVoice  # noqa: PLC0415

        json_path = str(onnx_path) + ".json"
        pv = PiperVoice.load(str(onnx_path), config_path=json_path)
        _voice_cache[voice_name] = pv
        logger.info("Piper voice loaded: %s", voice_name)
        return pv
    except Exception:
        logger.warning("Failed to load Piper voice %r", voice_name, exc_info=True)
        return None


def _piper_to_wav(piper_voice, text: str) -> bytes | None:
    """Synthesize *text* with *piper_voice* and return WAV bytes."""
    import io  # noqa: PLC0415
    import wave  # noqa: PLC0415

    try:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            # piper-tts 1.4.x API: synthesize_wav(text, wav_writer)
            # Older versions use synthesize(text, wav_writer) — try both
            if hasattr(piper_voice, "synthesize_wav"):
                piper_voice.synthesize_wav(text, wav_file)
            else:
                piper_voice.synthesize(text, wav_file)
        wav_bytes = buf.getvalue()
        return wav_bytes if wav_bytes else None
    except Exception:
        logger.warning("Piper synthesize failed", exc_info=True)
        return None


class PiperTts:
    """Piper ONNX TTS backend.

    Implements the Tts protocol:
      synthesize(text, voice, settings) -> WAV bytes | None
      list_voices() -> list[dict]

    Audio format: WAV (RIFF), 22.05 kHz mono (voice-dependent).
    Torch-free: uses onnxruntime only.
    """

    def list_voices(self) -> list[dict]:
        """Return available Piper voice names.

        Returns the curated built-in list; voices are downloaded on demand in
        synthesize().  Voices already cached locally are marked "cached=True".
        """
        result = []
        for v in PIPER_VOICES:
            onnx_path = _voice_dir_for(v) / f"{v}.onnx"
            result.append(
                {
                    "id": v,
                    "name": v,
                    "category": "piper",
                    "cached": onnx_path.exists(),
                }
            )
        return result

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Synthesize *text* with Piper ONNX.

        Args:
            text: Input text (plain text).
            voice: Piper voice name (e.g. "en_US-amy-medium"). Falls back to default.
            settings: Dict; ``speed`` is not supported by piper-tts natively (ignored).

        Returns:
            WAV bytes (RIFF header) or None on failure.
        """
        voice_name = voice if voice and voice != "default" else _DEFAULT_VOICE_NAME

        pv = _load_voice(voice_name)
        if pv is None:
            logger.warning("PiperTts.synthesize: voice %r unavailable — returning None", voice_name)
            return None

        return _piper_to_wav(pv, text)
