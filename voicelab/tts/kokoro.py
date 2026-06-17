"""Kokoro TTS backend — ONNX, torch-free, CPU/cross-platform.

Package: kokoro-onnx (Apache-2.0)
Install: uv sync --extra kokoro

Model files are downloaded automatically from Hugging Face on first use and
cached under the HF hub cache directory (~/.cache/huggingface/).

Voices are built-in to the kokoro-onnx package; the full list is available
via ``Kokoro.get_voices()``.  A representative default set is hard-coded in
``KOKORO_VOICES`` for display without loading the model.

All methods degrade gracefully (return None / empty list) — never raises.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Silence HuggingFace + kokoro telemetry (per project policy)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")

# Module-level model cache — loaded once, reused across calls.
_kokoro_instance = None  # Kokoro session object
_kokoro_load_attempted: bool = False

# Representative built-in voices (kokoro-onnx >= 0.4).
# Full list differs by model version; these are reliably present.
KOKORO_VOICES: list[str] = [
    "af_heart",
    "af_bella",
    "af_sarah",
    "af_nova",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]

_DEFAULT_VOICE = "af_heart"
_SAMPLE_RATE = 24000  # kokoro-onnx outputs 24 kHz

# GitHub releases URL for kokoro-onnx model files
# See: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
# Use the int8 quantized model for best CPU performance (~88 MB vs ~310 MB fp32)
_ONNX_FILENAME = "kokoro-v1.0.int8.onnx"
_VOICES_FILENAME = "voices-v1.0.bin"

# Local cache dir for model files
_KOKORO_CACHE = Path(os.environ.get("KOKORO_CACHE_DIR", Path.home() / ".cache" / "kokoro-onnx"))


def _download_model_file(filename: str) -> Path | None:
    """Download *filename* from GitHub releases into _KOKORO_CACHE.

    Returns local path or None on failure.
    """
    local_path = _KOKORO_CACHE / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    _KOKORO_CACHE.mkdir(parents=True, exist_ok=True)
    url = f"{_RELEASE_BASE}/{filename}"
    try:
        import urllib.request  # noqa: PLC0415

        logger.info("Downloading Kokoro model file %r from %s", filename, url)
        urllib.request.urlretrieve(url, str(local_path))
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        logger.warning("Kokoro model file download empty: %s", local_path)
        return None
    except Exception:
        logger.warning(
            "Failed to download Kokoro model file %r from %s",
            filename,
            url,
            exc_info=True,
        )
        # Clean up partial download
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        return None


def _load_kokoro():
    """Lazy-load the Kokoro ONNX session (cached module-level).

    Downloads model weights (~88 MB int8) from GitHub releases on first use.
    Returns the Kokoro instance or None on failure.
    """
    global _kokoro_instance, _kokoro_load_attempted
    if _kokoro_load_attempted:
        return _kokoro_instance
    _kokoro_load_attempted = True
    try:
        from kokoro_onnx import Kokoro  # noqa: PLC0415

        onnx_path = _download_model_file(_ONNX_FILENAME)
        voices_path = _download_model_file(_VOICES_FILENAME)
        if onnx_path is None or voices_path is None:
            logger.warning("Kokoro model files unavailable — KokoroTts disabled")
            return None

        _kokoro_instance = Kokoro(str(onnx_path), str(voices_path))
        logger.info("Kokoro ONNX session loaded from %s", _KOKORO_CACHE)
    except Exception:
        logger.warning("Failed to load Kokoro ONNX model", exc_info=True)
        _kokoro_instance = None
    return _kokoro_instance


def _pcm_to_wav(samples, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Convert float32/int16 numpy array to WAV bytes (RIFF/WAVE)."""
    import io  # noqa: PLC0415
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    # Normalise to int16
    arr = np.asarray(samples)
    if arr.dtype.kind == "f":
        arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        arr = arr.astype(np.int16)
    if arr.ndim > 1:
        arr = arr[:, 0]  # take first channel

    pcm = arr.tobytes()
    n_channels = 1
    sample_width = 2  # int16
    byte_rate = sample_rate * n_channels * sample_width
    block_align = n_channels * sample_width
    data_size = len(pcm)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", n_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


class KokoroTts:
    """Kokoro ONNX TTS backend.

    Implements the Tts protocol:
      synthesize(text, voice, settings) -> WAV bytes | None
      list_voices() -> list[dict]

    Audio format: WAV (RIFF), 24 kHz mono int16.
    Torch-free: uses onnxruntime only.
    """

    def list_voices(self) -> list[dict]:
        """Return Kokoro built-in voice names.

        Attempts to load the model to get the full dynamic list; falls back
        to the hard-coded ``KOKORO_VOICES`` constant if unavailable.
        """
        kokoro = _load_kokoro()
        if kokoro is not None:
            try:
                # kokoro-onnx exposes get_voices() -> list[str]
                raw = kokoro.get_voices()
                if raw:
                    return [{"id": v, "name": v, "category": "kokoro"} for v in sorted(raw)]
            except Exception:
                logger.warning(
                    "KokoroTts.list_voices dynamic failed — using built-in list",
                    exc_info=True,
                )

        return [{"id": v, "name": v, "category": "kokoro"} for v in KOKORO_VOICES]

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Synthesize *text* with Kokoro ONNX.

        Args:
            text: Input text (plain text, no SSML).
            voice: Kokoro voice name (e.g. "af_heart"). Falls back to default.
            settings: Dict; ``speed`` (float, default 1.0) is honoured.

        Returns:
            WAV bytes (RIFF header) or None on failure.
        """
        kokoro = _load_kokoro()
        if kokoro is None:
            logger.warning("KokoroTts.synthesize: model not loaded — returning None")
            return None

        try:
            voice_name = voice if voice and voice != "default" else _DEFAULT_VOICE
            speed = float(settings.get("speed", 1.0))

            # kokoro-onnx >=0.4: create(text, voice, speed) -> (samples, sample_rate)
            samples, sample_rate = kokoro.create(text, voice=voice_name, speed=speed)
            return _pcm_to_wav(samples, sample_rate)
        except Exception:
            logger.warning("KokoroTts.synthesize failed for voice=%r", voice, exc_info=True)
            return None
