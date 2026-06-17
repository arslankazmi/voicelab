"""Chatterbox TTS backend — keyless voice cloning (MIT).

Package: chatterbox-tts (MIT) — available via ``[clone]`` extra.

**This backend is intentionally NOT installed by default.**  Chatterbox
pulls PyTorch (heavy dependency, 1–2 GB) and is only useful when you want
keyless local voice cloning.  To enable:

    uv sync --extra clone

Torch dependency means this backend is excluded from CI runs.  It is
registered in the registry so ``/api/v1/engines`` can describe it.

Audio format: WAV (RIFF), 22.05 kHz mono (Chatterbox default).

Usage
-----
Once installed, this backend works identically to the others::

    from voicelab.tts.chatterbox import ChatterboxTts
    tts = ChatterboxTts()
    wav = tts.synthesize("Hello world", "default", {})

For voice cloning, pass a ``reference_audio`` path in settings::

    wav = tts.synthesize("Hello", "default", {"reference_audio": "/path/to/sample.wav"})
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Module-level model cache
_model = None
_model_load_attempted: bool = False

_SAMPLE_RATE = 22050


def _load_model():  # noqa: ANN201
    """Lazy-load Chatterbox model (cached module-level).

    Downloads ~1.5 GB of model weights on first use.
    Returns model or None if unavailable.
    """
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        from chatterbox.tts import ChatterboxTTS  # noqa: PLC0415

        _model = ChatterboxTTS.from_pretrained(device="cpu")
        logger.info("Chatterbox model loaded")
    except ImportError:
        logger.warning("chatterbox-tts not installed. Install with: uv sync --extra clone")
        _model = None
    except Exception:
        logger.warning("Failed to load Chatterbox model", exc_info=True)
        _model = None
    return _model


def _tensor_to_wav(audio_tensor, sample_rate: int = _SAMPLE_RATE) -> bytes | None:
    """Convert a torch tensor to WAV bytes."""
    import io  # noqa: PLC0415
    import struct  # noqa: PLC0415

    try:
        # Detach + convert to numpy
        arr = audio_tensor.squeeze().detach().cpu().numpy()
        # Normalise to int16
        import numpy as np  # noqa: PLC0415

        arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
        pcm = arr.tobytes()

        n_channels = 1
        sample_width = 2
        byte_rate = sample_rate * n_channels * sample_width
        block_align = n_channels * sample_width
        data_size = len(pcm)

        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 1))
        buf.write(struct.pack("<H", n_channels))
        buf.write(struct.pack("<I", sample_rate))
        buf.write(struct.pack("<I", byte_rate))
        buf.write(struct.pack("<H", block_align))
        buf.write(struct.pack("<H", sample_width * 8))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(pcm)
        return buf.getvalue()
    except Exception:
        logger.warning("_tensor_to_wav failed", exc_info=True)
        return None


class ChatterboxTts:
    """Chatterbox TTS backend — keyless local voice cloning.

    Implements the Tts protocol:
      synthesize(text, voice, settings) -> WAV bytes | None
      list_voices() -> list[dict]

    ``settings`` may include:
      ``reference_audio`` (str path): audio sample for voice cloning.
      ``exaggeration`` (float, 0.0–1.0): emotion exaggeration.
      ``cfg_weight`` (float): classifier-free guidance weight.

    **Requires the [clone] extra (pulls torch).**
    """

    def list_voices(self) -> list[dict]:
        """Return voice options.  Chatterbox uses reference audio for cloning."""
        return [
            {
                "id": "default",
                "name": "Default (no reference)",
                "category": "chatterbox",
            },
            {
                "id": "cloned",
                "name": "Cloned (supply reference_audio in settings)",
                "category": "chatterbox",
            },
        ]

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Synthesize *text* with optional voice cloning.

        Args:
            text: Input text.
            voice: Ignored (chatterbox uses reference audio for cloning).
            settings: Dict; ``reference_audio`` (path) enables cloning;
                      ``exaggeration`` and ``cfg_weight`` tune emotion.

        Returns:
            WAV bytes or None if chatterbox not installed or synthesis fails.
        """
        model = _load_model()
        if model is None:
            logger.warning(
                "ChatterboxTts.synthesize: model unavailable. Install with: uv sync --extra clone"
            )
            return None

        try:
            ref = settings.get("reference_audio")
            exaggeration = float(settings.get("exaggeration", 0.5))
            cfg_weight = float(settings.get("cfg_weight", 0.5))

            if ref:
                audio = model.generate(
                    text,
                    audio_prompt_path=ref,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
            else:
                audio = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight)

            return _tensor_to_wav(audio, sample_rate=model.sr)
        except Exception:
            logger.warning("ChatterboxTts.synthesize failed", exc_info=True)
            return None
