"""Chatterbox Turbo TTS backend — 350M one-step model (MIT).

Package: chatterbox-tts>=0.1.7 (MIT) — available via ``[clone]`` extra.

ChatterboxTurbo is a faster, smaller (~350M) single-step model compared to
the base Chatterbox.  It shares the same generate() API but uses the
``tts_turbo`` submodule introduced in chatterbox-tts 0.1.7.

**This backend is intentionally NOT installed by default.**  Like the base
Chatterbox backend it requires the [clone] extra:

    uv sync --extra clone

Audio format: WAV (RIFF), sample rate from ``model.sr``.

Usage
-----
Once installed::

    from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts
    tts = ChatterboxTurboTts()
    wav = tts.synthesize("Hello world", "default", {})

For voice cloning, pass ``reference_audio`` in settings::

    wav = tts.synthesize("Hello", "default", {"reference_audio": "/path/to/sample.wav"})
"""

from __future__ import annotations

import logging
import os

from voicelab.tts._chatter_common import (
    exaggeration_to_temperature,
    pick_device,
    supported_generate_kwargs,
    tensor_to_wav,
)
from voicelab.tts._model_manager import heavy_session

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# `_model` is a TEST-INJECTION hook only. Production does NOT cache the real
# model here — the single-residency model manager owns it so it can be evicted
# to free memory (base Chatterbox + Turbo never co-resident → no host OOM).
_model = None


def _load_model():  # noqa: ANN201
    """Build the ChatterboxTurbo model (the model manager owns/caches it).

    Uses ``chatterbox.tts_turbo.ChatterboxTurboTTS`` (chatterbox-tts>=0.1.7).
    Returns the model, or None on failure (never raises).
    """
    if _model is not None:
        return _model  # test-injected fake
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: PLC0415

        device = pick_device()
        logger.info("ChatterboxTurbo loading on device=%s", device)
        m = ChatterboxTurboTTS.from_pretrained(device)
        logger.info("ChatterboxTurbo model loaded (device=%s)", device)
        return m
    except ImportError:
        logger.warning(
            "chatterbox.tts_turbo not available. "
            "Install chatterbox-tts>=0.1.7 with: uv sync --extra clone"
        )
        return None
    except Exception:
        logger.warning("Failed to load ChatterboxTurbo model", exc_info=True)
        return None


class ChatterboxTurboTts:
    """Chatterbox Turbo TTS backend — 350M one-step model.

    Implements the Tts protocol:
      synthesize(text, voice, settings) -> WAV bytes | None
      list_voices() -> list[dict]

    ``settings`` may include:
      ``reference_audio`` (str path): audio sample for voice cloning.
      ``exaggeration`` (float, 0.0–1.0): emotion exaggeration; also mapped
          to sampling ``temperature`` via ``exaggeration_to_temperature``.
      ``cfg_weight`` (float): classifier-free guidance weight.

    **Requires the [clone] extra (chatterbox-tts>=0.1.7, pulls torch).**
    """

    def list_voices(self) -> list[dict]:
        """Return voice options.  Turbo uses reference audio for cloning."""
        return [
            {
                "id": "default",
                "name": "Default (no reference)",
                "category": "chatterbox-turbo",
            },
            {
                "id": "cloned",
                "name": "Cloned (supply reference_audio in settings)",
                "category": "chatterbox-turbo",
            },
        ]

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Synthesize *text* with optional voice cloning using the Turbo model.

        Args:
            text: Input text.
            voice: Ignored (turbo uses reference audio for cloning).
            settings: Dict; ``reference_audio`` (path) enables cloning;
                      ``exaggeration`` and ``cfg_weight`` tune emotion;
                      ``exaggeration`` is also mapped to sampling temperature.

        Returns:
            WAV bytes or None if chatterbox-tts not installed or synthesis fails.
        """
        try:
            ref = settings.get("reference_audio")
            exaggeration = float(settings.get("exaggeration", 0.0))
            cfg_weight = float(settings.get("cfg_weight", 0.0))
            temperature = exaggeration_to_temperature(exaggeration)

            kwargs: dict = {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
            }
            if ref:
                kwargs["audio_prompt_path"] = ref

            # Single-residency + memory-preflight + serialized heavy synth (prevents host OOM).
            with heavy_session("chatterbox-turbo", _load_model) as model:
                if model is None:
                    logger.warning(
                        "ChatterboxTurboTts.synthesize: model unavailable or memory low."
                    )
                    return None
                filtered = supported_generate_kwargs(model.generate, **kwargs)
                audio = model.generate(text, **filtered)
                return tensor_to_wav(audio, model.sr)
        except Exception:
            logger.warning("ChatterboxTurboTts.synthesize failed", exc_info=True)
            return None
