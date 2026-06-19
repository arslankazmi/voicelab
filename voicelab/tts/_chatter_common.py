"""Shared helpers for Chatterbox backends (base + turbo).

Extracted so both ``chatterbox.py`` and ``chatterbox_turbo.py`` can reuse:
  - ``pick_device``         — MPS→CPU device selection
  - ``tensor_to_wav``       — torch tensor → WAV bytes
  - ``exaggeration_to_temperature`` — UI knob (0–1) → sampling temperature
  - ``supported_generate_kwargs``   — safely filter kwargs for a generate fn
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)


def pick_device() -> str:
    """Return 'mps' if Apple Silicon MPS is available, else 'cpu'.

    Chatterbox does not support MPS in all ops; callers may choose to
    override to 'cpu' when they detect an MPS hang, but this is the
    out-of-the-box selection logic.
    """
    try:
        import torch  # noqa: PLC0415

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def tensor_to_wav(audio_tensor, sample_rate: int) -> bytes | None:
    """Convert a torch tensor to WAV bytes.

    Args:
        audio_tensor: Torch tensor output from a Chatterbox generate() call.
        sample_rate:  The model's sample rate (``model.sr``).

    Returns:
        RIFF/WAV bytes (16-bit mono PCM) or None if conversion fails.
    """
    import io  # noqa: PLC0415
    import struct  # noqa: PLC0415

    try:
        arr = audio_tensor.squeeze().detach().cpu().numpy()

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
        logger.warning("tensor_to_wav failed", exc_info=True)
        return None


def exaggeration_to_temperature(exag: float) -> float:
    """Map the UI exaggeration knob (0–1) to Chatterbox sampling temperature.

    The formula is ``clamp(0.5 + exag, 0.3, 1.5)`` which gives:
      - exag=0.0 → temperature=0.5  (flat/neutral)
      - exag=0.5 → temperature=1.0  (balanced)
      - exag=1.0 → temperature=1.5  (maximum expressiveness)

    The clamp prevents extreme values from breaking the sampler.  Bad or
    non-numeric input defaults to 0.5 (neutral temperature).

    Args:
        exag: Exaggeration value from the UI, typically in [0.0, 1.0].

    Returns:
        Sampling temperature clamped to [0.3, 1.5].
    """
    try:
        val = float(exag)
    except (TypeError, ValueError):
        logger.debug("exaggeration_to_temperature: bad input %r, defaulting to 0.5", exag)
        return 0.5
    return max(0.3, min(1.5, 0.5 + val))


def supported_generate_kwargs(generate_fn, **kwargs) -> dict:  # noqa: ANN001
    """Return only the kwargs that *generate_fn* accepts.

    Uses ``inspect.signature`` to determine accepted parameters.  If the
    function accepts ``**kwargs`` (VAR_KEYWORD) or if introspection fails,
    all provided kwargs are returned unchanged so the caller doesn't need
    to worry about unsupported keys breaking things.

    Args:
        generate_fn: A callable (e.g. ``model.generate``).
        **kwargs:    Candidate keyword arguments.

    Returns:
        Dict containing only those kwargs that the function will accept.
    """
    try:
        sig = inspect.signature(generate_fn)
        params = sig.parameters
        # If the function accepts **kwargs, pass everything through.
        for p in params.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}
    except (ValueError, TypeError):
        logger.debug(
            "supported_generate_kwargs: could not introspect %r — passing all kwargs through",
            generate_fn,
        )
        return dict(kwargs)
