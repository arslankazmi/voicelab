"""Memory-guarded single-residency manager for heavy (torch) TTS models.

On shared-memory Macs (e.g. 16 GB unified), loading several multi-GB torch
models at once — base Chatterbox + Turbo + … — can exhaust unified memory and
**crash the whole host**. This manager makes that impossible:

  * **Single residency** — at most ONE heavy model is loaded at a time.
    Requesting a different heavy model evicts the current one (drop ref +
    ``gc.collect()`` + ``torch.mps.empty_cache()``).
  * **Serialized heavy work** — a global lock is held for the whole
    load+synthesize session, so a parallel grid fan-out can never co-load two
    heavy models.
  * **System-RAM preflight** — before loading, require a configurable floor of
    *system* free memory (``min_free_memory_mb``). If below it, refuse (yield
    ``None``) rather than triggering an OS-level OOM. Checking *system* memory
    (not just this process) also protects against another process (e.g. the
    rehearsal-room flagship) already holding a heavy model.

Light backends (ONNX Kokoro/Piper, ElevenLabs API, system TTS) are small and do
NOT go through this manager.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MIN_FREE_MB = 3000.0

# One global lock + one resident-model slot for ALL heavy backends.
_LOCK = threading.RLock()
_current_name: str | None = None
_current_model: Any = None


def _free_mb() -> float | None:
    """System available memory in MB, or None if psutil is unavailable."""
    try:
        import psutil  # noqa: PLC0415

        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return None  # no psutil → skip preflight; single-residency still guards


def _empty_mps_cache() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _min_free_mb() -> float:
    try:
        from voicelab.config.settings import get_settings  # noqa: PLC0415

        return float(getattr(get_settings(), "min_free_memory_mb", _DEFAULT_MIN_FREE_MB))
    except Exception:
        return _DEFAULT_MIN_FREE_MB


def evict() -> None:
    """Drop the resident heavy model and reclaim memory. Caller must hold ``_LOCK``."""
    global _current_name, _current_model
    if _current_model is not None:
        logger.info("Evicting heavy TTS model %r to free memory", _current_name)
    _current_model = None
    _current_name = None
    gc.collect()
    _empty_mps_cache()


def resident_model_name() -> str | None:
    """Name of the currently-resident heavy model (for tests / introspection)."""
    return _current_name


@contextlib.contextmanager
def heavy_session(name: str, loader: Callable[[], Any]) -> Iterator[Any]:
    """Yield a heavy model under single-residency + memory guards.

    Holds the global heavy lock for the whole ``with`` block, so only one heavy
    load/synthesis runs at a time. Yields the model, or ``None`` when it can't be
    loaded safely (caller must degrade gracefully — never raise to the host).

    Args:
        name:   Stable backend id (e.g. ``"chatterbox"``, ``"chatterbox-turbo"``).
        loader: Zero-arg callable that constructs + returns the model (may raise).
    """
    global _current_name, _current_model
    with _LOCK:
        # Already resident → reuse (warm).
        if _current_name == name and _current_model is not None:
            yield _current_model
            return

        # Switching heavy models → evict the other one FIRST (free its memory
        # before allocating the next), so peak residency stays at one model.
        if _current_model is not None and _current_name != name:
            evict()

        # System-RAM preflight — refuse rather than OOM the host.
        free = _free_mb()
        floor = _min_free_mb()
        if free is not None and free < floor:
            logger.warning(
                "Refusing to load heavy TTS model %r: only %.0f MB free (< %.0f MB floor). "
                "Free memory or lower min_free_memory_mb.",
                name,
                free,
                floor,
            )
            yield None
            return

        try:
            _current_model = loader()
            _current_name = name
        except Exception:
            logger.warning("Heavy TTS model %r failed to load", name, exc_info=True)
            evict()
            yield None
            return

        yield _current_model
