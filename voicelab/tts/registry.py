"""TTS backend registry — hardware-aware selection (wallgen pattern).

Each entry describes a backend: name, priority (lower = preferred), how to
check availability, license, and notes.  ``get_tts_for_engine`` resolves a
named engine (or "auto") to the best available backend instance.

Selection order when engine="auto":
    1. ElevenLabs   (cloud, if ELEVENLABS_API_KEY set + SDK installed)
    2. Kokoro       (local ONNX, if kokoro-onnx installed)
    3. Piper        (local ONNX, if piper-tts installed)
    4. Local        (system TTS: pyttsx3 / macOS say fallback)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

EngineKey = Literal["auto", "elevenlabs", "kokoro", "piper", "local"]

_ALL_ENGINES: tuple[str, ...] = ("elevenlabs", "kokoro", "piper", "local")


@dataclass
class BackendInfo:
    """Descriptor for a TTS backend."""

    name: str
    priority: int  # lower = higher preference
    package: str | None  # PyPI package that must be importable (None = always available)
    import_check: str | None  # module to import-check (may differ from package name)
    license: str
    notes: str
    requires_key: bool = False
    tags: list[str] = field(default_factory=list)


_REGISTRY: list[BackendInfo] = [
    BackendInfo(
        name="elevenlabs",
        priority=10,
        package="elevenlabs",
        import_check="elevenlabs.client",
        license="Proprietary (ElevenLabs SDK)",
        notes="Cloud TTS; requires ELEVENLABS_API_KEY.",
        requires_key=True,
        tags=["cloud", "high-quality"],
    ),
    BackendInfo(
        name="kokoro",
        priority=20,
        package="kokoro-onnx",
        import_check="kokoro_onnx",
        license="Apache-2.0",
        notes="ONNX runtime, torch-free, CPU/cross-platform. Voices: af_heart, af_bella, …",
        requires_key=False,
        tags=["local", "onnx", "torch-free"],
    ),
    BackendInfo(
        name="piper",
        priority=30,
        package="piper-tts",
        import_check="piper",
        license="MIT",
        notes="ONNX runtime, torch-free. Voices downloaded as .onnx on first use.",
        requires_key=False,
        tags=["local", "onnx", "torch-free"],
    ),
    BackendInfo(
        name="local",
        priority=40,
        package=None,
        import_check=None,
        license="Apache-2.0 (voicelab)",
        notes="System TTS: pyttsx3 or macOS say/afconvert. Always available.",
        requires_key=False,
        tags=["local", "system"],
    ),
]


def _backend_by_name(name: str) -> BackendInfo | None:
    return next((b for b in _REGISTRY if b.name == name), None)


def is_available(info: BackendInfo, *, api_key: str | None = None) -> bool:
    """Return True if this backend can be instantiated right now."""
    if info.requires_key and not api_key:
        return False
    if info.import_check is None:
        return True
    try:
        importlib.import_module(info.import_check)
        return True
    except Exception:
        return False


def list_backends(api_key: str | None = None) -> list[dict]:
    """Return a list of dicts describing every backend + availability."""
    return [
        {
            "name": b.name,
            "priority": b.priority,
            "available": is_available(b, api_key=api_key),
            "license": b.license,
            "notes": b.notes,
            "tags": b.tags,
            "requires_key": b.requires_key,
        }
        for b in sorted(_REGISTRY, key=lambda x: x.priority)
    ]


def get_tts_for_engine(engine: str, *, api_key: str | None = None):  # noqa: ANN201
    """Resolve *engine* name (or "auto") to a backend instance.

    Returns the best available backend, never raises.  Falls back along the
    priority chain when the requested engine is unavailable.

    Args:
        engine: One of "auto", "elevenlabs", "kokoro", "piper", "local".
        api_key: ElevenLabs API key (required for elevenlabs backend).

    Returns:
        A backend instance implementing the Tts protocol.
    """
    from voicelab.tts.local import LocalTts  # always available

    if engine == "auto":
        candidates = sorted(_REGISTRY, key=lambda x: x.priority)
    else:
        # Put requested engine first, fall through to lower-priority on failure
        info = _backend_by_name(engine)
        if info is None:
            logger.warning("Unknown engine %r — falling back to auto", engine)
            return get_tts_for_engine("auto", api_key=api_key)
        rest = [b for b in sorted(_REGISTRY, key=lambda x: x.priority) if b.name != engine]
        candidates = [info] + rest

    for candidate in candidates:
        if not is_available(candidate, api_key=api_key):
            continue
        try:
            instance = _instantiate(candidate, api_key=api_key)
            if instance is not None:
                logger.info("TTS engine resolved: %s", candidate.name)
                return instance
        except Exception:
            logger.warning("Failed to instantiate %s backend", candidate.name, exc_info=True)
            continue

    logger.warning("All TTS backends failed — falling back to LocalTts")
    return LocalTts()


def _instantiate(info: BackendInfo, *, api_key: str | None = None):  # noqa: ANN201
    """Construct backend instance for *info*.  Returns None on failure."""
    if info.name == "elevenlabs":
        from voicelab.tts.elevenlabs import ElevenLabsTts

        return ElevenLabsTts(api_key=api_key)  # type: ignore[arg-type]

    if info.name == "kokoro":
        from voicelab.tts.kokoro import KokoroTts

        return KokoroTts()

    if info.name == "piper":
        from voicelab.tts.piper import PiperTts

        return PiperTts()

    if info.name == "local":
        from voicelab.tts.local import LocalTts

        return LocalTts()

    return None
