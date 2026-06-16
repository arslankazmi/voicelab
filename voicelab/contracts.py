"""TTS protocol contract — all backends implement this interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tts(Protocol):
    """Common interface for all TTS backends."""

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Convert *text* to audio bytes using *voice* and *settings*.

        Args:
            text: The text to synthesize.
            voice: Voice identifier (name or ID, backend-specific).
            settings: Backend-specific parameters (stability, similarity_boost, etc.).

        Returns:
            Raw audio bytes (WAV or MP3) or None if synthesis fails.
        """
        ...

    def list_voices(self) -> list[dict]:
        """Return available voices as a list of dicts with at least ``id`` and ``name``."""
        ...
