"""ElevenLabs TTS backend — gated on ELEVENLABS_API_KEY.

Only constructed when a key is present. The elevenlabs SDK is a soft dep
(in [project.optional-dependencies] cloud); imports are lazy so the module
can be imported without the package installed.

All SDK calls are routed through :func:`~voicelab.resilience.resilient_call`
for timeout enforcement and transient-failure retries. Failures degrade
gracefully (return None / empty list) — the event loop never hangs and no
exception escapes to callers that rely on graceful fallback.
"""

from __future__ import annotations

import io
import logging
import time

from voicelab.metrics import tts_calls_total, tts_latency_seconds
from voicelab.resilience import NonTransientError, TransientError, resilient_call

logger = logging.getLogger(__name__)

# Default resilience parameters — overridden per-instance from Settings.
_DEFAULT_TIMEOUT: float = 30.0
_DEFAULT_RETRIES: int = 2
_DEFAULT_BACKOFF: float = 0.5


def _resolve_voice_id(voice: str, voices: list[dict]) -> str | None:
    """Resolve a voice name / id / 'default' to a valid account voice_id.

    The ElevenLabs API needs a real ``voice_id`` (e.g. ``CwhRBWXzGAHq8TQ4Fs17``),
    not a display name. We accept an id (pass-through), a name (exact or prefix
    match, case-insensitive), or ``default``/empty (→ the account's first voice).
    Returns None only when the account exposes no voices.
    """
    if not voices:
        return voice if voice and voice not in ("default", "auto", "") else None
    ids = {v["id"] for v in voices}
    if voice in ids:
        return voice
    if voice and voice not in ("default", "auto", ""):
        vl = voice.lower()
        for v in voices:
            name = str(v.get("name", "")).lower()
            if name == vl or name.startswith(vl):
                return v["id"]
    return voices[0]["id"]


class ElevenLabsTts:
    """TTS backend using the ElevenLabs SDK.

    Raises ImportError at construction time if the ``elevenlabs`` package
    is not installed.
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = _DEFAULT_TIMEOUT,
        retries: int = _DEFAULT_RETRIES,
        backoff_base: float = _DEFAULT_BACKOFF,
    ) -> None:
        try:
            from elevenlabs.client import ElevenLabs  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ElevenLabs SDK not installed. Run: uv pip install 'voicelab[cloud]'"
            ) from exc

        self._client = ElevenLabs(api_key=api_key)
        self._timeout = timeout
        self._retries = retries
        self._backoff_base = backoff_base
        self._voices_cache: list[dict] | None = None

    def _voices(self) -> list[dict]:
        """Cached account voices (one fetch per instance)."""
        if self._voices_cache is None:
            self._voices_cache = self.list_voices()
        return self._voices_cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resilient(self, fn, *, label: str):
        """Run *fn* through resilient_call; return result or raise."""
        return resilient_call(
            fn,
            timeout=self._timeout,
            retries=self._retries,
            backoff_base=self._backoff_base,
        )

    def _record(self, label: str, start: float, outcome: str) -> None:
        elapsed = time.monotonic() - start
        tts_calls_total.labels(backend="elevenlabs", outcome=outcome).inc()
        tts_latency_seconds.labels(backend="elevenlabs", outcome=outcome).observe(elapsed)
        logger.info(
            "elevenlabs.%s outcome=%s latency_s=%.3f",
            label,
            outcome,
            elapsed,
        )

    # ------------------------------------------------------------------
    # Tts protocol implementation
    # ------------------------------------------------------------------

    def list_voices(self) -> list[dict]:
        """Return all voices available on the account."""
        start = time.monotonic()
        try:
            resp = self._resilient(
                lambda: self._client.voices.get_all(),
                label="list_voices",
            )
            voices = resp.voices if hasattr(resp, "voices") else []
            result = [
                {"id": v.voice_id, "name": v.name, "category": getattr(v, "category", "premade")}
                for v in voices
            ]
            self._record("list_voices", start, "ok")
            return result
        except (TransientError, NonTransientError) as exc:
            logger.warning("ElevenLabs list_voices failed: %s", exc)
            self._record("list_voices", start, "error")
            return []
        except Exception:
            logger.exception("ElevenLabs list_voices failed")
            self._record("list_voices", start, "error")
            return []

    def synthesize(self, text: str, voice: str, settings: dict) -> bytes | None:
        """Synthesize *text* with *voice*.

        Args:
            text: Input text.
            voice: Voice name or voice_id.
            settings: Dict with optional keys: stability, similarity_boost, style.

        Returns:
            MP3 bytes or None on failure.
        """
        start = time.monotonic()
        try:
            from elevenlabs import VoiceSettings  # noqa: PLC0415

            voice_settings = VoiceSettings(
                stability=float(settings.get("stability", 0.75)),
                similarity_boost=float(settings.get("similarity_boost", 0.75)),
                style=float(settings.get("style", 0.0)),
                use_speaker_boost=True,
            )

            voice_id = _resolve_voice_id(voice, self._voices())
            if voice_id is None:
                logger.warning("ElevenLabs: no voices available on account for voice=%r", voice)
                self._record("synthesize", start, "error")
                return None

            def _call() -> bytes | None:
                audio_iter = self._client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2",
                    voice_settings=voice_settings,
                )
                buf = io.BytesIO()
                for chunk in audio_iter:
                    buf.write(chunk)
                return buf.getvalue() or None

            result = self._resilient(_call, label="synthesize")
            self._record("synthesize", start, "ok")
            return result
        except (TransientError, NonTransientError) as exc:
            logger.warning("ElevenLabs synthesize failed for voice=%r: %s", voice, exc)
            self._record("synthesize", start, "error")
            return None
        except Exception:
            logger.exception("ElevenLabs synthesize failed for voice=%r", voice)
            self._record("synthesize", start, "error")
            return None

    # ------------------------------------------------------------------
    # Extended / gated features
    # ------------------------------------------------------------------

    def clone_voice(self, sample_path: str, name: str) -> str | None:
        """Clone a voice from an audio sample file.

        **Requires ElevenLabs Professional plan or higher.**
        This is a stub — it documents the capability but does not fully
        implement error recovery or file validation.

        Args:
            sample_path: Absolute path to the audio sample (WAV / MP3).
            name: Display name for the cloned voice.

        Returns:
            New voice_id string, or None on failure.
        """
        start = time.monotonic()
        try:

            def _call() -> str | None:
                with open(sample_path, "rb") as f:
                    resp = self._client.voices.add(name=name, files=[f])
                return getattr(resp, "voice_id", None)

            result = self._resilient(_call, label="clone_voice")
            self._record("clone_voice", start, "ok")
            return result
        except (TransientError, NonTransientError) as exc:
            logger.warning("ElevenLabs clone_voice failed for sample=%r: %s", sample_path, exc)
            self._record("clone_voice", start, "error")
            return None
        except Exception:
            logger.exception("ElevenLabs clone_voice failed for sample=%r", sample_path)
            self._record("clone_voice", start, "error")
            return None
