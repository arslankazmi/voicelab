"""REST API v1 — /api/v1/* endpoints for VoiceLab."""

from __future__ import annotations

import base64
import dataclasses
import logging
import tempfile
import time
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from voicelab.auth import require_auth
from voicelab.config.settings import get_settings
from voicelab.engine import audio_media_type, compare_across_engines
from voicelab.personas import load_personas
from voicelab.tts.registry import list_backends

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Optional slowapi rate limiting — graceful no-op if not installed
# ---------------------------------------------------------------------------

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter: Any = Limiter(key_func=get_remote_address)

    def _limit(rate: str) -> Callable[[_F], _F]:
        return limiter.limit(rate)

except ImportError:  # pragma: no cover
    logger.warning(
        "slowapi not installed — rate limiting disabled. Install with: uv pip install slowapi"
    )

    class _NoOpLimiter:
        """Stub limiter exported so main.py can do `app.state.limiter = api_limiter`."""

        pass

    limiter = _NoOpLimiter()

    def _limit(rate: str) -> Callable[[_F], _F]:  # noqa: F811
        def decorator(fn: _F) -> _F:
            return fn

        return decorator


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class VoiceSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stability: float = Field(default=0.75, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    style: float = Field(default=0.0, ge=0.0, le=1.0)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    # Chatterbox-specific params — ignored by other backends
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class SynthesizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(max_length=5000)
    voice: str = "default"
    engine: str = "auto"
    settings: VoiceSettings = Field(default_factory=VoiceSettings)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(max_length=5000)
    voice_a: str = "default"
    voice_b: str = "default"
    engine_a: str = "auto"
    engine_b: str = "auto"
    settings: VoiceSettings = Field(default_factory=VoiceSettings)


class EngineVoicePair(BaseModel):
    """An (engine, voice) pair for cross-engine comparison."""

    engine: str = "auto"
    voice: str = "default"


class CrossCompareRequest(BaseModel):
    """Compare the same text across multiple (engine, voice) pairs."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(max_length=5000)
    pairs: list[EngineVoicePair] = Field(min_length=1, max_length=6)
    settings: VoiceSettings = Field(default_factory=VoiceSettings)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"],
    dependencies=[require_auth],
)


# ---------------------------------------------------------------------------
# GET /api/v1/engines
# ---------------------------------------------------------------------------


@router.get("/engines")
@_limit("60/minute")
async def list_engines(request: Request) -> dict[str, Any]:
    """List all TTS backends with availability status."""
    settings = get_settings()
    backends = list_backends(api_key=settings.elevenlabs_api_key)
    return {"engines": backends, "active": settings.tts_engine}


# ---------------------------------------------------------------------------
# GET /api/v1/voices
# ---------------------------------------------------------------------------


@router.get("/voices")
@_limit("60/minute")
async def list_voices(request: Request, engine: str = "auto") -> dict[str, Any]:
    """Return available voices from the requested TTS backend.

    Pass ``?engine=kokoro`` (or piper/elevenlabs/local) to query a specific backend.
    """
    from voicelab.tts.registry import get_tts_for_engine  # noqa: PLC0415

    settings = get_settings()
    tts = get_tts_for_engine(engine, api_key=settings.elevenlabs_api_key)
    voices = tts.list_voices()
    return {"voices": voices, "engine": engine}


# ---------------------------------------------------------------------------
# GET /api/v1/personas
# ---------------------------------------------------------------------------


@router.get("/personas")
@_limit("60/minute")
async def list_personas(request: Request) -> dict[str, Any]:
    """Return all persona presets."""
    personas = load_personas()
    return {"personas": [dataclasses.asdict(p) for p in personas]}


# ---------------------------------------------------------------------------
# POST /api/v1/synthesize
# ---------------------------------------------------------------------------


@router.post("/synthesize")
@_limit("10/minute")
async def synthesize(request: Request, body: SynthesizeRequest) -> StreamingResponse:
    """Synthesize text to audio (WAV for local/kokoro/piper; MP3 for ElevenLabs).

    Emits timing headers for the self-evaluating grid:
      - ``X-Synth-Seconds``  — the ``tts.synthesize()`` call (model inference; for
        cloud backends this includes the provider round-trip).
      - ``X-Server-Seconds`` — total time inside this handler.
      - ``X-Engine`` / ``X-Audio-Format``.
    """
    from voicelab.tts.registry import get_tts_for_engine  # noqa: PLC0415

    server_t0 = time.perf_counter()
    settings = get_settings()
    tts = get_tts_for_engine(body.engine, api_key=settings.elevenlabs_api_key)

    synth_t0 = time.perf_counter()
    audio_bytes = tts.synthesize(
        body.text,
        body.voice,
        body.settings.model_dump(),
    )
    synth_seconds = time.perf_counter() - synth_t0

    if audio_bytes is None:
        raise HTTPException(status_code=503, detail="Synthesis failed")

    media = audio_media_type(tts)
    headers = {
        "X-Synth-Seconds": f"{synth_seconds:.4f}",
        "X-Server-Seconds": f"{time.perf_counter() - server_t0:.4f}",
        "X-Engine": body.engine,
        "X-Audio-Format": "wav" if media == "audio/wav" else "mp3",
    }
    return StreamingResponse(iter([audio_bytes]), media_type=media, headers=headers)


# ---------------------------------------------------------------------------
# POST /api/v1/compare
# ---------------------------------------------------------------------------


@router.post("/compare")
@_limit("60/minute")
async def compare(request: Request, body: CompareRequest) -> dict[str, Any]:
    """Synthesize text with two (engine, voice) pairs; return base64-encoded audio."""
    from voicelab.tts.registry import get_tts_for_engine  # noqa: PLC0415

    settings = get_settings()
    settings_dict = body.settings.model_dump()

    tts_a = get_tts_for_engine(body.engine_a, api_key=settings.elevenlabs_api_key)
    tts_b = get_tts_for_engine(body.engine_b, api_key=settings.elevenlabs_api_key)

    audio_a = tts_a.synthesize(body.text, body.voice_a, settings_dict)
    audio_b = tts_b.synthesize(body.text, body.voice_b, settings_dict)

    return {
        "voice_a": base64.b64encode(audio_a).decode() if audio_a is not None else None,
        "voice_b": base64.b64encode(audio_b).decode() if audio_b is not None else None,
        "engine_a": body.engine_a,
        "engine_b": body.engine_b,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/cross-compare
# ---------------------------------------------------------------------------


@router.post("/cross-compare")
@_limit("10/minute")
async def cross_compare(request: Request, body: CrossCompareRequest) -> dict[str, Any]:
    """Synthesize text across multiple (engine, voice) pairs for cross-engine comparison."""
    app_settings = get_settings()
    pairs = [(p.engine, p.voice) for p in body.pairs]
    results = compare_across_engines(
        body.text,
        pairs,
        body.settings.model_dump(),
        app_settings=app_settings,
    )
    return {
        "results": [
            {
                "engine": engine,
                "voice": voice,
                "audio": base64.b64encode(audio).decode() if audio is not None else None,
            }
            for engine, voice, audio in results
        ]
    }


# ---------------------------------------------------------------------------
# POST /api/v1/clone
# ---------------------------------------------------------------------------


@router.post("/clone")
@_limit("5/minute")
async def clone_voice(
    request: Request,
    consent: bool = Form(...),  # noqa: B008
    voice_name: str = Form(...),  # noqa: B008
    sample: UploadFile = File(...),  # noqa: B008
    text: str = Form(default="Hello, this is a voice clone test."),  # noqa: B008
    exaggeration: float = Form(default=0.5),  # noqa: B008
    cfg_weight: float = Form(default=0.5),  # noqa: B008
) -> dict[str, Any]:
    """Clone a voice from an uploaded audio sample.

    Requires ``consent=true``.

    If Chatterbox is installed (``[clone]`` extra), it is used for **keyless** local
    cloning and returns ``audio`` (base64 WAV).  If only ElevenLabs is configured,
    it performs cloud cloning and returns ``voice_id``.
    """
    if not consent:
        raise HTTPException(status_code=400, detail="Consent required for voice cloning")

    # Write upload to a temp file
    suffix = ""
    if sample.filename:
        for ext in (".mp3", ".wav", ".ogg", ".m4a", ".flac"):
            if sample.filename.lower().endswith(ext):
                suffix = ext
                break

    with tempfile.NamedTemporaryFile(
        suffix=suffix or ".bin", prefix="voicelab_clone_", dir="/tmp", delete=False
    ) as tmp:
        tmp.write(await sample.read())
        tmp_path = tmp.name

    from voicelab.validation import validate_audio

    try:
        validate_audio(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # --- Prefer Chatterbox (keyless local cloning) when installed ---
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec("chatterbox") is not None:
        from voicelab.tts.chatterbox import ChatterboxTts  # noqa: PLC0415

        tts = ChatterboxTts()
        clone_settings = {
            "reference_audio": tmp_path,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
        audio_bytes = tts.synthesize(text, "cloned", clone_settings)
        if audio_bytes is None:
            raise HTTPException(
                status_code=502,
                detail="Chatterbox voice cloning failed — check installation.",
            )
        import base64  # noqa: PLC0415

        return {
            "engine": "chatterbox",
            "voice_name": voice_name,
            "audio": base64.b64encode(audio_bytes).decode(),
        }

    # --- Fall back to ElevenLabs cloud cloning ---
    settings = get_settings()
    if not settings.elevenlabs_api_key:
        raise HTTPException(
            status_code=501,
            detail=(
                "Voice cloning requires either Chatterbox (install with 'uv sync --extra clone') "
                "or an ElevenLabs API key (set ELEVENLABS_API_KEY)."
            ),
        )

    from voicelab.tts.elevenlabs import ElevenLabsTts  # noqa: PLC0415

    el_tts = ElevenLabsTts(api_key=settings.elevenlabs_api_key)
    voice_id = el_tts.clone_voice(tmp_path, voice_name)

    if voice_id is None:
        raise HTTPException(
            status_code=502,
            detail="Voice cloning failed — check your ElevenLabs plan tier.",
        )

    return {"engine": "elevenlabs", "voice_id": voice_id}
