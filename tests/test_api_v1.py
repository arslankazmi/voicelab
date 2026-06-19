"""Tests for /api/v1/* REST endpoints — keyless, LocalTts/mock backend.

All tests run without ELEVENLABS_API_KEY (keyless). The LocalTts backend
is either mocked to return stub bytes or allowed to degrade to None.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def local_tts_stub():
    """Patch registry to return LocalTts and stub LocalTts.synthesize."""
    from voicelab.tts.local import LocalTts

    # Minimal valid WAV header for testing (split to keep line-length clean)
    fake_wav = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    _local = LocalTts()
    with patch("voicelab.tts.registry.get_tts_for_engine", return_value=_local):
        with patch("voicelab.tts.local.LocalTts.synthesize", return_value=fake_wav):
            yield fake_wav


@pytest.fixture
async def client():
    """Async HTTP client against the voicelab FastAPI app."""
    from voicelab.app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/v1/voices
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_voices_returns_list(client):
    resp = await client.get("/api/v1/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert "voices" in data
    assert isinstance(data["voices"], list)
    assert len(data["voices"]) >= 1


# ---------------------------------------------------------------------------
# GET /api/v1/personas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_personas_returns_list(client):
    resp = await client.get("/api/v1/personas")
    assert resp.status_code == 200
    data = resp.json()
    assert "personas" in data
    assert isinstance(data["personas"], list)
    assert len(data["personas"]) == 5  # known presets


@pytest.mark.asyncio
async def test_get_personas_have_required_keys(client):
    resp = await client.get("/api/v1/personas")
    assert resp.status_code == 200
    for p in resp.json()["personas"]:
        assert "name" in p
        assert "description" in p
        assert "default_settings" in p


# ---------------------------------------------------------------------------
# POST /api/v1/synthesize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_returns_audio_local_fallback(client, local_tts_stub):
    resp = await client.post(
        "/api/v1/synthesize",
        json={"text": "Hello world", "voice": "default"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/")
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_synthesize_emits_timing_headers(client, local_tts_stub):
    """The grid relies on numeric X-Synth-Seconds / X-Server-Seconds + X-Engine headers."""
    resp = await client.post(
        "/api/v1/synthesize",
        json={"text": "Hello world", "voice": "default", "engine": "kokoro"},
    )
    assert resp.status_code == 200
    assert float(resp.headers["x-synth-seconds"]) >= 0.0
    assert float(resp.headers["x-server-seconds"]) >= 0.0
    assert resp.headers["x-engine"] == "kokoro"
    assert resp.headers["x-audio-format"] in ("wav", "mp3")


@pytest.mark.asyncio
async def test_synthesize_local_returns_wav_media_type(client, local_tts_stub):
    """Local backend (no ElevenLabs key) must advertise audio/wav, not audio/mpeg."""
    resp = await client.post(
        "/api/v1/synthesize",
        json={"text": "Hello world", "voice": "default"},
    )
    assert resp.status_code == 200
    content_type = resp.headers["content-type"]
    assert "audio/wav" in content_type, (
        f"Expected audio/wav for local backend, got {content_type!r}"
    )


@pytest.mark.asyncio
async def test_synthesize_text_too_long(client):
    resp = await client.post(
        "/api/v1/synthesize",
        json={"text": "x" * 5001, "voice": "default"},
    )
    assert resp.status_code == 422  # pydantic validation error


@pytest.mark.asyncio
async def test_synthesize_voice_settings_out_of_range(client):
    """stability > 1.0 must be rejected."""
    resp = await client.post(
        "/api/v1/synthesize",
        json={"text": "hello", "voice": "default", "settings": {"stability": 1.5}},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_synthesize_speed_out_of_range(client):
    resp = await client.post(
        "/api/v1/synthesize",
        json={"text": "hi", "voice": "default", "settings": {"speed": 0.1}},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/compare
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_returns_two_clips(client, local_tts_stub):
    resp = await client.post(
        "/api/v1/compare",
        json={"text": "Testing", "voice_a": "default", "voice_b": "default"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Both clips are present (base64 string or null)
    assert "voice_a" in data
    assert "voice_b" in data


@pytest.mark.asyncio
async def test_compare_base64_decodable(client, local_tts_stub):
    import base64

    resp = await client.post(
        "/api/v1/compare",
        json={"text": "Hello", "voice_a": "default", "voice_b": "default"},
    )
    assert resp.status_code == 200
    data = resp.json()
    if data["voice_a"] is not None:
        decoded = base64.b64decode(data["voice_a"])
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# POST /api/v1/clone — consent gate + keyless 501
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clone_without_consent_returns_400(client):
    resp = await client.post(
        "/api/v1/clone",
        data={"consent": "false", "voice_name": "Test"},
        files={"sample": ("test.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")},
    )
    assert resp.status_code == 400
    assert "consent" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_clone_with_consent_no_key_returns_501(client):
    """With consent=true, no ElevenLabs key, and chatterbox absent → 501."""
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "chatterbox":
            return None
        return original_find_spec(name, *args, **kwargs)

    # Patch validate_audio (no-op) so minimal fake WAV passes validation.
    # Patch find_spec so chatterbox appears absent — exercises ElevenLabs fallback path.
    with patch("importlib.util.find_spec", side_effect=_fake_find_spec):
        with patch("voicelab.validation.validate_audio", return_value=None):
            resp = await client.post(
                "/api/v1/clone",
                data={"consent": "true", "voice_name": "Test"},
                files={"sample": ("test.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")},
            )
    assert resp.status_code == 501
    detail = resp.json()["detail"].lower()
    assert "elevenlabs" in detail or "api key" in detail or "clone" in detail


# ---------------------------------------------------------------------------
# CORS — not wildcard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_not_wildcard(client):
    """CORS allow_origins must not be ['*'] in the app config."""
    from voicelab.config.settings import get_settings

    settings = get_settings()
    assert settings.cors_origins != ["*"], (
        "cors_origins must not be wildcard ['*'] — use explicit origins"
    )
    # Also verify it's a list of strings
    assert isinstance(settings.cors_origins, list)
    for origin in settings.cors_origins:
        assert isinstance(origin, str)


# ---------------------------------------------------------------------------
# Auth — 401 when token set + missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_401_when_token_set_and_missing(monkeypatch):
    """When auth_token is configured, missing bearer → 401."""
    import voicelab.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_settings", None)
    monkeypatch.setenv("AUTH_TOKEN", "super-secret-token")

    from voicelab.app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        resp = await c.get("/api/v1/voices")  # no Authorization header
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_passes_with_correct_token(monkeypatch):
    """Correct bearer token → 200."""
    import voicelab.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_settings", None)
    monkeypatch.setenv("AUTH_TOKEN", "my-token")

    from voicelab.app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        resp = await c.get(
            "/api/v1/voices",
            headers={"Authorization": "Bearer my-token"},
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /readyz
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readyz_200_with_backend(client):
    resp = await client.get("/readyz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backend" in data
    assert data["backend"]  # non-empty string


# ---------------------------------------------------------------------------
# /metrics — prometheus text format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_prometheus_format(client):
    resp = await client.get("/metrics")
    # /metrics may not be mounted if prometheus_fastapi_instrumentator is absent
    # In that case we skip — we're testing graceful degradation, not hard requirement
    if resp.status_code == 404:
        pytest.skip("prometheus_fastapi_instrumentator not installed — /metrics not exposed")
    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    # Prometheus text format
    assert "text/plain" in content_type or "openmetrics" in content_type


# ---------------------------------------------------------------------------
# Request-ID header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_id_header_present(client):
    resp = await client.get("/healthz")
    assert "x-request-id" in resp.headers, "X-Request-ID header must be present on every response"
    # Should be a valid UUID4 format
    import uuid

    try:
        uuid.UUID(resp.headers["x-request-id"], version=4)
    except ValueError:
        pytest.fail(f"X-Request-ID is not a valid UUID4: {resp.headers['x-request-id']!r}")


# ---------------------------------------------------------------------------
# Oversize clone sample rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clone_oversize_sample_rejected(client):
    """Upload a file that exceeds the 15 MB limit → 400/422."""
    # Create a fake 16 MB file content in memory — no real audio just lots of bytes
    big_data = b"x" * (16 * 1024 * 1024)

    # We need consent=true and a key to reach the validation step.
    # Without a key we hit 501 before size check — so we mock the settings.
    import voicelab.config.settings as settings_mod
    from voicelab.app.main import create_app

    original_settings = settings_mod._settings

    # Temporarily inject a fake key so we get past the keyless gate
    fake_settings = settings_mod.Settings(elevenlabs_api_key="fake-key-for-size-test")
    settings_mod._settings = fake_settings

    try:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as c:
            resp = await c.post(
                "/api/v1/clone",
                data={"consent": "true", "voice_name": "BigVoice"},
                files={"sample": ("big.wav", big_data, "audio/wav")},
            )
        # Should be rejected before hitting ElevenLabs
        assert resp.status_code in (400, 422), (
            f"Expected 400/422 for oversized upload, got {resp.status_code}"
        )
    finally:
        settings_mod._settings = original_settings
