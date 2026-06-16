"""Smoke tests — app boots, /healthz responds, telemetry disabled."""

from __future__ import annotations

import os

import pytest
from httpx import AsyncClient, ASGITransport


def test_gradio_analytics_disabled():
    """GRADIO_ANALYTICS_ENABLED must be falsy after importing voicelab."""
    import voicelab  # noqa: F401 — triggers __init__ telemetry-off boot

    val = os.environ.get("GRADIO_ANALYTICS_ENABLED", "")
    assert val.lower() in ("false", "0", ""), (
        f"GRADIO_ANALYTICS_ENABLED={val!r} — must be falsy"
    )


def test_hf_telemetry_disabled():
    import voicelab  # noqa: F401

    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
    assert os.environ.get("DISABLE_TELEMETRY") == "1"


@pytest.mark.asyncio
async def test_healthz_returns_ok():
    """GET /healthz returns 200 with status=ok."""
    from voicelab.app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backend" in data


@pytest.mark.asyncio
async def test_healthz_backend_key_keyless():
    """In keyless mode healthz.backend should mention local/fallback."""
    from voicelab.app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/healthz")
    data = resp.json()
    # Backend should be the local fallback label when no key
    assert "Local" in data["backend"] or "ElevenLabs" in data["backend"]
