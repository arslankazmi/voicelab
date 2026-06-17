# voicelab — Key Setup

---

## What runs keyless

Voicelab boots and runs without any API key. The **local TTS fallback** (`voicelab/tts/local.py`) uses `pyttsx3` — a system TTS engine with no network calls. If `pyttsx3` is not installed, TTS synthesis returns `None` and the UI continues in text-only mode. Voice listing, the Gradio playground, and all metrics endpoints are fully available keyless.

---

## Gate 1 — ElevenLabs (real TTS + voice cloning)

| Env var | Required | What it unlocks |
|---|---|---|
| `ELEVENLABS_API_KEY` | Yes | Real TTS synthesis via ElevenLabs cloud; full voice library; voice cloning (see caveat below) |

### Steps

1. Create a `.env` file in the repo root (already in `.gitignore`):

```bash
ELEVENLABS_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxx
```

2. Install the ElevenLabs SDK via the `cloud` extra:

```bash
uv sync --extra cloud
```

This installs `elevenlabs>=1.0` as declared in `pyproject.toml` `[project.optional-dependencies]`.

3. Restart the server. The engine will auto-select `ElevenLabsTts` when the key is present.

### Voice cloning

`clone_voice()` requires:

- **Consent from the voice owner** — you must have explicit permission to clone the voice.
- **ElevenLabs Professional plan or higher** — the `voices.add` API endpoint is not available on the free tier.

---

## Optional deploy-time settings

| Env var | Default | Purpose |
|---|---|---|
| `AUTH_TOKEN` | `None` (disabled) | Bearer token guard for public deployments. |
| `CORS_ORIGINS` | `["http://localhost:8001"]` | Allowed CORS origins. |

---

> **Caveat**: The ElevenLabs cloud path is built and covered by unit/mock tests but has not been exercised against the live API yet. The first live call may require minor adjustments (e.g. voice ID format, model ID).
