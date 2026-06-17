# VoiceLab — Security

## Threat Model

VoiceLab is a local/self-hosted playground. The primary threats are:

1. Unauthorised use of a paid ElevenLabs API key (cost amplification)
2. Unbounded uploads consuming server resources or being forwarded to ElevenLabs
3. Injection through text or voice-name fields
4. Secret leakage via logs
5. Voice impersonation through the cloning feature

---

## CORS

Allowed origins are configured via `CORS_ORIGINS` (default: `["http://localhost:8001"]`). Set this to the exact origin of your deployment — do not use `"*"` in any environment where the ElevenLabs API key is active.

`app/main.py` wires `CORSMiddleware` with the setting:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Authentication

All `/api/v1/*` routes have the `require_auth` dependency applied at the router level.

When `AUTH_TOKEN` is unset (default), all requests are accepted — suitable for local/internal use only.

When `AUTH_TOKEN` is set, every request to `/api/v1/*` must include:

```
Authorization: Bearer <token>
```

Missing or incorrect tokens return `HTTP 401 Unauthorized`. The Gradio UI (`/`) and ops endpoints (`/healthz`, `/readyz`, `/metrics`) are NOT protected by bearer auth.

**Recommendation:** Set `AUTH_TOKEN` whenever the server is reachable from any network beyond `localhost`.

---

## Rate Limiting

Rate limiting is provided by `slowapi` (installed by default). Limits are applied per remote IP:

| Endpoint | Limit | Reason |
|---|---|---|
| `GET /api/v1/voices` | 60/minute | List call; low cost |
| `GET /api/v1/personas` | 60/minute | Static data |
| `POST /api/v1/synthesize` | 10/minute | Hits ElevenLabs paid API per call |
| `POST /api/v1/compare` | 60/minute | Two synthesis calls per request |
| `POST /api/v1/clone` | 5/minute | Expensive API + Professional-plan resource |

Exceeded limits return `HTTP 429` with a `{"detail": "Rate limit exceeded: ..."}` body.

**Note on /compare:** `/compare` is rated at 60/minute despite triggering two synthesis calls. Consider lowering this if cost control is a priority for your account tier.

If `slowapi` is unavailable (e.g. stripped from a custom image), all limits become no-ops — a warning is logged at startup.

---

## Upload Limits (clone endpoint)

`POST /api/v1/clone` accepts a multipart audio upload. Validation is applied in `validation.py`:

| Limit | Value | Enforcement |
|---|---|---|
| File size | 15 MB | `os.path.getsize()` |
| Audio duration | 120 s | `soundfile.info().duration` (skipped if soundfile not installed) |
| Accepted extensions | `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac` | Suffix check for temp file naming only |

Files that exceed either limit return `HTTP 422 Unprocessable Entity` before being forwarded to ElevenLabs.

**Note:** Duration validation requires `soundfile` (`uv pip install 'voicelab[audio]'`). Without it, only file size is checked.

---

## Input Validation

Text synthesis input is validated at the Pydantic model layer in `api/v1.py`:

| Field | Constraint |
|---|---|
| `text` | `max_length=5000` characters |
| `stability` | `float`, 0.0 – 1.0 |
| `similarity_boost` | `float`, 0.0 – 1.0 |
| `style` | `float`, 0.0 – 1.0 |
| `speed` | `float`, 0.25 – 4.0 |

The `VoiceSettings` and `SynthesizeRequest` models both use `extra="ignore"` so unrecognised fields are silently dropped rather than causing errors or being forwarded downstream.

`voice` fields are passed directly to the ElevenLabs API as `voice_id`. ElevenLabs validates ownership — an invalid or unowned voice_id returns a non-transient 4xx which surfaces as `HTTP 503` to the client.

---

## Secret Handling

`ELEVENLABS_API_KEY` must be supplied via environment variable or `.env` file. It is:

- Never logged directly
- Filtered at the logging layer by `SecretRedactionFilter` in `logging_config.py`:
  - Patterns matched and replaced with `[REDACTED]`:
    - `sk-<10+ alphanumeric chars>`
    - `Bearer <10+ alphanumeric chars>`
    - `key=<20+ alphanumeric chars>` and `token=<20+ alphanumeric chars>`
- Never returned in any API response body
- Passed only to the ElevenLabs SDK, which transmits it as an HTTPS request header

Do not pass the key as a URL query parameter — it will not be redacted from URL-level access logs.

---

## Biometric Data / Voice Cloning

Voice audio is biometric data. The cloning feature presents significant misuse risk (impersonation, deepfakes).

### Consent gate

The `/api/v1/clone` endpoint requires an explicit `consent=true` form field. Requests with `consent=false` or the field absent receive `HTTP 400`:

```
{"detail": "Consent required for voice cloning"}
```

The Gradio UI only renders the clone form when `ELEVENLABS_API_KEY` is set; no hidden form is rendered in keyless mode.

### Misuse risk

Cloning a voice without the subject's knowledge or consent can enable:
- Impersonation fraud
- Non-consensual audio deepfakes
- Identity-based social engineering attacks

### Controls in place

| Control | Implementation |
|---|---|
| Explicit consent flag | `consent=true` required in every `/clone` request |
| API key gate | Cloning is impossible without `ELEVENLABS_API_KEY` |
| Rate limit | 5 requests/minute per IP |
| Upload size/duration limits | 15 MB / 120 s max |

### Operator responsibilities

- Ensure informed consent from the voice owner before any cloning call
- Do not expose the clone endpoint to untrusted clients without additional identity verification
- Review ElevenLabs' own policies on cloned voice usage and data retention

See also [PRIVACY.md](PRIVACY.md) for data handling details.

---

## Dependency Supply Chain

Production dependencies are pinned via `uv.lock`. The `[cloud]` extra (`elevenlabs>=1.0`) is optional — the base image has no cloud SDK, reducing attack surface for keyless deployments.

The `prometheus-fastapi-instrumentator` package is listed in `pyproject.toml` but is NOT used for the `/metrics` endpoint (its middleware walker is incompatible with Gradio's router). `prometheus_client` is used directly instead. The instrumentator is a dead dependency and may be removed in a future version.
