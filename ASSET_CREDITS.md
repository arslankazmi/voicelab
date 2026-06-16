# Asset & Dependency Credits

This project stands on the work of others. Thank you to:

## Voice services

- **ElevenLabs** — cloud text-to-speech synthesis and voice cloning API
  - Used via the `elevenlabs` Python SDK (`voicelab[cloud]` optional extra)
  - Key-gated: requires `ELEVENLABS_API_KEY` to activate
  - Voice cloning additionally requires an ElevenLabs Professional plan or higher
  - https://elevenlabs.io

## Libraries

- **FastAPI** (Sebastián Ramírez) — web framework powering the backend and `/healthz` endpoint. MIT.
  - https://fastapi.tiangolo.com
- **Gradio** (Hugging Face) — Blocks UI mounted on FastAPI at `/`. Apache-2.0.
  - https://gradio.app
- **pydantic-settings** — environment variable + `.env` loading for settings. MIT.
  - https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- **PyYAML** — YAML parser for `personas.yaml`. MIT.
  - https://pyyaml.org
- **Uvicorn** — ASGI server. BSD.
  - https://www.uvicorn.org
- **pyttsx3** (optional) — system TTS engine used by `LocalTts` when no ElevenLabs key is present. Mozilla Public License 2.0.
  - https://pyttsx3.readthedocs.io

## Trademark notice

ElevenLabs is a trademark of ElevenLabs, Inc. VoiceLab is not affiliated with or endorsed by ElevenLabs. All other trademarks are the property of their respective owners.
