# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-17

### Added
- Gradio + FastAPI UI with Synthesize, Compare, and Voice Cloning tabs
- ElevenLabs cloud TTS backend (gated on `ELEVENLABS_API_KEY` + `[cloud]` extra)
- Local system TTS fallback via pyttsx3 (keyless / CI mode)
- Persona preset system (`personas.yaml`) with 5 built-in presets
- `voicelab serve` CLI with `--port` and `--host` flags
- `/healthz` health-check endpoint
- Telemetry disabled at import time (Gradio analytics, HuggingFace Hub)
- Production hardening: per-request unique temp files (fixes concurrent-user race), ruff lint/format, mypy, pre-commit, CI workflow, Docker + Compose
