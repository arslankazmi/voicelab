"""FastAPI application factory — voicelab.

Mounts a Gradio UI at "/" and exposes REST at "/healthz".

Usage::

    uv run voicelab serve
    # or directly:
    uv run python -m voicelab.app.main
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from voicelab.config.settings import get_settings
from voicelab.engine import get_tts, synthesize_one
from voicelab.personas import load_personas, get_persona

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Application lifespan — startup/shutdown hooks."""
    yield


# ---------------------------------------------------------------------------
# Backend banner helpers
# ---------------------------------------------------------------------------

def _backend_label() -> str:
    settings = get_settings()
    if settings.elevenlabs_api_key:
        return "ElevenLabs"
    return "Local (system TTS fallback)"


def _backend_banner_md() -> str:
    label = _backend_label()
    if label == "ElevenLabs":
        return "**Active backend:** ElevenLabs (cloud) ✓"
    return (
        "**Active backend:** Local system TTS (pyttsx3 fallback) — "
        "set `ELEVENLABS_API_KEY` to unlock cloud voices."
    )


# ---------------------------------------------------------------------------
# Voice list helper
# ---------------------------------------------------------------------------

def _get_voices() -> list[str]:
    """Return list of voice names/ids from the active backend."""
    try:
        tts = get_tts()
        voices = tts.list_voices()
        return [v.get("name") or v.get("id", "default") for v in voices] or ["default"]
    except Exception:
        logger.warning("Could not list voices", exc_info=True)
        return ["default"]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _build_gradio_ui() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    personas = load_personas()
    persona_names = [p.name for p in personas]
    voice_choices = _get_voices()
    is_cloud = get_settings().elevenlabs_api_key is not None

    with gr.Blocks(
        title="VoiceLab",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# VoiceLab — Voice & Persona Design Playground")
        gr.Markdown(_backend_banner_md())

        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab 1: Synthesize
            # ---------------------------------------------------------------
            with gr.TabItem("Synthesize"):
                gr.Markdown("### Design a voice persona and preview it.")
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Text",
                            placeholder="Enter text to synthesize…",
                            lines=4,
                        )
                        persona_dd = gr.Dropdown(
                            choices=["(none)"] + persona_names,
                            value="(none)",
                            label="Persona preset",
                        )
                        persona_desc = gr.Markdown("")
                        voice_dd = gr.Dropdown(
                            choices=voice_choices,
                            value=voice_choices[0] if voice_choices else "default",
                            label="Voice",
                        )
                    with gr.Column(scale=1):
                        stability_sl = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Stability")
                        similarity_sl = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Similarity Boost")
                        style_sl = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Style")
                        speed_sl = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")

                synth_btn = gr.Button("Synthesize", variant="primary")
                synth_status = gr.Markdown("")
                synth_audio = gr.Audio(label="Output", type="numpy", interactive=False)

                # Persona auto-fill
                def on_persona_change(persona_name):
                    if persona_name == "(none)":
                        return "", 0.75, 0.75, 0.0, 1.0, ""
                    p = get_persona(persona_name)
                    if p is None:
                        return "", 0.75, 0.75, 0.0, 1.0, ""
                    desc = f"*{p.description}*\n\n**Sample line:** {p.sample_line}"
                    s = p.default_settings
                    return desc, s.stability, s.similarity_boost, s.style, s.speed, p.sample_line

                persona_dd.change(
                    on_persona_change,
                    inputs=[persona_dd],
                    outputs=[persona_desc, stability_sl, similarity_sl, style_sl, speed_sl, text_input],
                )

                def on_synthesize(text, voice, stability, similarity, style, speed):
                    if not text.strip():
                        return "⚠ Please enter some text.", None
                    settings = {
                        "stability": stability,
                        "similarity_boost": similarity,
                        "style": style,
                        "speed": speed,
                    }
                    audio_bytes = synthesize_one(text, voice, settings)
                    if audio_bytes is None:
                        return "⚠ Synthesis returned no audio (check backend/voice settings).", None
                    # Convert bytes to numpy array for gr.Audio
                    import numpy as np
                    import struct
                    # Parse WAV or treat as raw PCM if MP3-like (ElevenLabs returns MP3)
                    try:
                        import wave
                        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                            sample_rate = wf.getframerate()
                            n_frames = wf.getnframes()
                            raw = wf.readframes(n_frames)
                            n_channels = wf.getnchannels()
                            sampwidth = wf.getsampwidth()
                            if sampwidth == 2:
                                arr = np.frombuffer(raw, dtype=np.int16)
                            else:
                                arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
                            if n_channels == 2:
                                arr = arr[::2]  # take left channel
                        return "✓ Synthesis complete.", (sample_rate, arr)
                    except Exception:
                        # MP3 or other format — return raw bytes as temp file path
                        tmp_path = "/tmp/voicelab_out.mp3"
                        with open(tmp_path, "wb") as f:
                            f.write(audio_bytes)
                        return "✓ Synthesis complete (MP3).", tmp_path

                synth_btn.click(
                    on_synthesize,
                    inputs=[text_input, voice_dd, stability_sl, similarity_sl, style_sl, speed_sl],
                    outputs=[synth_status, synth_audio],
                )

            # ---------------------------------------------------------------
            # Tab 2: Compare
            # ---------------------------------------------------------------
            with gr.TabItem("Compare"):
                gr.Markdown("### Compare two voices side-by-side.")
                cmp_text = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to compare…",
                    lines=3,
                    value="The quick brown fox jumps over the lazy dog.",
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Voice A**")
                        voice_a = gr.Dropdown(
                            choices=voice_choices,
                            value=voice_choices[0] if voice_choices else "default",
                            label="Voice A",
                        )
                        persona_a = gr.Dropdown(
                            choices=["(none)"] + persona_names,
                            value="(none)",
                            label="Persona A",
                        )
                        stab_a = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Stability A")
                        sim_a = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Similarity A")
                        style_a = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Style A")
                        speed_a = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed A")
                    with gr.Column():
                        gr.Markdown("**Voice B**")
                        voice_b = gr.Dropdown(
                            choices=voice_choices,
                            value=voice_choices[-1] if len(voice_choices) > 1 else (voice_choices[0] if voice_choices else "default"),
                            label="Voice B",
                        )
                        persona_b = gr.Dropdown(
                            choices=["(none)"] + persona_names,
                            value="(none)",
                            label="Persona B",
                        )
                        stab_b = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Stability B")
                        sim_b = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Similarity B")
                        style_b = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Style B")
                        speed_b = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed B")

                # Persona auto-fill for compare tab
                def _fill_persona_settings(persona_name):
                    if persona_name == "(none)":
                        return 0.75, 0.75, 0.0, 1.0
                    p = get_persona(persona_name)
                    if p is None:
                        return 0.75, 0.75, 0.0, 1.0
                    s = p.default_settings
                    return s.stability, s.similarity_boost, s.style, s.speed

                persona_a.change(_fill_persona_settings, inputs=[persona_a], outputs=[stab_a, sim_a, style_a, speed_a])
                persona_b.change(_fill_persona_settings, inputs=[persona_b], outputs=[stab_b, sim_b, style_b, speed_b])

                cmp_btn = gr.Button("Compare", variant="primary")
                cmp_status = gr.Markdown("")
                with gr.Row():
                    audio_out_a = gr.Audio(label="Voice A Output", type="numpy", interactive=False)
                    audio_out_b = gr.Audio(label="Voice B Output", type="numpy", interactive=False)

                def on_compare(text, va, sa, sima, stya, spda, vb, sb, simb, styb, spdb):
                    if not text.strip():
                        return "⚠ Please enter some text.", None, None

                    def _bytes_to_audio(audio_bytes):
                        if audio_bytes is None:
                            return None
                        import numpy as np
                        import wave
                        try:
                            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                                sr = wf.getframerate()
                                raw = wf.readframes(wf.getnframes())
                                nc = wf.getnchannels()
                                sw = wf.getsampwidth()
                                arr = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.uint8).astype(np.int16)
                                if nc == 2:
                                    arr = arr[::2]
                            return (sr, arr)
                        except Exception:
                            p = "/tmp/voicelab_cmp.mp3"
                            with open(p, "wb") as f:
                                f.write(audio_bytes)
                            return p

                    settings_a = {"stability": sa, "similarity_boost": sima, "style": stya, "speed": spda}
                    settings_b = {"stability": sb, "similarity_boost": simb, "style": styb, "speed": spdb}
                    tts = get_tts()
                    res_a = tts.synthesize(text, va, settings_a)
                    res_b = tts.synthesize(text, vb, settings_b)

                    out_a = _bytes_to_audio(res_a)
                    out_b = _bytes_to_audio(res_b)
                    status = "✓ Compare complete." if (res_a or res_b) else "⚠ Both synthesises failed."
                    return status, out_a, out_b

                cmp_btn.click(
                    on_compare,
                    inputs=[cmp_text, voice_a, stab_a, sim_a, style_a, speed_a, voice_b, stab_b, sim_b, style_b, speed_b],
                    outputs=[cmp_status, audio_out_a, audio_out_b],
                )

            # ---------------------------------------------------------------
            # Tab 3: Voice Cloning
            # ---------------------------------------------------------------
            with gr.TabItem("Voice Cloning"):
                if is_cloud:
                    gr.Markdown(
                        "### Clone a voice from an audio sample.\n"
                        "_Requires ElevenLabs Professional plan or higher._"
                    )
                    clone_sample = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Upload audio sample (WAV / MP3)",
                    )
                    clone_name = gr.Textbox(label="Voice name", placeholder="My Custom Voice")
                    clone_btn = gr.Button("Clone Voice", variant="primary")
                    clone_status = gr.Markdown("")

                    def on_clone(sample_path, name):
                        if not sample_path:
                            return "⚠ Please upload an audio sample."
                        if not name.strip():
                            return "⚠ Please enter a voice name."
                        from voicelab.tts.elevenlabs import ElevenLabsTts
                        settings = get_settings()
                        tts = ElevenLabsTts(api_key=settings.elevenlabs_api_key)
                        voice_id = tts.clone_voice(sample_path, name)
                        if voice_id:
                            return f"✓ Voice cloned! voice_id = `{voice_id}`"
                        return "⚠ Voice cloning failed — check your ElevenLabs plan tier."

                    clone_btn.click(on_clone, inputs=[clone_sample, clone_name], outputs=[clone_status])
                else:
                    gr.Markdown(
                        "### Voice Cloning — Requires ElevenLabs Key\n\n"
                        "This feature is **disabled** in keyless mode.\n\n"
                        "To enable:\n"
                        "1. Get an ElevenLabs API key at https://elevenlabs.io\n"
                        "2. Set `ELEVENLABS_API_KEY=your_key` in your `.env` file\n"
                        "3. Install the cloud extras: `uv pip install 'voicelab[cloud]'`\n"
                        "4. Restart the server\n\n"
                        "_Voice cloning also requires an ElevenLabs Professional plan or higher._"
                    )

    return demo


def create_app() -> FastAPI:
    """Factory that builds and wires the full FastAPI application."""
    app = FastAPI(
        title="VoiceLab",
        description="ElevenLabs voice/persona design + comparison playground.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict:
        return {"status": "ok", "backend": _backend_label()}

    gradio_app = _build_gradio_ui()
    app = gr.mount_gradio_app(app, gradio_app, path="/")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("voicelab.app.main:app", host="0.0.0.0", port=s.app_port, reload=False)
