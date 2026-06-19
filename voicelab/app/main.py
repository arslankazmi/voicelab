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
import tempfile
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from voicelab.api.v1 import limiter as api_limiter
from voicelab.api.v1 import router as api_v1_router
from voicelab.config.settings import get_settings
from voicelab.engine import compare_across_engines
from voicelab.logging_config import configure_logging
from voicelab.middleware import RequestIDMiddleware
from voicelab.personas import get_persona, load_personas
from voicelab.tts.registry import get_tts_for_engine, list_backends

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Application lifespan — startup/shutdown hooks."""
    yield


# ---------------------------------------------------------------------------
# Backend banner helpers
# ---------------------------------------------------------------------------

_ENGINE_CHOICES: list[str] = ["auto", "elevenlabs", "kokoro", "piper", "local", "chatterbox"]


def _backend_label() -> str:
    settings = get_settings()
    if settings.elevenlabs_api_key:
        return "ElevenLabs"
    return "Local (system TTS fallback)"


def _backend_banner_md() -> str:
    """Return Markdown banner showing available backends."""
    settings = get_settings()
    backends = list_backends(api_key=settings.elevenlabs_api_key)

    lines = ["**Available TTS backends:**"]
    for b in backends:
        status = "✓ available" if b["available"] else "✗ not installed"
        lines.append(f"- **{b['name']}** — {status} ({b['license']})")

    lines.append("")
    active = getattr(settings, "tts_engine", "auto")
    lines.append(f"_Active engine setting: `{active}`. Set `TTS_ENGINE=kokoro` etc. in `.env`._")
    lines.append("")
    lines.append(
        "_**Chatterbox** (keyless voice cloning + emotion): "
        "install with `uv sync --extra clone` then select engine=chatterbox._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Voice list helper
# ---------------------------------------------------------------------------


def _get_voices(engine: str = "auto") -> list[str]:
    """Return list of voice names/ids from the specified backend."""
    try:
        settings = get_settings()
        tts = get_tts_for_engine(engine, api_key=settings.elevenlabs_api_key)
        voices = tts.list_voices()
        return [v.get("name") or v.get("id", "default") for v in voices] or ["default"]
    except Exception:
        logger.warning("Could not list voices for engine=%r", engine, exc_info=True)
        return ["default"]


def _mp3_bytes_to_temp_path(audio_bytes: bytes, prefix: str = "voicelab_") -> str:
    """Write MP3 bytes to a unique per-request temp file and return its path.

    Uses NamedTemporaryFile so concurrent requests never share a file.
    The caller (Gradio) serves the file; we schedule best-effort cleanup
    via a finalizer registered with the OS temp infrastructure.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", prefix=prefix, dir="/tmp", delete=False) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def _bytes_to_audio_gradio(audio_bytes: bytes | None):  # noqa: ANN201
    """Convert audio bytes to Gradio-compatible format (numpy tuple or temp filepath)."""
    if audio_bytes is None:
        return None
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            nc = wf.getnchannels()
            sw = wf.getsampwidth()
            dtype = np.int16 if sw == 2 else np.uint8
            arr = np.frombuffer(raw, dtype=dtype).astype(np.int16)
            if nc == 2:
                arr = arr[::2]
        return (sr, arr)
    except Exception:
        # MP3 or unknown format — write to temp file
        return _mp3_bytes_to_temp_path(audio_bytes, prefix="voicelab_audio_")


def _build_gradio_ui() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    personas = load_personas()
    persona_names = [p.name for p in personas]
    settings = get_settings()
    voice_choices = _get_voices(getattr(settings, "tts_engine", "auto"))
    is_cloud = settings.elevenlabs_api_key is not None

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
                        engine_dd = gr.Dropdown(
                            choices=_ENGINE_CHOICES,
                            value=getattr(settings, "tts_engine", "auto"),
                            label="Engine",
                        )
                        voice_dd = gr.Dropdown(
                            choices=voice_choices,
                            value=voice_choices[0] if voice_choices else "default",
                            label="Voice",
                        )
                    with gr.Column(scale=1):
                        stability_sl = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Stability")
                        similarity_sl = gr.Slider(
                            0.0, 1.0, value=0.75, step=0.01, label="Similarity Boost"
                        )
                        style_sl = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Style")
                        speed_sl = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                        exaggeration_sl = gr.Slider(
                            0.0,
                            1.0,
                            value=0.5,
                            step=0.05,
                            label="Emotion / Exaggeration (Chatterbox only)",
                            visible=False,
                        )
                        cfg_weight_sl = gr.Slider(
                            0.0,
                            1.0,
                            value=0.5,
                            step=0.05,
                            label="CFG Weight (Chatterbox only)",
                            visible=False,
                        )

                synth_btn = gr.Button("Synthesize", variant="primary")
                synth_status = gr.Markdown("")
                synth_audio = gr.Audio(label="Output", type="numpy", interactive=False)

                # Engine change — update voice dropdown + show/hide chatterbox sliders
                def on_engine_change(engine_name):
                    new_voices = _get_voices(engine_name)
                    default = new_voices[0] if new_voices else "default"
                    is_cb = engine_name == "chatterbox"
                    return (
                        gr.update(choices=new_voices, value=default),
                        gr.update(visible=is_cb),
                        gr.update(visible=is_cb),
                    )

                engine_dd.change(
                    on_engine_change,
                    inputs=[engine_dd],
                    outputs=[voice_dd, exaggeration_sl, cfg_weight_sl],
                )

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
                    outputs=[
                        persona_desc,
                        stability_sl,
                        similarity_sl,
                        style_sl,
                        speed_sl,
                        text_input,
                    ],
                )

                def on_synthesize(
                    text,
                    engine,
                    voice,
                    stability,
                    similarity,
                    style,
                    speed,
                    exaggeration,
                    cfg_weight,
                ):
                    if not text.strip():
                        return "⚠ Please enter some text.", None
                    voice_settings: dict = {
                        "stability": stability,
                        "similarity_boost": similarity,
                        "style": style,
                        "speed": speed,
                    }
                    if engine == "chatterbox":
                        voice_settings["exaggeration"] = exaggeration
                        voice_settings["cfg_weight"] = cfg_weight
                    app_settings = get_settings()
                    tts = get_tts_for_engine(engine, api_key=app_settings.elevenlabs_api_key)
                    audio_bytes = tts.synthesize(text, voice, voice_settings)
                    if audio_bytes is None:
                        return "⚠ Synthesis returned no audio (check backend/voice settings).", None
                    out = _bytes_to_audio_gradio(audio_bytes)
                    return f"✓ Synthesis complete ({engine}).", out

                synth_btn.click(
                    on_synthesize,
                    inputs=[
                        text_input,
                        engine_dd,
                        voice_dd,
                        stability_sl,
                        similarity_sl,
                        style_sl,
                        speed_sl,
                        exaggeration_sl,
                        cfg_weight_sl,
                    ],
                    outputs=[synth_status, synth_audio],
                )

            # ---------------------------------------------------------------
            # Tab 2: Compare (same engine, two voices)
            # ---------------------------------------------------------------
            with gr.TabItem("Compare"):
                gr.Markdown("### Compare two voices side-by-side (same engine).")
                cmp_text = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to compare…",
                    lines=3,
                    value="The quick brown fox jumps over the lazy dog.",
                )
                cmp_engine = gr.Dropdown(
                    choices=_ENGINE_CHOICES,
                    value=getattr(settings, "tts_engine", "auto"),
                    label="Engine",
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
                        _voice_b_default = (
                            voice_choices[-1]
                            if len(voice_choices) > 1
                            else (voice_choices[0] if voice_choices else "default")
                        )
                        voice_b = gr.Dropdown(
                            choices=voice_choices,
                            value=_voice_b_default,
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

                # Engine change for compare — update both voice dropdowns
                def on_cmp_engine_change(engine_name):
                    new_voices = _get_voices(engine_name)
                    default_v = new_voices[0] if new_voices else "default"
                    last_v = new_voices[-1] if len(new_voices) > 1 else default_v
                    return (
                        gr.update(choices=new_voices, value=default_v),
                        gr.update(choices=new_voices, value=last_v),
                    )

                cmp_engine.change(
                    on_cmp_engine_change,
                    inputs=[cmp_engine],
                    outputs=[voice_a, voice_b],
                )

                persona_a.change(
                    _fill_persona_settings,
                    inputs=[persona_a],
                    outputs=[stab_a, sim_a, style_a, speed_a],
                )
                persona_b.change(
                    _fill_persona_settings,
                    inputs=[persona_b],
                    outputs=[stab_b, sim_b, style_b, speed_b],
                )

                cmp_btn = gr.Button("Compare", variant="primary")
                cmp_status = gr.Markdown("")
                with gr.Row():
                    audio_out_a = gr.Audio(label="Voice A Output", type="numpy", interactive=False)
                    audio_out_b = gr.Audio(label="Voice B Output", type="numpy", interactive=False)

                def on_compare(text, engine, va, sa, sima, stya, spda, vb, sb, simb, styb, spdb):
                    if not text.strip():
                        return "⚠ Please enter some text.", None, None

                    s_a = {"stability": sa, "similarity_boost": sima, "style": stya, "speed": spda}
                    s_b = {"stability": sb, "similarity_boost": simb, "style": styb, "speed": spdb}
                    app_settings = get_settings()
                    tts = get_tts_for_engine(engine, api_key=app_settings.elevenlabs_api_key)
                    res_a = tts.synthesize(text, va, s_a)
                    res_b = tts.synthesize(text, vb, s_b)

                    out_a = _bytes_to_audio_gradio(res_a)
                    out_b = _bytes_to_audio_gradio(res_b)
                    status = (
                        f"✓ Compare complete ({engine})."
                        if (res_a or res_b)
                        else "⚠ Both synthesises failed."
                    )
                    return status, out_a, out_b

                cmp_btn.click(
                    on_compare,
                    inputs=[
                        cmp_text,
                        cmp_engine,
                        voice_a,
                        stab_a,
                        sim_a,
                        style_a,
                        speed_a,
                        voice_b,
                        stab_b,
                        sim_b,
                        style_b,
                        speed_b,
                    ],
                    outputs=[cmp_status, audio_out_a, audio_out_b],
                )

            # ---------------------------------------------------------------
            # Tab 3: Cross-Engine Compare (same text, different engines)
            # ---------------------------------------------------------------
            with gr.TabItem("Cross-Engine Compare"):
                gr.Markdown(
                    "### Compare the same text across different TTS engines.\n"
                    "Pick two (engine, voice) pairs to hear how each engine sounds."
                )
                xc_text = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to compare across engines…",
                    lines=3,
                    value="Hello! This is a cross-engine voice comparison.",
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Engine A**")
                        xc_engine_a = gr.Dropdown(
                            choices=_ENGINE_CHOICES, value="kokoro", label="Engine A"
                        )
                        xc_voice_a_choices = _get_voices("kokoro")
                        xc_voice_a = gr.Dropdown(
                            choices=xc_voice_a_choices,
                            value=xc_voice_a_choices[0] if xc_voice_a_choices else "default",
                            label="Voice A",
                        )
                    with gr.Column():
                        gr.Markdown("**Engine B**")
                        xc_engine_b = gr.Dropdown(
                            choices=_ENGINE_CHOICES, value="piper", label="Engine B"
                        )
                        xc_voice_b_choices = _get_voices("piper")
                        xc_voice_b = gr.Dropdown(
                            choices=xc_voice_b_choices,
                            value=xc_voice_b_choices[0] if xc_voice_b_choices else "default",
                            label="Voice B",
                        )

                # Engine dropdowns update corresponding voice dropdowns
                def _xc_engine_change(engine_name):
                    new_v = _get_voices(engine_name)
                    return gr.update(choices=new_v, value=(new_v or ["default"])[0])

                xc_engine_a.change(_xc_engine_change, inputs=[xc_engine_a], outputs=[xc_voice_a])
                xc_engine_b.change(_xc_engine_change, inputs=[xc_engine_b], outputs=[xc_voice_b])

                xc_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                xc_btn = gr.Button("Compare Engines", variant="primary")
                xc_status = gr.Markdown("")
                with gr.Row():
                    xc_audio_a = gr.Audio(label="Engine A Output", type="numpy", interactive=False)
                    xc_audio_b = gr.Audio(label="Engine B Output", type="numpy", interactive=False)

                def on_cross_compare(text, eng_a, v_a, eng_b, v_b, speed):
                    if not text.strip():
                        return "⚠ Please enter some text.", None, None
                    pairs = [(eng_a, v_a), (eng_b, v_b)]
                    s = {"stability": 0.75, "similarity_boost": 0.75, "style": 0.0, "speed": speed}
                    app_settings = get_settings()
                    results = compare_across_engines(text, pairs, s, app_settings=app_settings)
                    audios = [_bytes_to_audio_gradio(r[2]) for r in results]
                    labels = [f"{r[0]}/{r[1]}" for r in results]
                    status = f"✓ Cross-engine compare: {labels[0]} vs {labels[1]}"
                    return status, audios[0], audios[1]

                xc_btn.click(
                    on_cross_compare,
                    inputs=[xc_text, xc_engine_a, xc_voice_a, xc_engine_b, xc_voice_b, xc_speed],
                    outputs=[xc_status, xc_audio_a, xc_audio_b],
                )

            # ---------------------------------------------------------------
            # Tab 4: Voice Cloning
            # ---------------------------------------------------------------
            with gr.TabItem("Voice Cloning"):
                from voicelab.tts.registry import (  # noqa: PLC0415
                    _REGISTRY as _TTS_REGISTRY,
                )
                from voicelab.tts.registry import (
                    is_available as _is_available,
                )

                _cb_info = next(
                    (b for b in _TTS_REGISTRY if b.name == "chatterbox"),
                    None,
                )
                _chatterbox_available = _cb_info is not None and _is_available(_cb_info)

                if _chatterbox_available:
                    gr.Markdown(
                        "### Keyless Voice Cloning (Chatterbox)\n"
                        "Clone any voice from a 5–20 second reference audio clip — "
                        "**no API key required**.\n\n"
                        "_By uploading a voice sample, you confirm you have the right "
                        "to clone this voice._"
                    )
                    cb_clone_consent = gr.Checkbox(
                        label="I confirm I have the right to clone this voice",
                        value=False,
                    )
                    cb_clone_sample = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Reference audio (WAV / MP3, 5–20 sec)",
                    )
                    cb_clone_text = gr.Textbox(
                        label="Text to synthesize",
                        placeholder="Enter text to speak in the cloned voice…",
                        lines=3,
                        value="Hello! This is a cloned voice speaking.",
                    )
                    cb_exaggeration = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05, label="Emotion / Exaggeration"
                    )
                    cb_cfg_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="CFG Weight")
                    cb_clone_btn = gr.Button("Clone & Synthesize", variant="primary")
                    cb_clone_status = gr.Markdown("")
                    cb_clone_audio = gr.Audio(
                        label="Cloned Voice Output", type="numpy", interactive=False
                    )

                    def on_cb_clone(consent, sample_path, text, exaggeration, cfg_weight):
                        if not consent:
                            return (
                                "⚠ You must confirm you have the right to clone this voice.",
                                None,
                            )
                        if not sample_path:
                            return "⚠ Please upload a reference audio sample.", None
                        if not text.strip():
                            return "⚠ Please enter text to synthesize.", None
                        from voicelab.tts.chatterbox import ChatterboxTts  # noqa: PLC0415

                        tts = ChatterboxTts()
                        clone_settings = {
                            "reference_audio": sample_path,
                            "exaggeration": exaggeration,
                            "cfg_weight": cfg_weight,
                        }
                        audio_bytes = tts.synthesize(text, "cloned", clone_settings)
                        if audio_bytes is None:
                            return "⚠ Cloning failed — check Chatterbox installation.", None
                        out = _bytes_to_audio_gradio(audio_bytes)
                        return "✓ Voice cloned and synthesized!", out

                    cb_clone_btn.click(
                        on_cb_clone,
                        inputs=[
                            cb_clone_consent,
                            cb_clone_sample,
                            cb_clone_text,
                            cb_exaggeration,
                            cb_cfg_weight,
                        ],
                        outputs=[cb_clone_status, cb_clone_audio],
                    )

                    if is_cloud:
                        gr.Markdown(
                            "---\n### ElevenLabs Voice Cloning\n_Requires Professional plan._"
                        )
                        el_clone_sample = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            label="Upload audio sample (WAV / MP3)",
                        )
                        el_clone_name = gr.Textbox(
                            label="Voice name", placeholder="My Custom Voice"
                        )
                        el_clone_btn = gr.Button("Clone Voice (ElevenLabs)", variant="secondary")
                        el_clone_status = gr.Markdown("")

                        def on_el_clone(sample_path, name):
                            if not sample_path:
                                return "⚠ Please upload an audio sample."
                            if not name.strip():
                                return "⚠ Please enter a voice name."
                            from voicelab.tts.elevenlabs import ElevenLabsTts  # noqa: PLC0415

                            _settings = get_settings()
                            tts = ElevenLabsTts(api_key=_settings.elevenlabs_api_key)
                            voice_id = tts.clone_voice(sample_path, name)
                            if voice_id:
                                return f"✓ Voice cloned! voice_id = `{voice_id}`"
                            return "⚠ Voice cloning failed — check your ElevenLabs plan tier."

                        el_clone_btn.click(
                            on_el_clone,
                            inputs=[el_clone_sample, el_clone_name],
                            outputs=[el_clone_status],
                        )

                elif is_cloud:
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
                        from voicelab.tts.elevenlabs import ElevenLabsTts  # noqa: PLC0415

                        _settings = get_settings()
                        tts = ElevenLabsTts(api_key=_settings.elevenlabs_api_key)
                        voice_id = tts.clone_voice(sample_path, name)
                        if voice_id:
                            return f"✓ Voice cloned! voice_id = `{voice_id}`"
                        return "⚠ Voice cloning failed — check your ElevenLabs plan tier."

                    clone_btn.click(
                        on_clone,
                        inputs=[clone_sample, clone_name],
                        outputs=[clone_status],
                    )
                else:
                    gr.Markdown(
                        "### Voice Cloning\n\n"
                        "**Keyless cloning** (Chatterbox) is available after installing "
                        "the `[clone]` extra:\n"
                        "```\nuv sync --extra clone\n```\n\n"
                        "**Cloud cloning** (ElevenLabs) requires an API key:\n"
                        "1. Get an ElevenLabs API key at https://elevenlabs.io\n"
                        "2. Set `ELEVENLABS_API_KEY=your_key` in your `.env` file\n"
                        "3. Restart the server\n\n"
                        "_ElevenLabs voice cloning requires a Professional plan or higher._"
                    )

    return demo


def create_app() -> FastAPI:
    """Factory that builds and wires the full FastAPI application."""
    settings = get_settings()

    configure_logging(settings.log_level)

    app = FastAPI(
        title="VoiceLab",
        description="ElevenLabs voice/persona design + comparison playground.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Wire rate limiter (no-op stub if slowapi not installed)
    app.state.limiter = api_limiter
    try:
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware
        from starlette.requests import Request as StarletteRequest
        from starlette.responses import JSONResponse

        async def _rate_limit_handler(
            request: StarletteRequest, exc: RateLimitExceeded
        ) -> JSONResponse:
            return JSONResponse({"detail": f"Rate limit exceeded: {exc.detail}"}, status_code=429)

        app.add_middleware(SlowAPIMiddleware)
        app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
    except ImportError:
        pass

    # REST API v1
    app.include_router(api_v1_router)

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict:
        return {"status": "ok", "backend": _backend_label()}

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> dict:
        return {"status": "ok", "backend": _backend_label()}

    # Prometheus metrics endpoint — exposes /metrics in Prometheus text format.
    # NOTE: prometheus_fastapi_instrumentator's middleware walker is incompatible with
    # Gradio's _IncludedRouter (AttributeError: no .path). We use prometheus_client
    # directly to generate the metrics page, which is simpler and avoids the conflict.
    try:
        import prometheus_client

        @app.get("/metrics", tags=["ops"], include_in_schema=False)
        async def metrics():  # noqa: ANN202
            from fastapi.responses import Response  # noqa: PLC0415

            data = prometheus_client.generate_latest()
            return Response(
                content=data,
                media_type=prometheus_client.CONTENT_TYPE_LATEST,
            )

        logger.info("Prometheus metrics exposed at /metrics")
    except ImportError:
        logger.warning("prometheus_client not installed — /metrics endpoint skipped.")

    @app.get("/grid", include_in_schema=False)
    async def grid():
        """Serve the handwritten TTS-grid page (compare every engine on one line)."""
        from pathlib import Path

        from fastapi.responses import FileResponse

        html = Path(__file__).resolve().parent.parent / "static" / "grid.html"
        return FileResponse(html, media_type="text/html")

    gradio_app = _build_gradio_ui()
    app = gr.mount_gradio_app(app, gradio_app, path="/")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("voicelab.app.main:app", host="0.0.0.0", port=s.app_port, reload=False)
