"""voicelab — ElevenLabs voice/persona design + comparison playground.

Privacy first: disable every external telemetry/analytics channel *before* any
third-party library (gradio, huggingface_hub) is imported anywhere in the process.
Nothing here phones home.
"""

from __future__ import annotations

import os as _os

# Must run at import time, before gradio/HF libs are imported by any submodule.
_os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("DISABLE_TELEMETRY", "1")
_os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# OOM safety on shared-memory Macs: cap MPS (Apple GPU) allocation so torch
# raises a catchable error instead of exhausting unified memory and crashing the
# host, and let unsupported ops fall back to CPU. Overridable via env.
_os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

__all__ = ["__version__"]
