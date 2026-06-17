"""CLI entry point — voicelab serve."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voicelab",
        description="VoiceLab — ElevenLabs voice/persona design + comparison playground.",
    )
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the VoiceLab server.")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: from config.yaml or 8001).",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0).",
    )

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        from voicelab.config.settings import get_settings

        settings = get_settings()
        port = args.port or settings.app_port
        uvicorn.run(
            "voicelab.app.main:app",
            host=args.host,
            port=port,
            reload=False,
        )
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
