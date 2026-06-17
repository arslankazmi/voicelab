# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install production deps only (no dev group)
RUN uv sync --no-dev --frozen

# Copy application source
COPY voicelab/ voicelab/
COPY config.yaml ./

# Non-root user
RUN useradd --system --create-home --uid 1000 voicelab
USER voicelab

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz')"

CMD ["uv", "run", "voicelab", "serve", "--port", "8001", "--host", "0.0.0.0"]
