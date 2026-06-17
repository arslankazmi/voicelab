"""Optional bearer token auth for /api/v1/* routes."""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request

from voicelab.config.settings import Settings, get_settings


async def get_optional_auth(
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> None:
    """Dependency that enforces bearer-token auth only when settings.auth_token is set.

    Uses hmac.compare_digest for constant-time comparison to prevent timing attacks.

    Raises:
        HTTPException(401): if auth_token is configured and the request is missing or
            provides an incorrect Authorization header.
    """
    if settings.auth_token is None:
        return

    auth_header = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = auth_header[len(prefix) :]
    if not hmac.compare_digest(token, settings.auth_token):
        raise HTTPException(status_code=401, detail="Unauthorized")


require_auth = Depends(get_optional_auth)
