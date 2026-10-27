"""Administrative authentication: a bearer token compared in constant time."""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


def require_admin(request: Request) -> None:
    """FastAPI dependency for endpoints that change data.

    - no token configured  -> 403 (the endpoints are disabled, never open by default)
    - missing or malformed -> 401
    - wrong token          -> 403
    The token is never logged or echoed.
    """
    configured = request.app.state.services.settings.admin_token
    if configured is None:
        raise HTTPException(
            status_code=403, detail="Administrative endpoints are disabled: LKA_ADMIN_TOKEN is not configured."
        )
    presented = _bearer(request.headers.get("authorization"))
    if presented is None:
        raise HTTPException(
            status_code=401, detail="Administrative token required.", headers={"WWW-Authenticate": "Bearer"}
        )
    if not secrets.compare_digest(presented.encode("utf-8"), configured.get_secret_value().encode("utf-8")):
        logger.warning("rejected administrative request: invalid token")
        raise HTTPException(status_code=403, detail="Invalid administrative token.")


def _bearer(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    value = value.strip()
    return value if scheme.lower() == "bearer" and value else None


class AdminRoute(APIRoute):
    """A route that authenticates *before* FastAPI reads or validates the request body.

    With a plain dependency FastAPI would parse the JSON first, so an unauthenticated caller could
    still trigger body parsing and get validation messages back.
    """

    def get_route_handler(self):  # noqa: ANN201
        handler = super().get_route_handler()

        async def guarded(request: Request):  # noqa: ANN202
            require_admin(request)
            return await handler(request)

        return guarded
