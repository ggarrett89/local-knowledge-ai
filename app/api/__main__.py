"""Run the API: ``python -m app.api`` (listens on LKA_API_HOST:LKA_API_PORT, 127.0.0.1 by default)."""

from __future__ import annotations

import logging

import uvicorn

from app.api.app import create_app
from app.core.config import load_settings


def main() -> None:
    settings = load_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
