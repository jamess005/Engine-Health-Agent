"""Centralised logging configuration for Engine Health Agent.

Call configure_logging() once at application startup (CLI or API entry point).
All modules obtain loggers via logging.getLogger(__name__) — no configuration
required in individual modules.
"""

from __future__ import annotations

import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure root logger with a consistent format.

    Args:
        level: Override log level (DEBUG/INFO/WARNING/ERROR).
               Falls back to LOG_LEVEL env var, then INFO.
    """
    raw_level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric = getattr(logging, raw_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=numeric,
    )
    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "fastmcp", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
