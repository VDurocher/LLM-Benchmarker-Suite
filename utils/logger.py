"""
Structured logger for the LLM-Benchmarker suite.

Uses the standard `logging` module with a custom formatter
that produces readable lines in development and JSON in production.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

LOG_FORMAT: Final[str] = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Log level configurable via environment variable
_LOG_LEVEL: str = os.environ.get("LLM_BENCH_LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured with the calling module's name.
    Idempotent: calling multiple times with the same name returns the same logger.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if the function is called multiple times
    if logger.handlers:
        return logger

    # Force UTF-8 on Windows to support Unicode symbols in logs
    stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)  # noqa: WPS515
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    logger.propagate = False

    return logger
