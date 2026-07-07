"""
logger.py
=========
Centralized logging configuration for the JusticeLens AI backend.

Provides a single ``get_logger`` factory function that every module in the
package uses to obtain a correctly configured ``logging.Logger`` instance.
Configuring logging in exactly one place (rather than letting each module
call ``logging.basicConfig`` independently) avoids duplicate handlers,
inconsistent formatting, and log-level surprises when modules are imported
in different orders (a common source of subtle bugs in data pipelines).

Design goals:
    * Console output for interactive development (Streamlit / notebooks /
      CLI runs).
    * Rotating file output so long-running pipeline jobs do not produce
      unbounded log files.
    * Idempotent configuration: calling ``get_logger`` multiple times for
      the same name never attaches duplicate handlers.
    * Log level configurable via ``JUSTICELENS_LOG_LEVEL`` environment
      variable (see config.py), defaulting to INFO.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from justicelens import config

#: Tracks which logger names have already been configured so repeated calls
#: to ``get_logger`` are safe and cheap.
_CONFIGURED_LOGGERS: set = set()


def _build_console_handler() -> logging.Handler:
    """Create a stream handler that writes formatted log records to stdout.

    Returns:
        A configured ``logging.StreamHandler``.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    )
    return handler


def _build_file_handler() -> logging.Handler:
    """Create a rotating file handler that writes formatted log records to
    the configured log file path, rotating when the size threshold is
    exceeded.

    Returns:
        A configured ``logging.handlers.RotatingFileHandler``.
    """
    handler = RotatingFileHandler(
        filename=str(config.LOG_FILE_PATH),
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(fmt=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    )
    return handler


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a fully configured logger for the given module name.

    The first call for a given ``name`` attaches a console handler and a
    rotating file handler and sets the log level. Subsequent calls for the
    same ``name`` return the same logger instance without re-attaching
    handlers, so this function is safe to call at the top of every module
    (``logger = get_logger(__name__)``).

    Args:
        name: Logger name, conventionally the caller's ``__name__``.
        level: Optional explicit log level (e.g. "DEBUG", "INFO",
            "WARNING"). When omitted, falls back to
            ``JUSTICELENS_LOG_LEVEL`` from config.py (default "INFO").

    Returns:
        A configured ``logging.Logger`` instance.

    Raises:
        ValueError: If an invalid log level string is supplied.
    """
    logger = logging.getLogger(name)
    resolved_level = (level or config.LOG_LEVEL).upper()

    if resolved_level not in logging._nameToLevel:  # noqa: SLF001 (intentional)
        valid_levels = ", ".join(sorted(logging._nameToLevel.keys()))  # noqa: SLF001
        raise ValueError(
            f"Invalid log level '{resolved_level}'. Must be one of: {valid_levels}"
        )

    logger.setLevel(resolved_level)

    if name not in _CONFIGURED_LOGGERS:
        logger.addHandler(_build_console_handler())
        logger.addHandler(_build_file_handler())
        # Prevent double-logging via the root logger if the host application
        # (e.g. Streamlit) also configures logging.
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)

    return logger
