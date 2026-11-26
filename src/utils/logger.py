"""
Unified Logging Protocol for Pedagogical AI Audit.

Provides dual-sink logging across all pipeline stages:
  1. Console stream: concise timestamped progress for terminal & Jupyter notebooks.
  2. Persistent file stream (`logs/{run_name}.log`): full audit trail when executed
     via CLI scripts (automatically bypassed when running inside Jupyter Notebooks).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class UTCFormatter(logging.Formatter):
    """Enforce UTC timestamps across all log records for strict reproducibility."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def is_running_in_notebook() -> bool:
    """Detect whether the current Python process is executing inside a Jupyter kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and "IPKernelApp" in shell.config:
            return True
    except ImportError:
        pass
    return False


def setup_logger(
    name: str = "eduai_audit",
    log_dir: str | Path = "logs",
    log_filename: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    disable_file_in_notebook: bool = True,
) -> logging.Logger:
    """
    Configure and return a unified logger with console + optional file handlers.
    Automatically skips file creation in Jupyter Notebooks unless explicitly overridden.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    # 1. Console Handler (always active for both CLI and Jupyter)
    console_fmt = UTCFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 2. Persistent Audit File Handler (CLI only by default)
    skip_file = disable_file_in_notebook and is_running_in_notebook()
    if log_filename and not skip_file:
        log_path = Path(log_dir) / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_fmt = UTCFormatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger