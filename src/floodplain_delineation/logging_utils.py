"""Logging setup and a small collector for processing warnings.

Warnings are both logged and stored so they can be written to the processing
summary together with the detected limitations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

LOGGER_NAME = "floodplain_delineation"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger below the package root logger."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(log_file: Optional[Path] = None, verbose: bool = True, append: bool = False) -> logging.Logger:
    """Configure the package logger (console and optional file).

    Existing handlers are replaced so repeated calls (e.g. in tests) do not
    duplicate messages. ``verbose`` selects DEBUG instead of INFO. ``append`` keeps the
    existing log file (used when resuming a run) instead of truncating it.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a" if append else "w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class WarningCollector:
    """Collects human-readable processing warnings and logs them as they occur."""

    def __init__(self) -> None:
        self._messages: List[str] = []
        self._log = get_logger("warnings")

    def add(self, message: str) -> None:
        self._messages.append(message)
        self._log.warning(message)

    @property
    def messages(self) -> List[str]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
