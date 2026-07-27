"""
Logging utility — structured logging to console + rotating file.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Returns a logger that writes to stdout and optionally to a rotating file.

    Parameters
    ----------
    name      : Logger name (usually __name__).
    log_file  : Path to log file. If None, file handler is skipped.
    level     : Logging level.
    fmt       : Log format string.
    datefmt   : Date/time format.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers.

    logger.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # ── Console handler ────────────────────────────────────────────────────────
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # ── File handler ───────────────────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
