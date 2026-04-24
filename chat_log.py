"""Persistent chat log.

Writes every incoming message the bot sees and every reply the bot sends to a
daily-rotating file under ``logs/``. The format is a single line per event,
grep-friendly, separate from the normal Python `logging` output.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

CHAT_LOG_DIR = os.getenv("CHAT_LOG_DIR", "logs")
CHAT_LOG_FILENAME = os.getenv("CHAT_LOG_FILENAME", "chat.log")
CHAT_LOG_BACKUP_DAYS = int(os.getenv("CHAT_LOG_BACKUP_DAYS", "30"))

_lock = threading.Lock()
_logger: logging.Logger | None = None


def _build_logger() -> logging.Logger:
    os.makedirs(CHAT_LOG_DIR, exist_ok=True)
    lg = logging.getLogger("ai-discord-member.chat")
    lg.setLevel(logging.INFO)
    lg.propagate = False  # don't spam stdout with the big chat payload
    # Avoid adding handlers twice if this is re-imported.
    if not lg.handlers:
        handler = TimedRotatingFileHandler(
            os.path.join(CHAT_LOG_DIR, CHAT_LOG_FILENAME),
            when="midnight",
            utc=True,
            backupCount=CHAT_LOG_BACKUP_DAYS,
            encoding="utf-8",
        )
        handler.suffix = "%Y-%m-%d"
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(handler)
    return lg


def _get_logger() -> logging.Logger:
    global _logger
    with _lock:
        if _logger is None:
            _logger = _build_logger()
        return _logger


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def log_event(
    *,
    kind: str,
    guild: str | None,
    channel: str | None,
    author: str,
    author_id: int | None,
    content: str,
    extra: dict[str, object] | None = None,
) -> None:
    """Append one log line.

    ``kind`` is one of: ``"msg"`` (incoming), ``"reply"`` (bot sent), or
    ``"skip"`` (bot chose not to respond).
    """
    payload: dict[str, object] = {
        "ts": _iso_now(),
        "kind": kind,
        "guild": guild,
        "channel": channel,
        "author": author,
        "author_id": author_id,
        "content": content,
    }
    if extra:
        payload.update(extra)
    try:
        _get_logger().info(json.dumps(payload, ensure_ascii=False))
    except Exception:  # noqa: BLE001 - never let logging crash the bot
        logging.getLogger("ai-discord-member").exception(
            "Failed to write chat log entry"
        )
