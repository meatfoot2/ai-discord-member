"""Persistent memory (SQLite) for the AI Discord member.

Stores every message the bot sees and every reply it sends, so conversation
history survives process restarts. Also stores per-user "facts" the bot has
learned (things like "darlington likes LEGO Star Wars") so it can recall them
in future conversations.

Schema is created on first open. Safe to call from a single asyncio event
loop — sqlite3 calls are synchronous and fast; all queries are LIMIT-bounded.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Iterable

log = logging.getLogger("ai-discord-member.memory")

DEFAULT_DB_PATH = os.getenv("MEMORY_DB_PATH", "data/memory.db")
MAX_FACTS_PER_USER = int(os.getenv("MAX_FACTS_PER_USER", "40"))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    guild_id    INTEGER,
    channel_id  INTEGER NOT NULL,
    author_id   INTEGER,
    author_name TEXT    NOT NULL,
    is_self     INTEGER NOT NULL,
    content     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_channel_ts
    ON messages (channel_id, id);

CREATE TABLE IF NOT EXISTS user_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    guild_id    INTEGER,
    user_id     INTEGER NOT NULL,
    user_name   TEXT    NOT NULL,
    fact        TEXT    NOT NULL,
    UNIQUE (user_id, fact)
);
CREATE INDEX IF NOT EXISTS idx_facts_user
    ON user_facts (user_id, id);
"""


class MemoryStore:
    """SQLite-backed chat memory + user-facts store."""

    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        log.info("Memory store open at %s", path)

    # -- messages ------------------------------------------------------------

    def add_message(
        self,
        *,
        guild_id: int | None,
        channel_id: int,
        author_id: int | None,
        author_name: str,
        is_self: bool,
        content: str,
    ) -> None:
        if not content:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages "
                "(ts, guild_id, channel_id, author_id, author_name, is_self, content) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    guild_id,
                    channel_id,
                    author_id,
                    author_name,
                    1 if is_self else 0,
                    content,
                ),
            )
            self._conn.commit()

    def transcript(self, channel_id: int, limit: int) -> list[dict[str, str]]:
        """Return the last ``limit`` messages for ``channel_id``, formatted
        for the Groq chat API. Bot's own lines are ``assistant``; everyone
        else is ``user`` with the author name prefixed.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT author_name, is_self, content FROM messages "
                "WHERE channel_id = ? ORDER BY id DESC LIMIT ?",
                (channel_id, limit),
            ).fetchall()
        rows = list(reversed(rows))
        out: list[dict[str, str]] = []
        for r in rows:
            if r["is_self"]:
                out.append({"role": "assistant", "content": r["content"]})
            else:
                out.append(
                    {"role": "user", "content": f"{r['author_name']}: {r['content']}"}
                )
        return out

    # -- facts ---------------------------------------------------------------

    def add_fact(
        self,
        *,
        user_id: int,
        user_name: str,
        fact: str,
        guild_id: int | None,
    ) -> None:
        fact = (fact or "").strip()
        if not fact:
            return
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO user_facts (ts, guild_id, user_id, user_name, fact) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (time.time(), guild_id, user_id, user_name, fact),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                # Dupe fact — update timestamp so it stays recent.
                self._conn.execute(
                    "UPDATE user_facts SET ts = ? WHERE user_id = ? AND fact = ?",
                    (time.time(), user_id, fact),
                )
                self._conn.commit()
            # Trim the oldest entries so we never keep more than the cap.
            self._conn.execute(
                "DELETE FROM user_facts WHERE user_id = ? AND id NOT IN ("
                "SELECT id FROM user_facts WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ?"
                ")",
                (user_id, user_id, MAX_FACTS_PER_USER),
            )
            self._conn.commit()

    def facts_for(self, user_id: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT fact FROM user_facts WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, MAX_FACTS_PER_USER),
            ).fetchall()
        return [r["fact"] for r in rows]

    def facts_for_many(self, user_ids: Iterable[int]) -> dict[int, list[str]]:
        return {uid: self.facts_for(uid) for uid in set(user_ids)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
