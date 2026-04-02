"""
chat_history.py
---------------
Stores chat history in SQLite so conversations persist across server restarts.

SQLite is built into Python — no installation needed.
Table structure:
  sessions  → one row per conversation session
  messages  → one row per message (linked to a session)
"""

import sqlite3
import os
from datetime import datetime
from typing import List


DB_PATH = os.path.join("database", "chat_history.db")


class ChatHistoryDB:
    """SQLite-backed chat history store."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def _init_db(self):
        """Create tables if they don't already exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    doc_name    TEXT,
                    created_at  TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT,
                    role        TEXT,       -- 'user' or 'assistant'
                    content     TEXT,
                    created_at  TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # rows behave like dicts
        return conn

    # ------------------------------------------------------------------ #
    #  Sessions                                                            #
    # ------------------------------------------------------------------ #

    def create_session(self, session_id: str, doc_name: str = ""):
        """Create a new chat session."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, doc_name, created_at) VALUES (?, ?, ?)",
                (session_id, doc_name, datetime.utcnow().isoformat()),
            )

    def list_sessions(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Messages                                                            #
    # ------------------------------------------------------------------ #

    def add_message(self, session_id: str, role: str, content: str):
        """Append a message to a session."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, datetime.utcnow().isoformat()),
            )

    def get_messages(self, session_id: str) -> List[dict]:
        """Return all messages for a session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_history_for_llm(self, session_id: str, last_n: int = 6) -> List[dict]:
        """
        Return the last N messages in OpenAI message format.
        We cap at last_n to avoid bloating the LLM context window.
        """
        messages = self.get_messages(session_id)
        recent = messages[-last_n:] if len(messages) > last_n else messages
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
