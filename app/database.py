"""SQLite database initialization and connection management for Standalone Voice Transcription."""

import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager

from app.config import settings


def get_db_path() -> str:
    """Get database file path, creating parent directories if needed."""
    db_path = settings.DATABASE_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection() -> sqlite3.Connection:
    """Create a new database connection with row factory."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database schema for meetings, speaker profiles, and participants."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meetings (
                id TEXT PRIMARY KEY,
                meeting_date TEXT NOT NULL,
                transcript TEXT,
                transcript_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS meeting_participants (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id   TEXT    NOT NULL,
                user_id      TEXT    NOT NULL,
                speaker_id   TEXT    NOT NULL,
                display_name TEXT    NOT NULL,
                role         TEXT    NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id),
                UNIQUE(meeting_id, speaker_id)
            );

            CREATE TABLE IF NOT EXISTS speaker_profiles (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             TEXT    NOT NULL,
                speaker_id          TEXT    NOT NULL UNIQUE,
                display_name        TEXT    NOT NULL,
                role                TEXT    NOT NULL,
                embedding_dimension INTEGER NOT NULL DEFAULT 192,
                sample_count        INTEGER NOT NULL,
                embeddings_json     TEXT    NOT NULL,
                centroid_json       TEXT    NOT NULL,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Ensure demo meeting exists
        conn.execute(
            "INSERT OR IGNORE INTO meetings (id, meeting_date) VALUES ('voice_chat_demo', '2026-08-19')"
        )
