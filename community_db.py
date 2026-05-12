"""
community_db.py — SQLite-based community storage.
Zero cost, no external server, built into Python.
On Railway: set COMMUNITY_DB_PATH=/data/community.db and attach a Volume at /data
to persist data across deploys. Without a Volume, data resets on redeploy (acceptable
for a small community starting out).
"""
import sqlite3
import os

DB_PATH = os.getenv("COMMUNITY_DB_PATH", os.path.join(os.path.dirname(__file__), "community.db"))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_tables() -> None:
    conn = get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Community (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL DEFAULT 0,
                user_name     TEXT    NOT NULL DEFAULT 'Guest',
                title         TEXT,
                category      TEXT,
                tags          TEXT,
                body          TEXT    NOT NULL,
                like_count    INTEGER NOT NULL DEFAULT 0,
                dislike_count INTEGER NOT NULL DEFAULT 0,
                comment_count INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT    NOT NULL
                                DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS CommunityComments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id    INTEGER NOT NULL,
                user_name  TEXT    DEFAULT 'Guest',
                body       TEXT,
                created_at TEXT    NOT NULL
                             DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_community_created
            ON Community(created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_comments_post
            ON CommunityComments(post_id)
        """)
        conn.commit()
    finally:
        conn.close()
