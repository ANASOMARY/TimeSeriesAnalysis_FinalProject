"""SQLite storage."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  role TEXT NOT NULL CHECK (role IN ('admin','reviewer')),
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE COLLATE NOCASE,
  password_hash TEXT,
  status TEXT NOT NULL DEFAULT 'invited' CHECK (status IN ('invited','active','suspended')),
  invite_token_hash TEXT,
  invite_token_enc BLOB,
  invite_expires_at TEXT,
  invite_sent_at TEXT,
  max_devices INTEGER NOT NULL DEFAULT 2,
  failed_logins INTEGER NOT NULL DEFAULT 0,
  locked_until TEXT,
  nda_accepted_at TEXT,
  note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  last_login_at TEXT
);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  original_filename TEXT NOT NULL,
  source_format TEXT NOT NULL,
  page_count INTEGER NOT NULL,
  pages_json TEXT NOT NULL,
  paras_json TEXT,
  size_bytes INTEGER NOT NULL,
  blob_name TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','suspended')),
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grants (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  starts_at TEXT,
  expires_at TEXT,
  max_views INTEGER,
  views_used INTEGER NOT NULL DEFAULT 0,
  seconds_read INTEGER NOT NULL DEFAULT 0,
  away_count INTEGER NOT NULL DEFAULT 0,
  away_brief INTEGER NOT NULL DEFAULT 0,
  allow_comments INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','revoked')),
  created_at TEXT NOT NULL,
  revoked_at TEXT,
  last_viewed_at TEXT,
  reviewed_at TEXT,
  UNIQUE (user_id, document_id)
);
CREATE TABLE IF NOT EXISTS devices (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  ip TEXT NOT NULL DEFAULT '',
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  UNIQUE (user_id, token_hash)
);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY,
  token_hash TEXT NOT NULL UNIQUE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  device_id INTEGER REFERENCES devices(id) ON DELETE CASCADE,
  csrf TEXT NOT NULL,
  flash TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  ip TEXT NOT NULL DEFAULT '',
  user_agent TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS views (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  grant_id INTEGER REFERENCES grants(id) ON DELETE CASCADE,
  stamp TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  pinged_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
-- The audit trail deliberately has no foreign keys: it must outlive the
-- reviewers and chapters it describes, so names are copied into each row.
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  user_id INTEGER,
  document_id INTEGER,
  view_id INTEGER,
  page INTEGER,
  actor TEXT NOT NULL DEFAULT '',
  subject TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT '',
  ip TEXT NOT NULL DEFAULT '',
  user_agent TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
CREATE INDEX IF NOT EXISTS events_user_doc ON events (user_id, document_id, kind);
CREATE INDEX IF NOT EXISTS events_view_page ON events (view_id, page);
CREATE INDEX IF NOT EXISTS events_kind_ts ON events (kind, ts);
CREATE TABLE IF NOT EXISTS comments (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  page INTEGER NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL,
  -- Optional marked area the note is pinned to, as fractions of the page, and the text under it.
  x0 REAL, y0 REAL, x1 REAL, y1 REAL,
  quote TEXT NOT NULL DEFAULT '',
  edited_at TEXT
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- Kinds of note a reviewer can choose from. A category that has been used is retired
-- (active = 0) rather than deleted, so notes already written keep their label.
CREATE TABLE IF NOT EXISTS categories (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  description TEXT NOT NULL DEFAULT '',
  color TEXT NOT NULL DEFAULT 'slate',
  position INTEGER NOT NULL DEFAULT 0,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
-- Which paragraphs each reviewer has covered, and how: "seen" (on screen long enough to have been
-- read), "confirmed" (ticked by hand) or "noted" (a note was pinned to it).
CREATE TABLE IF NOT EXISTS coverage (
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER NOT NULL,
  para INTEGER NOT NULL,
  how TEXT NOT NULL CHECK (how IN ('seen', 'confirmed', 'noted')),
  at TEXT NOT NULL,
  PRIMARY KEY (user_id, document_id, page, para)
);
CREATE TABLE IF NOT EXISTS comment_categories (
  comment_id INTEGER NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
  category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
  PRIMARY KEY (comment_id, category_id)
);
"""

# Colours a category can have. Each is a class in app.css, dark enough for white text.
CATEGORY_COLORS = ("violet", "blue", "teal", "green", "amber", "orange", "rose", "slate")

STARTER_CATEGORIES = (
    ("Sounds AI-written", "Reads as machine-generated: generic, formulaic or over-polished phrasing a human "
                          "author would be unlikely to write.", "violet"),
    ("Hard to follow", "The information is there, but it takes effort to get: long or tangled sentences, too many "
                       "ideas at once, or an unclear order.", "blue"),
    ("Awkward wording", "Clumsy or unnatural phrasing that makes you stumble, whatever the cause.", "orange"),
    ("Filler", "Words without substance. It could be cut and nothing would be lost.", "slate"),
    ("Repetitive", "Repeats a point or phrase already made.", "amber"),
    ("Eyebrow text", "A short label sitting above a heading (a kicker) that adds nothing or reads as decoration.", "rose"),
    ("Works well", "A passage worth keeping as it is.", "green"),
)

ISO = "%Y-%m-%dT%H:%M:%SZ"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(ISO)


def now_iso() -> str:
    return iso(utcnow())


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, ISO).replace(tzinfo=timezone.utc)


def in_future(**delta) -> str:
    return iso(utcnow() + timedelta(**delta))


def connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def get() -> sqlite3.Connection:
    if "db" not in g:
        g.db = connect(current_app.config["VAULT"].db_path)
    return g.db


def all(sql: str, *args) -> list[sqlite3.Row]:  # noqa: A001 - mirrors cursor.fetchall
    return get().execute(sql, args).fetchall()


def one(sql: str, *args) -> sqlite3.Row | None:
    return get().execute(sql, args).fetchone()


def scalar(sql: str, *args):
    row = get().execute(sql, args).fetchone()
    return row[0] if row else None


def run(sql: str, *args) -> sqlite3.Cursor:
    return get().execute(sql, args)


# Columns added after the first release. Existing databases gain them on start-up.
ADDED_COLUMNS = {
    "users": {"invite_sent_at": "TEXT"},
    "documents": {"paras_json": "TEXT"},
    "grants": {"reviewed_at": "TEXT", "away_count": "INTEGER NOT NULL DEFAULT 0",
               "away_brief": "INTEGER NOT NULL DEFAULT 0"},
    "comments": {"x0": "REAL", "y0": "REAL", "x1": "REAL", "y1": "REAL", "quote": "TEXT NOT NULL DEFAULT ''",
                 "edited_at": "TEXT"},
}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, declaration in columns.items():
            if name not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _seed_categories(conn: sqlite3.Connection) -> None:
    """Give a vault its starter categories once. The author's later changes, including
    removing every one of them, are never undone by a restart."""
    if conn.execute("SELECT 1 FROM settings WHERE key = 'categories_seeded'").fetchone():
        return
    if not conn.execute("SELECT 1 FROM categories LIMIT 1").fetchone():
        for position, (name, description, color) in enumerate(STARTER_CATEGORIES):
            conn.execute("INSERT INTO categories (name, description, color, position, created_at) VALUES (?,?,?,?,?)",
                         (name, description, color, position, now_iso()))
    conn.execute("INSERT INTO settings (key, value) VALUES ('categories_seeded', '1')")


def _carry_over_reading_progress(conn: sqlite3.Connection) -> None:
    """Progress used to be counted from pages that were *loaded*; it is now counted from pages
    that were actually on screen for a few seconds ("page_read"). So that nobody's progress
    drops to zero on upgrade, pages loaded before the change are carried over once."""
    if conn.execute("SELECT 1 FROM settings WHERE key = 'page_read_carried_over'").fetchone():
        return
    conn.execute(
        "INSERT INTO events (ts, kind, user_id, document_id, page, actor, subject, detail) "
        "SELECT MIN(ts), 'page_read', user_id, document_id, page, actor, subject, 'carried over from pages opened' "
        "FROM events WHERE kind = 'page_view' AND user_id IS NOT NULL AND document_id IS NOT NULL AND page IS NOT NULL "
        "GROUP BY user_id, document_id, page")
    conn.execute("INSERT INTO settings (key, value) VALUES ('page_read_carried_over', '1')")


def init_app(app) -> None:
    conn = connect(app.config["VAULT"].db_path)
    conn.executescript(SCHEMA)
    _migrate(conn)
    _seed_categories(conn)
    _carry_over_reading_progress(conn)
    conn.close()

    @app.teardown_appcontext
    def _close(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------- settings

_NDA_OPENING = (
    "The chapters you are about to read are unpublished, confidential drafts shared with you "
    "solely so that you can review them.\n\n"
    "By continuing you agree that you will not copy, photograph, screenshot, record, forward or "
    "otherwise share any part of this material, and that you will not discuss its contents with "
    "anyone other than the author without the author's written permission.\n\n"
)
# The terms as they read before reading activity was recorded. A vault still showing exactly
# this wording has never had its terms edited, so it moves to the current wording by itself.
_OLD_NDA = _NDA_OPENING + "Every page you open is marked with your name and email address, and your access is logged."
DEFAULT_NDA = _NDA_OPENING + (
    "You also agree to read the chapters yourself: you will not open them with automated tools, "
    "and you will not give any part of them to an AI assistant or any other service to read, "
    "summarise or review.\n\n"
    "Every page you open is marked with your name and email address. Your activity in the reader "
    "is recorded for the author: what you have read and when, attempts to print, copy or capture "
    "a page, and how often the reading window is left."
)

DEFAULT_SETTINGS = {
    "book_title": "",
    "author_name": "",
    "default_expiry_days": "30",
    "default_max_views": "",
    "default_max_devices": "2",
    "require_nda": "1",
    "nda_text": DEFAULT_NDA,
    # The author writes both watermark texts. Placeholders in braces are filled in per reader.
    "watermark_text": "{name}  ·  {email}\n{date}  ·  {ip}",
    "watermark_footer": "Confidential review copy - do not copy or distribute  ·  Issued to {email}  ·  {date}  ·  Ref {ref}",
    "watermark_opacity": "14",
    # Outgoing email. The password is stored encrypted; blank host means email is switched off.
    "smtp_host": "",
    "smtp_port": "587",
    "smtp_security": "starttls",
    "smtp_username": "",
    "smtp_password_enc": "",
    "mail_from_name": "",
    "mail_from_address": "",
    "mail_reply_to": "",
    "invite_subject": "",
    "invite_message": "",
}

_OLD_NOTICE = "Confidential review copy - do not copy or distribute"


def settings() -> dict[str, str]:
    if "settings" not in g:
        values = dict(DEFAULT_SETTINGS)
        stored = {r["key"]: r["value"] for r in all("SELECT key, value FROM settings")}
        values.update(stored)
        # Saved through the settings form, the old wording comes back with the browser's line endings.
        if stored.get("nda_text", "").replace("\r", "").strip() == _OLD_NDA:
            values["nda_text"] = DEFAULT_NDA
        # Before the footer became fully editable, only its opening notice was stored. Carry a
        # customised notice over into the new footer text.
        notice = stored.get("watermark_notice")
        if "watermark_footer" not in stored and notice is not None and notice != _OLD_NOTICE:
            values["watermark_footer"] = "  ·  ".join(
                part for part in (notice.strip(), "Issued to {email}", "{date}", "Ref {ref}") if part)
        g.settings = values
    return g.settings


def set_setting(key: str, value: str) -> None:
    run(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        key,
        value,
    )
    g.pop("settings", None)
