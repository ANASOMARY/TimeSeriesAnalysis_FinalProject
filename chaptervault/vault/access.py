"""The single place that decides whether a grant currently lets a reviewer read a chapter."""
from __future__ import annotations

from datetime import datetime

from . import db


def evaluate(grant, document, user, now: datetime | None = None) -> tuple[str, str]:
    """Return ``(code, message)``. ``code == "ok"`` means reading is allowed right now.

    The open-count limit is deliberately not checked here: it only gates starting a
    *new* reading session (see ``can_open``), so a reader is not cut off mid-chapter.
    """
    now = now or db.utcnow()
    if user["status"] != "active":
        return "user_suspended", "Your account is not active."
    if grant is None or grant["status"] == "revoked":
        return "revoked", "The author has withdrawn access to this chapter."
    if document["status"] != "active":
        return "doc_suspended", "This chapter is temporarily unavailable."
    starts = db.parse_iso(grant["starts_at"])
    if starts and now < starts:
        return "not_started", f"This chapter becomes available on {starts:%d %b %Y}."
    expires = db.parse_iso(grant["expires_at"])
    if expires and now > expires:
        return "expired", f"Your access to this chapter ended on {expires:%d %b %Y}."
    return "ok", ""


def can_open(grant, document, user, now: datetime | None = None) -> tuple[str, str]:
    code, message = evaluate(grant, document, user, now)
    if code != "ok":
        return code, message
    if grant["max_views"] is not None and grant["views_used"] >= grant["max_views"]:
        return "views_exhausted", "You have used all the opens the author allowed for this chapter."
    return "ok", ""


BADGES = {
    "ok": ("Active", "ok"),
    "revoked": ("Revoked", "bad"),
    "doc_suspended": ("Chapter suspended", "warn"),
    "not_started": ("Scheduled", "info"),
    "expired": ("Expired", "bad"),
    "views_exhausted": ("No opens left", "warn"),
    "user_suspended": ("Reviewer suspended", "warn"),
    "invited": ("Invite pending", "info"),
}


def badge(grant, document, user) -> tuple[str, str]:
    """Label and colour for the author's access tables."""
    if user["status"] == "invited" and grant["status"] == "active":
        code, _ = can_open(grant, document, {**dict(user), "status": "active"})
        if code == "ok":
            return BADGES["invited"]
        return BADGES[code]
    code, _ = can_open(grant, document, user)
    return BADGES[code]
