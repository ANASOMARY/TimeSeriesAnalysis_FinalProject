"""Append-only audit trail."""
from __future__ import annotations

from flask import g, has_request_context, request

from . import db

LABELS = {
    "setup": "Author account created",
    "login": "Signed in",
    "login_failed": "Failed sign-in",
    "login_locked": "Sign-in locked after repeated failures",
    "login_blocked_device": "Sign-in blocked: device limit reached",
    "logout": "Signed out",
    "invite_created": "Invite link created",
    "invite_accepted": "Invite accepted, password set",
    "password_changed": "Password changed",
    "nda_accepted": "Confidentiality terms accepted",
    "doc_uploaded": "Chapter uploaded",
    "doc_updated": "Chapter details updated",
    "doc_suspended": "Chapter suspended",
    "doc_activated": "Chapter re-activated",
    "doc_deleted": "Chapter deleted",
    "doc_open": "Chapter opened",
    "doc_preview": "Author preview opened",
    "page_view": "Page viewed",
    "view_denied": "Access denied",
    "throttled": "Page requests throttled",
    "print_attempt": "Print attempt blocked",
    "capture_key": "Screen-capture key pressed",
    "copy_attempt": "Copy attempt blocked",
    "comment_added": "Feedback added",
    "comment_edited": "Feedback edited",
    "comment_deleted": "Feedback deleted",
    "grant_created": "Access granted",
    "grant_updated": "Access rules changed",
    "grant_revoked": "Access revoked",
    "grant_restored": "Access restored",
    "grant_views_reset": "Open count reset",
    "reviewer_added": "Reviewer added",
    "reviewer_updated": "Reviewer details updated",
    "reviewer_suspended": "Reviewer suspended",
    "reviewer_activated": "Reviewer re-activated",
    "reviewer_deleted": "Reviewer deleted",
    "devices_reset": "Registered devices cleared",
    "settings_updated": "Settings updated",
}

# Events that deserve the author's attention in the activity feed.
ALERTS = {
    "login_failed", "login_locked", "login_blocked_device", "view_denied",
    "throttled", "print_attempt", "capture_key", "copy_attempt",
}


def client_ip() -> str:
    return (request.remote_addr or "") if has_request_context() else ""


def log(kind: str, *, user=None, document=None, view_id=None, page=None, detail: str = "",
        actor: str | None = None) -> None:
    if user is None and has_request_context():
        user = g.get("user")
    if actor is None:
        actor = f"{user['name']} <{user['email']}>" if user is not None else ""
    db.run(
        "INSERT INTO events (ts, kind, user_id, document_id, view_id, page, actor, subject, detail, ip, user_agent) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        db.now_iso(),
        kind,
        user["id"] if user is not None else None,
        document["id"] if document is not None else None,
        view_id,
        page,
        actor,
        document["title"] if document is not None else "",
        detail[:500],
        client_ip(),
        (request.headers.get("User-Agent", "")[:300] if has_request_context() else ""),
    )
