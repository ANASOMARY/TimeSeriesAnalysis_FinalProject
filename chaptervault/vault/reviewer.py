"""The reviewer's library and the secure page-image reader."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import timedelta
from functools import wraps

from flask import (Blueprint, Response, abort, current_app, g, jsonify, redirect, render_template,
                   request, url_for)

from . import access, audit, db, documents
from .auth import flash, role_required

bp = Blueprint("reviewer", __name__)

VIEW_IDLE = timedelta(minutes=45)
VIEW_LIFETIME = timedelta(hours=8)
PAGES_PER_MINUTE = 90
CLIENT_EVENTS = {"print_attempt", "capture_key", "copy_attempt"}
MAX_COMMENT = 5000

_page_lock = threading.Lock()


def _throttled(session_id: int) -> bool:
    """Cap how fast one session can pull pages, to blunt automated harvesting."""
    now = time.monotonic()
    with _page_lock:
        bucket = current_app.config.setdefault("PAGE_HITS", defaultdict(deque))[session_id]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= PAGES_PER_MINUTE:
            return True
        bucket.append(now)
        return False


def _needs_terms() -> bool:
    return (g.user["role"] == "reviewer" and db.settings()["require_nda"] == "1"
            and not g.user["nda_accepted_at"])


def api(view):
    """JSON endpoints used by the reader: answer 401 instead of redirecting."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        if g.user is None:
            return jsonify(error="signed_out", message="You have been signed out. Sign in again to continue."), 401
        return view(*args, **kwargs)
    return wrapper


# ------------------------------------------------------------------ library

@bp.get("/library")
@role_required("reviewer")
def library():
    if _needs_terms():
        return redirect(url_for("reviewer.terms"))
    rows = db.all(
        "SELECT g.id AS grant_id, d.id AS doc_id FROM grants g JOIN documents d ON d.id = g.document_id "
        "WHERE g.user_id = ? AND g.status = 'active' ORDER BY d.title COLLATE NOCASE", g.user["id"])
    items = []
    for row in rows:
        grant = db.one("SELECT * FROM grants WHERE id = ?", row["grant_id"])
        doc = db.one("SELECT * FROM documents WHERE id = ?", row["doc_id"])
        code, message = access.can_open(grant, doc, g.user)
        pages_read = db.scalar(
            "SELECT COUNT(DISTINCT page) FROM events WHERE kind = 'page_view' AND user_id = ? AND document_id = ?",
            g.user["id"], doc["id"])
        expires = db.parse_iso(grant["expires_at"])
        items.append({
            "doc": doc, "grant": grant, "code": code, "message": message,
            "pages_read": min(pages_read, doc["page_count"]),
            "days_left": (expires - db.utcnow()).days if expires else None,
            "opens_left": (grant["max_views"] - grant["views_used"]) if grant["max_views"] is not None else None,
            "comments": db.scalar("SELECT COUNT(*) FROM comments WHERE user_id = ? AND document_id = ?",
                                  g.user["id"], doc["id"]),
        })
    return render_template("reviewer/library.html", items=items)


@bp.route("/terms", methods=["GET", "POST"])
@role_required("reviewer")
def terms():
    if request.method == "POST":
        if request.form.get("agree") != "1":
            return render_template("reviewer/terms.html", error="Tick the box to confirm that you agree."), 400
        db.run("UPDATE users SET nda_accepted_at = ? WHERE id = ?", db.now_iso(), g.user["id"])
        audit.log("nda_accepted")
        return redirect(url_for("reviewer.library"))
    return render_template("reviewer/terms.html", error=None)


# ------------------------------------------------------------ opening a chapter

def start_view(document, grant):
    """Create a reading session row and return its id."""
    now = db.utcnow()
    stamp = f"{now:%d %b %Y %H:%M} UTC"
    ip = audit.client_ip()
    if ip:
        stamp += f"  ·  {ip}"
    return db.run(
        "INSERT INTO views (session_id, user_id, document_id, grant_id, stamp, created_at, last_seen, pinged_at, expires_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        g.session["id"], g.user["id"], document["id"], grant["id"] if grant is not None else None, stamp,
        db.iso(now), db.iso(now), db.iso(now), db.iso(now + VIEW_LIFETIME),
    ).lastrowid


@bp.post("/read/<int:doc_id>/open")
@role_required("reviewer")
def open_chapter(doc_id: int):
    if _needs_terms():
        return redirect(url_for("reviewer.terms"))
    doc = db.one("SELECT * FROM documents WHERE id = ?", doc_id)
    grant = db.one("SELECT * FROM grants WHERE user_id = ? AND document_id = ?", g.user["id"], doc_id)
    if doc is None or grant is None:
        abort(404)
    code, message = access.evaluate(grant, doc, g.user)
    if code != "ok":
        audit.log("view_denied", document=doc, detail=code)
        flash(message, "bad")
        return redirect(url_for("reviewer.library"))

    now = db.utcnow()
    live = db.one(
        "SELECT id FROM views WHERE session_id = ? AND document_id = ? AND grant_id = ? "
        "AND expires_at > ? AND last_seen > ? ORDER BY id DESC LIMIT 1",
        g.session["id"], doc_id, grant["id"], db.iso(now), db.iso(now - VIEW_IDLE))
    if live is not None:  # coming back to a chapter already open in this sitting is not a new "open"
        return redirect(url_for("reviewer.viewer", view_id=live["id"]))

    counted = db.run(
        "UPDATE grants SET views_used = views_used + 1, last_viewed_at = ? "
        "WHERE id = ? AND (max_views IS NULL OR views_used < max_views)", db.iso(now), grant["id"])
    if counted.rowcount != 1:
        _, message = access.can_open(grant, doc, g.user)
        audit.log("view_denied", document=doc, detail="views_exhausted")
        flash(message or "This chapter cannot be opened right now.", "bad")
        return redirect(url_for("reviewer.library"))
    view_id = start_view(doc, grant)
    audit.log("doc_open", document=doc, view_id=view_id)
    return redirect(url_for("reviewer.viewer", view_id=view_id))


def _view_state(view_id: int):
    """Load a reading session and re-check, on every request, that it may still be used."""
    view = db.one("SELECT * FROM views WHERE id = ? AND session_id = ?", view_id, g.session["id"])
    if view is None:
        abort(404)
    doc = db.one("SELECT * FROM documents WHERE id = ?", view["document_id"])
    grant = None
    now = db.utcnow()
    if view["grant_id"] is None:
        code, message = ("ok", "") if g.user["role"] == "admin" else ("revoked", "Access denied.")
    else:
        grant = db.one("SELECT * FROM grants WHERE id = ?", view["grant_id"])
        code, message = access.evaluate(grant, doc, g.user, now)
    if code == "ok" and (db.parse_iso(view["expires_at"]) < now or now - db.parse_iso(view["last_seen"]) > VIEW_IDLE):
        code, message = "view_expired", ("This reading session has timed out. "
                                         "Go back to your library and open the chapter again.")
    return view, doc, grant, code, message


def _deny(view, doc, code: str, message: str, page: int | None = None):
    already = db.scalar("SELECT 1 FROM events WHERE kind = 'view_denied' AND view_id = ? AND detail = ?",
                        view["id"], code)
    if not already:
        audit.log("view_denied", document=doc, view_id=view["id"], page=page, detail=code)
    return jsonify(error=code, message=message), 403


def _exit_url() -> str:
    return url_for("admin.chapters") if g.user["role"] == "admin" else url_for("reviewer.library")


@bp.get("/view/<int:view_id>")
def viewer(view_id: int):
    if g.user is None:
        return redirect(url_for("auth.login"))
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        flash(message, "bad")
        return redirect(_exit_url())
    import json as _json
    config = {
        "pages": _json.loads(doc["pages_json"]),
        "pageUrl": url_for("reviewer.page", view_id=view_id, number=0).rsplit("/", 1)[0] + "/",
        "pingUrl": url_for("reviewer.ping", view_id=view_id),
        "eventUrl": url_for("reviewer.client_event", view_id=view_id),
        "commentsUrl": url_for("reviewer.comments", view_id=view_id),
        "exitUrl": _exit_url() if grant is not None else url_for("admin.chapter", doc_id=doc["id"]),
        "csrf": g.session["csrf"],
        "canComment": bool(grant is not None and grant["allow_comments"]),
        "isAuthor": grant is None,
    }
    return render_template("viewer.html", doc=doc, config=config, grant=grant)


def _watermark(view) -> documents.Watermark:
    settings = db.settings()
    opacity = max(4, min(40, int(settings["watermark_opacity"] or 14))) / 100
    notice = settings["watermark_notice"].strip()
    if view["grant_id"] is None:
        lines = ("AUTHOR PREVIEW", g.user["email"])
        footer = f"Author preview  ·  {view['stamp']}"
    else:
        lines = (f"{g.user['name']}  ·  {g.user['email']}", view["stamp"])
        footer = "  ·  ".join(part for part in (
            notice, f"Issued to {g.user['email']}", view["stamp"], f"Ref V{view['id']:05d}") if part)
    return documents.Watermark(lines=lines, footer=footer, opacity=opacity)


@bp.get("/view/<int:view_id>/page/<int:number>")
@api
def page(view_id: int, number: int):
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message, number)
    if not 1 <= number <= doc["page_count"]:
        abort(404)
    if _throttled(g.session["id"]):
        recent = db.scalar("SELECT 1 FROM events WHERE kind = 'throttled' AND view_id = ? AND ts > ?",
                           view_id, db.iso(db.utcnow() - timedelta(minutes=5)))
        if not recent:
            audit.log("throttled", document=doc, view_id=view_id, page=number,
                      detail=f"more than {PAGES_PER_MINUTE} pages per minute")
        return jsonify(error="throttled", message="Slow down a little."), 429

    cfg = current_app.config["VAULT"]
    try:
        jpeg = documents.render_page(cfg, current_app.config["BLOB_KEY"], doc, number - 1,
                                     request.args.get("w", default=1200, type=int), _watermark(view))
    except (OSError, ValueError):
        current_app.logger.exception("Could not render page %s of document %s", number, doc["id"])
        return jsonify(error="render_failed", message="This page could not be displayed."), 500

    now = db.now_iso()
    if not db.scalar("SELECT 1 FROM events WHERE view_id = ? AND page = ? AND kind = 'page_view'", view_id, number):
        audit.log("page_view", document=doc, view_id=view_id, page=number)
    db.run("UPDATE views SET last_seen = ? WHERE id = ?", now, view_id)
    if grant is not None:
        db.run("UPDATE grants SET last_viewed_at = ? WHERE id = ?", now, grant["id"])
    response = Response(jpeg, mimetype="image/jpeg")
    response.headers["Content-Disposition"] = "inline"
    return response


@bp.post("/view/<int:view_id>/ping")
@api
def ping(view_id: int):
    """Heartbeat from an open reader: keeps the session alive, records reading time and lets
    the reader notice within seconds when access has been withdrawn."""
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message)
    now = db.utcnow()
    elapsed = int((now - db.parse_iso(view["pinged_at"])).total_seconds())
    if grant is not None and 0 < elapsed <= 90:
        db.run("UPDATE grants SET seconds_read = seconds_read + ? WHERE id = ?", elapsed, grant["id"])
    db.run("UPDATE views SET last_seen = ?, pinged_at = ? WHERE id = ?", db.iso(now), db.iso(now), view_id)
    return jsonify(ok=True)


@bp.post("/view/<int:view_id>/event")
@api
def client_event(view_id: int):
    view, doc, _grant, _code, _message = _view_state(view_id)
    payload = request.get_json(silent=True) or {}
    kind = payload.get("kind")
    if kind not in CLIENT_EVENTS:
        abort(400)
    if db.scalar("SELECT COUNT(*) FROM events WHERE view_id = ? AND kind = ?", view_id, kind) < 25:
        audit.log(kind, document=doc, view_id=view_id, page=payload.get("page") if isinstance(payload.get("page"), int) else None,
                  detail=str(payload.get("detail", ""))[:80])
    return jsonify(ok=True)


# ----------------------------------------------------------------- feedback

def _comment_json(row) -> dict:
    return {"id": row["id"], "page": row["page"], "body": row["body"], "author": row["name"],
            "created": row["created_at"], "mine": row["user_id"] == g.user["id"]}


@bp.route("/view/<int:view_id>/comments", methods=["GET", "POST"])
@api
def comments(view_id: int):
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message)
    if request.method == "POST":
        if grant is None or not grant["allow_comments"]:
            abort(403)
        payload = request.get_json(silent=True) or {}
        body = str(payload.get("body", "")).strip()
        page_no = payload.get("page")
        if not body or len(body) > MAX_COMMENT or not isinstance(page_no, int) or not 1 <= page_no <= doc["page_count"]:
            return jsonify(error="invalid", message=f"Write a note of up to {MAX_COMMENT} characters."), 400
        comment_id = db.run("INSERT INTO comments (document_id, user_id, page, body, created_at) VALUES (?,?,?,?,?)",
                            doc["id"], g.user["id"], page_no, body, db.now_iso()).lastrowid
        audit.log("comment_added", document=doc, view_id=view_id, page=page_no)
        row = db.one("SELECT c.*, u.name FROM comments c JOIN users u ON u.id = c.user_id WHERE c.id = ?", comment_id)
        return jsonify(_comment_json(row)), 201
    if grant is None:  # the author sees everyone's notes while previewing
        rows = db.all("SELECT c.*, u.name FROM comments c JOIN users u ON u.id = c.user_id "
                      "WHERE c.document_id = ? ORDER BY c.page, c.id", doc["id"])
    else:
        rows = db.all("SELECT c.*, u.name FROM comments c JOIN users u ON u.id = c.user_id "
                      "WHERE c.document_id = ? AND c.user_id = ? ORDER BY c.page, c.id", doc["id"], g.user["id"])
    return jsonify([_comment_json(r) for r in rows])


@bp.post("/view/<int:view_id>/comments/<int:comment_id>/delete")
@api
def delete_comment(view_id: int, comment_id: int):
    view, doc, _grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message)
    deleted = db.run("DELETE FROM comments WHERE id = ? AND user_id = ? AND document_id = ?",
                     comment_id, g.user["id"], doc["id"])
    if deleted.rowcount != 1:
        abort(404)
    audit.log("comment_deleted", document=doc, view_id=view_id)
    return jsonify(ok=True)
