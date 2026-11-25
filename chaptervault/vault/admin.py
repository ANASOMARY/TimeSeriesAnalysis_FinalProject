"""The author's console: chapters, reviewers, access rules, activity and settings."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from flask import (Blueprint, Response, abort, current_app, g, redirect, render_template, request,
                   url_for)

from . import access, audit, db, documents, security
from .auth import create_invite, end_sessions, flash, invite_link
from .reviewer import start_view

bp = Blueprint("admin", __name__)


@bp.before_request
def _authors_only():
    if g.user is None:
        return redirect(url_for("auth.login")) if request.method == "GET" else abort(401)
    if g.user["role"] != "admin":
        abort(404)
    return None


# ------------------------------------------------------------------ helpers

def _doc_or_404(doc_id: int):
    return db.one("SELECT * FROM documents WHERE id = ?", doc_id) or abort(404)


def _reviewer_or_404(user_id: int):
    return db.one("SELECT * FROM users WHERE id = ? AND role = 'reviewer'", user_id) or abort(404)


def _back(default: str):
    """Return to the page a form was submitted from (same-site paths only)."""
    target = request.form.get("next", "")
    if target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return redirect(target)
    return redirect(default)


def _parse_date(value: str, end_of_day: bool) -> str | None:
    value = (value or "").strip()
    if not value:
        return None
    day = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        day = day + timedelta(hours=23, minutes=59, seconds=59)
    return db.iso(day)


def _rules_from_form() -> dict:
    """Read the access-rule fields shared by every grant form. Raises ValueError with a message."""
    try:
        starts = _parse_date(request.form.get("starts_at", ""), end_of_day=False)
        expires = _parse_date(request.form.get("expires_at", ""), end_of_day=True)
    except ValueError:
        raise ValueError("Enter dates in the form YYYY-MM-DD.") from None
    if starts and expires and expires < starts:
        raise ValueError("The end date is before the start date.")
    raw = request.form.get("max_views", "").strip()
    max_views = None
    if raw:
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError("The number of opens must be a whole number of 1 or more (or left blank).")
        max_views = int(raw)
    return {"starts_at": starts, "expires_at": expires, "max_views": max_views,
            "allow_comments": 1 if request.form.get("allow_comments") == "1" else 0}


def _default_rules() -> dict:
    settings = db.settings()
    days = settings["default_expiry_days"].strip()
    expires = None
    if days.isdigit() and int(days) > 0:
        end = (db.utcnow() + timedelta(days=int(days))).replace(hour=23, minute=59, second=59)
        expires = db.iso(end)
    views = settings["default_max_views"].strip()
    return {"starts_at": None, "expires_at": expires,
            "max_views": int(views) if views.isdigit() and int(views) > 0 else None, "allow_comments": 1}


def _give_access(user, doc, rules: dict) -> None:
    existing = db.one("SELECT * FROM grants WHERE user_id = ? AND document_id = ?", user["id"], doc["id"])
    who = f"{user['name']} <{user['email']}>"
    if existing is None:
        db.run(
            "INSERT INTO grants (user_id, document_id, starts_at, expires_at, max_views, allow_comments, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            user["id"], doc["id"], rules["starts_at"], rules["expires_at"], rules["max_views"],
            rules["allow_comments"], db.now_iso())
        audit.log("grant_created", document=doc, detail=who)
    else:
        db.run(
            "UPDATE grants SET starts_at = ?, expires_at = ?, max_views = ?, allow_comments = ?, "
            "status = 'active', revoked_at = NULL WHERE id = ?",
            rules["starts_at"], rules["expires_at"], rules["max_views"], rules["allow_comments"], existing["id"])
        audit.log("grant_restored" if existing["status"] == "revoked" else "grant_updated", document=doc, detail=who)


def _revoke(grant) -> None:
    # Open reading sessions are left in place on purpose: their next page request or heartbeat
    # re-checks the grant, is refused, and tells the reader why the chapter closed.
    db.run("UPDATE grants SET status = 'revoked', revoked_at = ? WHERE id = ?", db.now_iso(), grant["id"])


def _grant_rows(where: str, *args, order: str = "id") -> list[dict]:
    """Grants joined with their reviewer and chapter, plus reading progress, for the access tables."""
    rows = []
    for grant in db.all(f"SELECT * FROM grants WHERE {where} ORDER BY {order}", *args):
        user = db.one("SELECT * FROM users WHERE id = ?", grant["user_id"])
        doc = db.one("SELECT * FROM documents WHERE id = ?", grant["document_id"])
        label, tone = access.badge(grant, doc, user)
        pages_read = db.scalar(
            "SELECT COUNT(DISTINCT page) FROM events WHERE kind = 'page_view' AND user_id = ? AND document_id = ?",
            user["id"], doc["id"])
        rows.append({
            "grant": grant, "user": user, "doc": doc, "label": label, "tone": tone,
            "pages_read": min(pages_read, doc["page_count"]),
            "comments": db.scalar("SELECT COUNT(*) FROM comments WHERE user_id = ? AND document_id = ?",
                                  user["id"], doc["id"]),
        })
    return rows


def _safe_cell(value) -> str:
    """Neutralise spreadsheet formulas in exported CSV cells."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _csv(filename: str, header: list[str], rows) -> Response:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(header)
    for row in rows:
        writer.writerow([_safe_cell(cell) for cell in row])
    response = Response("﻿" + out.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


# ---------------------------------------------------------------- dashboard

@bp.get("/")
def dashboard():
    week_ago = db.iso(db.utcnow() - timedelta(days=7))
    alert_kinds = ",".join(f"'{k}'" for k in sorted(audit.ALERTS))
    stats = {
        "chapters": db.scalar("SELECT COUNT(*) FROM documents"),
        "reviewers": db.scalar("SELECT COUNT(*) FROM users WHERE role = 'reviewer'"),
        "pending": db.scalar("SELECT COUNT(*) FROM users WHERE role = 'reviewer' AND status = 'invited'"),
        "grants": db.scalar("SELECT COUNT(*) FROM grants WHERE status = 'active'"),
        "opens": db.scalar("SELECT COUNT(*) FROM events WHERE kind = 'doc_open' AND ts > ?", week_ago),
        "feedback": db.scalar("SELECT COUNT(*) FROM comments"),
        "alerts": db.scalar(f"SELECT COUNT(*) FROM events WHERE kind IN ({alert_kinds}) AND ts > ?", week_ago),
    }
    events = db.all("SELECT * FROM events WHERE kind != 'page_view' ORDER BY id DESC LIMIT 14")
    reading = _grant_rows("last_viewed_at IS NOT NULL", order="last_viewed_at DESC LIMIT 6")
    return render_template("admin/dashboard.html", stats=stats, events=events, reading=reading)


# ----------------------------------------------------------------- chapters

@bp.get("/chapters")
def chapters():
    docs = db.all(
        "SELECT d.*, "
        "(SELECT COUNT(*) FROM grants g WHERE g.document_id = d.id AND g.status = 'active') AS reviewers, "
        "(SELECT COALESCE(SUM(views_used), 0) FROM grants g WHERE g.document_id = d.id) AS opens, "
        "(SELECT COUNT(*) FROM comments c WHERE c.document_id = d.id) AS feedback "
        "FROM documents d ORDER BY d.title COLLATE NOCASE")
    return render_template("admin/chapters.html", docs=docs, accepted=sorted(documents.ACCEPTED))


def _title_from(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(stem.split())[:200] or "Untitled chapter"


@bp.post("/chapters/upload")
def upload():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        flash("Choose at least one file to upload.", "bad")
        return redirect(url_for("admin.chapters"))
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()[:2000]
    page_range = request.form.get("page_range", "").strip() if len(files) == 1 else ""
    cfg = current_app.config["VAULT"]
    added, problems = [], []
    for upload_file in files:
        name = Path(upload_file.filename.replace("\\", "/")).name
        data = upload_file.read()
        try:
            result = documents.ingest(data, name, page_range)
        except documents.IngestError as exc:
            problems.append(f"{name}: {exc}")
            continue
        blob_name = security.new_token(18) + ".bin"
        documents.store(cfg, current_app.config["BLOB_KEY"], blob_name, result["pdf"])
        doc_id = db.run(
            "INSERT INTO documents (title, description, original_filename, source_format, page_count, pages_json, "
            "size_bytes, blob_name, sha256, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (title if len(files) == 1 and title else _title_from(name)), description, name[:255],
            result["source_format"], result["page_count"], result["pages_json"], len(result["pdf"]),
            blob_name, result["sha256"], db.now_iso()).lastrowid
        doc = _doc_or_404(doc_id)
        audit.log("doc_uploaded", document=doc, detail=f"{name}, {result['page_count']} pages")
        added.append(doc)
    for problem in problems:
        flash(problem, "bad")
    if len(added) == 1:
        flash(f"“{added[0]['title']}” was uploaded and encrypted ({added[0]['page_count']} pages). "
              "Choose who may read it below.")
        return redirect(url_for("admin.chapter", doc_id=added[0]["id"]))
    if added:
        flash(f"{len(added)} chapters were uploaded and encrypted.")
    return redirect(url_for("admin.chapters"))


@bp.get("/chapters/<int:doc_id>")
def chapter(doc_id: int):
    doc = _doc_or_404(doc_id)
    grants = _grant_rows("document_id = ?", doc_id)
    granted_ids = {row["user"]["id"] for row in grants}
    available = [u for u in db.all("SELECT * FROM users WHERE role = 'reviewer' ORDER BY name COLLATE NOCASE")
                 if u["id"] not in granted_ids]
    feedback = db.all(
        "SELECT c.*, u.name, u.email FROM comments c JOIN users u ON u.id = c.user_id "
        "WHERE c.document_id = ? ORDER BY c.page, c.id", doc_id)
    events = db.all("SELECT * FROM events WHERE document_id = ? AND kind != 'page_view' ORDER BY id DESC LIMIT 15",
                    doc_id)
    return render_template("admin/chapter.html", doc=doc, grants=grants, available=available,
                           feedback=feedback, events=events, defaults=_default_rules())


@bp.post("/chapters/<int:doc_id>/update")
def chapter_update(doc_id: int):
    doc = _doc_or_404(doc_id)
    title = request.form.get("title", "").strip()[:200]
    if not title:
        flash("A chapter needs a title.", "bad")
    else:
        db.run("UPDATE documents SET title = ?, description = ? WHERE id = ?",
               title, request.form.get("description", "").strip()[:2000], doc_id)
        audit.log("doc_updated", document=doc, detail=title)
        flash("Chapter details saved.")
    return redirect(url_for("admin.chapter", doc_id=doc_id))


@bp.post("/chapters/<int:doc_id>/replace")
def chapter_replace(doc_id: int):
    """Swap in a revised file while keeping every reviewer's access rules."""
    doc = _doc_or_404(doc_id)
    upload_file = request.files.get("file")
    if not upload_file or not upload_file.filename:
        flash("Choose the revised file to upload.", "bad")
        return redirect(url_for("admin.chapter", doc_id=doc_id))
    name = Path(upload_file.filename.replace("\\", "/")).name
    try:
        result = documents.ingest(upload_file.read(), name, request.form.get("page_range", ""))
    except documents.IngestError as exc:
        flash(str(exc), "bad")
        return redirect(url_for("admin.chapter", doc_id=doc_id))
    cfg = current_app.config["VAULT"]
    blob_name = security.new_token(18) + ".bin"
    documents.store(cfg, current_app.config["BLOB_KEY"], blob_name, result["pdf"])
    db.run(
        "UPDATE documents SET original_filename = ?, source_format = ?, page_count = ?, pages_json = ?, "
        "size_bytes = ?, blob_name = ?, sha256 = ? WHERE id = ?",
        name[:255], result["source_format"], result["page_count"], result["pages_json"], len(result["pdf"]),
        blob_name, result["sha256"], doc_id)
    db.run("DELETE FROM views WHERE document_id = ?", doc_id)
    documents.remove(cfg, doc["blob_name"], doc_id)
    audit.log("doc_uploaded", document=doc, detail=f"replaced with {name}, {result['page_count']} pages")
    flash(f"The chapter file was replaced ({result['page_count']} pages). Existing access rules are unchanged.")
    return redirect(url_for("admin.chapter", doc_id=doc_id))


@bp.post("/chapters/<int:doc_id>/status")
def chapter_status(doc_id: int):
    doc = _doc_or_404(doc_id)
    suspend = request.form.get("action") == "suspend"
    db.run("UPDATE documents SET status = ? WHERE id = ?", "suspended" if suspend else "active", doc_id)
    audit.log("doc_suspended" if suspend else "doc_activated", document=doc)
    flash("The chapter is suspended: nobody can read it until you re-activate it." if suspend
          else "The chapter is active again.", "warn" if suspend else "ok")
    return redirect(url_for("admin.chapter", doc_id=doc_id))


@bp.post("/chapters/<int:doc_id>/delete")
def chapter_delete(doc_id: int):
    doc = _doc_or_404(doc_id)
    audit.log("doc_deleted", document=doc)
    db.run("DELETE FROM documents WHERE id = ?", doc_id)
    documents.remove(current_app.config["VAULT"], doc["blob_name"], doc_id)
    flash(f"“{doc['title']}” and all access to it were permanently deleted.")
    return redirect(url_for("admin.chapters"))


@bp.post("/chapters/<int:doc_id>/preview")
def chapter_preview(doc_id: int):
    doc = _doc_or_404(doc_id)
    view_id = start_view(doc, None)
    audit.log("doc_preview", document=doc, view_id=view_id)
    return redirect(url_for("reviewer.viewer", view_id=view_id))


@bp.get("/chapters/<int:doc_id>/feedback.csv")
def chapter_feedback_csv(doc_id: int):
    doc = _doc_or_404(doc_id)
    rows = db.all(
        "SELECT c.page, u.name, u.email, c.created_at, c.body FROM comments c JOIN users u ON u.id = c.user_id "
        "WHERE c.document_id = ? ORDER BY c.page, c.id", doc_id)
    return _csv(f"feedback-chapter-{doc['id']}.csv", ["Page", "Reviewer", "Email", "Written (UTC)", "Feedback"],
                [tuple(r) for r in rows])


# ---------------------------------------------------------------- reviewers

@bp.get("/reviewers")
def reviewers():
    people = db.all(
        "SELECT u.*, "
        "(SELECT COUNT(*) FROM grants g WHERE g.user_id = u.id AND g.status = 'active') AS chapters, "
        "(SELECT COUNT(*) FROM devices d WHERE d.user_id = u.id) AS devices, "
        "(SELECT MAX(last_viewed_at) FROM grants g WHERE g.user_id = u.id) AS last_read "
        "FROM users u WHERE u.role = 'reviewer' ORDER BY u.name COLLATE NOCASE")
    docs = db.all("SELECT id, title FROM documents ORDER BY title COLLATE NOCASE")
    return render_template("admin/reviewers.html", people=people, docs=docs, defaults=_default_rules())


@bp.post("/reviewers/add")
def reviewer_add():
    name = request.form.get("name", "").strip()[:120]
    email = request.form.get("email", "").strip().lower()[:200]
    if not name or "@" not in email or " " in email:
        flash("Enter the reviewer's name and a valid email address.", "bad")
        return redirect(url_for("admin.reviewers"))
    if db.scalar("SELECT 1 FROM users WHERE email = ?", email):
        flash(f"There is already an account for {email}.", "bad")
        return redirect(url_for("admin.reviewers"))
    devices = db.settings()["default_max_devices"]
    user_id = db.run(
        "INSERT INTO users (role, name, email, status, max_devices, note, created_at) VALUES ('reviewer',?,?,'invited',?,?,?)",
        name, email, int(devices) if devices.isdigit() and int(devices) > 0 else 2,
        request.form.get("note", "").strip()[:500], db.now_iso()).lastrowid
    create_invite(user_id)
    user = _reviewer_or_404(user_id)
    audit.log("reviewer_added", detail=f"{name} <{email}>")
    rules = _default_rules()
    granted = 0
    for raw in request.form.getlist("documents"):
        doc = db.one("SELECT * FROM documents WHERE id = ?", raw) if raw.isdigit() else None
        if doc is not None:
            _give_access(user, doc, rules)
            granted += 1
    flash(f"{name} was added" + (f" with access to {granted} chapter{'s' if granted != 1 else ''}" if granted else "")
          + ". Send them the invite link below so they can set a password.")
    return redirect(url_for("admin.reviewer", user_id=user_id))


@bp.get("/reviewers/<int:user_id>")
def reviewer(user_id: int):
    person = _reviewer_or_404(user_id)
    grants = _grant_rows("user_id = ?", user_id)
    granted_ids = {row["doc"]["id"] for row in grants}
    available = [d for d in db.all("SELECT * FROM documents ORDER BY title COLLATE NOCASE")
                 if d["id"] not in granted_ids]
    link = invite_link(person)
    mailto = None
    if link:
        settings = db.settings()
        book = settings["book_title"] or "my manuscript"
        body = (f"Hello {person['name']},\n\nI would like to share some chapters of {book} with you for review.\n\n"
                f"Use this private link to set your password and start reading:\n{link}\n\n"
                "The link is personal to you. Please do not forward it.\n\n"
                f"Thank you,\n{settings['author_name'] or g.user['name']}")
        mailto = f"mailto:{quote(person['email'])}?subject={quote('Chapters for your review')}&body={quote(body)}"
    return render_template(
        "admin/reviewer.html", person=person, grants=grants, available=available, link=link, mailto=mailto,
        devices=db.all("SELECT * FROM devices WHERE user_id = ? ORDER BY last_seen DESC", user_id),
        events=db.all("SELECT * FROM events WHERE user_id = ? AND kind != 'page_view' ORDER BY id DESC LIMIT 15",
                      user_id),
        defaults=_default_rules())


@bp.post("/reviewers/<int:user_id>/update")
def reviewer_update(user_id: int):
    person = _reviewer_or_404(user_id)
    name = request.form.get("name", "").strip()[:120]
    email = request.form.get("email", "").strip().lower()[:200]
    devices = request.form.get("max_devices", "").strip()
    if not name or "@" not in email or " " in email:
        flash("Enter a name and a valid email address.", "bad")
    elif not devices.isdigit() or not 1 <= int(devices) <= 10:
        flash("The device limit must be between 1 and 10.", "bad")
    elif db.scalar("SELECT 1 FROM users WHERE email = ? AND id != ?", email, user_id):
        flash(f"Another account already uses {email}.", "bad")
    else:
        db.run("UPDATE users SET name = ?, email = ?, max_devices = ?, note = ? WHERE id = ?",
               name, email, int(devices), request.form.get("note", "").strip()[:500], user_id)
        audit.log("reviewer_updated", detail=f"{person['name']} <{person['email']}>")
        flash("Reviewer details saved.")
    return redirect(url_for("admin.reviewer", user_id=user_id))


@bp.post("/reviewers/<int:user_id>/invite")
def reviewer_invite(user_id: int):
    person = _reviewer_or_404(user_id)
    create_invite(user_id)
    audit.log("invite_created", detail=f"{person['name']} <{person['email']}>")
    flash("A new link is ready below. Any earlier link for this reviewer no longer works."
          + ("" if person["status"] == "invited" else " Their current password keeps working until they use it."))
    return redirect(url_for("admin.reviewer", user_id=user_id))


@bp.post("/reviewers/<int:user_id>/status")
def reviewer_status(user_id: int):
    person = _reviewer_or_404(user_id)
    who = f"{person['name']} <{person['email']}>"
    if request.form.get("action") == "suspend":
        db.run("UPDATE users SET status = 'suspended' WHERE id = ?", user_id)
        end_sessions(user_id)
        audit.log("reviewer_suspended", detail=who)
        flash(f"{person['name']} is suspended and has been signed out everywhere.", "warn")
    else:
        db.run("UPDATE users SET status = ?, failed_logins = 0, locked_until = NULL WHERE id = ?",
               "active" if person["password_hash"] else "invited", user_id)
        audit.log("reviewer_activated", detail=who)
        flash(f"{person['name']} can sign in again.")
    return redirect(url_for("admin.reviewer", user_id=user_id))


@bp.post("/reviewers/<int:user_id>/devices/reset")
def reviewer_devices_reset(user_id: int):
    person = _reviewer_or_404(user_id)
    db.run("DELETE FROM devices WHERE user_id = ?", user_id)
    end_sessions(user_id)
    audit.log("devices_reset", detail=f"{person['name']} <{person['email']}>")
    flash("Registered devices were cleared. The next devices this reviewer signs in from will be registered.")
    return redirect(url_for("admin.reviewer", user_id=user_id))


@bp.post("/reviewers/<int:user_id>/delete")
def reviewer_delete(user_id: int):
    person = _reviewer_or_404(user_id)
    audit.log("reviewer_deleted", detail=f"{person['name']} <{person['email']}>")
    db.run("DELETE FROM users WHERE id = ?", user_id)
    flash(f"{person['name']} was deleted, along with their access and feedback.")
    return redirect(url_for("admin.reviewers"))


# ------------------------------------------------------------------- grants

@bp.post("/grants/create")
def grants_create():
    """Give access from either side: one chapter to many reviewers, or one reviewer to many chapters."""
    fallback = url_for("admin.access_matrix")
    try:
        rules = _rules_from_form()
    except ValueError as exc:
        flash(str(exc), "bad")
        return _back(fallback)
    user_ids = [v for v in request.form.getlist("users") if v.isdigit()]
    doc_ids = [v for v in request.form.getlist("documents") if v.isdigit()]
    count = 0
    for uid in user_ids:
        user = db.one("SELECT * FROM users WHERE id = ? AND role = 'reviewer'", uid)
        for did in doc_ids:
            doc = db.one("SELECT * FROM documents WHERE id = ?", did)
            if user is not None and doc is not None:
                _give_access(user, doc, rules)
                count += 1
    if count:
        flash(f"Access granted ({count} {'grant' if count == 1 else 'grants'}).")
    else:
        flash("Tick at least one reviewer and one chapter.", "bad")
    return _back(fallback)


@bp.post("/grants/<int:grant_id>/update")
def grant_update(grant_id: int):
    grant = db.one("SELECT * FROM grants WHERE id = ?", grant_id) or abort(404)
    doc = _doc_or_404(grant["document_id"])
    user = db.one("SELECT * FROM users WHERE id = ?", grant["user_id"])
    who = f"{user['name']} <{user['email']}>"
    action = request.form.get("action", "save")
    if action == "revoke":
        _revoke(grant)
        audit.log("grant_revoked", document=doc, detail=who)
        flash(f"{user['name']} can no longer read “{doc['title']}”. Any open copy closes within seconds.", "warn")
    elif action == "restore":
        db.run("UPDATE grants SET status = 'active', revoked_at = NULL WHERE id = ?", grant_id)
        audit.log("grant_restored", document=doc, detail=who)
        flash(f"Access restored for {user['name']}.")
    elif action == "reset_views":
        db.run("UPDATE grants SET views_used = 0 WHERE id = ?", grant_id)
        audit.log("grant_views_reset", document=doc, detail=who)
        flash(f"The open count for {user['name']} was reset.")
    else:
        try:
            rules = _rules_from_form()
        except ValueError as exc:
            flash(str(exc), "bad")
            return _back(url_for("admin.chapter", doc_id=doc["id"]))
        db.run("UPDATE grants SET starts_at = ?, expires_at = ?, max_views = ?, allow_comments = ? WHERE id = ?",
               rules["starts_at"], rules["expires_at"], rules["max_views"], rules["allow_comments"], grant_id)
        audit.log("grant_updated", document=doc, detail=who)
        flash(f"Access rules for {user['name']} saved.")
    return _back(url_for("admin.chapter", doc_id=doc["id"]))


@bp.route("/access", methods=["GET", "POST"])
def access_matrix():
    """One grid of every reviewer against every chapter."""
    people = db.all("SELECT * FROM users WHERE role = 'reviewer' ORDER BY name COLLATE NOCASE")
    docs = db.all("SELECT * FROM documents ORDER BY title COLLATE NOCASE")
    existing = {(g_["user_id"], g_["document_id"]): g_ for g_ in db.all("SELECT * FROM grants")}
    if request.method == "POST":
        ticked = set(request.form.getlist("cell"))
        rules = _default_rules()
        added = removed = 0
        for person in people:
            for doc in docs:
                grant = existing.get((person["id"], doc["id"]))
                wanted = f"{person['id']}:{doc['id']}" in ticked
                active = grant is not None and grant["status"] == "active"
                if wanted and not active:
                    if grant is None:
                        _give_access(person, doc, rules)
                    else:
                        db.run("UPDATE grants SET status = 'active', revoked_at = NULL WHERE id = ?", grant["id"])
                        audit.log("grant_restored", document=doc, detail=f"{person['name']} <{person['email']}>")
                    added += 1
                elif active and not wanted:
                    _revoke(grant)
                    audit.log("grant_revoked", document=doc, detail=f"{person['name']} <{person['email']}>")
                    removed += 1
        flash(f"Access updated: {added} granted, {removed} revoked." if added or removed else "Nothing changed.")
        return redirect(url_for("admin.access_matrix"))
    cells = {}
    for (uid, did), grant in existing.items():
        person = next((p for p in people if p["id"] == uid), None)
        doc = next((d for d in docs if d["id"] == did), None)
        if person is not None and doc is not None:
            label, tone = access.badge(grant, doc, person)
            cells[(uid, did)] = {"active": grant["status"] == "active", "label": label, "tone": tone}
    return render_template("admin/access.html", people=people, docs=docs, cells=cells, defaults=_default_rules())


# ----------------------------------------------------------------- activity

EVENT_GROUPS = {
    "alerts": sorted(audit.ALERTS),
    "reading": ["doc_open", "page_view", "comment_added", "comment_deleted", "nda_accepted"],
    "signin": ["login", "login_failed", "login_locked", "login_blocked_device", "logout", "invite_accepted",
               "password_changed"],
}


@bp.get("/activity")
def activity():
    where, args = ["1 = 1"], []
    group = request.args.get("show", "")
    if group in EVENT_GROUPS:
        where.append("kind IN (%s)" % ",".join("?" * len(EVENT_GROUPS[group])))
        args += EVENT_GROUPS[group]
    elif group != "everything":
        where.append("kind != 'page_view'")
    reviewer_id = request.args.get("reviewer", type=int)
    if reviewer_id:
        where.append("user_id = ?")
        args.append(reviewer_id)
    doc_id = request.args.get("chapter", type=int)
    if doc_id:
        where.append("document_id = ?")
        args.append(doc_id)
    sql = f"SELECT * FROM events WHERE {' AND '.join(where)}"
    if request.args.get("format") == "csv":
        rows = db.all(sql + " ORDER BY id DESC LIMIT 50000", *args)
        return _csv("chaptervault-activity.csv",
                    ["Time (UTC)", "Event", "Who", "Chapter", "Page", "Detail", "IP address", "Browser"],
                    [(r["ts"], audit.LABELS.get(r["kind"], r["kind"]), r["actor"], r["subject"], r["page"] or "",
                      r["detail"], r["ip"], r["user_agent"]) for r in rows])
    before = request.args.get("before", type=int)
    if before:
        sql += " AND id < ?"
        args.append(before)
    events = db.all(sql + " ORDER BY id DESC LIMIT 101", *args)
    return render_template(
        "admin/activity.html", events=events[:100], older=events[99]["id"] if len(events) > 100 else None,
        people=db.all("SELECT id, name FROM users WHERE role = 'reviewer' ORDER BY name COLLATE NOCASE"),
        docs=db.all("SELECT id, title FROM documents ORDER BY title COLLATE NOCASE"),
        show=group, reviewer_id=reviewer_id, doc_id=doc_id)


# ----------------------------------------------------------------- settings

@bp.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        form = request.form
        problems = []
        for key, label, low, high, blank_ok in (
                ("default_expiry_days", "Default access period", 1, 3650, True),
                ("default_max_views", "Default number of opens", 1, 100000, True),
                ("default_max_devices", "Default device limit", 1, 10, False),
                ("watermark_opacity", "Watermark strength", 4, 40, False)):
            value = form.get(key, "").strip()
            if not value and blank_ok:
                continue
            if not value.isdigit() or not low <= int(value) <= high:
                problems.append(f"{label} must be a number from {low} to {high}"
                                + (" (or blank for no limit)." if blank_ok else "."))
        if problems:
            for problem in problems:
                flash(problem, "bad")
        else:
            for key in ("book_title", "author_name", "default_expiry_days", "default_max_views",
                        "default_max_devices", "watermark_notice", "watermark_opacity"):
                db.set_setting(key, form.get(key, "").strip()[:300])
            db.set_setting("nda_text", form.get("nda_text", "").strip()[:8000])
            db.set_setting("require_nda", "1" if form.get("require_nda") == "1" else "0")
            audit.log("settings_updated")
            flash("Settings saved.")
        return redirect(url_for("admin.settings"))
    return render_template("admin/settings.html")
