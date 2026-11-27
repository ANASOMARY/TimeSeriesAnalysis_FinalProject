"""The author's console: chapters, reviewers, access rules, activity and settings."""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from flask import (Blueprint, Response, abort, current_app, g, redirect, render_template, request,
                   url_for)

from . import access, audit, coverage, db, documents, mailer, security
from . import progress as reporting
from .auth import create_invite, end_sessions, flash, invite_link
from .reviewer import MAX_WATERMARK_LINES, categories_of, start_view

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
            "SELECT COUNT(DISTINCT page) FROM events WHERE kind = 'page_read' AND user_id = ? AND document_id = ?",
            user["id"], doc["id"])
        rows.append({
            "grant": grant, "user": user, "doc": doc, "label": label, "tone": tone,
            "coverage": coverage.state(user["id"], doc),
            "pages_read": min(pages_read, doc["page_count"]),
            "comments": db.scalar("SELECT COUNT(*) FROM comments WHERE user_id = ? AND document_id = ?",
                                  user["id"], doc["id"]),
        })
    return rows


def _mail() -> dict | None:
    return mailer.configuration(db.settings(), current_app.config["MAIL_KEY"])


def _email_invite(person) -> str | None:
    """Email a reviewer their personal link. Returns None on success, or the reason it failed.

    The link already shown to the author is reused while it is still valid, so a resent
    invitation does not silently break the one sent before; a lapsed link is replaced.
    """
    link = invite_link(person)
    if link is None:
        create_invite(person["id"])
        person = _reviewer_or_404(person["id"])
        link = invite_link(person)
        audit.log("invite_created", detail=f"{person['name']} <{person['email']}>")
    who = f"{person['name']} <{person['email']}>"
    subject, body = mailer.invitation(db.settings(), person, link,
                                      f"{db.parse_iso(person['invite_expires_at']):%d %B %Y}", g.user["name"])
    try:
        mailer.send(_mail(), person["name"], person["email"], subject, body)
    except mailer.MailError as exc:
        audit.log("email_failed", detail=f"{who}: {exc}")
        return str(exc)
    db.run("UPDATE users SET invite_sent_at = ? WHERE id = ?", db.now_iso(), person["id"])
    audit.log("invite_emailed", detail=who)
    return None


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
    events = db.all("SELECT * FROM events WHERE kind NOT IN ('page_view', 'page_read') ORDER BY id DESC LIMIT 14")
    reading = _grant_rows("last_viewed_at IS NOT NULL", order="last_viewed_at DESC LIMIT 6")
    return render_template("admin/dashboard.html", stats=stats, events=events, reading=reading)


# ----------------------------------------------------------------- progress

@bp.get("/progress")
def progress():
    """The report shown at the start of each weekly call: progress, trajectory and forecast."""
    snap = reporting.snapshot()
    return render_template(
        "admin/progress.html", p=snap, chart=reporting.trajectory_chart(snap) if snap["people"] else None,
        summary=reporting.chart_summary(snap) if snap["people"] else "",
        spark=lambda person: reporting.sparkline(person, snap), icon=reporting.status_icon,
        short_date=reporting.short_date, pace_window=reporting.PACE_WINDOW_DAYS,
        at_risk_share=int(reporting.AT_RISK_SHARE * 100))


@bp.post("/progress/target")
def progress_target():
    raw = request.form.get("target", "").strip()
    if raw:
        try:
            datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            flash("Enter the target date in the form YYYY-MM-DD.", "bad")
            return redirect(url_for("admin.progress"))
    db.set_setting("review_target_date", raw)
    audit.log("settings_updated", detail=f"review target date: {raw or 'cleared'}")
    flash("Target date saved." if raw else "Target date cleared.")
    return redirect(url_for("admin.progress"))


@bp.get("/progress.csv")
def progress_csv():
    snap = reporting.snapshot()
    stamp = lambda moment: db.iso(moment) if moment else ""   # noqa: E731
    return _csv(
        f"review-progress-{snap['now']:%Y-%m-%d}.csv",
        ["Reviewer", "Email", "Status", "Pages assigned", "Pages read", "Percent read", "Read in last 7 days",
         "Read in the 7 days before", "Pace (pages a day)", "Pace needed (pages a day)", "Projected finish (UTC)",
         "Chapters finished", "Chapters assigned", "Notes", "Notes in last 7 days", "Last read (UTC)"],
        [(p.name, p.email, p.label, p.total, p.read, f"{p.percent:.1f}", p.read_7d, p.read_prev_7d, f"{p.pace:.2f}",
          "" if p.needed is None else f"{p.needed:.2f}", stamp(p.projected), p.chapters_finished, p.chapters,
          p.notes, p.notes_7d, stamp(p.last_read_at)) for p in snap["people"]])


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
            "paras_json, size_bytes, blob_name, sha256, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (title if len(files) == 1 and title else _title_from(name)), description, name[:255],
            result["source_format"], result["page_count"], result["pages_json"], result["paras_json"],
            len(result["pdf"]), blob_name, result["sha256"], db.now_iso()).lastrowid
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
    notes = db.all(
        "SELECT c.*, u.name, u.email FROM comments c JOIN users u ON u.id = c.user_id "
        "WHERE c.document_id = ? ORDER BY c.page, c.y0, c.id", doc_id)
    labels = categories_of([n["id"] for n in notes])
    # How often each category was used on this chapter, for the summary row above the list.
    tally: dict[int, dict] = {}
    for note in notes:
        for category in labels[note["id"]]:
            tally.setdefault(category["id"], {**category, "count": 0})["count"] += 1
    unlabelled = sum(1 for n in notes if not labels[n["id"]])
    chosen = request.args.get("category", "")
    if chosen == "none":
        shown = [n for n in notes if not labels[n["id"]]]
    elif chosen.isdigit() and int(chosen) in tally:
        shown = [n for n in notes if any(c["id"] == int(chosen) for c in labels[n["id"]])]
    else:
        chosen, shown = "", notes
    feedback = [{"note": n, "categories": labels[n["id"]]} for n in shown]
    events = db.all("SELECT * FROM events WHERE document_id = ? AND kind NOT IN ('page_view', 'page_read') ORDER BY id DESC LIMIT 15",
                    doc_id)
    return render_template("admin/chapter.html", doc=doc, grants=grants, available=available,
                           feedback=feedback, events=events, defaults=_default_rules(),
                           feedback_total=len(notes), tally=list(tally.values()), unlabelled=unlabelled,
                           chosen_category=chosen)


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
        "paras_json = ?, size_bytes = ?, blob_name = ?, sha256 = ? WHERE id = ?",
        name[:255], result["source_format"], result["page_count"], result["pages_json"], result["paras_json"],
        len(result["pdf"]), blob_name, result["sha256"], doc_id)
    db.run("DELETE FROM views WHERE document_id = ?", doc_id)
    # The old file's pages and paragraphs no longer exist, so what was read of them cannot carry over.
    coverage.reset(doc_id)
    documents.remove(cfg, doc["blob_name"])
    audit.log("doc_uploaded", document=doc, detail=f"replaced with {name}, {result['page_count']} pages")
    flash(f"The chapter file was replaced ({result['page_count']} pages). Access rules and notes are unchanged; "
          "reviewers' reading progress on this chapter starts again.")
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
    documents.remove(current_app.config["VAULT"], doc["blob_name"])
    flash(f"“{doc['title']}” and all access to it were permanently deleted.")
    return redirect(url_for("admin.chapters"))


@bp.post("/chapters/<int:doc_id>/preview")
def chapter_preview(doc_id: int):
    doc = _doc_or_404(doc_id)
    view_id = start_view(doc, None)
    audit.log("doc_preview", document=doc, view_id=view_id)
    target = url_for("reviewer.viewer", view_id=view_id)
    # "Show on page" buttons in the feedback list open the preview at one particular note.
    page, note = request.form.get("page", type=int), request.form.get("note", type=int)
    if page:
        target += f"#p={page}" + (f"&n={note}" if note else "")
    return redirect(target)


@bp.get("/chapters/<int:doc_id>/feedback.csv")
def chapter_feedback_csv(doc_id: int):
    doc = _doc_or_404(doc_id)
    rows = db.all(
        "SELECT c.id, c.page, u.name, u.email, c.created_at, c.quote, c.body, c.edited_at FROM comments c "
        "JOIN users u ON u.id = c.user_id WHERE c.document_id = ? ORDER BY c.page, c.y0, c.id", doc_id)
    labels = categories_of([r["id"] for r in rows])
    return _csv(f"feedback-chapter-{doc['id']}.csv",
                ["Page", "Reviewer", "Email", "Written (UTC)", "Category", "Passage", "Feedback", "Last edited (UTC)"],
                [(r["page"], r["name"], r["email"], r["created_at"], "; ".join(c["name"] for c in labels[r["id"]]),
                  r["quote"], r["body"], r["edited_at"]) for r in rows])


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
    return render_template("admin/reviewers.html", people=people, docs=docs, defaults=_default_rules(),
                           mail_ready=_mail() is not None)


@bp.post("/reviewers/add")
def reviewer_add():
    name = request.form.get("name", "").strip()[:120]
    email = request.form.get("email", "").strip().lower()[:200]
    if not name or not mailer.valid_address(email):
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
    added = f"{name} was added" + (f" with access to {granted} chapter{'s' if granted != 1 else ''}" if granted else "")
    if _mail() is None:
        flash(added + ". Send them the invite link below so they can set a password.")
    elif request.form.get("send_email") != "1":
        flash(added + ". No email was sent: use the button below when you are ready, or copy the link.")
    else:
        problem = _email_invite(user)
        if problem is None:
            flash(added + f". Their invitation was emailed to {email}.")
        else:
            flash(added + ", but the invitation could not be emailed. You can copy the link below instead.", "warn")
            flash(problem, "bad")
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
        mail_ready=_mail() is not None,
        devices=db.all("SELECT * FROM devices WHERE user_id = ? ORDER BY last_seen DESC", user_id),
        events=db.all("SELECT * FROM events WHERE user_id = ? AND kind NOT IN ('page_view', 'page_read') ORDER BY id DESC LIMIT 15",
                      user_id),
        defaults=_default_rules())


@bp.post("/reviewers/<int:user_id>/update")
def reviewer_update(user_id: int):
    person = _reviewer_or_404(user_id)
    name = request.form.get("name", "").strip()[:120]
    email = request.form.get("email", "").strip().lower()[:200]
    devices = request.form.get("max_devices", "").strip()
    if not name or not mailer.valid_address(email):
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


@bp.post("/reviewers/<int:user_id>/send-invite")
def reviewer_send_invite(user_id: int):
    """Email the invitation (or, for someone who already joined, a password-reset link)."""
    person = _reviewer_or_404(user_id)
    if _mail() is None:
        flash("Email is not set up yet. Add a mail account under Settings, or copy the link and send it yourself.", "bad")
    elif person["status"] == "suspended":
        flash(f"{person['name']} is suspended. Re-activate them before sending a link.", "bad")
    else:
        problem = _email_invite(person)
        if problem is None:
            flash(f"The {'invitation' if person['status'] == 'invited' else 'password reset link'} "
                  f"was emailed to {person['email']}.")
        else:
            flash(problem, "bad")
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

@bp.get("/grants/<int:grant_id>/coverage")
def grant_coverage(grant_id: int):
    """What one reviewer has and has not covered in one chapter, paragraph by paragraph."""
    grant = db.one("SELECT * FROM grants WHERE id = ?", grant_id) or abort(404)
    doc = _doc_or_404(grant["document_id"])
    person = _reviewer_or_404(grant["user_id"])
    standing = coverage.state(person["id"], doc)
    cfg, key = current_app.config["VAULT"], current_app.config["BLOB_KEY"]
    skipped, shown = [], 0
    for page in standing["pages"]:
        if not page["left"]:
            continue
        entry = {"n": page["n"], "paragraphs": page["paragraphs"], "items": [], "more": 0}
        if page["paragraphs"]:
            try:
                outlines = documents.page_blocks(cfg, key, doc, page["n"] - 1)
            except (OSError, ValueError, RuntimeError):
                outlines = []
            for index in page["left"]:
                if shown >= 150 or index >= len(outlines):   # keep the page readable for a barely started chapter
                    entry["more"] += 1
                    continue
                entry["items"].append(documents.passage_text(cfg, key, doc, page["n"] - 1, outlines[index], limit=220))
                shown += 1
        skipped.append(entry)
    how = Counter(v for page in standing["pages"] for v in page["covered"].values())
    return render_template("admin/coverage.html", grant=grant, doc=doc, person=person, standing=standing,
                           skipped=skipped, how=how)


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


# --------------------------------------------------------------- categories

def _category_fields() -> tuple[str, str, str] | None:
    """Name, explanation and colour from a category form, or None after flashing what is wrong."""
    name = " ".join(request.form.get("name", "").split())[:40]
    description = " ".join(request.form.get("description", "").split())[:240]
    color = request.form.get("color", "")
    if not name:
        flash("Give the category a short name.", "bad")
        return None
    if color not in db.CATEGORY_COLORS:
        # "Automatic": the colour used least so far, so new categories stay easy to tell apart.
        used = [r["color"] for r in db.all("SELECT color FROM categories")]
        color = min(db.CATEGORY_COLORS, key=lambda c: (used.count(c), db.CATEGORY_COLORS.index(c)))
    return name, description, color


@bp.get("/categories")
def categories():
    rows = db.all(
        "SELECT c.*, (SELECT COUNT(*) FROM comment_categories cc WHERE cc.category_id = c.id) AS used "
        "FROM categories c ORDER BY c.active DESC, c.position, c.id")
    return render_template("admin/categories.html", rows=rows, colors=db.CATEGORY_COLORS)


@bp.post("/categories/add")
def category_add():
    fields = _category_fields()
    if fields is not None:
        name, description, color = fields
        if db.scalar("SELECT 1 FROM categories WHERE name = ?", name):
            flash(f"There is already a category called “{name}”.", "bad")
        else:
            position = (db.scalar("SELECT MAX(position) FROM categories") or 0) + 1
            db.run("INSERT INTO categories (name, description, color, position, created_at) VALUES (?,?,?,?,?)",
                   name, description, color, position, db.now_iso())
            audit.log("categories_changed", detail=f"added: {name}")
            flash(f"“{name}” was added. Reviewers can choose it straight away.")
    return redirect(url_for("admin.categories"))


@bp.post("/categories/<int:category_id>/update")
def category_update(category_id: int):
    category = db.one("SELECT * FROM categories WHERE id = ?", category_id) or abort(404)
    action = request.form.get("action", "save")
    if action in ("up", "down"):
        order = [r["id"] for r in db.all("SELECT id FROM categories WHERE active = 1 ORDER BY position, id")]
        if category_id in order:
            at = order.index(category_id)
            to = at - 1 if action == "up" else at + 1
            if 0 <= to < len(order):
                order[at], order[to] = order[to], order[at]
                for position, cid in enumerate(order):
                    db.run("UPDATE categories SET position = ? WHERE id = ?", position, cid)
    elif action == "retire":
        db.run("UPDATE categories SET active = 0 WHERE id = ?", category_id)
        audit.log("categories_changed", detail=f"retired: {category['name']}")
        flash(f"“{category['name']}” is retired. Reviewers can no longer choose it; notes that have it keep it.", "warn")
    elif action == "restore":
        db.run("UPDATE categories SET active = 1 WHERE id = ?", category_id)
        audit.log("categories_changed", detail=f"restored: {category['name']}")
        flash(f"“{category['name']}” is available to reviewers again.")
    elif action == "delete":
        if db.scalar("SELECT COUNT(*) FROM comment_categories WHERE category_id = ?", category_id):
            db.run("UPDATE categories SET active = 0 WHERE id = ?", category_id)
            flash(f"“{category['name']}” is on notes already written, so it was retired instead of deleted. "
                  "Those notes keep their label.", "warn")
        else:
            db.run("DELETE FROM categories WHERE id = ?", category_id)
            flash(f"“{category['name']}” was deleted.")
        audit.log("categories_changed", detail=f"removed: {category['name']}")
    else:
        fields = _category_fields()
        if fields is not None:
            name, description, color = fields
            if db.scalar("SELECT 1 FROM categories WHERE name = ? AND id != ?", name, category_id):
                flash(f"There is already a category called “{name}”.", "bad")
            else:
                db.run("UPDATE categories SET name = ?, description = ?, color = ? WHERE id = ?",
                       name, description, color, category_id)
                audit.log("categories_changed", detail=f"edited: {name}")
                flash(f"“{name}” saved.")
    return redirect(url_for("admin.categories"))


# ----------------------------------------------------------------- activity

EVENT_GROUPS = {
    "alerts": sorted(audit.ALERTS),
    "reading": ["doc_open", "page_view", "page_read", "comment_added", "comment_edited", "comment_deleted",
                "nda_accepted"],
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
        where.append("kind NOT IN ('page_view', 'page_read')")
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
        watermark_lines = [line.strip() for line in form.get("watermark_text", "").replace("\r", "").split("\n")]
        watermark_lines = [line for line in watermark_lines if line]
        if len(watermark_lines) > MAX_WATERMARK_LINES or any(len(line) > 120 for line in watermark_lines):
            problems.append(f"The watermark can have up to {MAX_WATERMARK_LINES} lines of up to 120 characters each.")
        host = form.get("smtp_host", "").strip().lower()
        sender = form.get("mail_from_address", "").strip()
        port = form.get("smtp_port", "").strip() or "587"
        if host or sender:
            if not host or not re.fullmatch(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?", host):
                problems.append("Enter the mail server's name, for example smtp.gmail.com.")
            if not mailer.valid_address(sender):
                problems.append("Enter the email address invitations should come from.")
            if not port.isdigit() or not 1 <= int(port) <= 65535:
                problems.append("The mail server port must be a number, usually 587 or 465.")
            if form.get("smtp_security") not in mailer.SECURITY_MODES:
                problems.append("Choose the connection type for the mail server.")
        reply_to = form.get("mail_reply_to", "").strip()
        if reply_to and not mailer.valid_address(reply_to):
            problems.append("The address for replies is not a valid email address.")
        if problems:
            for problem in problems:
                flash(problem, "bad")
        else:
            for key in ("book_title", "author_name", "default_expiry_days", "default_max_views",
                        "default_max_devices", "watermark_footer", "watermark_opacity"):
                db.set_setting(key, form.get(key, "").strip()[:300])
            db.set_setting("smtp_host", host)
            db.set_setting("smtp_port", port)
            db.set_setting("smtp_security", form.get("smtp_security", "starttls"))
            db.set_setting("smtp_username", form.get("smtp_username", "").strip()[:200])
            db.set_setting("mail_from_address", sender)
            db.set_setting("mail_reply_to", reply_to)
            db.set_setting("mail_from_name", " ".join(form.get("mail_from_name", "").split())[:120])
            db.set_setting("invite_subject", " ".join(form.get("invite_subject", "").split())[:200])
            db.set_setting("invite_message", form.get("invite_message", "").replace("\r", "").strip()[:4000])
            # A blank password box means "keep the one already saved"; it is never sent back to the browser.
            if form.get("smtp_password"):
                db.set_setting("smtp_password_enc",
                               mailer.seal_password(current_app.config["MAIL_KEY"], form["smtp_password"]))
            elif not host:
                db.set_setting("smtp_password_enc", "")
            db.set_setting("watermark_text", "\n".join(watermark_lines))
            db.set_setting("nda_text", form.get("nda_text", "").strip()[:8000])
            db.set_setting("require_nda", "1" if form.get("require_nda") == "1" else "0")
            audit.log("settings_updated")
            flash("Settings saved.")
        return redirect(url_for("admin.settings"))
    return render_template("admin/settings.html", mail_ready=_mail() is not None,
                           security_modes=mailer.SECURITY_MODES, default_subject=mailer.DEFAULT_INVITE_SUBJECT,
                           default_message=mailer.DEFAULT_INVITE_MESSAGE)


@bp.post("/settings/test-email")
def settings_test_email():
    """Send a message to the author's own address, to prove the saved mail account works."""
    config = _mail()
    if config is None:
        flash("Save a mail server and a sender address first.", "bad")
    else:
        try:
            mailer.send(config, g.user["name"], g.user["email"], "ChapterVault test email",
                        "This is a test from your ChapterVault server.\n\n"
                        "If you can read it, invitations to reviewers will be delivered the same way.")
        except mailer.MailError as exc:
            audit.log("email_failed", detail=f"test: {exc}")
            flash(str(exc), "bad")
        else:
            audit.log("email_test", detail=g.user["email"])
            flash(f"A test email was sent to {g.user['email']}. Check your inbox (and the spam folder).")
    return redirect(url_for("admin.settings"))
