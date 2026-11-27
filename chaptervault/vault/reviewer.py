"""The reviewer's library and the secure page-image reader."""
from __future__ import annotations

import re
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import timedelta
from functools import wraps

from flask import (Blueprint, Response, abort, current_app, g, jsonify, redirect, render_template,
                   request, url_for)

from . import access, audit, coverage, db, documents
from . import progress as reporting
from .auth import flash, role_required

bp = Blueprint("reviewer", __name__)

VIEW_IDLE = timedelta(minutes=45)
VIEW_LIFETIME = timedelta(hours=8)
PAGES_PER_MINUTE = 90
CLIENT_EVENTS = {"print_attempt", "capture_key", "copy_attempt", "automation"}
BRIEF_ABSENCE = 15   # seconds: leaving the reader for less than this is counted separately
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
        standing = coverage.state(g.user["id"], doc)
        noted = Counter(r["page"] for r in db.all(
            "SELECT page FROM comments WHERE user_id = ? AND document_id = ?", g.user["id"], doc["id"]))
        # One entry per page, for the page map: fully read, partly read or untouched, and where notes are.
        pages = [{"n": page["n"], "state": page["state"], "done": page["done"], "total": page["total"],
                  "paragraphs": page["paragraphs"], "notes": noted.get(page["n"], 0)} for page in standing["pages"]]
        unread = [page["n"] for page in pages if page["state"] != "full"]
        expires = db.parse_iso(grant["expires_at"])
        items.append({
            "doc": doc, "grant": grant, "code": code, "message": message,
            "pages": pages, "unread": unread, "pages_read": doc["page_count"] - len(unread),
            "left": standing["left"], "total": standing["total"], "reviewed": bool(grant["reviewed_at"]),
            "days_left": (expires - db.utcnow()).days if expires else None,
            "opens_left": (grant["max_views"] - grant["views_used"]) if grant["max_views"] is not None else None,
            "comments": sum(noted.values()),
        })

    # The reviewer's own line from the author's progress report: the same figures, theirs only.
    snap = reporting.snapshot()
    mine = next((person for person in snap["people"] if person.id == g.user["id"]), None)
    summary = None
    if mine is not None and mine.total:
        heading, detail = reporting.outlook(mine, snap)
        kinds = db.all(
            "SELECT c.name, c.color, c.description, COUNT(*) AS used FROM comment_categories cc "
            "JOIN comments m ON m.id = cc.comment_id JOIN categories c ON c.id = cc.category_id "
            "WHERE m.user_id = ? GROUP BY c.id ORDER BY c.position, c.id", g.user["id"])
        summary = {"me": mine, "heading": heading, "detail": detail, "kinds": kinds, "target": snap["target"],
                   "days_left": snap["days_left"], "icon": reporting.status_icon(mine.tone),
                   "chart": reporting.sparkline(mine, snap, width=300, height=84, css="pv-spark pv-spark-large")}
    return render_template("reviewer/library.html", items=items, summary=summary, short_date=reporting.short_date)


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
    def reader(view_id: int):
        """The reader's address, at the page asked for when the reviewer clicked a cell of the page map."""
        page = request.form.get("page", type=int)
        at = f"#p={page}" if page and 1 <= page <= doc["page_count"] else ""
        return redirect(url_for("reviewer.viewer", view_id=view_id) + at)

    if live is not None:  # coming back to a chapter already open in this sitting is not a new "open"
        return reader(live["id"])

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
    return reader(view_id)


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
        "readUrl": url_for("reviewer.mark_read", view_id=view_id),
        "coverUrl": url_for("reviewer.cover", view_id=view_id),
        "reviewedUrl": url_for("reviewer.mark_reviewed", view_id=view_id),
        "coverage": _coverage_for_reader(doc, grant),
        "commentsUrl": url_for("reviewer.comments", view_id=view_id),
        "blocksUrl": url_for("reviewer.blocks", view_id=view_id, number=0).rsplit("/", 1)[0] + "/",
        "exitUrl": _exit_url() if grant is not None else url_for("admin.chapter", doc_id=doc["id"]),
        "csrf": g.session["csrf"],
        "canComment": bool(grant is not None and grant["allow_comments"]),
        "categories": active_categories(),
        "isAuthor": grant is None,
        "briefAbsence": BRIEF_ABSENCE,
    }
    return render_template("viewer.html", doc=doc, config=config, grant=grant)


WATERMARK_FIELDS = ("name", "email", "date", "ip", "ref", "book", "author")
_PLACEHOLDER = re.compile(r"\{(%s)\}" % "|".join(WATERMARK_FIELDS))
MAX_WATERMARK_LINES = 3


def _fill(template: str, values: dict[str, str]) -> str:
    """Replace the known {placeholders}; anything else in the author's text is left exactly as typed."""
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template).strip(" ·|-\t")


def watermark_for(view, user, settings) -> documents.Watermark:
    """Build the watermark for one reading session from the texts the author wrote in Settings."""
    opacity = max(4, min(40, int(settings["watermark_opacity"] or 14))) / 100
    date, _, ip = view["stamp"].partition("  ·  ")
    if view["grant_id"] is None:
        # The author's preview shows what a reviewer will get, so it uses stand-in reader details
        # rather than the author's own name and sign-in address.
        values = {"name": "Reviewer Name", "email": "reviewer@example.com", "ip": "203.0.113.7"}
    else:
        values = {"name": user["name"], "email": user["email"], "ip": ip}
    values.update(date=date, ref=f"V{view['id']:05d}", book=settings["book_title"], author=settings["author_name"])
    lines = [_fill(line, values) for line in settings["watermark_text"].splitlines()]
    return documents.Watermark(lines=tuple(line for line in lines if line)[:MAX_WATERMARK_LINES],
                               footer=_fill(settings["watermark_footer"], values), opacity=opacity)


def _watermark(view) -> documents.Watermark:
    return watermark_for(view, g.user, db.settings())


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


def _small_count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 60 else 0


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
    # How often the reader window was left since the last heartbeat. It is kept as a plain count:
    # switching to another program to take notes and switching to one that captures the screen
    # look the same from here.
    away = (request.get_json(silent=True) or {}).get("away")
    if grant is not None and isinstance(away, dict):
        times = _small_count(away.get("n"))
        brief = min(_small_count(away.get("brief")), times)
        if times:
            db.run("UPDATE grants SET away_count = away_count + ?, away_brief = away_brief + ? WHERE id = ?",
                   times, brief, grant["id"])
    db.run("UPDATE views SET last_seen = ?, pinged_at = ? WHERE id = ?", db.iso(now), db.iso(now), view_id)
    return jsonify(ok=True)


@bp.post("/view/<int:view_id>/read")
@api
def mark_read(view_id: int):
    """The reader reports pages with no paragraphs (a figure, a scan) that stayed on screen.

    Pages with text are counted as read through their paragraphs instead. A page is only accepted
    if this reading session was actually served it, and counts once per reviewer and chapter.
    """
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message)
    pages = (request.get_json(silent=True) or {}).get("pages")
    if not isinstance(pages, list) or len(pages) > 50:
        abort(400)
    recorded = 0
    if grant is not None:   # the author's own preview is not anyone's progress
        words = coverage.words_by_page(doc)
        for number in pages:
            if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= doc["page_count"]:
                continue
            if coverage.required(words[number - 1] if number <= len(words) else []):
                continue   # a page with paragraphs is read when its paragraphs are: see cover()
            served = db.scalar("SELECT 1 FROM events WHERE kind = 'page_view' AND view_id = ? AND page = ?",
                               view_id, number)
            already = db.scalar("SELECT 1 FROM events WHERE kind = 'page_read' AND user_id = ? AND document_id = ? "
                                "AND page = ?", g.user["id"], doc["id"], number)
            if served and not already:
                audit.log("page_read", document=doc, view_id=view_id, page=number)
                recorded += 1
    return jsonify(ok=True, recorded=recorded)


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
    if kind == "automation" and g.user["role"] == "reviewer":
        # The reader found itself inside a browser driven by a program. It closes itself; ending
        # the reading session here as well stops the pages being served to whatever is driving it.
        db.run("DELETE FROM views WHERE id = ?", view_id)
    return jsonify(ok=True)


# ----------------------------------------------------------------- feedback

MAX_CATEGORIES_PER_NOTE = 8


def active_categories() -> list[dict]:
    """The kinds of note reviewers can choose from, in the author's order."""
    return [{"id": r["id"], "name": r["name"], "description": r["description"], "color": r["color"]}
            for r in db.all("SELECT * FROM categories WHERE active = 1 ORDER BY position, id")]


def categories_of(comment_ids: list[int]) -> dict[int, list[dict]]:
    """The categories attached to each of the given notes (retired ones included)."""
    found: dict[int, list[dict]] = {cid: [] for cid in comment_ids}
    if comment_ids:
        marks = ",".join("?" * len(comment_ids))
        for r in db.all(
                f"SELECT cc.comment_id, c.id, c.name, c.description, c.color FROM comment_categories cc "
                f"JOIN categories c ON c.id = cc.category_id WHERE cc.comment_id IN ({marks}) "
                f"ORDER BY c.position, c.id", *comment_ids):
            found[r["comment_id"]].append({"id": r["id"], "name": r["name"], "description": r["description"],
                                           "color": r["color"]})
    return found


def _comment_json(row, for_author: bool, categories: list[dict]) -> dict:
    data = {"id": row["id"], "page": row["page"], "body": row["body"], "author": row["name"],
            "created": row["created_at"], "edited": row["edited_at"], "mine": row["user_id"] == g.user["id"],
            "box": [row["x0"], row["y0"], row["x1"], row["y1"]] if row["x0"] is not None else None,
            "categories": categories}
    if for_author:
        # The quoted passage goes to the author only. Sending it to reviewers would hand them the
        # manuscript's text one marked paragraph at a time.
        data["quote"] = row["quote"]
    return data


def _one_comment(comment_id: int, for_author: bool = False) -> dict:
    row = db.one("SELECT c.*, u.name FROM comments c JOIN users u ON u.id = c.user_id WHERE c.id = ?", comment_id)
    return _comment_json(row, for_author, categories_of([comment_id])[comment_id])


def _clean_categories(value, allowed: set[int]) -> list[int]:
    """Validate the category ids sent with a note. Raises ValueError on anything unexpected."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_CATEGORIES_PER_NOTE:
        raise ValueError
    chosen: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item not in allowed:
            raise ValueError
        if item not in chosen:
            chosen.append(item)
    return chosen


def _clean_box(value) -> list[float] | None:
    """Validate a marked area sent by the reader: four fractions of the page, left/top/right/bottom."""
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
        raise ValueError
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in value)
    if x1 - x0 < 0.005 or y1 - y0 < 0.003:
        raise ValueError
    return [round(v, 4) for v in (x0, y0, x1, y1)]


def _coverage_for_reader(doc, grant) -> dict | None:
    """What the reader needs to guide a reviewer through a chapter: how much is left, and where."""
    if grant is None:
        return None
    standing = coverage.state(g.user["id"], doc)
    return {"total": standing["total"], "left": standing["left"], "reviewed": bool(grant["reviewed_at"]),
            "pagesLeft": [len(page["left"]) for page in standing["pages"]],
            "minWords": coverage.MIN_WORDS, "wordsPerMinute": coverage.MAX_WPM}


@bp.get("/view/<int:view_id>/blocks/<int:number>")
@api
def blocks(view_id: int, number: int):
    """Paragraph outlines for one page: for pinning notes, and for tracking which have been read.

    Rectangles and word counts only. No text is sent.
    """
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message, number)
    if not 1 <= number <= doc["page_count"]:
        abort(404)
    if grant is None:   # the author's preview neither pins notes nor tracks reading
        return jsonify(blocks=[])
    try:
        paragraphs = documents.page_paragraphs(current_app.config["VAULT"], current_app.config["BLOB_KEY"],
                                               doc, number - 1)
    except (OSError, ValueError, RuntimeError):
        paragraphs = []
    done = {row["para"]: row["how"] for row in db.all(
        "SELECT para, how FROM coverage WHERE user_id = ? AND document_id = ? AND page = ?",
        g.user["id"], doc["id"], number)}
    return jsonify(blocks=[box for box, _words in paragraphs], words=[words for _box, words in paragraphs],
                   covered={str(index): how for index, how in done.items()})


@bp.post("/view/<int:view_id>/cover")
@api
def cover(view_id: int):
    """The reader reports paragraphs that have been read (on screen long enough) or ticked by hand."""
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message)
    payload = request.get_json(silent=True) or {}
    number, paragraphs, how = payload.get("page"), payload.get("paras"), payload.get("how")
    if (isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= doc["page_count"]
            or not isinstance(paragraphs, list) or len(paragraphs) > 400 or how not in ("seen", "confirmed")
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in paragraphs)):
        abort(400)
    if grant is None:
        return jsonify(ok=True, left=0)
    # Only pages this reading session was actually given can be reported as read.
    if db.scalar("SELECT 1 FROM events WHERE kind = 'page_view' AND view_id = ? AND page = ?", view_id, number):
        coverage.record(g.user, doc, view_id, number, paragraphs, how)
    standing = coverage.state(g.user["id"], doc)
    return jsonify(ok=True, left=standing["left"], pageLeft=len(standing["pages"][number - 1]["left"]))


@bp.post("/view/<int:view_id>/reviewed")
@api
def mark_reviewed(view_id: int):
    """The reviewer declares the chapter reviewed. Refused while any paragraph is still uncovered."""
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message)
    if grant is None:
        abort(403)
    if (request.get_json(silent=True) or {}).get("done") is False:
        if grant["reviewed_at"]:
            db.run("UPDATE grants SET reviewed_at = NULL WHERE id = ?", grant["id"])
            audit.log("chapter_reopened", document=doc, view_id=view_id)
        return jsonify(ok=True, reviewed=False)
    left = coverage.state(g.user["id"], doc)["left"]
    if left:
        return jsonify(error="incomplete", left=left,
                       message=f"{left} paragraph{'' if left == 1 else 's'} still to read before this chapter "
                               "can be marked as reviewed."), 409
    if not grant["reviewed_at"]:
        db.run("UPDATE grants SET reviewed_at = ? WHERE id = ?", db.now_iso(), grant["id"])
        audit.log("chapter_reviewed", document=doc, view_id=view_id)
    return jsonify(ok=True, reviewed=True)


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
        try:
            box = _clean_box(payload.get("box"))
        except ValueError:
            return jsonify(error="invalid", message="That selection could not be used. Select the passage again."), 400
        try:
            chosen = _clean_categories(payload.get("categories"), {c["id"] for c in active_categories()})
        except ValueError:
            return jsonify(error="invalid", message="One of those categories is no longer available. "
                                                    "Reload the page and try again."), 400
        if (len(body) > MAX_COMMENT or isinstance(page_no, bool) or not isinstance(page_no, int)
                or not 1 <= page_no <= doc["page_count"]):
            return jsonify(error="invalid", message=f"Write a note of up to {MAX_COMMENT} characters."), 400
        if not body and not chosen:
            return jsonify(error="invalid", message="Pick a category or write a note."), 400
        quote = ""
        if box is not None:
            try:
                quote = documents.passage_text(current_app.config["VAULT"], current_app.config["BLOB_KEY"],
                                               doc, page_no - 1, box)
            except (OSError, ValueError, RuntimeError):
                quote = ""
        x0, y0, x1, y1 = box or (None, None, None, None)
        comment_id = db.run(
            "INSERT INTO comments (document_id, user_id, page, body, created_at, x0, y0, x1, y1, quote) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            doc["id"], g.user["id"], page_no, body, db.now_iso(), x0, y0, x1, y1, quote).lastrowid
        for category_id in chosen:
            db.run("INSERT INTO comment_categories (comment_id, category_id) VALUES (?, ?)", comment_id, category_id)
        audit.log("comment_added", document=doc, view_id=view_id, page=page_no)
        if box is not None:   # commenting on a paragraph is the clearest sign it was reviewed
            try:
                outlines = documents.page_blocks(current_app.config["VAULT"], current_app.config["BLOB_KEY"],
                                                 doc, page_no - 1)
                coverage.record(g.user, doc, view_id, page_no, coverage.overlapped(outlines, box), "noted")
            except (OSError, ValueError, RuntimeError):
                pass
        return jsonify(_one_comment(comment_id)), 201
    if grant is None:  # the author sees everyone's notes while previewing
        rows = db.all("SELECT c.*, u.name FROM comments c JOIN users u ON u.id = c.user_id "
                      "WHERE c.document_id = ? ORDER BY c.page, c.id", doc["id"])
    else:
        rows = db.all("SELECT c.*, u.name FROM comments c JOIN users u ON u.id = c.user_id "
                      "WHERE c.document_id = ? AND c.user_id = ? ORDER BY c.page, c.id", doc["id"], g.user["id"])
    attached = categories_of([r["id"] for r in rows])
    return jsonify([_comment_json(r, grant is None, attached[r["id"]]) for r in rows])


@bp.post("/view/<int:view_id>/comments/<int:comment_id>/edit")
@api
def edit_comment(view_id: int, comment_id: int):
    """Reword one of your own notes. The passage it is pinned to stays where it is."""
    view, doc, grant, code, message = _view_state(view_id)
    if code != "ok":
        return _deny(view, doc, code, message)
    if grant is None or not grant["allow_comments"]:
        abort(403)
    payload = request.get_json(silent=True) or {}
    body = str(payload.get("body", "")).strip()
    existing = db.one("SELECT * FROM comments WHERE id = ? AND user_id = ? AND document_id = ?",
                      comment_id, g.user["id"], doc["id"])
    if existing is None:
        abort(404)
    current = [c["id"] for c in categories_of([comment_id])[comment_id]]
    chosen = current
    if "categories" in payload:
        try:  # a retired category already on the note may stay; it cannot be newly added
            chosen = _clean_categories(payload["categories"], {c["id"] for c in active_categories()} | set(current))
        except ValueError:
            return jsonify(error="invalid", message="One of those categories is no longer available. "
                                                    "Reload the page and try again."), 400
    if len(body) > MAX_COMMENT:
        return jsonify(error="invalid", message=f"Write a note of up to {MAX_COMMENT} characters."), 400
    if not body and not chosen:
        return jsonify(error="invalid", message="Pick a category or write a note."), 400
    if body != existing["body"] or set(chosen) != set(current):
        db.run("UPDATE comments SET body = ?, edited_at = ? WHERE id = ?", body, db.now_iso(), comment_id)
        db.run("DELETE FROM comment_categories WHERE comment_id = ?", comment_id)
        for category_id in chosen:
            db.run("INSERT INTO comment_categories (comment_id, category_id) VALUES (?, ?)", comment_id, category_id)
        audit.log("comment_edited", document=doc, view_id=view_id, page=existing["page"])
    return jsonify(_one_comment(comment_id))


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
