"""Accounts, sessions, device binding and CSRF protection."""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from datetime import timedelta
from functools import wraps

from flask import (Blueprint, abort, current_app, g, redirect, render_template, request,
                   url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from . import audit, db, security

bp = Blueprint("auth", __name__)

SESSION_COOKIE = "cv_session"
DEVICE_COOKIE = "cv_device"
CSRF_COOKIE = "cv_csrf"

IDLE_TIMEOUT = {"admin": timedelta(hours=2), "reviewer": timedelta(minutes=60)}
SESSION_LIFETIME = timedelta(hours=12)
INVITE_DAYS = 14
MAX_FAILED_LOGINS = 5
LOCK_MINUTES = 15
MIN_PASSWORD = 10

# A throwaway hash so that sign-in takes the same time whether or not the email exists.
_DUMMY_HASH = generate_password_hash("not-a-real-password")

_attempts_lock = threading.Lock()
IP_FAILURE_LIMIT = 25      # failed sign-ins allowed from one address ...
IP_FAILURE_WINDOW = 600    # ... within this many seconds


def _ip_bucket(ip: str) -> deque:
    bucket = current_app.config["LOGIN_FAILURES"][ip]
    now = time.monotonic()
    while bucket and now - bucket[0] > IP_FAILURE_WINDOW:
        bucket.popleft()
    return bucket


def _ip_blocked(ip: str) -> bool:
    with _attempts_lock:
        return len(_ip_bucket(ip)) >= IP_FAILURE_LIMIT


def _note_failure(ip: str) -> None:
    with _attempts_lock:
        _ip_bucket(ip).append(time.monotonic())


# ------------------------------------------------------------------ helpers

def password_problem(password: str, confirm: str, email: str = "") -> str | None:
    if len(password) < MIN_PASSWORD:
        return f"Choose a password of at least {MIN_PASSWORD} characters."
    if password != confirm:
        return "The two passwords do not match."
    if email and password.lower() == email.lower():
        return "Your password cannot be your email address."
    return None


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def external_url(path: str) -> str:
    base = current_app.config["VAULT"].base_url or request.url_root.rstrip("/")
    return base + path


def device_label(user_agent: str) -> str:
    ua = user_agent or ""
    browser = next((name for key, name in (("Edg/", "Edge"), ("OPR/", "Opera"), ("Firefox/", "Firefox"),
                                            ("Chrome/", "Chrome"), ("Safari/", "Safari")) if key in ua),
                   "Browser")
    system = next((name for key, name in (("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"),
                                           ("Windows", "Windows"), ("Mac OS X", "Mac"), ("Linux", "Linux"))
                   if key in ua), "unknown device")
    return f"{browser} on {system}"


def flash(message: str, kind: str = "ok") -> None:
    """Queue a one-time message, stored server-side with the session."""
    if g.session is None:
        return
    pending = json.loads(db.scalar("SELECT flash FROM sessions WHERE id = ?", g.session["id"]) or "[]")
    pending.append([kind, message])
    db.run("UPDATE sessions SET flash = ? WHERE id = ?", json.dumps(pending), g.session["id"])


def take_flashes() -> list[list[str]]:
    if g.get("session") is None:
        return []
    raw = db.scalar("SELECT flash FROM sessions WHERE id = ?", g.session["id"]) or "[]"
    if raw != "[]":
        db.run("UPDATE sessions SET flash = '[]' WHERE id = ?", g.session["id"])
    return json.loads(raw)


def csrf_token() -> str:
    if g.get("session") is not None:
        return g.session["csrf"]
    raw = request.cookies.get(CSRF_COOKIE) or g.get("new_csrf")
    if not raw:
        raw = g.new_csrf = security.new_token(18)
    return security.mac(current_app.config["CSRF_KEY"], raw)


def _check_csrf() -> None:
    sent = request.form.get("_csrf") or request.headers.get("X-CSRF-Token", "")
    if g.session is not None:
        expected = g.session["csrf"]
    else:
        raw = request.cookies.get(CSRF_COOKIE)
        expected = security.mac(current_app.config["CSRF_KEY"], raw) if raw else ""
    if not sent or not expected or not security.safe_eq(sent, expected):
        abort(400, "This form has expired. Go back, reload the page and try again.")


def home_for(user) -> str:
    return url_for("admin.dashboard") if user["role"] == "admin" else url_for("reviewer.library")


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("auth.login")) if request.method == "GET" else abort(401)
        return view(*args, **kwargs)
    return wrapper


def role_required(role: str):
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if g.user is None:
                return redirect(url_for("auth.login")) if request.method == "GET" else abort(401)
            if g.user["role"] != role:
                abort(404)
            return view(*args, **kwargs)
        return wrapper
    return decorator


# ----------------------------------------------------------------- sessions

def _load_session() -> None:
    g.user = None
    g.session = None
    token = request.cookies.get(SESSION_COOKIE)
    if token and not request.path.startswith("/static/"):
        row = db.one("SELECT * FROM sessions WHERE token_hash = ?", security.hash_token(token))
        if row is not None:
            user = db.one("SELECT * FROM users WHERE id = ?", row["user_id"])
            now = db.utcnow()
            last_seen = db.parse_iso(row["last_seen"])
            alive = (
                user is not None
                and user["status"] == "active"
                and db.parse_iso(row["expires_at"]) > now
                and now - last_seen <= IDLE_TIMEOUT[user["role"]]
            )
            if not alive:
                db.run("DELETE FROM sessions WHERE id = ?", row["id"])
            else:
                g.user, g.session = user, row
                if (now - last_seen).total_seconds() > 20:
                    db.run("UPDATE sessions SET last_seen = ? WHERE id = ?", db.iso(now), row["id"])
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        _check_csrf()


def _set_cookies(response):
    raw = g.pop("new_csrf", None)
    if raw:
        response.set_cookie(CSRF_COOKIE, raw, httponly=True, samesite="Strict",
                            secure=request.is_secure, path="/")
    return response


def init_app(app) -> None:
    app.config["LOGIN_FAILURES"] = defaultdict(deque)
    app.before_request(_load_session)
    app.after_request(_set_cookies)

    @app.context_processor
    def _inject():
        return {"csrf_token": csrf_token, "me": g.get("user"), "take_flashes": take_flashes,
                "site": db.settings()}


def end_sessions(user_id: int, keep: int | None = None) -> None:
    if keep is None:
        db.run("DELETE FROM sessions WHERE user_id = ?", user_id)
    else:
        db.run("DELETE FROM sessions WHERE user_id = ? AND id != ?", user_id, keep)


def _complete_login(user):
    """Bind the device, start a session and return ``(response, error)``."""
    now = db.now_iso()
    device_id = None
    new_device_token = None
    if user["role"] == "reviewer":
        raw = request.cookies.get(DEVICE_COOKIE)
        device = None
        if raw:
            device = db.one("SELECT * FROM devices WHERE user_id = ? AND token_hash = ?",
                            user["id"], security.hash_token(raw))
        if device is None:
            registered = db.scalar("SELECT COUNT(*) FROM devices WHERE user_id = ?", user["id"])
            if registered >= user["max_devices"]:
                audit.log("login_blocked_device", user=user,
                          detail=f"{device_label(request.headers.get('User-Agent', ''))}; limit {user['max_devices']}")
                return None, ("This account is already registered on the maximum number of devices. "
                              "Sign in from a device you have used before, or ask the author to reset your devices.")
            new_device_token = raw or security.new_token()
            device_id = db.run(
                "INSERT INTO devices (user_id, token_hash, label, ip, first_seen, last_seen) VALUES (?,?,?,?,?,?)",
                user["id"], security.hash_token(new_device_token),
                device_label(request.headers.get("User-Agent", "")), audit.client_ip(), now, now,
            ).lastrowid
        else:
            device_id = device["id"]
            db.run("UPDATE devices SET last_seen = ?, ip = ? WHERE id = ?", now, audit.client_ip(), device_id)
        # One live session per reviewer: signing in elsewhere ends the previous session,
        # which makes sharing a login impractical.
        end_sessions(user["id"])

    db.run("DELETE FROM sessions WHERE expires_at < ?", now)
    token = security.new_token()
    db.run(
        "INSERT INTO sessions (token_hash, user_id, device_id, csrf, created_at, last_seen, expires_at, ip, user_agent) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        security.hash_token(token), user["id"], device_id, security.new_token(24), now, now,
        db.iso(db.utcnow() + SESSION_LIFETIME), audit.client_ip(),
        request.headers.get("User-Agent", "")[:300],
    )
    db.run("UPDATE users SET last_login_at = ?, failed_logins = 0, locked_until = NULL WHERE id = ?",
           now, user["id"])
    audit.log("login", user=user, detail=device_label(request.headers.get("User-Agent", "")))

    response = redirect(home_for(user))
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="Lax",
                        secure=request.is_secure, path="/")
    if new_device_token:
        response.set_cookie(DEVICE_COOKIE, new_device_token, max_age=400 * 86400, httponly=True,
                            samesite="Lax", secure=request.is_secure, path="/")
    return response, None


# ------------------------------------------------------------------ invites

def create_invite(user_id: int) -> str:
    token = security.new_token()
    key = current_app.config["INVITE_KEY"]
    db.run(
        "UPDATE users SET invite_token_hash = ?, invite_token_enc = ?, invite_expires_at = ? WHERE id = ?",
        security.hash_token(token), security.encrypt(key, token.encode(), b"invite"),
        db.in_future(days=INVITE_DAYS), user_id,
    )
    return token


def invite_link(user) -> str | None:
    """The pending invite URL for a reviewer, or None if there is none or it has lapsed."""
    if not user["invite_token_enc"] or not user["invite_expires_at"]:
        return None
    if db.parse_iso(user["invite_expires_at"]) < db.utcnow():
        return None
    try:
        token = security.decrypt(current_app.config["INVITE_KEY"], bytes(user["invite_token_enc"]), b"invite")
    except ValueError:
        return None
    return external_url(url_for("auth.invite", token=token.decode()))


# ------------------------------------------------------------------- routes

@bp.get("/")
def index():
    if g.user is not None:
        return redirect(home_for(g.user))
    if not db.scalar("SELECT COUNT(*) FROM users WHERE role = 'admin'"):
        return redirect(url_for("auth.setup"))
    return redirect(url_for("auth.login"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(home_for(g.user))
    notice = {"password_set": "Your password is set. Sign in to continue.",
              "signed_out": "You have been signed out.",
              "setup_done": "Your author account is ready. Sign in to start uploading chapters."
              }.get(request.args.get("m", ""))
    if request.method == "GET":
        return render_template("auth/login.html", notice=notice, error=None, email="")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    def fail(message: str, status: int = 401):
        return render_template("auth/login.html", notice=None, error=message, email=email), status

    if _ip_blocked(audit.client_ip()):
        return fail("Too many sign-in attempts from this network. Wait a few minutes and try again.", 429)

    user = db.one("SELECT * FROM users WHERE email = ?", email)
    now = db.utcnow()
    if user is not None and user["locked_until"] and db.parse_iso(user["locked_until"]) > now:
        check_password_hash(_DUMMY_HASH, password)
        return fail("Too many failed attempts. Sign-in for this account is paused for a few minutes.", 429)

    valid = check_password_hash(user["password_hash"] if user is not None and user["password_hash"]
                                else _DUMMY_HASH, password)
    if user is None or not user["password_hash"] or not valid:
        _note_failure(audit.client_ip())
        if user is not None:
            fails = user["failed_logins"] + 1
            if fails >= MAX_FAILED_LOGINS:
                db.run("UPDATE users SET failed_logins = 0, locked_until = ? WHERE id = ?",
                       db.in_future(minutes=LOCK_MINUTES), user["id"])
                audit.log("login_locked", user=user)
            else:
                db.run("UPDATE users SET failed_logins = ? WHERE id = ?", fails, user["id"])
                audit.log("login_failed", user=user)
        else:
            audit.log("login_failed", actor=email[:120], detail="unknown email")
        return fail("Incorrect email or password.")

    if user["status"] == "suspended":
        audit.log("view_denied", user=user, detail="sign-in while suspended")
        return fail("This account has been suspended by the author.", 403)

    response, error = _complete_login(user)
    if error:
        return fail(error, 403)
    return response


@bp.post("/logout")
def logout():
    if g.session is not None:
        audit.log("logout")
        db.run("DELETE FROM sessions WHERE id = ?", g.session["id"])
    response = redirect(url_for("auth.login", m="signed_out"))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@bp.route("/setup", methods=["GET", "POST"])
def setup():
    """First-run creation of the author account, gated by a one-time token printed at start-up."""
    if db.scalar("SELECT COUNT(*) FROM users WHERE role = 'admin'"):
        return redirect(url_for("auth.login"))
    token = request.values.get("token", "")
    expected = db.settings().get("setup_token_hash", "")
    if not token or not expected or not security.safe_eq(security.hash_token(token), expected):
        return render_template("auth/setup.html", locked=True, error=None, form={}), 403
    if request.method == "GET":
        return render_template("auth/setup.html", locked=False, error=None, form={}, token=token)

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    error = None
    if not name or "@" not in email:
        error = "Enter your name and a valid email address."
    else:
        error = password_problem(password, request.form.get("confirm", ""), email)
    if error:
        return render_template("auth/setup.html", locked=False, error=error, token=token,
                               form={"name": name, "email": email}), 400
    now = db.now_iso()
    user_id = db.run(
        "INSERT INTO users (role, name, email, password_hash, status, created_at) VALUES ('admin',?,?,?,'active',?)",
        name, email, hash_password(password), now,
    ).lastrowid
    db.run("DELETE FROM settings WHERE key = 'setup_token_hash'")
    db.set_setting("author_name", name)
    audit.log("setup", user=db.one("SELECT * FROM users WHERE id = ?", user_id))
    return redirect(url_for("auth.login", m="setup_done"))


@bp.route("/invite/<token>", methods=["GET", "POST"])
def invite(token: str):
    user = db.one("SELECT * FROM users WHERE invite_token_hash = ?", security.hash_token(token))
    if (user is None or not user["invite_expires_at"]
            or db.parse_iso(user["invite_expires_at"]) < db.utcnow() or user["status"] == "suspended"):
        return render_template("error.html", title="This invite link is no longer valid",
                               message="Invite links can be used once and expire after "
                                       f"{INVITE_DAYS} days. Ask the author to send you a new one."), 410
    if request.method == "GET":
        return render_template("auth/invite.html", user=user, error=None)
    password = request.form.get("password", "")
    error = password_problem(password, request.form.get("confirm", ""), user["email"])
    if error:
        return render_template("auth/invite.html", user=user, error=error), 400
    db.run(
        "UPDATE users SET password_hash = ?, status = 'active', invite_token_hash = NULL, "
        "invite_token_enc = NULL, invite_expires_at = NULL, failed_logins = 0, locked_until = NULL WHERE id = ?",
        hash_password(password), user["id"],
    )
    end_sessions(user["id"])
    audit.log("invite_accepted", user=user)
    if g.session is not None:  # someone else was signed in on this browser
        db.run("DELETE FROM sessions WHERE id = ?", g.session["id"])
    response = redirect(url_for("auth.login", m="password_set"))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@bp.route("/account", methods=["GET", "POST"])
@login_required
def account():
    error = None
    if request.method == "POST":
        if not check_password_hash(g.user["password_hash"], request.form.get("current", "")):
            error = "Your current password is not correct."
        else:
            error = password_problem(request.form.get("password", ""), request.form.get("confirm", ""),
                                     g.user["email"])
        if not error:
            db.run("UPDATE users SET password_hash = ? WHERE id = ?",
                   hash_password(request.form["password"]), g.user["id"])
            end_sessions(g.user["id"], keep=g.session["id"])
            audit.log("password_changed")
            flash("Your password has been changed.")
            return redirect(url_for("auth.account"))
    devices = db.all("SELECT * FROM devices WHERE user_id = ? ORDER BY last_seen DESC", g.user["id"])
    return render_template("account.html", error=error, devices=devices), (400 if error else 200)
