"""ChapterVault: share manuscript chapters with reviewers under the author's control."""
from __future__ import annotations

import os
from datetime import timezone

from flask import Flask, g, render_template, request
from werkzeug.exceptions import Forbidden, HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from . import audit, auth, db, security
from .config import Config

__version__ = "1.10.0"

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
    "connect-src 'self'; font-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)


def create_app(data_dir: str | os.PathLike | None = None) -> Flask:
    cfg = Config(data_dir)
    cfg.ensure_dirs()
    app = Flask(__name__)
    app.config["VAULT"] = cfg
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_upload_mb * 1024 * 1024
    master = security.load_master_key(cfg)
    # Independent sub-keys so that one purpose can never be confused with another.
    app.config["BLOB_KEY"] = security.derive(master, "blobs")
    app.config["INVITE_KEY"] = security.derive(master, "invites")
    app.config["CSRF_KEY"] = security.derive(master, "csrf")
    app.config["MAIL_KEY"] = security.derive(master, "mail")
    if cfg.behind_proxy:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    db.init_app(app)
    auth.init_app(app)
    from .admin import bp as admin_bp
    from .reviewer import bp as reviewer_bp
    app.register_blueprint(auth.bp)
    app.register_blueprint(reviewer_bp)
    app.register_blueprint(admin_bp, url_prefix="/admin")
    _register_filters(app)

    @app.after_request
    def _harden(response):
        headers = response.headers
        headers["Content-Security-Policy"] = CSP
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-Frame-Options"] = "DENY"
        headers["Referrer-Policy"] = "no-referrer"
        headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), display-capture=()"
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        headers["Cross-Origin-Resource-Policy"] = "same-origin"
        headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        if request.path.startswith("/static/"):
            headers["Cache-Control"] = "public, max-age=600"
        else:
            # Nothing that carries manuscript content or personal data may be kept by a browser or proxy.
            headers["Cache-Control"] = "no-store, private"
            headers["Pragma"] = "no-cache"
        if request.is_secure:
            headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        if request.path.startswith("/view/") and request.path.count("/") > 2:  # the reader's JSON calls
            message = ("This chapter was updated or withdrawn. Go back and open it again."
                       if exc.code == 404 else exc.description or exc.name)
            return {"error": exc.name, "message": message}, exc.code
        titles = {400: "That request could not be processed", 401: "Please sign in",
                  403: "You do not have access to this", 404: "Page not found",
                  413: "That file is too large", 429: "Too many requests"}
        explained = exc.code == 400 or (exc.code == 403 and exc.description != Forbidden.description)
        message = exc.description if explained else {
            401: "Your session has ended. Sign in again to continue.",
            404: "The page you asked for does not exist, or you no longer have access to it.",
            413: f"Uploads are limited to {cfg.max_upload_mb} MB.",
        }.get(exc.code, "")
        return render_template("error.html", title=titles.get(exc.code, exc.name), message=message), exc.code

    @app.get("/robots.txt")
    def _robots():
        return "User-agent: *\nDisallow: /\n", 200, {"Content-Type": "text/plain"}

    _bootstrap(app)
    return app


def _bootstrap(app: Flask) -> None:
    """Create the author account from the environment, or issue a one-time setup link."""
    app.config["SETUP_TOKEN"] = None
    with app.app_context():
        if db.scalar("SELECT COUNT(*) FROM users WHERE role = 'admin'"):
            return
        email = os.environ.get("CHAPTERVAULT_ADMIN_EMAIL", "").strip().lower()
        password = os.environ.get("CHAPTERVAULT_ADMIN_PASSWORD", "")
        if email and password:
            if len(password) < auth.MIN_PASSWORD:
                raise RuntimeError(f"CHAPTERVAULT_ADMIN_PASSWORD must be at least {auth.MIN_PASSWORD} characters")
            name = os.environ.get("CHAPTERVAULT_ADMIN_NAME", "Author")
            db.run(
                "INSERT INTO users (role, name, email, password_hash, status, created_at) "
                "VALUES ('admin',?,?,?,'active',?)", name, email, auth.hash_password(password), db.now_iso())
            db.set_setting("author_name", name)
            return
        # A deployment script may fix the token so the setup link survives restarts; otherwise
        # a fresh one is issued on every start until the author account exists.
        token = os.environ.get("CHAPTERVAULT_SETUP_TOKEN", "").strip()
        if token and len(token) < 32:
            raise RuntimeError("CHAPTERVAULT_SETUP_TOKEN must be at least 32 characters")
        token = token or security.new_token(24)
        db.set_setting("setup_token_hash", security.hash_token(token))
        app.config["SETUP_TOKEN"] = token


def _register_filters(app: Flask) -> None:
    @app.template_filter("dt")
    def _dt(value, fmt="%d %b %Y, %H:%M"):
        parsed = db.parse_iso(value)
        return parsed.astimezone(timezone.utc).strftime(fmt) if parsed else "—"

    @app.template_filter("day")
    def _day(value):
        parsed = db.parse_iso(value)
        return parsed.strftime("%d %b %Y") if parsed else "—"

    @app.template_filter("dateinput")
    def _dateinput(value):
        parsed = db.parse_iso(value)
        return parsed.strftime("%Y-%m-%d") if parsed else ""

    @app.template_filter("ago")
    def _ago(value):
        parsed = db.parse_iso(value)
        if not parsed:
            return "never"
        seconds = int((db.utcnow() - parsed).total_seconds())
        if seconds < 60:
            return "just now"
        for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
            if seconds >= size:
                count = seconds // size
                if unit == "day" and count > 60:
                    return parsed.strftime("%d %b %Y")
                return f"{count} {unit}{'s' if count != 1 else ''} ago"
        return "just now"

    @app.template_filter("duration")
    def _duration(seconds):
        seconds = int(seconds or 0)
        if seconds < 60:
            return "under a minute" if seconds else "—"
        hours, minutes = divmod(seconds // 60, 60)
        return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"

    @app.template_filter("filesize")
    def _filesize(size):
        size = float(size or 0)
        for unit in ("bytes", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit in ("bytes", "KB") else f"{size:.1f} {unit}"
            size /= 1024
        return ""

    @app.template_filter("event_label")
    def _event_label(kind):
        return audit.LABELS.get(kind, kind.replace("_", " ").capitalize())

    @app.template_filter("is_alert")
    def _is_alert(kind):
        return kind in audit.ALERTS

    app.jinja_env.globals["app_version"] = __version__
