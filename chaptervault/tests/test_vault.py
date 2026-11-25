"""End-to-end tests of the access-control guarantees, driven through the real HTTP routes."""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vault import auth, create_app, db, documents  # noqa: E402

AUTHOR = ("author@example.com", "correct horse battery")
SECRET_TEXT = "The lighthouse keeper never told anyone"


def make_pdf(pages: int = 5) -> bytes:
    doc = pymupdf.open()
    for number in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 100), f"{SECRET_TEXT} - page {number}", fontsize=14)
    return doc.tobytes()


def make_docx(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml",
                   '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   f'<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>')
    return buf.getvalue()


class Browser:
    """A cookie jar plus the CSRF handling a real browser gets from the rendered forms."""

    def __init__(self, app):
        self.client = app.test_client()
        self.token = None

    def get(self, url, **kwargs):
        return self.client.get(url, **kwargs)

    def _csrf(self, form_page):
        html = self.client.get(form_page).get_data(as_text=True)
        return re.search(r'name="_csrf" value="([^"]+)"', html).group(1)

    def post(self, url, data=None, form_page=None, **kwargs):
        data = dict(data or {})
        data["_csrf"] = self._csrf(form_page) if form_page else self.token
        return self.client.post(url, data=data, **kwargs)

    def json(self, url, payload=None):
        return self.client.post(url, json=payload or {}, headers={"X-CSRF-Token": self.token})

    def login(self, email, password):
        response = self.post("/login", {"email": email, "password": password}, form_page="/login")
        if response.status_code == 302:
            home = self.client.get(response.headers["Location"], follow_redirects=True).get_data(as_text=True)
            self.token = re.search(r'name="_csrf" value="([^"]+)"', home).group(1)
        return response

    def logout(self):
        return self.post("/logout")


@pytest.fixture()
def app(tmp_path):
    return create_app(tmp_path / "data")


@pytest.fixture()
def author(app):
    browser = Browser(app)
    setup_url = f"/setup?token={app.config['SETUP_TOKEN']}"
    response = browser.post("/setup", {"token": app.config["SETUP_TOKEN"], "name": "Ann Author", "email": AUTHOR[0],
                                       "password": AUTHOR[1], "confirm": AUTHOR[1]}, form_page=setup_url)
    assert response.status_code == 302
    assert browser.login(*AUTHOR).status_code == 302
    return browser


def upload(author, name="chapter-one.pdf", data=None, **fields):
    payload = {"files": (io.BytesIO(data or make_pdf()), name), **fields}
    response = author.post("/admin/chapters/upload", payload, content_type="multipart/form-data")
    assert response.status_code == 302, response.get_data(as_text=True)
    return response


def query(app, sql, *args):
    with app.app_context():
        return [dict(row) for row in db.all(sql, *args)]


def add_reviewer(app, author, name, email, documents_=(), password="reviewer password 1"):
    """Add a reviewer through the console, follow their invite link, and return a signed-in browser."""
    response = author.post("/admin/reviewers/add", {"name": name, "email": email, "documents": list(documents_)})
    page = author.get(response.headers["Location"]).get_data(as_text=True)
    link = re.search(r'id="invite-link" value="([^"]+)"', page).group(1)
    path = "/invite/" + link.rsplit("/invite/", 1)[1]
    browser = Browser(app)
    accepted = browser.post(path, {"password": password, "confirm": password}, form_page=path)
    assert accepted.status_code == 302
    assert browser.login(email, password).status_code == 302
    browser.post("/terms", {"agree": "1"})
    return browser


def open_chapter(browser, doc_id=1):
    response = browser.post(f"/read/{doc_id}/open")
    return response.headers.get("Location", "")


# --------------------------------------------------------------------- tests

def test_setup_requires_the_one_time_token(app):
    browser = Browser(app)
    assert browser.get("/setup").status_code == 403
    assert browser.get("/setup?token=wrong").status_code == 403
    assert browser.get(f"/setup?token={app.config['SETUP_TOKEN']}").status_code == 200


def test_setup_token_can_be_fixed_by_the_deployment(tmp_path, monkeypatch):
    token = "t" * 40
    monkeypatch.setenv("CHAPTERVAULT_SETUP_TOKEN", token)
    first = create_app(tmp_path / "data")
    again = create_app(tmp_path / "data")          # a restart keeps the same link valid
    assert first.config["SETUP_TOKEN"] == again.config["SETUP_TOKEN"] == token
    assert Browser(again).get(f"/setup?token={token}").status_code == 200
    monkeypatch.setenv("CHAPTERVAULT_SETUP_TOKEN", "too-short")
    with pytest.raises(RuntimeError):
        create_app(tmp_path / "other")


def test_setup_is_closed_once_an_author_exists(app, author):
    assert Browser(app).get(f"/setup?token={app.config['SETUP_TOKEN']}").status_code == 302


def test_upload_is_encrypted_at_rest(app, author):
    upload(author)
    doc = query(app, "SELECT * FROM documents")[0]
    assert doc["page_count"] == 5
    blob = (app.config["VAULT"].blob_dir / doc["blob_name"]).read_bytes()
    assert not blob.startswith(b"%PDF")
    assert SECRET_TEXT.encode() not in blob and b"lighthouse" not in blob


def test_page_range_keeps_only_the_selected_chapter(app, author):
    upload(author, name="whole-book.pdf", data=make_pdf(10), page_range="3-5, 9", title="Chapter Three")
    doc = query(app, "SELECT * FROM documents")[0]
    assert (doc["title"], doc["page_count"]) == ("Chapter Three", 4)


def test_bad_uploads_are_rejected_with_a_reason(app, author):
    response = author.post("/admin/chapters/upload", {"files": (io.BytesIO(b"not a pdf"), "broken.pdf")},
                           content_type="multipart/form-data", follow_redirects=True)
    assert "could not be read" in response.get_data(as_text=True)
    response = author.post("/admin/chapters/upload", {"files": (io.BytesIO(b"MZ"), "tool.exe")},
                           content_type="multipart/form-data", follow_redirects=True)
    assert "Unsupported file type" in response.get_data(as_text=True)
    assert query(app, "SELECT COUNT(*) AS n FROM documents")[0]["n"] == 0


def test_word_file_converts_without_libreoffice(monkeypatch):
    monkeypatch.setattr(documents.shutil, "which", lambda _name: None)
    result = documents.ingest(make_docx("A chapter written in Word"), "chapter.docx")
    assert result["page_count"] >= 1
    with pymupdf.open(stream=result["pdf"], filetype="pdf") as pdf:
        assert "written in Word" in pdf[0].get_text()


@pytest.mark.parametrize("spec, expected", [("", None), ("2-4", [1, 2, 3]), ("1, 3", [0, 2]), ("4-", [3, 4]),
                                            ("-2", [0, 1]), ("2-3, 3-4", [1, 2, 3])])
def test_page_range_parsing(spec, expected):
    assert documents.parse_page_range(spec, 5) == expected


@pytest.mark.parametrize("spec", ["0", "6", "4-2", "abc", "1-9"])
def test_page_range_rejects_nonsense(spec):
    with pytest.raises(documents.IngestError):
        documents.parse_page_range(spec, 5)


def test_reviewer_reads_watermarked_images_never_the_file(app, author):
    upload(author)
    reviewer = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    library = reviewer.get("/library").get_data(as_text=True)
    assert "chapter one" in library

    view_url = open_chapter(reviewer)
    assert re.fullmatch(r"/view/\d+", view_url)
    viewer = reviewer.get(view_url)
    assert viewer.status_code == 200 and "no-store" in viewer.headers["Cache-Control"]
    assert SECRET_TEXT not in viewer.get_data(as_text=True)

    page = reviewer.get(f"{view_url}/page/1?w=1000")
    assert page.status_code == 200 and page.mimetype == "image/jpeg"
    assert page.data[:3] == b"\xff\xd8\xff" and b"%PDF" not in page.data
    assert "no-store" in page.headers["Cache-Control"]
    assert reviewer.get(f"{view_url}/page/6").status_code == 404

    # The stamped image must differ from a clean render of the same page.
    with app.app_context():
        doc = db.one("SELECT * FROM documents")
        clean = documents.render_page(app.config["VAULT"], app.config["BLOB_KEY"], doc, 0, 1000, None)
    assert clean != page.data

    grant = query(app, "SELECT * FROM grants")[0]
    assert grant["views_used"] == 1
    kinds = {e["kind"] for e in query(app, "SELECT kind FROM events")}
    assert {"doc_open", "page_view", "nda_accepted", "invite_accepted"} <= kinds


def test_watermark_names_the_reader(app, author):
    """Render a blank page so the only ink on it is the watermark, then check it is really there."""
    blank = pymupdf.open()
    blank.new_page()
    upload(author, data=blank.tobytes())
    with app.app_context():
        doc = db.one("SELECT * FROM documents")
        cfg, key = app.config["VAULT"], app.config["BLOB_KEY"]
        mark = documents.Watermark(lines=("Rae Reader  ·  rae@example.com", "25 Nov 2026"), footer="Ref V00001",
                                   opacity=0.14)
        stamped = pymupdf.Pixmap(documents.render_page(cfg, key, doc, 0, 800, mark))
        clean = pymupdf.Pixmap(documents.render_page(cfg, key, doc, 0, 800, None))
    darker = sum(1 for a, b in zip(stamped.samples[::3], clean.samples[::3]) if a < b - 8)
    assert darker > 2000  # thousands of pixels carry watermark ink, spread over the page


def test_reviewers_only_see_what_they_were_granted(app, author):
    upload(author)
    upload(author, name="chapter-two.pdf")
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    sam = add_reviewer(app, author, "Sam Skimmer", "sam@example.com", documents_=["2"])

    assert rae.post("/read/2/open").status_code == 404
    assert "chapter two" not in rae.get("/library").get_data(as_text=True)

    rae_view = open_chapter(rae, 1)
    assert sam.get(rae_view).status_code == 404                # someone else's reading session
    assert sam.get(f"{rae_view}/page/1").status_code == 404
    anonymous = Browser(app)
    assert anonymous.get(f"{rae_view}/page/1").status_code == 401
    assert anonymous.get(rae_view).status_code == 302


def test_reviewers_cannot_reach_the_author_console(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    for path in ("/admin/", "/admin/chapters", "/admin/reviewers", "/admin/activity", "/admin/chapters/1"):
        assert rae.get(path).status_code == 404
    assert rae.post("/admin/chapters/1/delete").status_code == 404
    assert rae.post("/admin/chapters/1/preview").status_code == 404
    assert query(app, "SELECT COUNT(*) AS n FROM documents")[0]["n"] == 1


def test_revoking_cuts_off_an_open_reader_immediately(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    assert rae.get(f"{view_url}/page/1").status_code == 200

    author.post("/admin/grants/1/update", {"action": "revoke"})
    denied = rae.get(f"{view_url}/page/2")
    assert denied.status_code == 403 and denied.get_json()["error"] == "revoked"
    assert rae.json(f"{view_url}/ping").status_code == 403
    assert "chapter one" not in rae.get("/library").get_data(as_text=True)
    assert rae.post("/read/1/open").status_code == 302 and open_chapter(rae) == "/library"

    author.post("/admin/grants/1/update", {"action": "restore"})
    assert re.fullmatch(r"/view/\d+", open_chapter(rae))


def test_suspending_a_chapter_blocks_every_reader(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    author.post("/admin/chapters/1/status", {"action": "suspend"})
    denied = rae.get(f"{view_url}/page/1")
    assert denied.status_code == 403 and denied.get_json()["error"] == "doc_suspended"


def test_expired_and_future_grants_cannot_be_opened(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)

    author.post("/admin/grants/1/update", {"action": "save", "expires_at": "2020-01-31", "allow_comments": "1"})
    assert rae.get(f"{view_url}/page/1").get_json()["error"] == "expired"
    assert open_chapter(rae) == "/library"
    assert "ended on 31 Jan 2020" in rae.get("/library").get_data(as_text=True)

    author.post("/admin/grants/1/update", {"action": "save", "starts_at": "2999-01-01"})
    assert rae.get(f"{view_url}/page/1").get_json()["error"] == "not_started"


def test_open_limit_is_enforced(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    author.post("/admin/grants/1/update", {"action": "save", "max_views": "1"})

    first = open_chapter(rae)
    assert re.fullmatch(r"/view/\d+", first)
    assert open_chapter(rae) == first                  # same sitting: not counted again
    assert query(app, "SELECT views_used FROM grants")[0]["views_used"] == 1

    rae.logout()
    assert rae.login("rae@example.com", "reviewer password 1").status_code == 302
    assert open_chapter(rae) == "/library"             # a new sitting would be a second open
    author.post("/admin/grants/1/update", {"action": "reset_views"})
    assert re.fullmatch(r"/view/\d+", open_chapter(rae))


def test_account_is_locked_to_registered_devices(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    author.post("/admin/reviewers/2/update", {"name": "Rae Reader", "email": "rae@example.com", "max_devices": "1"})

    other_device = Browser(app)
    blocked = other_device.login("rae@example.com", "reviewer password 1")
    assert blocked.status_code == 403 and "maximum number of devices" in blocked.get_data(as_text=True)
    assert rae.get("/library").status_code == 200       # the registered device is unaffected

    author.post("/admin/reviewers/2/devices/reset")
    assert rae.get("/library").status_code == 302       # signed out by the reset
    assert other_device.login("rae@example.com", "reviewer password 1").status_code == 302


def test_signing_in_elsewhere_ends_the_previous_session(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    second = Browser(app)
    assert second.login("rae@example.com", "reviewer password 1").status_code == 302
    assert rae.get(f"{view_url}/page/1").status_code == 401
    assert rae.get("/library").status_code == 302


def test_suspended_reviewer_is_signed_out_and_kept_out(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    author.post("/admin/reviewers/2/status", {"action": "suspend"})
    assert rae.get("/library").status_code == 302
    assert rae.login("rae@example.com", "reviewer password 1").status_code == 403


def test_repeated_wrong_passwords_lock_the_account(app, author):
    upload(author)
    add_reviewer(app, author, "Rae Reader", "rae@example.com")
    attacker = Browser(app)
    for _ in range(auth.MAX_FAILED_LOGINS):
        assert attacker.login("rae@example.com", "guess").status_code == 401
    assert attacker.login("rae@example.com", "reviewer password 1").status_code == 429


def test_invite_link_works_once(app, author):
    response = author.post("/admin/reviewers/add", {"name": "Rae Reader", "email": "rae@example.com"})
    page = author.get(response.headers["Location"]).get_data(as_text=True)
    path = "/invite/" + re.search(r'id="invite-link" value="([^"]+)"', page).group(1).rsplit("/invite/", 1)[1]
    visitor = Browser(app)
    assert visitor.post(path, {"password": "short", "confirm": "short"}, form_page=path).status_code == 400
    assert visitor.post(path, {"password": "long enough pw", "confirm": "long enough pw"}, form_page=path).status_code == 302
    assert Browser(app).get(path).status_code == 410
    assert Browser(app).get("/invite/not-a-real-token").status_code == 410


def test_forms_without_a_csrf_token_are_refused(app, author):
    upload(author)
    assert author.client.post("/admin/chapters/1/delete").status_code == 400
    assert author.client.post("/admin/chapters/1/delete", data={"_csrf": "forged"}).status_code == 400
    assert Browser(app).client.post("/login", data={"email": AUTHOR[0], "password": AUTHOR[1]}).status_code == 400
    assert query(app, "SELECT COUNT(*) AS n FROM documents")[0]["n"] == 1


def test_feedback_is_private_to_each_reviewer_and_visible_to_the_author(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    sam = add_reviewer(app, author, "Sam Skimmer", "sam@example.com", documents_=["1"])
    rae_view, sam_view = open_chapter(rae), open_chapter(sam)

    created = rae.json(f"{rae_view}/comments", {"page": 2, "body": "The twist lands too early."})
    assert created.status_code == 201
    assert rae.json(f"{rae_view}/comments", {"page": 99, "body": "x"}).status_code == 400
    assert [c["body"] for c in rae.get(f"{rae_view}/comments").get_json()] == ["The twist lands too early."]
    assert sam.get(f"{sam_view}/comments").get_json() == []
    assert sam.json(f"{sam_view}/comments/{created.get_json()['id']}/delete").status_code == 404

    assert "The twist lands too early." in author.get("/admin/chapters/1").get_data(as_text=True)
    export = author.get("/admin/chapters/1/feedback.csv").get_data(as_text=True)
    assert "Rae Reader" in export and "The twist lands too early." in export

    author.post("/admin/grants/2/update", {"action": "save"})      # feedback box left unticked for Sam
    assert sam.json(f"{sam_view}/comments", {"page": 1, "body": "hello"}).status_code == 403


def test_access_grid_grants_and_revokes(app, author):
    upload(author)
    upload(author, name="chapter-two.pdf")
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com")
    author.post("/admin/access", {"cell": ["2:1", "2:2"]})
    assert len(query(app, "SELECT * FROM grants WHERE status = 'active'")) == 2
    author.post("/admin/access", {"cell": ["2:2"]})
    active = query(app, "SELECT document_id FROM grants WHERE status = 'active'")
    assert [g["document_id"] for g in active] == [2]
    library = rae.get("/library").get_data(as_text=True)
    assert "chapter two" in library and "chapter one" not in library


def test_replacing_a_file_keeps_access_rules(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    old_blob = query(app, "SELECT blob_name FROM documents")[0]["blob_name"]
    author.post("/admin/chapters/1/replace", {"file": (io.BytesIO(make_pdf(8)), "draft-2.pdf")},
                content_type="multipart/form-data")
    doc = query(app, "SELECT * FROM documents")[0]
    assert doc["page_count"] == 8 and doc["blob_name"] != old_blob
    assert not (app.config["VAULT"].blob_dir / old_blob).exists()
    view_url = open_chapter(rae)
    assert rae.get(f"{view_url}/page/8").status_code == 200


def test_deleting_a_chapter_removes_the_file(app, author):
    upload(author)
    blob = app.config["VAULT"].blob_dir / query(app, "SELECT blob_name FROM documents")[0]["blob_name"]
    assert blob.exists()
    author.post("/admin/chapters/1/delete")
    assert not blob.exists() and not query(app, "SELECT * FROM documents")


def test_author_preview_and_activity_export(app, author):
    upload(author)
    preview = author.post("/admin/chapters/1/preview").headers["Location"]
    assert author.get(f"{preview}/page/1").mimetype == "image/jpeg"
    export = author.get("/admin/activity?format=csv&show=everything")
    assert export.mimetype == "text/csv" and "Chapter uploaded" in export.get_data(as_text=True)


def test_security_headers(app, author):
    response = author.get("/admin/")
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "no-store" in response.headers["Cache-Control"]


def test_every_author_page_renders(app, author):
    upload(author)
    add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    for path in ("/admin/", "/admin/chapters", "/admin/chapters/1", "/admin/reviewers", "/admin/reviewers/2",
                 "/admin/access", "/admin/activity", "/admin/activity?show=alerts&reviewer=2&chapter=1",
                 "/admin/settings", "/account"):
        assert author.get(path).status_code == 200, path
    saved = author.post("/admin/settings", {"book_title": "The Lighthouse", "author_name": "Ann Author",
                                            "default_expiry_days": "14", "default_max_views": "",
                                            "default_max_devices": "2", "watermark_notice": "Review copy",
                                            "watermark_opacity": "18", "require_nda": "1", "nda_text": "Keep it secret."},
                        follow_redirects=True)
    assert "Settings saved." in saved.get_data(as_text=True)
    assert "The Lighthouse" in Browser(app).get("/login").get_data(as_text=True)
