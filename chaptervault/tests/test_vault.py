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
                                            "default_max_devices": "2", "watermark_footer": "Review copy",
                                            "watermark_text": "{name}", "watermark_opacity": "18",
                                            "require_nda": "1", "nda_text": "Keep it secret."},
                        follow_redirects=True)
    assert "Settings saved." in saved.get_data(as_text=True)
    assert "The Lighthouse" in Browser(app).get("/login").get_data(as_text=True)


# ------------------------------------------------- notes pinned to a passage

def test_notes_can_be_pinned_to_a_passage(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    sam = add_reviewer(app, author, "Sam Skimmer", "sam@example.com", documents_=["1"])
    author.post("/admin/grants/2/update", {"action": "save"})        # Sam may read but not comment
    rae_view, sam_view = open_chapter(rae), open_chapter(sam)

    outlines = rae.get(f"{rae_view}/blocks/2").get_json()["blocks"]
    assert len(outlines) == 1 and all(0 <= v <= 1 for v in outlines[0])
    assert SECRET_TEXT not in rae.get(f"{rae_view}/blocks/2").get_data(as_text=True)   # rectangles and counts only
    # Sam may not comment, but still gets the outlines: they are also how reading is tracked.
    assert sam.get(f"{sam_view}/blocks/2").get_json() == {"blocks": outlines, "words": [9], "covered": {}}
    assert rae.get(f"{rae_view}/blocks/9").status_code == 404

    created = rae.json(f"{rae_view}/comments", {"page": 2, "body": "This line is the hook.", "box": outlines[0]})
    assert created.status_code == 201
    note = created.get_json()
    assert note["box"] == outlines[0] and "quote" not in note
    listed = rae.get(f"{rae_view}/comments").get_json()
    assert listed[0]["box"] == outlines[0] and "quote" not in listed[0]          # never sent to reviewers
    # commenting on a paragraph counts as having covered it
    assert rae.get(f"{rae_view}/blocks/2").get_json()["covered"] == {"0": "noted"}
    assert sam.get(f"{sam_view}/blocks/2").get_json()["covered"] == {}

    for bad in ([0.1, 0.1, 0.1, 0.1], [0.2, 0.2], "everything", [0.1, 0.1, "x", 0.5], [True, 0, 1, 1]):
        assert rae.json(f"{rae_view}/comments", {"page": 2, "body": "x", "box": bad}).status_code == 400

    # The author sees which passage the note is about, in the console, the export and the preview.
    passage = f"{SECRET_TEXT} - page 2"
    assert passage in author.get("/admin/chapters/1").get_data(as_text=True)
    assert passage in author.get("/admin/chapters/1/feedback.csv").get_data(as_text=True)
    preview = author.post("/admin/chapters/1/preview", {"page": "2", "note": str(note["id"])}).headers["Location"]
    assert preview.endswith(f"#p=2&n={note['id']}")
    seen = author.get(preview.split("#")[0] + "/comments").get_json()
    assert seen[0]["quote"] == passage and seen[0]["box"] == outlines[0]


def test_a_roughly_dragged_box_quotes_whole_lines(app, author):
    source = pymupdf.open()
    page = source.new_page(width=400, height=600)
    for i, text in enumerate(("The first line of the paragraph runs here", "and the second line follows it closely",
                              "while the third line ends the thought.")):
        page.insert_text((50, 100 + i * 16), text, fontsize=11)
    upload(author, data=source.tobytes())
    with app.app_context():
        doc = db.one("SELECT * FROM documents")
        cfg, key = app.config["VAULT"], app.config["BLOB_KEY"]
        # A box drawn short of both margins, covering lines two and three only.
        rough = [0.25, 108 / 600, 0.45, 138 / 600]
        assert documents.passage_text(cfg, key, doc, 0, rough) == (
            "and the second line follows it closely while the third line ends the thought.")
        assert documents.passage_text(cfg, key, doc, 0, [0.1, 0.8, 0.9, 0.9]) == ""     # empty part of the page


def test_whole_page_notes_still_work(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    created = rae.json(f"{view_url}/comments", {"page": 3, "body": "Pacing drags here."})
    assert created.status_code == 201 and created.get_json()["box"] is None
    assert "Pacing drags here." in author.get("/admin/chapters/1").get_data(as_text=True)


def test_passages_line_up_on_rotated_pages(app, author):
    source = pymupdf.open()
    page = source.new_page(width=400, height=600)
    page.insert_textbox(pymupdf.Rect(50, 60, 350, 140), "A sideways paragraph for the rotation test.", fontsize=12)
    page.set_rotation(90)
    upload(author, data=source.tobytes())
    with app.app_context():
        doc = db.one("SELECT * FROM documents")
        cfg, key = app.config["VAULT"], app.config["BLOB_KEY"]
        (box,) = documents.page_blocks(cfg, key, doc, 0)
        assert documents.passage_text(cfg, key, doc, 0, box) == "A sideways paragraph for the rotation test."
        # The outline must sit on the ink of the page as rendered (rotated: 600 wide, 400 tall).
        image = pymupdf.Pixmap(documents.render_page(cfg, key, doc, 0, 600, None))
    inside = outside = 0
    for y in range(0, image.height, 3):
        for x in range(0, image.width, 3):
            if image.pixel(x, y)[0] < 120:
                if box[0] <= x / image.width <= box[2] and box[1] <= y / image.height <= box[3]:
                    inside += 1
                else:
                    outside += 1
    assert inside > 20 and outside == 0


def test_existing_database_gains_the_new_columns(tmp_path):
    """A vault created before pinned notes existed must upgrade in place and keep its notes."""
    import sqlite3
    first = create_app(tmp_path / "data")
    path = first.config["VAULT"].db_path
    conn = sqlite3.connect(path)
    conn.executescript(
        "DROP TABLE comments;"
        "CREATE TABLE comments (id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, user_id INTEGER NOT NULL, "
        "page INTEGER NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);"
        "INSERT INTO comments (document_id, user_id, page, body, created_at) VALUES (1, 1, 4, 'an old note', '2026-01-01T00:00:00Z');")
    conn.commit()
    conn.close()

    upgraded = create_app(tmp_path / "data")
    rows = query(upgraded, "SELECT * FROM comments")
    assert rows == [{"id": 1, "document_id": 1, "user_id": 1, "page": 4, "body": "an old note",
                     "created_at": "2026-01-01T00:00:00Z", "x0": None, "y0": None, "x1": None, "y1": None, "quote": "",
                     "edited_at": None}]
    create_app(tmp_path / "data")      # starting again changes nothing


# ------------------------------------------------ author-written watermark text

def test_author_writes_the_watermark_text(app, author):
    from vault import reviewer
    view = {"id": 7, "grant_id": 3, "stamp": "25 Nov 2026 18:21 UTC  ·  198.51.100.4"}
    rae = {"name": "Rae Reader", "email": "rae@example.com"}
    settings = dict(db.DEFAULT_SETTINGS, book_title="The Lighthouse", author_name="Ann Author")

    default = reviewer.watermark_for(view, rae, settings)
    assert default.lines == ("Rae Reader  ·  rae@example.com", "25 Nov 2026 18:21 UTC  ·  198.51.100.4")
    assert "Issued to rae@example.com" in default.footer and default.footer.endswith("Ref V00007")

    settings.update(watermark_text="Property of {author} - {book}\nLicensed to {name}\n{unknown} {email.__class__}",
                    watermark_footer="")
    custom = reviewer.watermark_for(view, rae, settings)
    assert custom.lines == ("Property of Ann Author - The Lighthouse", "Licensed to Rae Reader",
                            "{unknown} {email.__class__}")       # only the known words are filled in
    assert custom.footer == ""

    settings.update(watermark_text="", watermark_footer="")
    assert reviewer.watermark_for(view, rae, settings).lines == ()

    # The author's own preview shows a stand-in reviewer, never the author's sign-in address.
    preview = reviewer.watermark_for({**view, "grant_id": None}, {"name": "Ann Author", "email": AUTHOR[0]},
                                     dict(db.DEFAULT_SETTINGS))
    assert AUTHOR[0] not in " ".join(preview.lines) + preview.footer
    assert "Ann Author" not in " ".join(preview.lines) + preview.footer
    assert preview.lines[0] == "Reviewer Name  ·  reviewer@example.com"


def test_watermark_settings_change_what_reviewers_see(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    before = rae.get(f"{view_url}/page/1?w=800").data

    form = {"book_title": "", "author_name": "Ann Author", "default_expiry_days": "30", "default_max_views": "",
            "default_max_devices": "2", "watermark_opacity": "14", "require_nda": "1", "nda_text": "Keep it secret.",
            "watermark_text": "DRAFT - NOT FOR CIRCULATION\r\nLicensed to {name}", "watermark_footer": "Copy {ref}"}
    saved = author.post("/admin/settings", form, follow_redirects=True).get_data(as_text=True)
    assert "Settings saved." in saved and "DRAFT - NOT FOR CIRCULATION\nLicensed to {name}" in saved
    after = rae.get(f"{view_url}/page/1?w=800").data
    assert after[:3] == b"\xff\xd8\xff" and after != before

    too_long = dict(form, watermark_text="one\ntwo\nthree\nfour")
    assert "up to 3 lines" in author.post("/admin/settings", too_long, follow_redirects=True).get_data(as_text=True)
    assert query(app, "SELECT value FROM settings WHERE key = 'watermark_text'")[0]["value"].count("\n") == 1


def test_customised_notice_from_an_older_version_is_kept(app, author):
    with app.app_context():
        db.set_setting("watermark_notice", "Advance copy for the editorial board")
    with app.app_context():
        footer = db.settings()["watermark_footer"]
    assert footer == "Advance copy for the editorial board  ·  Issued to {email}  ·  {date}  ·  Ref {ref}"


# ------------------------------------------------------------- editing a note

def test_reviewer_can_edit_their_own_note(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    sam = add_reviewer(app, author, "Sam Skimmer", "sam@example.com", documents_=["1"])
    rae_view, sam_view = open_chapter(rae), open_chapter(sam)
    box = rae.get(f"{rae_view}/blocks/2").get_json()["blocks"][0]
    note = rae.json(f"{rae_view}/comments", {"page": 2, "body": "Frist draft of my thought.", "box": box}).get_json()
    assert note["edited"] is None

    edited = rae.json(f"{rae_view}/comments/{note['id']}/edit", {"body": "  First draft of my thought, fixed.  "})
    assert edited.status_code == 200
    data = edited.get_json()
    assert data["body"] == "First draft of my thought, fixed." and data["edited"] and "quote" not in data
    assert data["box"] == box and data["page"] == 2                       # still pinned to the same passage
    assert [c["body"] for c in rae.get(f"{rae_view}/comments").get_json()] == ["First draft of my thought, fixed."]

    # refused: empty text, too long, someone else's note, a note that does not exist
    assert rae.json(f"{rae_view}/comments/{note['id']}/edit", {"body": "   "}).status_code == 400
    assert rae.json(f"{rae_view}/comments/{note['id']}/edit", {"body": "x" * 5001}).status_code == 400
    assert sam.json(f"{sam_view}/comments/{note['id']}/edit", {"body": "hijacked"}).status_code == 404
    assert rae.json(f"{rae_view}/comments/999/edit", {"body": "nothing there"}).status_code == 404
    assert query(app, "SELECT body FROM comments")[0]["body"] == "First draft of my thought, fixed."

    # the author sees the new wording, the passage it refers to, and that it was edited
    page = author.get("/admin/chapters/1").get_data(as_text=True)
    assert "First draft of my thought, fixed." in page and "Frist" not in page and "edited " in page
    assert f"{SECRET_TEXT} - page 2" in page
    assert "Feedback edited" in author.get("/admin/activity").get_data(as_text=True)

    # saving unchanged text is not recorded as an edit
    stamp = query(app, "SELECT edited_at FROM comments")[0]["edited_at"]
    rae.json(f"{rae_view}/comments/{note['id']}/edit", {"body": "First draft of my thought, fixed."})
    assert query(app, "SELECT COUNT(*) AS n FROM events WHERE kind = 'comment_edited'")[0]["n"] == 1
    assert query(app, "SELECT edited_at FROM comments")[0]["edited_at"] == stamp

    # once the author switches feedback off or revokes access, notes can no longer be changed
    author.post("/admin/grants/1/update", {"action": "save"})             # feedback box unticked
    assert rae.json(f"{rae_view}/comments/{note['id']}/edit", {"body": "too late"}).status_code == 403
    author.post("/admin/grants/1/update", {"action": "revoke"})
    assert rae.json(f"{rae_view}/comments/{note['id']}/edit", {"body": "too late"}).status_code == 403
    assert query(app, "SELECT body FROM comments")[0]["body"] == "First draft of my thought, fixed."


# ------------------------------------------------------------ invitation email

import smtplib  # noqa: E402

from vault import mailer  # noqa: E402

SETTINGS_FORM = {"book_title": "The Lighthouse", "author_name": "Ann Author", "default_expiry_days": "30",
                 "default_max_views": "", "default_max_devices": "2", "watermark_opacity": "14",
                 "watermark_text": "{name}", "watermark_footer": "", "require_nda": "1", "nda_text": "Keep it secret."}
MAIL_FORM = {"smtp_host": "smtp.example.com", "smtp_port": "587", "smtp_security": "starttls",
             "smtp_username": "ann.login", "smtp_password": "app-password-123",
             "mail_from_address": "ann@example.com", "mail_from_name": "Ann Author"}


class FakeMailServer:
    """Stands in for smtplib.SMTP / SMTP_SSL and records what the app does with it."""
    outbox: list = []
    log: list = []
    fail_with: Exception | None = None

    def __init__(self, host, port, timeout=None, context=None):
        FakeMailServer.log.append(("connect", type(self).__name__, host, port))
        if isinstance(FakeMailServer.fail_with, OSError) and not isinstance(FakeMailServer.fail_with, smtplib.SMTPException):
            raise FakeMailServer.fail_with

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        FakeMailServer.log.append(("starttls", context is not None))

    def login(self, username, password):
        FakeMailServer.log.append(("login", username, password))
        if isinstance(FakeMailServer.fail_with, smtplib.SMTPAuthenticationError):
            raise FakeMailServer.fail_with

    def send_message(self, message):
        if FakeMailServer.fail_with is not None:
            raise FakeMailServer.fail_with
        FakeMailServer.outbox.append(message)


class FakeMailServerSSL(FakeMailServer):
    pass


@pytest.fixture()
def mail(monkeypatch):
    FakeMailServer.outbox, FakeMailServer.log, FakeMailServer.fail_with = [], [], None
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeMailServer)
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", FakeMailServerSSL)
    return FakeMailServer


def set_up_email(author, **overrides):
    page = author.post("/admin/settings", {**SETTINGS_FORM, **MAIL_FORM, **overrides}, follow_redirects=True)
    return page.get_data(as_text=True)


def link_in(message) -> str:
    return re.search(r"https?://\S+/invite/\S+", message.get_content()).group(0)


def test_invitation_is_emailed_when_a_reviewer_is_added(app, author, mail):
    assert "Settings saved." in set_up_email(author)
    stored = {r["key"]: r["value"] for r in query(app, "SELECT key, value FROM settings")}
    assert "app-password-123" not in "".join(stored.values())                 # encrypted at rest
    assert "app-password-123" not in author.get("/admin/settings").get_data(as_text=True)   # never sent back

    upload(author)
    added = author.post("/admin/reviewers/add", {"name": "Rae Reader", "email": "rae@example.com",
                                                 "documents": ["1"], "send_email": "1"}, follow_redirects=True)
    page = added.get_data(as_text=True)
    assert "Their invitation was emailed to rae@example.com." in page and "Resend by email" in page

    (message,) = mail.outbox
    assert message["To"] == "Rae Reader <rae@example.com>" and message["From"] == "Ann Author <ann@example.com>"
    assert message["Subject"] == "Chapters for your review"
    body = message.get_content()
    assert body.startswith("Hello Rae Reader,") and "chapters of The Lighthouse" in body and body.rstrip().endswith("Ann Author")
    assert ("connect", "FakeMailServer", "smtp.example.com", 587) in mail.log and ("starttls", True) in mail.log
    assert ("login", "ann.login", "app-password-123") in mail.log
    assert mail.log.index(("starttls", True)) < mail.log.index(("login", "ann.login", "app-password-123"))

    # the emailed link is the reviewer's real invitation
    path = "/invite/" + link_in(message).rsplit("/invite/", 1)[1]
    visitor = Browser(app)
    assert visitor.get(path).status_code == 200
    assert visitor.post(path, {"password": "reviewer password 1", "confirm": "reviewer password 1"}, form_page=path).status_code == 302
    kinds = [e["kind"] for e in query(app, "SELECT kind FROM events")]
    assert "invite_emailed" in kinds and "email_failed" not in kinds


def test_invitation_can_be_resent_and_replaced_when_it_has_lapsed(app, author, mail):
    set_up_email(author)
    author.post("/admin/reviewers/add", {"name": "Rae Reader", "email": "rae@example.com", "send_email": "1"})
    again = author.post("/admin/reviewers/2/send-invite", follow_redirects=True).get_data(as_text=True)
    assert "The invitation was emailed to rae@example.com." in again
    first, second = (link_in(m) for m in mail.outbox)
    assert first == second                                   # resending does not break the link already sent

    with app.app_context():                                  # let the invitation lapse
        db.run("UPDATE users SET invite_expires_at = '2020-01-01T00:00:00Z' WHERE id = 2")
    assert "Email a new invitation" in author.get("/admin/reviewers/2").get_data(as_text=True)
    author.post("/admin/reviewers/2/send-invite")
    third = link_in(mail.outbox[2])
    assert third != first
    assert Browser(app).get("/invite/" + first.rsplit("/invite/", 1)[1]).status_code == 410
    assert Browser(app).get("/invite/" + third.rsplit("/invite/", 1)[1]).status_code == 200


def test_reviewer_who_already_joined_gets_a_password_reset_email(app, author, mail):
    set_up_email(author)
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    mail.outbox.clear()
    assert "Email a password reset link" in author.get("/admin/reviewers/2").get_data(as_text=True)
    sent = author.post("/admin/reviewers/2/send-invite", follow_redirects=True).get_data(as_text=True)
    assert "The password reset link was emailed to rae@example.com." in sent
    assert rae.get("/library").status_code == 200            # still signed in; the old password works until the link is used
    path = "/invite/" + link_in(mail.outbox[0]).rsplit("/invite/", 1)[1]
    visitor = Browser(app)
    assert visitor.post(path, {"password": "a brand new password", "confirm": "a brand new password"}, form_page=path).status_code == 302
    assert Browser(app).login("rae@example.com", "reviewer password 1").status_code == 401

    author.post("/admin/reviewers/2/status", {"action": "suspend"})
    refused = author.post("/admin/reviewers/2/send-invite", follow_redirects=True).get_data(as_text=True)
    assert "is suspended" in refused and len(mail.outbox) == 1


def test_a_failed_email_does_not_lose_the_reviewer(app, author, mail):
    set_up_email(author)
    mail.fail_with = smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")
    page = author.post("/admin/reviewers/add", {"name": "Rae Reader", "email": "rae@example.com", "send_email": "1"},
                       follow_redirects=True).get_data(as_text=True)
    assert "the invitation could not be emailed" in page and "rejected the username or password" in page
    assert 'id="invite-link"' in page and "Send by email" in page          # the author can still copy the link
    person = query(app, "SELECT * FROM users WHERE role = 'reviewer'")[0]
    assert person["status"] == "invited" and person["invite_sent_at"] is None
    assert "Email could not be sent" in author.get("/admin/activity?show=alerts").get_data(as_text=True)

    for error, expected in (
            (smtplib.SMTPDataError(554, b"Message rejected: Email address is not verified."), "Email address is not verified"),
            (smtplib.SMTPRecipientsRefused({"rae@example.com": (550, b"No such user")}), "No such user"),
            (ConnectionRefusedError(111, "refused"), "Could not connect to the mail server at smtp.example.com on port 587"),
            (TimeoutError("timed out"), "did not answer in time")):
        mail.fail_with = error
        assert expected in author.post("/admin/reviewers/2/send-invite", follow_redirects=True).get_data(as_text=True)
    assert mail.outbox == []


def test_without_a_mail_account_links_are_still_sent_by_hand(app, author, mail):
    page = author.post("/admin/reviewers/add", {"name": "Rae Reader", "email": "rae@example.com", "send_email": "1"},
                       follow_redirects=True).get_data(as_text=True)
    assert "Send them the invite link below" in page and "Write email" in page and "Send by email" not in page
    refused = author.post("/admin/reviewers/2/send-invite", follow_redirects=True).get_data(as_text=True)
    assert "Email is not set up yet" in refused and mail.outbox == []
    assert "Save a mail server and a sender address first." in author.post(
        "/admin/settings/test-email", follow_redirects=True).get_data(as_text=True)


def test_author_can_add_a_reviewer_without_emailing_yet(app, author, mail):
    set_up_email(author)
    page = author.post("/admin/reviewers/add", {"name": "Rae Reader", "email": "rae@example.com"},
                       follow_redirects=True).get_data(as_text=True)
    assert "No email was sent" in page and "Not emailed yet." in page and mail.outbox == []
    assert "not emailed" in author.get("/admin/reviewers").get_data(as_text=True)


def test_author_words_the_invitation(app, author, mail):
    set_up_email(author, invite_subject="{author} would value your eye on {book}",
                 invite_message="Dear {name},\r\nplease read by {expires}. {unknown} stays as typed.")
    author.post("/admin/reviewers/add", {"name": "Rae Reader", "email": "rae@example.com", "send_email": "1"})
    (message,) = mail.outbox
    assert message["Subject"] == "Ann Author would value your eye on The Lighthouse"
    body = message.get_content()
    assert body.startswith("Dear Rae Reader,\nplease read by ") and "{unknown} stays as typed." in body
    assert "/invite/" in body.splitlines()[-1]                # the link is added when the wording leaves it out


def test_replies_can_be_directed_to_another_address(app, author, mail):
    """Sending from a dedicated sending domain with no inbox, while replies reach a real mailbox."""
    assert "not a valid email address" in set_up_email(author, mail_reply_to="info at example.org")
    set_up_email(author, mail_from_address="info@mail.example.org", mail_reply_to="info@example.org")
    author.post("/admin/reviewers/add", {"name": "Rae Reader", "email": "rae@example.com", "send_email": "1"})
    author.post("/admin/settings/test-email")
    invitation, test = mail.outbox
    for message in (invitation, test):
        assert message["From"] == "Ann Author <info@mail.example.org>" and message["Reply-To"] == "info@example.org"

    set_up_email(author, mail_reply_to="", smtp_password="")           # cleared again: no Reply-To header at all
    author.post("/admin/reviewers/2/send-invite")
    assert mail.outbox[2]["Reply-To"] is None and mail.outbox[2]["From"] == "Ann Author <ann@example.com>"


def test_form_input_cannot_add_mail_headers(app, author, mail):
    set_up_email(author, mail_from_name="Ann\r\nBcc: spy@example.com")
    author.post("/admin/reviewers/add", {"name": "Rae\r\nBcc: other@example.com", "email": "rae@example.com",
                                         "send_email": "1"})
    (message,) = mail.outbox
    assert message["Bcc"] is None and message["Cc"] is None
    assert message["To"] == '"Rae Bcc: other@example.com" <rae@example.com>'
    for bad in ("rae@example.com\r\nBcc: x@example.com", "rae@@example.com", "rae example@x.com", "rae@nodot", "<rae@example.com>"):
        refused = author.post("/admin/reviewers/add", {"name": "Bad Address", "email": bad}, follow_redirects=True)
        assert "valid email address" in refused.get_data(as_text=True), bad
    assert len(query(app, "SELECT id FROM users WHERE role = 'reviewer'")) == 1


def test_mail_settings_are_validated_and_the_password_is_kept(app, author, mail):
    assert "must be a number, usually 587 or 465" in set_up_email(author, smtp_port="0")
    assert "address invitations should come from" in set_up_email(author, mail_from_address="not-an-address")
    assert "for example smtp.gmail.com" in set_up_email(author, smtp_host="smtp example com")
    assert query(app, "SELECT COUNT(*) AS n FROM settings WHERE key = 'smtp_host'")[0]["n"] == 0

    set_up_email(author, smtp_security="ssl", smtp_port="465")
    set_up_email(author, smtp_security="ssl", smtp_port="465", smtp_password="")      # blank keeps the saved password
    sent = author.post("/admin/settings/test-email", follow_redirects=True).get_data(as_text=True)
    assert f"A test email was sent to {AUTHOR[0]}." in sent
    assert ("connect", "FakeMailServerSSL", "smtp.example.com", 465) in mail.log
    assert ("login", "ann.login", "app-password-123") in mail.log and not any(e[0] == "starttls" for e in mail.log)
    assert mail.outbox[0]["To"] == f"Ann Author <{AUTHOR[0]}>"

    set_up_email(author, smtp_host="", mail_from_address="", smtp_password="")        # switching email off
    stored = {r["key"]: r["value"] for r in query(app, "SELECT key, value FROM settings")}
    assert stored["smtp_host"] == "" and stored["smtp_password_enc"] == ""
    assert "Not set up" in author.get("/admin/settings").get_data(as_text=True)


def test_existing_users_table_gains_the_invite_sent_column(tmp_path):
    import sqlite3
    first = create_app(tmp_path / "data")
    conn = sqlite3.connect(first.config["VAULT"].db_path)
    conn.execute("ALTER TABLE users DROP COLUMN invite_sent_at")
    conn.commit()
    conn.close()
    upgraded = create_app(tmp_path / "data")
    with upgraded.app_context():
        assert "invite_sent_at" in {r["name"] for r in db.all("PRAGMA table_info(users)")}


# A real mail server on this machine: proves the encrypted connection, sign-in and delivery
# actually work through smtplib, not just against the stand-in above.

def _certificates(folder):
    from datetime import datetime, timedelta, timezone
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.now(timezone.utc)
    usage = dict(content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False,
                 encipher_only=False, decipher_only=False)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test mail CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1))
          .not_valid_after(now + timedelta(days=2))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, **usage), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
            .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=2))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=False, crl_sign=False, **usage), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    paths = {name: folder / f"{name}.pem" for name in ("ca", "cert", "key")}
    paths["ca"].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    paths["cert"].write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    paths["key"].write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
    return paths


@pytest.mark.filterwarnings("ignore:Requiring AUTH while not requiring TLS")
@pytest.mark.parametrize("mode", ["starttls", "ssl"])
def test_real_delivery_over_an_encrypted_connection(tmp_path, monkeypatch, mode):
    pytest.importorskip("aiosmtpd")
    import socket
    import ssl
    from aiosmtpd.controller import Controller
    from aiosmtpd.smtp import AuthResult

    certs = _certificates(tmp_path)
    (tmp_path / "stranger").mkdir()
    stranger = _certificates(tmp_path / "stranger")           # an authority that did not sign the server's certificate
    monkeypatch.setattr(mailer, "TIMEOUT", 4)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certs["cert"], certs["key"])
    received = []

    class Inbox:
        async def handle_DATA(self, server, session, envelope):
            received.append((session.ssl is not None, envelope.mail_from, envelope.rcpt_tos, envelope.content.decode()))
            return "250 Message accepted"

    def authenticator(server, session, envelope, mechanism, data):
        return AuthResult(success=(data.login, data.password) == (b"ann.login", b"app-password-123"))

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # On the always-encrypted port this test server only offers sign-in when told the link is already secure.
    options = (dict(ssl_context=tls, auth_require_tls=False) if mode == "ssl"
               else dict(tls_context=tls, require_starttls=True))
    controller = Controller(Inbox(), hostname="127.0.0.1", port=port, authenticator=authenticator,
                            auth_required=True, **options)
    controller.start()
    try:
        config = {"host": "localhost", "port": port, "security": mode, "username": "ann.login",
                  "password": "app-password-123", "from_name": "Ann Author", "from_address": "ann@example.com"}
        send = lambda **changes: mailer.send({**config, **changes}, "Rae Reader", "rae@example.com",  # noqa: E731
                                             "Chapters for your review", "Hello Rae,\nhttps://vault.example/invite/abc")

        # 1. A certificate nobody vouches for is refused: no message leaves over a connection we cannot trust.
        monkeypatch.delenv("SSL_CERT_DIR", raising=False)
        monkeypatch.setenv("SSL_CERT_FILE", str(stranger["ca"]))
        with pytest.raises(mailer.MailError, match="secure connection"):
            send()
        assert received == []

        # (A wrong password is covered with the stand-in server above: this test server sends no
        # reply within the time limit after a failed sign-in, which would only test its own stall.)
        monkeypatch.setenv("SSL_CERT_FILE", str(certs["ca"]))

        # 2. The real thing: encrypted connection, sign-in, delivery.
        send()
        ((encrypted, sender, recipients, content),) = received
        # The server only records the upgrade on the STARTTLS port; on the always-encrypted port the
        # refusal in step 1 is the proof that the link is TLS with certificate checking in force.
        assert encrypted or mode == "ssl"
        assert sender == "ann@example.com" and recipients == ["rae@example.com"]
        assert "Subject: Chapters for your review" in content and "https://vault.example/invite/abc" in content
        assert "From: Ann Author <ann@example.com>" in content and "To: Rae Reader <rae@example.com>" in content

        # 3. Asking for the wrong kind of connection is explained rather than hanging or crashing.
        with pytest.raises(mailer.MailError):
            send(security="ssl" if mode == "starttls" else "starttls")
    finally:
        controller.stop()


# -------------------------------------------------------------- note categories

import json as _json  # noqa: E402

STARTERS = ["Sounds AI-written", "Hard to follow", "Awkward wording", "Filler", "Repetitive", "Eyebrow text", "Works well"]


def reader_config(browser, view_url) -> dict:
    html = browser.get(view_url).get_data(as_text=True)
    return _json.loads(re.search(r'<script type="application/json" id="viewer-config">(.*?)</script>', html, re.S).group(1))


def category_id(app, name) -> int:
    return query(app, "SELECT id FROM categories WHERE name = ?", name)[0]["id"]


def test_a_vault_starts_with_starter_categories_and_keeps_the_authors_changes(tmp_path):
    app = create_app(tmp_path / "data")
    rows = query(app, "SELECT name, description, active FROM categories ORDER BY position")
    assert [r["name"] for r in rows] == STARTERS
    assert all(r["active"] == 1 and len(r["description"]) > 20 for r in rows)

    with app.app_context():                                    # the author removes every category ...
        db.run("DELETE FROM categories")
    again = create_app(tmp_path / "data")                      # ... and a restart does not bring them back
    assert query(again, "SELECT COUNT(*) AS n FROM categories")[0]["n"] == 0


def test_a_vault_from_before_categories_gains_them(tmp_path):
    import sqlite3
    first = create_app(tmp_path / "data")
    conn = sqlite3.connect(first.config["VAULT"].db_path)
    conn.executescript("DROP TABLE comment_categories; DROP TABLE categories; "
                       "DELETE FROM settings WHERE key = 'categories_seeded';"
                       "INSERT INTO comments (document_id, user_id, page, body, created_at) VALUES (1, 1, 2, 'old note', '2026-01-01T00:00:00Z');")
    conn.commit()
    conn.close()
    upgraded = create_app(tmp_path / "data")
    assert [r["name"] for r in query(upgraded, "SELECT name FROM categories ORDER BY position")] == STARTERS
    assert query(upgraded, "SELECT body FROM comments")[0]["body"] == "old note"


def test_author_manages_the_categories(app, author):
    page = author.get("/admin/categories").get_data(as_text=True)
    assert all(name in page for name in STARTERS) and "7 offered to reviewers" in page

    added = author.post("/admin/categories/add", {"name": "  Needs   a source ", "description": "A claim with nothing to back it up.", "color": ""},
                        follow_redirects=True).get_data(as_text=True)
    assert "“Needs a source” was added" in added
    new = query(app, "SELECT * FROM categories WHERE name = 'Needs a source'")[0]
    assert new["color"] == "teal" and new["position"] == 7        # the one colour no starter uses; added last

    assert "already a category called" in author.post("/admin/categories/add", {"name": "needs a SOURCE"}, follow_redirects=True).get_data(as_text=True)
    assert "Give the category a short name." in author.post("/admin/categories/add", {"name": "   "}, follow_redirects=True).get_data(as_text=True)
    assert query(app, "SELECT COUNT(*) AS n FROM categories")[0]["n"] == 8

    cid = new["id"]
    author.post(f"/admin/categories/{cid}/update", {"action": "save", "name": "Needs evidence", "description": "Says something without showing why.", "color": "rose"})
    assert query(app, "SELECT name, description, color FROM categories WHERE id = ?", cid)[0] == {
        "name": "Needs evidence", "description": "Says something without showing why.", "color": "rose"}
    clash = author.post(f"/admin/categories/{cid}/update", {"action": "save", "name": "Filler", "color": "rose"}, follow_redirects=True)
    assert "already a category called" in clash.get_data(as_text=True)

    order = lambda: [r["name"] for r in query(app, "SELECT name FROM categories WHERE active = 1 ORDER BY position, id")]  # noqa: E731
    author.post(f"/admin/categories/{cid}/update", {"action": "up"})
    assert order()[-2:] == ["Needs evidence", "Works well"]
    author.post(f"/admin/categories/{cid}/update", {"action": "down"})
    author.post(f"/admin/categories/{cid}/update", {"action": "down"})                 # already last: stays put
    assert order()[-2:] == ["Works well", "Needs evidence"]

    author.post(f"/admin/categories/{cid}/update", {"action": "retire"})
    assert "Needs evidence" not in order()
    author.post(f"/admin/categories/{cid}/update", {"action": "restore"})
    assert "Needs evidence" in order()
    author.post(f"/admin/categories/{cid}/update", {"action": "delete"})               # never used: really deleted
    assert not query(app, "SELECT 1 FROM categories WHERE id = ?", cid)
    assert author.post("/admin/categories/9999/update", {"action": "save", "name": "x"}).status_code == 404
    assert "Note categories changed" in author.get("/admin/activity").get_data(as_text=True)


def test_reviewer_tags_notes_with_categories(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    ai, hard, filler = (category_id(app, n) for n in ("Sounds AI-written", "Hard to follow", "Filler"))

    offered = reader_config(rae, view_url)["categories"]
    assert [c["name"] for c in offered] == STARTERS
    assert offered[0] == {"id": ai, "name": "Sounds AI-written", "color": "violet", "description": offered[0]["description"]}
    assert offered[0]["description"].startswith("Reads as machine-generated")

    box = rae.get(f"{view_url}/blocks/2").get_json()["blocks"][0]
    quick = rae.json(f"{view_url}/comments", {"page": 2, "box": box, "categories": [ai]})        # one tap, no typing
    assert quick.status_code == 201
    assert quick.get_json()["body"] == "" and [c["name"] for c in quick.get_json()["categories"]] == ["Sounds AI-written"]

    both = rae.json(f"{view_url}/comments", {"page": 3, "body": "Two ideas fighting in one sentence.",
                                             "categories": [hard, ai, hard]})                   # several, with details
    assert both.status_code == 201
    assert [c["name"] for c in both.get_json()["categories"]] == ["Sounds AI-written", "Hard to follow"]   # author's order, no repeats

    plain = rae.json(f"{view_url}/comments", {"page": 4, "body": "Just a thought."})             # details only still works
    assert plain.status_code == 201 and plain.get_json()["categories"] == []

    for bad in ({"page": 2}, {"page": 2, "body": "  ", "categories": []}, {"page": 2, "categories": [9999]},
                {"page": 2, "categories": "all"}, {"page": 2, "categories": [True]}, {"page": 2, "categories": [ai] * 9}):
        assert rae.json(f"{view_url}/comments", bad).status_code == 400, bad
    assert len(rae.get(f"{view_url}/comments").get_json()) == 3

    # editing: change the categories, add or remove the text
    note = quick.get_json()["id"]
    changed = rae.json(f"{view_url}/comments/{note}/edit", {"body": "Reads like a brochure.", "categories": [filler]}).get_json()
    assert changed["body"] == "Reads like a brochure." and [c["name"] for c in changed["categories"]] == ["Filler"] and changed["edited"]
    kept = rae.json(f"{view_url}/comments/{note}/edit", {"body": "Reads like a brochure, honestly."}).get_json()
    assert [c["name"] for c in kept["categories"]] == ["Filler"]                                 # categories untouched when not sent
    assert rae.json(f"{view_url}/comments/{note}/edit", {"body": "", "categories": [filler]}).status_code == 200
    assert rae.json(f"{view_url}/comments/{note}/edit", {"body": "", "categories": []}).status_code == 400

    # a retired category: gone from the choices, kept on notes that have it, and may stay through an edit
    author.post(f"/admin/categories/{filler}/update", {"action": "retire"})
    assert "Filler" not in [c["name"] for c in reader_config(rae, view_url)["categories"]]
    assert rae.json(f"{view_url}/comments", {"page": 2, "categories": [filler]}).status_code == 400
    listed = {c["id"]: c for c in rae.get(f"{view_url}/comments").get_json()}
    assert [c["name"] for c in listed[note]["categories"]] == ["Filler"]
    still = rae.json(f"{view_url}/comments/{note}/edit", {"body": "Padding.", "categories": [filler, hard]})
    assert still.status_code == 200 and {c["name"] for c in still.get_json()["categories"]} == {"Filler", "Hard to follow"}

    # deleting a category that is in use retires it instead, so the notes keep their label
    page = author.post(f"/admin/categories/{hard}/update", {"action": "delete"}, follow_redirects=True).get_data(as_text=True)
    assert "retired instead of deleted" in page
    assert query(app, "SELECT active FROM categories WHERE id = ?", hard)[0]["active"] == 0
    assert len(query(app, "SELECT * FROM comment_categories WHERE category_id = ?", hard)) == 2


def test_author_sees_categories_in_the_feedback(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    ai, hard = category_id(app, "Sounds AI-written"), category_id(app, "Hard to follow")
    rae.json(f"{view_url}/comments", {"page": 1, "categories": [ai]})
    rae.json(f"{view_url}/comments", {"page": 2, "body": "Dense paragraph.", "categories": [ai, hard]})
    rae.json(f"{view_url}/comments", {"page": 3, "body": "No category on this one."})

    page = author.get("/admin/chapters/1").get_data(as_text=True)
    assert re.search(r"All <b>3</b>", page) and re.search(r"Sounds AI-written <b>2</b>", page)
    assert re.search(r"Hard to follow <b>1</b>", page) and re.search(r"No category <b>1</b>", page)
    assert page.count('class="cat-label c-violet"') == 2 and page.count('class="cat-label c-blue"') == 1

    only_hard = author.get(f"/admin/chapters/1?category={hard}").get_data(as_text=True)
    assert "Dense paragraph." in only_hard and "No category on this one." not in only_hard
    without = author.get("/admin/chapters/1?category=none").get_data(as_text=True)
    assert "No category on this one." in without and "Dense paragraph." not in without
    assert "Dense paragraph." in author.get("/admin/chapters/1?category=nonsense").get_data(as_text=True)   # ignored

    rows = list(csv_rows(author.get("/admin/chapters/1/feedback.csv").get_data(as_text=True)))
    assert rows[0][4] == "Category"
    assert [r[4] for r in rows[1:]] == ["Sounds AI-written", "Sounds AI-written; Hard to follow", ""]

    # the author's preview shows every reviewer's labels
    preview = author.post("/admin/chapters/1/preview").headers["Location"]
    seen = author.get(preview + "/comments").get_json()
    assert [[c["name"] for c in n["categories"]] for n in seen] == [["Sounds AI-written"], ["Sounds AI-written", "Hard to follow"], []]


def csv_rows(text):
    import csv
    import io
    return csv.reader(io.StringIO(text.lstrip("﻿")))


# ------------------------------------------------------- reading progress report

from datetime import datetime as _dt, timedelta as _td, timezone as _tz  # noqa: E402

from vault import progress as reporting  # noqa: E402


def test_paragraphs_are_covered_one_by_one_before_a_chapter_can_be_marked_reviewed(app, author):
    upload(author)                                           # five pages, one nine-word paragraph on each
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    sam = add_reviewer(app, author, "Sam Skimmer", "sam@example.com", documents_=["1"])
    view_url, sam_view = open_chapter(rae), open_chapter(sam)
    config = reader_config(rae, view_url)
    assert config["coverUrl"] == f"{view_url}/cover" and config["reviewedUrl"] == f"{view_url}/reviewed"
    assert config["coverage"] == {"total": 5, "left": 5, "reviewed": False, "pagesLeft": [1, 1, 1, 1, 1],
                                  "minWords": 5, "wordsPerMinute": 600}

    for page in (1, 2):                                      # pages loaded (and preloaded) by the reader ...
        assert rae.get(f"{view_url}/page/{page}").status_code == 200
    library = rae.get("/library").get_data(as_text=True)     # ... are not progress
    assert "<b>0 of 5</b> pages read" in library and "5 paragraphs to go" in library

    assert rae.json(f"{view_url}/cover", {"page": 1, "paras": [0], "how": "seen"}).get_json() == {
        "ok": True, "left": 4, "pageLeft": 0}
    assert "<b>1 of 5</b> pages read" in rae.get("/library").get_data(as_text=True)
    assert rae.get(f"{view_url}/blocks/1").get_json()["covered"] == {"0": "seen"}
    assert len(query(app, "SELECT * FROM events WHERE kind = 'page_read'")) == 1          # the page is read once its paragraphs are

    same = rae.json(f"{view_url}/cover", {"page": 1, "paras": [0], "how": "confirmed"}).get_json()
    assert same["left"] == 4 and query(app, "SELECT how FROM coverage")[0]["how"] == "seen"   # first record stands
    assert rae.json(f"{view_url}/cover", {"page": 3, "paras": [0], "how": "seen"}).get_json()["left"] == 4   # never served
    assert rae.json(f"{view_url}/cover", {"page": 2, "paras": [7, 99], "how": "seen"}).get_json() == {
        "ok": True, "left": 4, "pageLeft": 1}                                              # no such paragraphs
    for bad in ({"page": 2, "paras": "all", "how": "seen"}, {"page": 2, "paras": [0], "how": "skimmed"},
                {"page": 99, "paras": [0], "how": "seen"}, {"page": 2, "paras": [-1], "how": "seen"},
                {"page": 2, "paras": [True], "how": "seen"}, {"paras": [0], "how": "seen"}):
        assert rae.json(f"{view_url}/cover", bad).status_code == 400, bad
    # the page-level rule is only for pages without text, so it cannot be used to skip paragraphs
    assert rae.json(f"{view_url}/read", {"pages": [2]}).get_json() == {"ok": True, "recorded": 0}
    assert sam.json(f"{view_url}/cover", {"page": 1, "paras": [0], "how": "seen"}).status_code == 404   # not Sam's session
    assert sam_view != view_url and reader_config(sam, sam_view)["coverage"]["left"] == 5

    refused = rae.json(f"{view_url}/reviewed", {"done": True})
    assert refused.status_code == 409 and refused.get_json()["left"] == 4
    assert "4 paragraphs still to read" in refused.get_json()["message"]

    for page in (3, 4, 5):
        rae.get(f"{view_url}/page/{page}")
    rae.json(f"{view_url}/cover", {"page": 2, "paras": [0], "how": "seen"})
    rae.json(f"{view_url}/cover", {"page": 3, "paras": [0], "how": "confirmed"})           # ticked by hand
    rae.json(f"{view_url}/cover", {"page": 4, "paras": [0], "how": "confirmed"})
    assert rae.json(f"{view_url}/reviewed", {"done": True}).status_code == 409               # one paragraph short
    last = rae.json(f"{view_url}/cover", {"page": 5, "paras": [0], "how": "seen"}).get_json()
    assert last == {"ok": True, "left": 0, "pageLeft": 0}
    assert query(app, "SELECT reviewed_at FROM grants WHERE user_id = 2")[0]["reviewed_at"] is None   # reading is not declaring

    assert rae.json(f"{view_url}/reviewed", {"done": True}).get_json() == {"ok": True, "reviewed": True}
    assert query(app, "SELECT reviewed_at FROM grants WHERE user_id = 2")[0]["reviewed_at"]
    assert reader_config(rae, view_url)["coverage"]["reviewed"] is True
    library = rae.get("/library").get_data(as_text=True)
    assert "<b>5 of 5</b> pages read" in library and "Marked as reviewed" in library
    assert "Chapter marked as reviewed" in author.get("/admin/activity").get_data(as_text=True)
    assert sorted(r["how"] for r in query(app, "SELECT how FROM coverage WHERE user_id = 2")) == [
        "confirmed", "confirmed", "seen", "seen", "seen"]

    assert rae.json(f"{view_url}/reviewed", {"done": False}).get_json() == {"ok": True, "reviewed": False}
    assert query(app, "SELECT reviewed_at FROM grants WHERE user_id = 2")[0]["reviewed_at"] is None

    preview = author.post("/admin/chapters/1/preview").headers["Location"]                  # the author's preview
    author.get(f"{preview}/page/1")
    assert reader_config(author, preview)["coverage"] is None
    assert author.json(f"{preview}/cover", {"page": 1, "paras": [0], "how": "seen"}).get_json() == {"ok": True, "left": 0}
    assert author.json(f"{preview}/reviewed", {"done": True}).status_code == 403
    assert len(query(app, "SELECT * FROM coverage")) == 5

    author.post("/admin/grants/1/update", {"action": "revoke"})
    assert rae.json(f"{view_url}/cover", {"page": 1, "paras": [0], "how": "seen"}).status_code == 403
    assert rae.json(f"{view_url}/reviewed", {"done": True}).status_code == 403
    listed = author.get("/admin/activity").get_data(as_text=True)                           # not in the everyday log ...
    assert "Page read" not in listed
    assert "Page read" in author.get("/admin/activity?show=reading").get_data(as_text=True)  # ... but in the reading view


def test_pages_without_text_count_when_looked_at_and_headings_are_not_required(app, author):
    source = pymupdf.open()
    first = source.new_page()
    first.insert_text((72, 90), "Chapter One", fontsize=20)                                  # a heading: too short to be required
    first.insert_text((72, 160), "A real paragraph with easily more than five words in it.", fontsize=12)
    source.new_page()                                                                        # a blank page, like a figure or a scan
    upload(author, data=source.tobytes())
    assert _json.loads(query(app, "SELECT paras_json FROM documents")[0]["paras_json"]) == [[2, 11], []]

    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    assert reader_config(rae, view_url)["coverage"]["pagesLeft"] == [1, 1]                  # one paragraph, one page to look at
    for page in (1, 2):
        rae.get(f"{view_url}/page/{page}")
    assert rae.json(f"{view_url}/cover", {"page": 1, "paras": [0], "how": "seen"}).get_json()["left"] == 2   # the heading does not count
    assert rae.json(f"{view_url}/cover", {"page": 1, "paras": [1], "how": "seen"}).get_json() == {"ok": True, "left": 1, "pageLeft": 0}
    assert rae.json(f"{view_url}/cover", {"page": 2, "paras": [0], "how": "seen"}).get_json()["left"] == 1   # nothing to tick on a blank page
    assert rae.json(f"{view_url}/reviewed", {"done": True}).status_code == 409
    assert rae.json(f"{view_url}/read", {"pages": [1, 2]}).get_json() == {"ok": True, "recorded": 1}        # only the blank page
    assert rae.json(f"{view_url}/reviewed", {"done": True}).status_code == 200


def test_author_sees_what_each_reviewer_has_not_covered(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    for page in (1, 2, 3):
        rae.get(f"{view_url}/page/{page}")
    rae.json(f"{view_url}/cover", {"page": 1, "paras": [0], "how": "seen"})
    rae.json(f"{view_url}/cover", {"page": 2, "paras": [0], "how": "confirmed"})
    box = rae.get(f"{view_url}/blocks/3").get_json()["blocks"][0]
    rae.json(f"{view_url}/comments", {"page": 3, "body": "Good.", "box": box})

    chapter = author.get("/admin/chapters/1").get_data(as_text=True)
    assert "3 of 5 paragraphs" in chapter and "not marked as reviewed" in chapter and "2 paragraphs not covered" in chapter
    detail = author.get("/admin/grants/1/coverage").get_data(as_text=True)
    assert "What Rae Reader has covered" in detail and "3 of 5 paragraphs covered" in detail
    for line in (f"{SECRET_TEXT} - page 4", f"{SECRET_TEXT} - page 5"):                       # the author sees the skipped text
        assert line in detail
    assert f"{SECRET_TEXT} - page 1" not in detail
    facts = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", detail))
    assert "Read on screen 1 " in facts and "Commented on 1 " in facts and "Ticked by hand 1 " in facts and "Still to cover 2" in facts
    assert rae.get("/admin/grants/1/coverage").status_code == 404 and author.get("/admin/grants/99/coverage").status_code == 404

    # a revised file: paragraphs no longer line up, so reading starts again; notes and access stay
    replaced = author.post("/admin/chapters/1/replace", {"file": (io.BytesIO(make_pdf(3)), "draft-2.pdf")},
                           content_type="multipart/form-data", follow_redirects=True).get_data(as_text=True)
    assert "reading progress on this chapter starts again" in replaced
    assert query(app, "SELECT COUNT(*) AS n FROM coverage")[0]["n"] == 0
    assert query(app, "SELECT COUNT(*) AS n FROM events WHERE kind = 'page_read'")[0]["n"] == 0
    assert query(app, "SELECT COUNT(*) AS n FROM comments")[0]["n"] == 1 and query(app, "SELECT status FROM grants")[0]["status"] == "active"
    assert "Nothing skipped" not in author.get("/admin/grants/1/coverage").get_data(as_text=True)


def test_chapters_uploaded_before_coverage_get_their_paragraphs_counted(app, author):
    upload(author)
    with app.app_context():
        db.run("UPDATE documents SET paras_json = NULL")                                     # as an older version left it
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    assert "5 paragraphs to go" in rae.get("/library").get_data(as_text=True)
    assert _json.loads(query(app, "SELECT paras_json FROM documents")[0]["paras_json"]) == [[9]] * 5


def test_reviewer_sees_a_summary_of_their_own_review(app, author):
    now = stage_review(app, author, target_in_days=5)                                         # Ada, Ben, Cy and Di, with history
    with app.app_context():
        for user_id, email in ((3, "ben@example.com"),):
            db.run("UPDATE users SET status = 'active', password_hash = ?, nda_accepted_at = ? WHERE id = ?",
                   auth.hash_password("reviewer password 1"), db.iso(now), user_id)
        db.run("INSERT INTO comment_categories (comment_id, category_id) VALUES (1, (SELECT id FROM categories WHERE name = 'Filler'))")
    ben = Browser(app)
    assert ben.login("ben@example.com", "reviewer password 1").status_code == 302
    page = ben.get("/library").get_data(as_text=True)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))

    assert "60%" in page and "of your pages read" in page and "6 of 10 pages" in text and "1 of 2 chapters finished" in text
    assert "A faster pace is needed" in page and "At your current pace you would finish on" in page
    assert "after the target" in page and "would finish by" in page
    assert "5 pages read in the last 7 days" in text and "2 notes written" in text and "Filler 1" in text
    assert 'class="pv-spark pv-spark-large"' in page and "An even pace to the target" in page
    assert "Cy Slow" not in page and "Ada Done" not in page and "ada@example.com" not in page   # their own figures only
    assert "style=" not in page

    # the page map: one cell per page; chapter one was read before paragraph tracking, so by
    # coverage nothing is ticked yet and every paragraph is still to go
    assert page.count('name="page"') == 10 and page.count("pm pm-none") >= 10
    assert 'title="Page 3: not read yet (0 of 1 paragraphs)"' in page
    there = ben.post("/read/1/open", {"page": "3"}).headers["Location"]
    assert re.fullmatch(r"/view/\d+#p=3", there)
    assert "#" not in ben.post("/read/1/open", {"page": "99"}).headers["Location"]
    view_url = there.split("#")[0]
    ben.get(f"{view_url}/page/3")
    ben.json(f"{view_url}/cover", {"page": 3, "paras": [0], "how": "seen"})
    assert 'title="Page 3: read"' in ben.get("/library").get_data(as_text=True)

    # someone with nothing assigned sees no summary, and nobody else's
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com")
    empty = rae.get("/library").get_data(as_text=True)
    assert "Nothing to read yet" in empty and "of your pages read" not in empty


def test_outlook_says_plainly_where_a_reviewer_stands():
    now = _dt(2026, 12, 1, 12, tzinfo=_tz.utc)
    target = _dt(2026, 12, 31, tzinfo=_tz.utc).date()
    snap = {"now": now, "target": target, "target_end": _dt(2026, 12, 31, 23, 59, 59, tzinfo=_tz.utc)}
    person = lambda **k: reporting.Person(1, "Rae", "rae@example.com", **k)                 # noqa: E731

    assert reporting.outlook(person(total=10, read=10, status="finished"), snap)[0] == "You have finished"
    heading, detail = reporting.outlook(person(total=100, read=0, status="not_started", needed=3.3), snap)
    assert heading == "Not started yet" and detail == "100 pages to read. About 4 pages a day from today would finish by 31 Dec 2026."
    heading, detail = reporting.outlook(person(total=100, read=80, status="on_track", pace=4.0, needed=0.67,
                                               projected=now + _td(days=5)), snap)
    assert heading == "On course" and detail == "At your current pace you finish on 6 Dec 2026, 25 days before the target of 31 Dec 2026."
    heading, detail = reporting.outlook(person(total=100, read=40, status="at_risk", pace=1.5, needed=2.0,
                                               projected=now + _td(days=40)), snap)
    assert heading == "A faster pace is needed"
    assert detail == "At your current pace you would finish on 10 Jan 2027, 10 days after the target. About 2 pages a day would finish by 31 Dec 2026."
    heading, detail = reporting.outlook(person(total=100, read=95, status="behind", pace=0.0, needed=0.17), snap)
    assert detail == ("You have not read in the last 14 days and have 5 pages left. "
                      "About 2 pages a week from today would finish by 31 Dec 2026.")
    assert reporting.outlook(person(total=10, read=4, status="overdue"), snap)[0] == "The target date has passed"
    untargeted = {"now": now, "target": None, "target_end": None}
    assert reporting.outlook(person(total=10, read=4, status="in_progress", projected=now + _td(days=3)), untargeted) == (
        "In progress", "6 pages left to read. At your current pace you would finish on 4 Dec 2026.")
    assert reporting.outlook(person(total=10, read=9, status="in_progress"), untargeted) == ("In progress", "1 page left to read.")


def test_progress_from_before_the_change_is_carried_over_once(tmp_path):
    import sqlite3
    first = create_app(tmp_path / "data")
    conn = sqlite3.connect(first.config["VAULT"].db_path)
    conn.execute("DELETE FROM settings WHERE key = 'page_read_carried_over'")
    for view, page, ts in ((1, 1, "2026-11-01T10:00:00Z"), (2, 1, "2026-11-03T10:00:00Z"), (1, 2, "2026-11-01T10:05:00Z")):
        conn.execute("INSERT INTO events (ts, kind, user_id, document_id, view_id, page, actor, subject) "
                     "VALUES (?, 'page_view', 7, 3, ?, ?, 'Rae <rae@example.com>', 'Chapter')", (ts, view, page))
    conn.commit()
    conn.close()
    upgraded = create_app(tmp_path / "data")
    carried = query(upgraded, "SELECT ts, user_id, document_id, page FROM events WHERE kind = 'page_read' ORDER BY page")
    assert carried == [{"ts": "2026-11-01T10:00:00Z", "user_id": 7, "document_id": 3, "page": 1},
                       {"ts": "2026-11-01T10:05:00Z", "user_id": 7, "document_id": 3, "page": 2}]
    create_app(tmp_path / "data")
    assert len(query(upgraded, "SELECT 1 FROM events WHERE kind = 'page_read'")) == 2          # not copied twice


def test_status_rules():
    c = reporting.classify
    assert c(100, 100, 0.0, None, 10) == "finished"
    assert c(100, 100, 0.0, None, -3) == "finished"               # finished stays finished after the date
    assert c(100, 40, 5.0, None, 0) == "overdue" and c(100, 40, 5.0, None, -2) == "overdue"
    assert c(100, 0, 0.0, 10.0, 10) == "not_started"
    assert c(100, 40, 3.0, None, None) == "in_progress"           # no target date set
    assert c(100, 40, 6.0, 6.0, 10) == "on_track" and c(100, 40, 9.0, 6.0, 10) == "on_track"
    assert c(100, 40, 3.6, 6.0, 10) == "at_risk"                  # exactly 60% of what is needed
    assert c(100, 40, 3.5, 6.0, 10) == "behind" and c(100, 40, 0.0, 6.0, 10) == "behind"


def test_names_are_listed_readably():
    people = [reporting.Person(i, name, "") for i, name in enumerate(["Ann", "Bo", "Cy", "Di", "Ed"])]
    assert reporting._names(people[:1]) == "Ann" and reporting._names(people[:2]) == "Ann and Bo"
    assert reporting._names(people[:3]) == "Ann, Bo and Cy" and reporting._names(people) == "Ann, Bo, Cy and 2 more"


def test_recent_pace_counts_quiet_days():
    now = _dt(2026, 12, 1, 12, tzinfo=_tz.utc)
    days = lambda n: now - _td(days=n)                                           # noqa: E731
    assert reporting.recent_pace([days(1)] * 30, days(3), now) == pytest.approx(10.0)       # assigned 3 days ago
    assert reporting.recent_pace([days(1)] * 28, days(60), now) == pytest.approx(2.0)       # averaged over 14 days
    assert reporting.recent_pace([days(20)] * 50, days(60), now) == 0.0                     # a burst, then silence
    assert reporting.recent_pace([now] * 5, now, now) == pytest.approx(5.0)                 # never divides by ~zero


def stage_review(app, author, target_in_days=5):
    """Two 5-page chapters, four reviewers assigned ten days ago, each with a known reading history."""
    upload(author)
    upload(author, name="chapter-two.pdf")
    for name in ("Ada Done", "Ben Steady", "Cy Slow", "Di Absent"):
        author.post("/admin/reviewers/add", {"name": name, "email": f"{name.split()[0].lower()}@example.com", "documents": ["1", "2"]})
    with app.app_context():
        now = db.utcnow()
        db.run("UPDATE grants SET created_at = ?, expires_at = NULL", db.iso(now - _td(days=10)))
        history = {
            2: [(doc, page, 9 - page) for doc in (1, 2) for page in range(1, 6)],          # Ada: everything
            3: [(1, page, 10 - 2 * page) for page in range(1, 6)] + [(2, 1, 0.5)],         # Ben: 6 pages, steadily
            4: [(1, 1, 9), (1, 2, 9)],                                                      # Cy: 2 pages, long ago
        }                                                                                    # Di: nothing
        for user_id, reads in history.items():
            for doc, page, days_ago in reads:
                db.run("INSERT INTO events (ts, kind, user_id, document_id, page) VALUES (?, 'page_read', ?, ?, ?)",
                       db.iso(now - _td(days=days_ago)), user_id, doc, page)
        db.run("INSERT INTO comments (document_id, user_id, page, body, created_at) VALUES (1, 3, 1, 'recent', ?)", db.iso(now - _td(days=2)))
        db.run("INSERT INTO comments (document_id, user_id, page, body, created_at) VALUES (1, 3, 2, 'older', ?)", db.iso(now - _td(days=9)))
        db.set_setting("review_target_date", (now + _td(days=target_in_days)).strftime("%Y-%m-%d"))
        return now


def test_progress_report_forecasts_each_reviewer(app, author):
    now = stage_review(app, author)
    with app.app_context():
        snap = reporting.snapshot(now)
        chart = reporting.trajectory_chart(snap)
        summary = reporting.chart_summary(snap)
        sparks = {p.name: str(reporting.sparkline(p, snap)) for p in snap["people"]}
    people = {p.name: p for p in snap["people"]}
    assert [p.name for p in snap["people"]] == ["Cy Slow", "Ben Steady", "Di Absent", "Ada Done"]     # worst first

    ada, ben, cy, di = (people[n] for n in ("Ada Done", "Ben Steady", "Cy Slow", "Di Absent"))
    assert (ada.status, ada.read, ada.total, ada.chapters_finished, ada.projected) == ("finished", 10, 10, 2, None)
    assert (di.status, di.read, di.pace, di.projected) == ("not_started", 0, 0.0, None)

    days_left = snap["days_left"]
    assert 5 <= days_left < 6                                             # to the end of the target day
    assert ben.read == 6 and ben.pace == pytest.approx(0.6) and ben.needed == pytest.approx(4 / days_left)
    assert ben.status == "at_risk" and ben.projected == now + _td(days=4 / 0.6)        # 6.7 days: after the target
    assert ben.projected > snap["target_end"]
    assert (ben.read_7d, ben.read_prev_7d, ben.notes, ben.notes_7d) == (5, 1, 2, 1)     # read 8, 6, 4, 2, 0.5 and 0 days ago
    assert cy.read == 2 and cy.pace == pytest.approx(0.2) and cy.status == "behind"
    assert cy.projected == now + _td(days=8 / 0.2)

    overall = snap["overall"]
    assert (overall.total, overall.read, round(overall.percent)) == (40, 18, 45)
    # The whole review follows its slowest reviewer: one is behind, and one has not started,
    # so there is no finish date at all, however fast the others read.
    assert overall.pace == pytest.approx(1.8) and overall.status == "behind" and overall.projected is None
    assert snap["on_course"] == 1 and snap["last"].name == "Cy Slow"
    assert [p.name for p in snap["not_started"]] == ["Di Absent"] and snap["not_reading"] == []
    assert [(key, count) for key, _label, _tone, count in snap["counts"]] == [
        ("behind", 1), ("at_risk", 1), ("not_started", 1), ("finished", 1)]
    chapters = {c["title"]: c for c in snap["chapters"]}
    assert chapters["chapter one"]["read"] == 12 and chapters["chapter one"]["percent"] == pytest.approx(60.0)
    assert chapters["chapter two"]["read"] == 6 and chapters["chapter two"]["finished"] == 1
    assert [c["title"] for c in snap["chapters"]] == ["chapter two", "chapter one"]                 # least read first
    assert chapters["chapter one"]["notes"] == 2 and snap["expiring"] == []

    # the cumulative series ends at today's total, never decreases, and the chart draws all three lines
    assert ben.cumulative[-1] == 6 and ben.cumulative == sorted(ben.cumulative) and len(ben.days) == 11
    assert overall.cumulative[-1] == 18
    svg = str(chart["svg"])
    for part in ('class="pv-actual"', 'class="pv-needed"', 'class="pv-projected"', "45% read", "Today", "Target", "Even pace"):
        assert part in svg, part
    assert chart["rows"][-1]["actual"] == pytest.approx(45.0) and 60 < chart["rows"][-1]["even"] < 70
    assert chart["data"]["points"][-1]["actual"] == 45.0 and len(chart["data"]["points"]) == 11
    assert summary == ("45% of assigned pages read. At the current pace the review levels off at 75% and never finishes: "
                       "Di Absent has not started.")
    with app.app_context():
        curve = reporting.combined_projection(snap, now + _td(days=60))
    assert [round(percent, 1) for _moment, percent in curve] == [45.0, 58.3, 75.0, 75.0]     # bends as Ben, then Cy, finish
    assert curve[1][0] == now + _td(days=4 / 0.6) and curve[2][0] == now + _td(days=40)

    # Once everyone is reading, the forecast is the date the last of them finishes.
    with app.app_context():
        db.run("DELETE FROM grants WHERE user_id = 5")                                          # Di is taken off the review
        snap = reporting.snapshot(now)
        summary = reporting.chart_summary(snap)
        chart = reporting.trajectory_chart(snap)
    assert snap["overall"].projected == now + _td(days=40) and snap["last"].name == "Cy Slow"
    assert snap["overall"].status == "behind" and snap["on_course"] == 1 and len(snap["people"]) == 3
    assert summary.startswith("60% of assigned pages read. At the current pace the last reviewer, Cy Slow, finishes on ")
    assert summary.endswith("days after the target of " + reporting.short_date(snap["target"], True) + ".")
    assert "At the current pace" in str(chart["svg"]) and chart["has_projection"]

    # And when the slow one catches up, the review is on track.
    with app.app_context():
        for page in range(3, 6):
            db.run("INSERT INTO events (ts, kind, user_id, document_id, page) VALUES (?, 'page_read', 4, 1, ?)", db.iso(now - _td(days=1)), page)
        for page in range(1, 5):
            db.run("INSERT INTO events (ts, kind, user_id, document_id, page) VALUES (?, 'page_read', 4, 2, ?)", db.iso(now - _td(days=1)), page)
        snap = reporting.snapshot(now)
        summary = reporting.chart_summary(snap)
    cy = next(p for p in snap["people"] if p.name == "Cy Slow")
    assert cy.read == 9 and cy.status == "on_track"
    assert {p.status for p in snap["people"]} == {"finished", "at_risk", "on_track"} and snap["overall"].status == "at_risk"
    assert snap["last"].name == "Ben Steady" and "the last reviewer, Ben Steady, finishes on" in summary
    assert all("<svg" in s and "pv-actual" in s for s in sparks.values())
    assert "an even pace would be at" in sparks["Ben Steady"]


def test_progress_page_export_and_target_date(app, author):
    empty = author.get("/admin/progress").get_data(as_text=True)
    assert "Nothing to report yet" in empty and "Print or save as PDF" not in empty

    stage_review(app, author)
    page = author.get("/admin/progress").get_data(as_text=True)
    for text in ("Review progress", "45%", "18 of 40 pages across 4 reviewers", "Ben Steady", "At risk", "Behind", "Not started",
                 "Finished", "after the target", "Projected finish",
                 "Print or save as PDF", "How this is worked out", 'id="pv-data"', "chapter two"):
        assert text in page, text
    table = page.split('class="pv-people"')[1]                 # names also appear in the "was added" banners above
    assert table.index("Cy Slow") < table.index("Ben Steady") < table.index("Di Absent") < table.index("Ada Done")
    assert "style=" not in page                                # the content-security policy allows no inline styles

    rows = list(csv_rows(author.get("/admin/progress.csv").get_data(as_text=True)))
    assert rows[0][:6] == ["Reviewer", "Email", "Status", "Pages assigned", "Pages read", "Percent read"]
    by_name = {r[0]: r for r in rows[1:]}
    assert by_name["Ben Steady"][2:9] == ["At risk", "10", "6", "60.0", "5", "1", "0.60"]
    assert by_name["Ada Done"][2] == "Finished" and by_name["Di Absent"][10] == ""

    # without a target date there is progress but no verdict
    assert "Target date cleared." in author.post("/admin/progress/target", {"target": ""}, follow_redirects=True).get_data(as_text=True)
    untargeted = author.get("/admin/progress").get_data(as_text=True)
    assert "Set a target date above" in untargeted and "In progress" in untargeted
    assert "At risk" not in untargeted and "Behind" not in untargeted and "needed" not in untargeted.split('class="pv-people"')[1]
    assert "pv-needed" not in untargeted
    assert "YYYY-MM-DD" in author.post("/admin/progress/target", {"target": "1 January"}, follow_redirects=True).get_data(as_text=True)

    # a date already gone by: unfinished reviewers are overdue, finished ones stay finished
    author.post("/admin/progress/target", {"target": "2020-01-01"})
    past = author.get("/admin/progress").get_data(as_text=True)
    assert "target date passed" in past and "Overdue" in past and "Finished" in past

    # access that runs out before the target is called out
    with app.app_context():
        soon = db.utcnow() + _td(days=2)
        db.set_setting("review_target_date", (db.utcnow() + _td(days=30)).strftime("%Y-%m-%d"))
        db.run("UPDATE grants SET expires_at = ? WHERE user_id IN (2, 3)", db.iso(soon))
    warned = author.get("/admin/progress").get_data(as_text=True)
    # Ada has finished everything and Ben has finished chapter one, so only Ben's chapter two is at stake
    assert "Access ends before the target date" in warned and "for 1 unfinished chapter:" in warned
    assert "Ben Steady" in warned.split("Access ends before the target date")[1].split("</div>")[0]

    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com")
    assert rae.get("/admin/progress").status_code == 404 and rae.get("/admin/progress.csv").status_code == 404


# ------------------------------------------------------ automated tools and reading signals

CHATGPT = ("Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; ChatGPT-User/1.0; "
           "+https://openai.com/bot")
SCRIPT = "python-requests/2.32.3"
REAL_BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/141.0.0.0 Safari/537.36")


def test_only_tools_that_say_what_they_are_are_recognised():
    from vault import automation
    people = [
        REAL_BROWSER,
        "Mozilla/5.0 (iPad; CPU OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
        "Mozilla/5.0 (Linux; Android 13; CUBOT KINGKONG 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 OPR/125.0.0.0",
        "Werkzeug/3.1.3",
    ]
    for agent in people:
        assert automation.declared({"User-Agent": agent}) is None, agent
    tools = {
        CHATGPT: "ChatGPT",
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; Claude-User/1.0; +Claude-User@anthropic.com)": "Claude",
        "Mozilla/5.0 (compatible; Perplexity-User/1.0; +https://perplexity.ai/perplexity-user)": "Perplexity",
        "Mozilla/5.0 (compatible; GPTBot/1.2; +https://openai.com/gptbot)": "OpenAI crawler",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/141.0.0.0 Safari/537.36": "a headless browser",
        SCRIPT: "a Python script",
        "curl/8.5.0": "curl",
        "": "a tool that does not identify itself",
    }
    for agent, name in tools.items():
        assert automation.declared({"User-Agent": agent}) == name, agent
    signed = automation.declared({"User-Agent": REAL_BROWSER, "Signature-Agent": '"https://chatgpt.com"'})
    assert signed == "a signed agent (https://chatgpt.com)"


def test_declared_tools_and_ai_agents_are_refused_for_reviewers(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    assert rae.get(f"{view_url}/page/1").mimetype == "image/jpeg"

    refused = rae.client.get("/library", headers={"User-Agent": CHATGPT})
    assert refused.status_code == 403
    assert "ordinary web browser" in refused.get_data(as_text=True) and "chapter one" not in refused.get_data(as_text=True)
    assert rae.client.get(f"{view_url}/page/2", headers={"User-Agent": CHATGPT}).status_code == 403   # no page image
    again = rae.client.post(f"{view_url}/ping", json={}, headers={"User-Agent": CHATGPT, "X-CSRF-Token": rae.token})
    assert again.status_code == 403 and "AI agents are not allowed" in again.get_json()["message"]
    logged = query(app, "SELECT detail, actor FROM events WHERE kind = 'automation_blocked'")
    assert logged == [{"detail": "ChatGPT", "actor": "Rae Reader <rae@example.com>"}]      # once, not once per request

    assert rae.client.get("/library", headers={"User-Agent": SCRIPT}).status_code == 403
    assert rae.client.get("/library", headers={"User-Agent": ""}).status_code == 403
    assert rae.client.get("/library", headers={"User-Agent": REAL_BROWSER, "Signature-Agent": '"https://chatgpt.com"'}).status_code == 403
    assert [e["detail"] for e in query(app, "SELECT detail FROM events WHERE kind = 'automation_blocked' ORDER BY id")] == [
        "ChatGPT", "a Python script", "a tool that does not identify itself", "a signed agent (https://chatgpt.com)"]

    # the person, in their ordinary browser, carries on; so does the author, whatever they use
    assert rae.client.get("/library", headers={"User-Agent": REAL_BROWSER}).status_code == 200
    assert rae.get(f"{view_url}/page/2").mimetype == "image/jpeg"
    assert author.client.get("/admin/chapters", headers={"User-Agent": "curl/8.5.0"}).status_code == 200
    assert Browser(app).client.get("/login", headers={"User-Agent": "curl/8.5.0"}).status_code == 200

    activity = author.get("/admin/activity").get_data(as_text=True)
    assert "Automated tool or AI agent blocked" in activity and "ChatGPT" in activity
    assert "Automated tool or AI agent blocked" in author.get("/admin/").get_data(as_text=True)   # an alert on the dashboard


def test_a_script_cannot_sign_in_as_a_reviewer_or_use_up_their_devices(app, author):
    add_reviewer(app, author, "Rae Reader", "rae@example.com", password="reviewer password 1")
    devices = query(app, "SELECT COUNT(*) AS n FROM devices")[0]["n"]
    script = Browser(app)
    script.client.environ_base["HTTP_USER_AGENT"] = SCRIPT
    response = script.login("rae@example.com", "reviewer password 1")
    assert response.status_code == 403 and "ordinary web browser" in response.get_data(as_text=True)
    assert query(app, "SELECT COUNT(*) AS n FROM devices")[0]["n"] == devices
    assert query(app, "SELECT COUNT(*) AS n FROM sessions WHERE user_agent = ?", SCRIPT)[0]["n"] == 0
    assert [e["detail"] for e in query(app, "SELECT detail FROM events WHERE kind = 'automation_blocked'")] == ["a Python script"]
    # the author signing in from a script (a backup job, say) is their own business
    robot = Browser(app)
    robot.client.environ_base["HTTP_USER_AGENT"] = SCRIPT
    assert robot.login(*AUTHOR).status_code == 302


def test_reader_closes_the_chapter_when_it_finds_itself_in_a_driven_browser(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    assert rae.get(f"{view_url}/page/1").mimetype == "image/jpeg"
    assert rae.json(f"{view_url}/event", {"kind": "automation", "detail": "webdriver", "page": 1}).status_code == 200
    assert rae.get(f"{view_url}/page/2").status_code == 404                     # that reading session is over
    event = query(app, "SELECT detail, subject, actor FROM events WHERE kind = 'automation'")
    assert event == [{"detail": "webdriver", "subject": "chapter one", "actor": "Rae Reader <rae@example.com>"}]
    assert "Automated browser detected, chapter closed" in author.get("/admin/activity").get_data(as_text=True)
    assert rae.get("/library").status_code == 200 and open_chapter(rae)        # a person can open it again
    assert rae.json(f"{view_url}/event", {"kind": "made_up"}).status_code in (400, 404)

    # the author's own preview is never closed, whatever the author's browser is
    preview = author.post("/admin/chapters/1/preview").headers["Location"]
    author.json(f"{preview}/event", {"kind": "automation", "detail": "webdriver"})
    assert author.get(f"{preview}/page/1").mimetype == "image/jpeg"


def test_leaving_the_reader_window_is_counted_for_the_author(app, author):
    upload(author)
    rae = add_reviewer(app, author, "Rae Reader", "rae@example.com", documents_=["1"])
    view_url = open_chapter(rae)
    assert reader_config(rae, view_url)["briefAbsence"] == 15
    rae.get(f"{view_url}/page/1")
    rae.json(f"{view_url}/cover", {"page": 1, "paras": [0], "how": "seen"})

    def counters():
        return query(app, "SELECT away_count, away_brief FROM grants")[0]

    assert rae.json(f"{view_url}/ping", {"page": 1}).status_code == 200 and counters() == {"away_count": 0, "away_brief": 0}
    rae.json(f"{view_url}/ping", {"page": 1, "away": {"n": 3, "brief": 2}})
    assert counters() == {"away_count": 3, "away_brief": 2}
    rae.json(f"{view_url}/ping", {"page": 1, "away": {"n": 1, "brief": 5}})       # brief ones cannot outnumber all of them
    assert counters() == {"away_count": 4, "away_brief": 3}
    for nonsense in ({"n": "9"}, {"n": -4, "brief": -4}, {"n": True}, {"n": 100000}, {"brief": 3}, "often", [3, 2], None):
        assert rae.json(f"{view_url}/ping", {"page": 1, "away": nonsense}).status_code == 200
    assert counters() == {"away_count": 4, "away_brief": 3}

    for kind in ("capture_key", "capture_key", "print_attempt"):
        rae.json(f"{view_url}/event", {"kind": kind, "detail": "PrintScreen", "page": 1})
    facts = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", author.get("/admin/grants/1/coverage").get_data(as_text=True)))
    assert "Left the reader window 4 times, 3 of them for under 15 seconds" in facts
    assert "Screen-capture key 2 " in facts and "Print attempts 1 " in facts and "Copy or save attempts 0 " in facts
    assert "Automated browser 0 " in facts and "Tool or AI agent refused 0 " in facts
    assert "not enough reading yet to say" in facts and "These are counts, not findings" in facts

    # pace: words read on screen over the time the chapter has been open
    with app.app_context():
        db.run("UPDATE documents SET paras_json = '[[450], [450], [450], [450], [450]]'")
        db.run("UPDATE grants SET seconds_read = 600")
    facts = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", author.get("/admin/grants/1/coverage").get_data(as_text=True)))
    assert "about 45 words a minute" in facts and "450 words read on screen" in facts and "up to 600 a minute" in facts

    # the author's own preview is not anyone's reading
    preview = author.post("/admin/chapters/1/preview").headers["Location"]
    assert author.json(f"{preview}/ping", {"page": 1, "away": {"n": 5, "brief": 5}}).status_code == 200
    assert counters() == {"away_count": 4, "away_brief": 3}


def test_terms_say_what_is_recorded_and_untouched_terms_are_brought_up_to_date(tmp_path):
    first = create_app(tmp_path / "data")
    with first.app_context():
        current = db.settings()["nda_text"]
        assert "how often the reading window is left" in current and "AI assistant" in current
        db.set_setting("nda_text", db._OLD_NDA)            # saved with the settings form before this version
    with create_app(tmp_path / "data").app_context():
        assert db.settings()["nda_text"] == db.DEFAULT_NDA
        db.set_setting("nda_text", db._OLD_NDA.replace("\n", "\r\n"))   # as a browser's text box sends it
    with create_app(tmp_path / "data").app_context():
        assert db.settings()["nda_text"] == db.DEFAULT_NDA
        db.set_setting("nda_text", "My own terms.")         # an author's own wording is never replaced
    with create_app(tmp_path / "data").app_context():
        assert db.settings()["nda_text"] == "My own terms."


def test_existing_grants_gain_the_window_counters(tmp_path):
    import sqlite3
    first = create_app(tmp_path / "data")
    conn = sqlite3.connect(first.config["VAULT"].db_path)
    conn.execute("ALTER TABLE grants DROP COLUMN away_count")
    conn.execute("ALTER TABLE grants DROP COLUMN away_brief")
    conn.commit()
    conn.close()
    with create_app(tmp_path / "data").app_context():
        assert {"away_count", "away_brief"} <= {r["name"] for r in db.all("PRAGMA table_info(grants)")}
