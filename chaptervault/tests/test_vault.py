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
    assert SECRET_TEXT not in rae.get(f"{rae_view}/blocks/2").get_data(as_text=True)   # rectangles only
    assert sam.get(f"{sam_view}/blocks/2").get_json() == {"blocks": []}
    assert rae.get(f"{rae_view}/blocks/9").status_code == 404

    created = rae.json(f"{rae_view}/comments", {"page": 2, "body": "This line is the hook.", "box": outlines[0]})
    assert created.status_code == 201
    note = created.get_json()
    assert note["box"] == outlines[0] and "quote" not in note
    listed = rae.get(f"{rae_view}/comments").get_json()
    assert listed[0]["box"] == outlines[0] and "quote" not in listed[0]          # never sent to reviewers

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
