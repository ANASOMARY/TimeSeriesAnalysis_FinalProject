"""Paragraph coverage: which paragraphs of a chapter a reviewer has got through, and what is left.

A chapter counts as fully covered when every paragraph of it has been on the reviewer's screen
long enough to have been read, has been ticked by hand, or carries one of their notes; and every
page without paragraphs (a figure, a scan) has been looked at. Only then can it be marked reviewed.
"""
from __future__ import annotations

import json

from flask import current_app

from . import audit, db, documents

MIN_WORDS = 5        # shorter blocks (page numbers, running heads, most headings) are not required
MAX_WPM = 600        # text shown for less time than this reading speed needs does not count as read


def words_by_page(doc) -> list[list[int]]:
    """Word count of each paragraph, page by page. Worked out once per stored file and kept."""
    if doc["paras_json"]:
        return json.loads(doc["paras_json"])
    try:
        words = documents.paragraph_words(current_app.config["VAULT"], current_app.config["BLOB_KEY"], doc)
    except (OSError, ValueError, RuntimeError):
        words = [[] for _ in range(doc["page_count"])]
    db.run("UPDATE documents SET paras_json = ? WHERE id = ?", json.dumps(words), doc["id"])
    return words


def required(words: list[int]) -> list[int]:
    """Indexes of the paragraphs on a page that have to be covered."""
    return [index for index, count in enumerate(words) if count >= MIN_WORDS]


def state(user_id: int, doc) -> dict:
    """Where one reviewer stands on one chapter, page by page."""
    words = words_by_page(doc)
    covered: dict[int, dict[int, str]] = {}
    for row in db.all("SELECT page, para, how FROM coverage WHERE user_id = ? AND document_id = ?", user_id, doc["id"]):
        covered.setdefault(row["page"], {})[row["para"]] = row["how"]
    read = {row["page"] for row in db.all(
        "SELECT DISTINCT page FROM events WHERE kind = 'page_read' AND user_id = ? AND document_id = ?",
        user_id, doc["id"])}
    pages, total, done, by_hand = [], 0, 0, 0
    for number in range(1, doc["page_count"] + 1):
        needed = required(words[number - 1]) if number <= len(words) else []
        got = covered.get(number, {})
        if needed:
            have = [index for index in needed if index in got]
            left = [index for index in needed if index not in got]
            by_hand += sum(1 for index in have if got[index] == "confirmed")
        else:   # nothing to tick off on this page: it counts once it has been looked at
            have, left = ([0] if number in read else []), ([] if number in read else [0])
        total += len(have) + len(left)
        done += len(have)
        pages.append({"n": number, "paragraphs": bool(needed), "total": len(have) + len(left), "done": len(have),
                      "left": left, "covered": got,
                      "state": "full" if not left else ("part" if have else "none")})
    return {"pages": pages, "total": total, "done": done, "left": total - done, "by_hand": by_hand}


def record(user, doc, view_id: int | None, page: int, paragraphs: list[int], how: str) -> None:
    """Mark paragraphs of a page as covered, and the page as read once nothing on it is left."""
    words = words_by_page(doc)
    needed = set(required(words[page - 1])) if 1 <= page <= len(words) else set()
    now = db.now_iso()
    for index in paragraphs:
        if index in needed:
            db.run("INSERT OR IGNORE INTO coverage (user_id, document_id, page, para, how, at) VALUES (?,?,?,?,?,?)",
                   user["id"], doc["id"], page, index, how, now)
    if needed:
        have = {row["para"] for row in db.all(
            "SELECT para FROM coverage WHERE user_id = ? AND document_id = ? AND page = ?", user["id"], doc["id"], page)}
        already = db.scalar("SELECT 1 FROM events WHERE kind = 'page_read' AND user_id = ? AND document_id = ? AND page = ?",
                            user["id"], doc["id"], page)
        if needed <= have and not already:
            audit.log("page_read", user=user, document=doc, view_id=view_id, page=page)


def overlapped(boxes: list[list[float]], area: list[float]) -> list[int]:
    """Paragraphs a marked area lands on: those it covers at least half of, or whose middle it contains."""
    hits = []
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        width = min(x1, area[2]) - max(x0, area[0])
        height = min(y1, area[3]) - max(y0, area[1])
        if width <= 0 or height <= 0:
            continue
        middle = area[0] <= (x0 + x1) / 2 <= area[2] and area[1] <= (y0 + y1) / 2 <= area[3]
        if middle or width * height >= 0.5 * (x1 - x0) * (y1 - y0):
            hits.append(index)
    return hits


def reset(doc_id: int) -> None:
    """Forget everything recorded against a chapter's old file: its paragraphs no longer exist."""
    db.run("DELETE FROM coverage WHERE document_id = ?", doc_id)
    db.run("DELETE FROM events WHERE kind = 'page_read' AND document_id = ?", doc_id)
    db.run("UPDATE grants SET reviewed_at = NULL WHERE document_id = ?", doc_id)
