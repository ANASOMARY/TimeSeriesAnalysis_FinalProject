"""Chapter ingestion, encrypted storage and watermarked page rendering.

Uploaded files are normalised to PDF, encrypted and written to disk. The PDF is
never sent to a browser: readers only ever receive individual pages rendered to
images with the reader's identity burned into the pixels.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops, ImageDraw, ImageFont

from . import security

MUPDF_FORMATS = {"epub", "xps", "fb2", "cbz", "mobi", "txt"}
OFFICE_FORMATS = {"docx", "doc", "odt", "rtf"}
ACCEPTED = {"pdf"} | MUPDF_FORMATS | OFFICE_FORMATS
MAX_PAGES = 3000
MIN_WIDTH, MAX_WIDTH = 400, 2200
MAX_HEIGHT = 6000


class IngestError(Exception):
    """A problem with an uploaded file that should be shown to the author."""


# ------------------------------------------------------------------ ingest

def parse_page_range(spec: str, page_count: int) -> list[int] | None:
    """Turn ``"3-12, 15, 40-"`` into zero-based page indices. Empty means all pages."""
    spec = (spec or "").strip()
    if not spec:
        return None
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d*)\s*(?:-|–|to)\s*(\d*)|(\d+)", part)
        if not match:
            raise IngestError(f"Could not read the page range “{part}”. Use a form like 12-34 or 5, 8-10.")
        if match.group(3):
            first = last = int(match.group(3))
        else:
            first = int(match.group(1)) if match.group(1) else 1
            last = int(match.group(2)) if match.group(2) else page_count
        if first < 1 or last > page_count or first > last:
            raise IngestError(
                f"Page range “{part}” is outside this file, which has {page_count} pages."
            )
        pages.extend(range(first - 1, last))
    seen: set[int] = set()
    ordered = [p for p in pages if not (p in seen or seen.add(p))]
    if not ordered:
        raise IngestError("The page range does not select any pages.")
    return ordered


def _office_to_pdf(data: bytes, ext: str) -> bytes:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        with tempfile.TemporaryDirectory(prefix="cv-conv-") as tmp:
            src = Path(tmp) / f"chapter.{ext}"
            src.write_bytes(data)
            try:
                subprocess.run(
                    [soffice, "--headless", "--norestore",
                     f"-env:UserInstallation=file://{tmp}/profile",
                     "--convert-to", "pdf", "--outdir", tmp, str(src)],
                    check=True, capture_output=True, timeout=180,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                raise IngestError("The document could not be converted to PDF.") from exc
            out = Path(tmp) / "chapter.pdf"
            if not out.exists():
                raise IngestError("The document could not be converted to PDF.")
            return out.read_bytes()
    if ext == "docx":
        return _docx_to_pdf_builtin(data)
    raise IngestError(
        f".{ext} files need LibreOffice on the server to be converted. "
        "Export the chapter to PDF and upload that instead."
    )


def _docx_to_pdf_builtin(data: bytes) -> bytes:
    """Fallback Word conversion without LibreOffice: reflows the text onto A4 pages."""
    try:
        import mammoth
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise IngestError("Word files cannot be converted on this server. Upload a PDF instead.") from exc
    try:
        html = mammoth.convert_to_html(io.BytesIO(data)).value
        story = pymupdf.Story(
            html=html,
            user_css="body{font-family:serif;font-size:11.5pt;line-height:1.45}"
                     "h1,h2,h3{font-family:sans-serif} img{max-width:100%} p{margin:0 0 8pt 0}",
        )
        buf = io.BytesIO()
        writer = pymupdf.DocumentWriter(buf)
        mediabox = pymupdf.paper_rect("a4")
        where = mediabox + (64, 64, -64, -64)
        more, count = 1, 0
        while more and count < MAX_PAGES:
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device)
            writer.end_page()
            count += 1
        writer.close()
        return buf.getvalue()
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError("The Word file could not be converted. Export it to PDF and upload that.") from exc


def ingest(data: bytes, filename: str, page_range: str = "") -> dict:
    """Validate an upload and return the normalised PDF plus its metadata."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ACCEPTED:
        raise IngestError(
            "Unsupported file type. Upload a PDF, Word (.docx), OpenDocument, RTF, EPUB or plain-text file."
        )
    if not data:
        raise IngestError("The uploaded file is empty.")
    if ext in OFFICE_FORMATS:
        data = _office_to_pdf(data, ext)
    try:
        if ext in MUPDF_FORMATS:
            with pymupdf.open(stream=data, filetype=ext) as src:
                data = src.convert_to_pdf()
        doc = pymupdf.open(stream=data, filetype="pdf")
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError("That file could not be read. Is it a valid, undamaged document?") from exc
    with doc:
        if doc.needs_pass:
            raise IngestError("This PDF is password-protected. Remove the password and upload it again.")
        if doc.page_count == 0:
            raise IngestError("The file contains no pages.")
        if doc.page_count > MAX_PAGES:
            raise IngestError(f"The file has more than {MAX_PAGES} pages.")
        selection = parse_page_range(page_range, doc.page_count)
        if selection is not None:
            doc.select(selection)
        pages = [[round(p.rect.width, 2), round(p.rect.height, 2)] for p in doc]
        pdf = doc.tobytes(garbage=3, deflate=True)
    return {
        "pdf": pdf,
        "page_count": len(pages),
        "pages_json": json.dumps(pages),
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "source_format": ext,
    }


# ----------------------------------------------------------------- storage

def store(cfg, key: bytes, blob_name: str, pdf: bytes) -> None:
    path = cfg.blob_dir / blob_name
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(security.encrypt(key, pdf, blob_name.encode()))
    tmp.replace(path)


def remove(cfg, blob_name: str) -> None:
    evict(blob_name)
    (cfg.blob_dir / blob_name).unlink(missing_ok=True)


# --------------------------------------------------------------- rendering

@dataclass(frozen=True)
class Watermark:
    lines: tuple[str, ...]
    footer: str
    opacity: float  # 0..1


# MuPDF documents are not thread-safe, so every open/render is serialised.
_lock = threading.Lock()
# Keyed by the stored file's random name, which is unique to one upload: a replaced or deleted
# chapter can never be served from a stale entry.
_open_docs: "OrderedDict[str, pymupdf.Document]" = OrderedDict()
_OPEN_LIMIT = 4


def evict(blob_name: str) -> None:
    with _lock:
        doc = _open_docs.pop(blob_name, None)
        if doc is not None:
            doc.close()


def _document(cfg, key: bytes, row) -> pymupdf.Document:
    name = row["blob_name"]
    doc = _open_docs.get(name)
    if doc is None:
        blob = (cfg.blob_dir / row["blob_name"]).read_bytes()
        pdf = security.decrypt(key, blob, row["blob_name"].encode())
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        _open_docs[name] = doc
        while len(_open_docs) > _OPEN_LIMIT:
            _, old = _open_docs.popitem(last=False)
            old.close()
    else:
        _open_docs.move_to_end(name)
    return doc


def clamp_width(width: int) -> int:
    """Snap requested widths to a few sizes so a reader cannot probe arbitrary resolutions."""
    width = max(MIN_WIDTH, min(MAX_WIDTH, width))
    return int(round(width / 200.0)) * 200


def render_page(cfg, key: bytes, row, index: int, width: int, watermark: Watermark | None) -> bytes:
    """Render one page to a JPEG with the watermark burned in."""
    width = clamp_width(width)
    with _lock:
        doc = _document(cfg, key, row)
        page = doc[index]
        # Bound the height too, so a freakishly tall page cannot exhaust memory.
        zoom = min(width / page.rect.width, MAX_HEIGHT / page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, colorspace=pymupdf.csRGB)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if watermark is not None:
        _stamp(image, watermark)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=86, subsampling=0)
    return out.getvalue()


def _paragraphs(lines: list[tuple[float, float, float, float]], page_width: float) -> list[list[float]]:
    """Group text lines, given in reading order, into paragraph rectangles.

    PDFs disagree about what a "block" is: some store a paragraph per block, others a line per
    block. So paragraphs are rebuilt from lines. A line starts a new paragraph when it is set
    apart by extra vertical space, sits in a different column, or is indented relative to the
    body of the paragraph above it (the usual book convention).
    """
    indent = page_width * 0.012
    groups: list[dict] = []
    for x0, y0, x1, y1 in lines:
        height = y1 - y0
        current = groups[-1] if groups else None
        joins = False
        if current is not None:
            last = current["lines"][-1]
            same_row = min(y1, last[3]) - max(y0, last[1]) > 0.5 * min(height, last[3] - last[1])
            gap = y0 - last[3]
            overlap = min(x1, current["x1"]) - max(x0, current["x0"])
            if same_row:
                joins = x0 - last[2] < 3 * height          # the rest of a line split into pieces
            elif -0.4 * height <= gap <= 0.65 * max(height, last[3] - last[1]) and overlap > 0:
                body = current["lines"][1:]
                # With two or more lines we know where the body's left edge is; a line starting
                # further right than that opens a new paragraph.
                joins = not body or x0 <= min(line[0] for line in body) + indent
        if joins:
            current["lines"].append((x0, y0, x1, y1))
            current["x0"], current["x1"] = min(current["x0"], x0), max(current["x1"], x1)
        else:
            groups.append({"lines": [(x0, y0, x1, y1)], "x0": x0, "x1": x1})
    return [[g["x0"], min(line[1] for line in g["lines"]), g["x1"], max(line[3] for line in g["lines"])]
            for g in groups]


def page_blocks(cfg, key: bytes, row, index: int) -> list[list[float]]:
    """Outlines of the paragraphs on a page, as fractions of its width and height.

    Only rectangles are returned, never text, so the reader can let someone point at a
    paragraph without any of the manuscript's text reaching their browser.
    """
    with _lock:
        page = _document(cfg, key, row)[index]
        width, height = page.rect.width, page.rect.height
        lines = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue  # images
            for line in block["lines"]:
                if not any(span["text"].strip() for span in line["spans"]):
                    continue
                # Text coordinates refer to the unrotated page; the image shows the rotated one.
                shown = (pymupdf.Rect(line["bbox"]) * page.rotation_matrix).normalize()
                lines.append((shown.x0, shown.y0, shown.x1, shown.y1))
    boxes = []
    for x0, y0, x1, y1 in _paragraphs(lines, width):
        box = [max(0.0, x0 / width), max(0.0, y0 / height), min(1.0, x1 / width), min(1.0, y1 / height)]
        if box[2] - box[0] > 0.01 and box[3] - box[1] > 0.004:
            boxes.append([round(v, 4) for v in box])
    return boxes


def passage_text(cfg, key: bytes, row, index: int, box: list[float], limit: int = 600) -> str:
    """The text under a marked area of a page, for showing the author what a note refers to.

    Whole lines are quoted: a line counts as marked when the area crosses its middle, so a box
    dragged a little short of the margins still yields readable sentences rather than cut words.
    """
    picked = []
    with _lock:
        page = _document(cfg, key, row)[index]
        width, height = page.rect.width, page.rect.height
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                shown = (pymupdf.Rect(line["bbox"]) * page.rotation_matrix).normalize()
                x0, y0, x1, y1 = shown.x0 / width, shown.y0 / height, shown.x1 / width, shown.y1 / height
                if shown.width >= shown.height:   # ordinary horizontal line
                    marked = box[1] <= (y0 + y1) / 2 <= box[3] and min(x1, box[2]) > max(x0, box[0])
                else:                             # a line running up or down a rotated page
                    marked = box[0] <= (x0 + x1) / 2 <= box[2] and min(y1, box[3]) > max(y0, box[1])
                if marked:
                    picked.append("".join(span["text"] for span in line["spans"]))
    text = " ".join(" ".join(picked).split())
    return text[:limit].rstrip() + ("…" if len(text) > limit else "")


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=24)
def _tiled_mask(size: tuple[int, int], lines: tuple[str, ...]) -> Image.Image:
    """Greyscale mask of the watermark text repeated diagonally across a page."""
    width, height = size
    font_size = max(13, width // 46)
    font = _font(font_size)
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    line_height = int(font_size * 1.4)
    tile_w = int(max(measure.textlength(line, font=font) for line in lines)) + font_size * 5
    tile_h = line_height * len(lines) + font_size * 6
    tile = Image.new("L", (tile_w, tile_h), 0)
    draw = ImageDraw.Draw(tile)
    for i, line in enumerate(lines):
        draw.text((font_size * 2, font_size * 3 + i * line_height), line, fill=255, font=font)
    diagonal = int((width ** 2 + height ** 2) ** 0.5) + tile_w
    sheet = Image.new("L", (diagonal, diagonal), 0)
    for row, y in enumerate(range(0, diagonal, tile_h)):
        offset = (row % 2) * (tile_w // 2)
        for x in range(-offset, diagonal, tile_w):
            sheet.paste(tile, (x, y))
    sheet = sheet.rotate(28, resample=Image.BICUBIC)
    left, top = (diagonal - width) // 2, (diagonal - height) // 2
    return sheet.crop((left, top, left + width, top + height))


def _ink(image: Image.Image, mask: Image.Image, opacity: float) -> None:
    """Lay text down as dark ink with a light halo so it survives on any background."""
    halo = ImageChops.offset(mask, 2, 2).point(lambda v: int(v * opacity * 0.9))
    dark = mask.point(lambda v: int(v * opacity))
    image.paste((255, 255, 255), (0, 0), halo)
    image.paste((40, 52, 70), (0, 0), dark)


def _stamp(image: Image.Image, wm: Watermark) -> None:
    width, height = image.size
    lines = tuple(line for line in wm.lines if line)
    if lines:
        _ink(image, _tiled_mask((width, height), lines), wm.opacity)
    if wm.footer:
        font_size = max(11, width // 78)
        font = _font(font_size)
        strip = Image.new("L", (width, font_size * 2), 0)
        draw = ImageDraw.Draw(strip)
        text = wm.footer
        while len(text) > 20 and draw.textlength(text, font=font) > width - font_size * 2:
            text = text[:-8] + "…"
        draw.text((width // 2, font_size), text, fill=255, font=font, anchor="mm")
        band = Image.new("L", (width, height), 0)
        band.paste(strip, (0, height - int(font_size * 2.6)))
        _ink(image, band, min(1.0, wm.opacity * 3.2))
