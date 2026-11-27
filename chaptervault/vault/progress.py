"""Review progress: how far each reviewer has got, how fast, and whether they will finish in time.

Everything here is derived from two things the vault already records: which pages each reviewer
has read (and when), and the notes they have written. Nothing is stored for the report itself,
so it is always current and can be regenerated for any weekly meeting.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from markupsafe import Markup, escape

from . import db

PACE_WINDOW_DAYS = 14     # recent pace is averaged over this many days ...
AT_RISK_SHARE = 0.6       # ... and "at risk" means at least this share of the pace needed

# status -> (label, tone). The tone picks a colour and an icon shape, so state is never colour alone.
STATUS = {
    "finished": ("Finished", "done"),
    "on_track": ("On track", "good"),
    "at_risk": ("At risk", "warning"),
    "behind": ("Behind", "critical"),
    "overdue": ("Overdue", "critical"),
    "not_started": ("Not started", "idle"),
    "in_progress": ("In progress", "neutral"),
}
STATUS_ORDER = ["behind", "overdue", "at_risk", "not_started", "in_progress", "on_track", "finished"]


def short_date(moment, year: bool = False) -> str:
    """'3 Jan' or '3 Jan 2027', without the platform-specific strftime flags."""
    return f"{moment.day} {moment:%b %Y}" if year else f"{moment.day} {moment:%b}"


@dataclass
class Person:
    id: int
    name: str
    email: str
    total: int = 0                    # pages assigned
    read: int = 0
    read_7d: int = 0
    read_prev_7d: int = 0
    notes: int = 0
    notes_7d: int = 0
    chapters: int = 0
    chapters_finished: int = 0
    chapters_reviewed: int = 0        # marked as reviewed by the reviewer
    assigned_at: datetime | None = None
    last_read_at: datetime | None = None
    pace: float = 0.0                 # pages a day, recent
    needed: float | None = None       # pages a day required from now to finish on the target date
    projected: datetime | None = None
    status: str = "not_started"
    days: list[date] = field(default_factory=list)
    cumulative: list[int] = field(default_factory=list)

    @property
    def percent(self) -> float:
        return 100.0 * self.read / self.total if self.total else 0.0

    @property
    def remaining(self) -> int:
        return self.total - self.read

    @property
    def label(self) -> str:
        return STATUS[self.status][0]

    @property
    def tone(self) -> str:
        return STATUS[self.status][1]


def classify(total: int, read: int, pace: float, needed: float | None, days_left: float | None) -> str:
    """Where a reviewer (or the whole review) stands against the target date."""
    if total and read >= total:
        return "finished"
    if days_left is not None and days_left <= 0:
        return "overdue"
    if read == 0:
        return "not_started"
    if needed is None:
        return "in_progress"            # no target date to measure against
    if pace >= needed:
        return "on_track"
    return "at_risk" if pace >= AT_RISK_SHARE * needed else "behind"


def recent_pace(reads: list[datetime], assigned_at: datetime, now: datetime) -> float:
    """Pages a day over the last two weeks, or since the chapters were assigned if that is more recent.

    Days with no reading inside that window count, so a reviewer who has gone quiet shows a
    falling pace rather than the speed of their last burst.
    """
    window_start = max(now - timedelta(days=PACE_WINDOW_DAYS), assigned_at)
    days = max((now - window_start).total_seconds() / 86400, 1.0)
    return sum(1 for r in reads if r >= window_start) / days


def _project(remaining: int, pace: float, now: datetime) -> datetime | None:
    if remaining <= 0 or pace <= 0:
        return None
    return now + timedelta(days=min(remaining / pace, 3650))


def _daily(reads: list[datetime], start: date, end: date) -> tuple[list[date], list[int]]:
    """Cumulative pages read at the end of each day from ``start`` to ``end``."""
    per_day = Counter(r.date() for r in reads)
    running = sum(count for day, count in per_day.items() if day < start)
    days, cumulative = [], []
    day = start
    while day <= end:
        running += per_day.get(day, 0)
        days.append(day)
        cumulative.append(running)
        day += timedelta(days=1)
    return days, cumulative


def target_date() -> date | None:
    raw = db.settings().get("review_target_date", "")
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date() if raw else None
    except ValueError:
        return None


def snapshot(now: datetime | None = None) -> dict:
    """Everything the progress page, its charts and its export show, as of ``now``."""
    now = now or db.utcnow()
    target = target_date()
    target_end = datetime(target.year, target.month, target.day, 23, 59, 59, tzinfo=timezone.utc) if target else None
    days_left = (target_end - now).total_seconds() / 86400 if target_end else None
    week_ago, fortnight_ago = now - timedelta(days=7), now - timedelta(days=14)

    grants = db.all(
        "SELECT g.user_id, g.document_id, g.created_at, g.expires_at, g.reviewed_at, d.title, d.page_count, u.name, u.email "
        "FROM grants g JOIN documents d ON d.id = g.document_id JOIN users u ON u.id = g.user_id "
        "WHERE g.status = 'active' AND u.role = 'reviewer' AND u.status != 'suspended' "
        "ORDER BY u.name COLLATE NOCASE, d.title COLLATE NOCASE")
    first_read: dict[tuple[int, int], dict[int, datetime]] = {}
    for row in db.all("SELECT user_id, document_id, page, MIN(ts) AS first FROM events "
                      "WHERE kind = 'page_read' GROUP BY user_id, document_id, page"):
        first_read.setdefault((row["user_id"], row["document_id"]), {})[row["page"]] = db.parse_iso(row["first"])
    noted: dict[tuple[int, int], list[datetime]] = {}
    for row in db.all("SELECT user_id, document_id, created_at FROM comments"):
        noted.setdefault((row["user_id"], row["document_id"]), []).append(db.parse_iso(row["created_at"]))

    people: dict[int, Person] = {}
    reads_of: dict[int, list[datetime]] = {}
    chapters: dict[int, dict] = {}
    expiring = []
    for g in grants:
        person = people.setdefault(g["user_id"], Person(g["user_id"], g["name"], g["email"]))
        pages = g["page_count"]
        # Only pages that still exist count: a chapter may have been replaced by a shorter file.
        reads = [ts for page, ts in first_read.get((g["user_id"], g["document_id"]), {}).items() if page <= pages]
        notes = noted.get((g["user_id"], g["document_id"]), [])
        granted = db.parse_iso(g["created_at"])
        person.total += pages
        person.read += len(reads)
        person.notes += len(notes)
        person.notes_7d += sum(1 for n in notes if n >= week_ago)
        person.chapters += 1
        person.chapters_finished += 1 if len(reads) >= pages else 0
        person.chapters_reviewed += 1 if g["reviewed_at"] else 0
        person.assigned_at = min(person.assigned_at, granted) if person.assigned_at else granted
        reads_of.setdefault(g["user_id"], []).extend(reads)

        chapter = chapters.setdefault(g["document_id"], {"id": g["document_id"], "title": g["title"], "pages": pages,
                                                         "reviewers": 0, "finished": 0, "read": 0, "notes": 0})
        chapter["reviewers"] += 1
        chapter["finished"] += 1 if len(reads) >= pages else 0
        chapter["read"] += len(reads)
        chapter["notes"] += len(notes)

        ends = db.parse_iso(g["expires_at"])
        if target_end and ends and ends < target_end and len(reads) < pages:
            expiring.append({"name": g["name"], "user_id": g["user_id"], "title": g["title"], "ends": ends})

    for person in people.values():
        reads = sorted(reads_of[person.id])
        person.read_7d = sum(1 for r in reads if r >= week_ago)
        person.read_prev_7d = sum(1 for r in reads if fortnight_ago <= r < week_ago)
        person.last_read_at = reads[-1] if reads else None
        person.pace = recent_pace(reads, person.assigned_at, now)
        if days_left is not None and days_left > 0:
            person.needed = person.remaining / days_left
        person.projected = _project(person.remaining, person.pace, now)
        person.status = classify(person.total, person.read, person.pace, person.needed, days_left)
        person.days, person.cumulative = _daily(reads, person.assigned_at.date(), now.date())

    ordered = sorted(people.values(), key=lambda p: (STATUS_ORDER.index(p.status), p.percent, p.name.lower()))
    total = sum(p.total for p in ordered)
    read = sum(p.read for p in ordered)
    every_read = sorted(r for reads in reads_of.values() for r in reads)
    started = min((p.assigned_at for p in ordered), default=None)
    overall = Person(0, "Everyone", "", total=total, read=read, assigned_at=started,
                     read_7d=sum(p.read_7d for p in ordered), read_prev_7d=sum(p.read_prev_7d for p in ordered),
                     notes=sum(p.notes for p in ordered), notes_7d=sum(p.notes_7d for p in ordered))
    if started:
        overall.pace = recent_pace(every_read, started, now)
        if days_left is not None and days_left > 0:
            overall.needed = overall.remaining / days_left
        overall.days, overall.cumulative = _daily(every_read, started.date(), now.date())

    # The review is finished when the last reviewer is, so its forecast follows the slowest of
    # them. Pooling everyone's pages would let one person's speed hide another who has stopped.
    unfinished = [p for p in ordered if p.status != "finished"]
    no_forecast = [p for p in unfinished if p.projected is None]
    last = max((p for p in unfinished if p.projected), key=lambda p: p.projected, default=None)
    overall.projected = last.projected if last and not no_forecast else None
    statuses = {p.status for p in ordered}
    if not unfinished:
        overall.status = "finished"
    elif days_left is not None and days_left <= 0:
        overall.status = "overdue"
    elif read == 0:
        overall.status = "not_started"
    elif days_left is None:
        overall.status = "in_progress"
    elif "behind" in statuses:
        overall.status = "behind"
    elif statuses & {"at_risk", "not_started"}:
        overall.status = "at_risk"
    else:
        overall.status = "on_track"

    for chapter in chapters.values():
        capacity = chapter["reviewers"] * chapter["pages"]
        chapter["percent"] = 100.0 * chapter["read"] / capacity if capacity else 0.0
    counts = Counter(p.status for p in ordered)
    return {
        "now": now, "target": target, "target_end": target_end, "days_left": days_left,
        "people": ordered, "overall": overall,
        "chapters": sorted(chapters.values(), key=lambda c: (c["percent"], c["title"].lower())),
        "counts": [(key, STATUS[key][0], STATUS[key][1], counts[key]) for key in STATUS_ORDER if counts[key]],
        "on_course": counts["finished"] + counts["on_track"], "last": last,
        "not_started": [p for p in no_forecast if p.read == 0], "not_reading": [p for p in no_forecast if p.read > 0],
        "expiring": sorted(expiring, key=lambda e: e["ends"]),
    }


# ------------------------------------------------------------------- charts
# Server-drawn SVG, so the report prints exactly as it looks and needs no script to be read.
# Colours and type come from classes in app.css (the page's content-security policy allows no
# inline styles); geometry is plain SVG attributes.

def _points(pairs) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in pairs)


def _series(person: Person, now: datetime) -> list[tuple[datetime, float]]:
    """Per-cent read over time: one point at the end of each day, the last one at this moment."""
    if not person.days:
        return []
    midnight = lambda day: datetime(day.year, day.month, day.day, tzinfo=timezone.utc)  # noqa: E731
    points = [(midnight(person.days[0]), 0.0)]
    for day, count in zip(person.days, person.cumulative):
        points.append((min(midnight(day) + timedelta(days=1), now), 100.0 * count / person.total if person.total else 0.0))
    return points


def combined_projection(snap: dict, until: datetime) -> list[tuple[datetime, float]]:
    """Where the current pace leads: every reviewer carries on at their own recent pace until
    their own pages run out. The line bends each time one of them finishes, and levels off short
    of 100% if anyone is not reading."""
    now, total = snap["now"], snap["overall"].total
    people = [p for p in snap["people"] if p.remaining > 0]
    if not total or not people:
        return []
    horizon = max((until - now).total_seconds() / 86400, 0.0)
    kinks = sorted({min(p.remaining / p.pace, horizon) for p in people if p.pace > 0} | {0.0, horizon})
    done = snap["overall"].read
    return [(now + timedelta(days=t), 100.0 * (done + sum(min(p.remaining, p.pace * t) for p in people)) / total)
            for t in kinks]


def _at(line: list[tuple[datetime, float]], moment: datetime) -> float:
    """Height of a line of (moment, per cent) points at ``moment``, interpolating between them."""
    if moment <= line[0][0]:
        return line[0][1]
    for (m0, p0), (m1, p1) in zip(line, line[1:]):
        if moment <= m1:
            return p0 + (p1 - p0) * (moment - m0).total_seconds() / max((m1 - m0).total_seconds(), 1.0)
    return line[-1][1]


def _tick_dates(start: datetime, end: datetime) -> list[datetime]:
    span = max((end - start).days, 1)
    step = next(s for s in (1, 2, 7, 14, 28, 56, 112, 365) if span / s <= 7)
    ticks, tick = [], datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    while tick <= end:
        if tick >= start:
            ticks.append(tick)
        tick += timedelta(days=step)
    return ticks


def trajectory_chart(snap: dict) -> dict | None:
    """The main chart: actual progress, the even pace needed to hit the target, and the projection.

    Returns the SVG plus the numbers behind it (for the hover read-out and the table view).
    """
    overall, now, target_end = snap["overall"], snap["now"], snap["target_end"]
    actual = _series(overall, now)
    if not actual:
        return None
    width, height, left, right, top, bottom = 880, 320, 46, 26, 34, 34
    start = actual[0][0]
    if target_end and target_end <= start:
        target_end = None      # a target before the review began: nothing to draw it against
    end = max(now, target_end or now)
    if overall.projected and target_end and overall.projected > target_end:
        end = min(overall.projected, target_end + (target_end - start) * 0.3)    # show how far it overshoots
    elif overall.projected and not target_end:
        end = min(overall.projected, now + (now - start) + timedelta(days=14))
    if end - start < timedelta(days=2):
        end = start + timedelta(days=2)
    span = (end - start).total_seconds()
    x = lambda moment: left + (width - left - right) * (moment - start).total_seconds() / span  # noqa: E731
    y = lambda percent: top + (height - top - bottom) * (1 - percent / 100.0)                      # noqa: E731

    out = [f'<svg class="pv-chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="pv-chart-title">',
           f'<title id="pv-chart-title">{escape(chart_summary(snap))}</title>']
    for percent in (0, 25, 50, 75, 100):
        out.append(f'<line class="pv-grid" x1="{left}" x2="{width - right}" y1="{y(percent):.1f}" y2="{y(percent):.1f}"/>')
        out.append(f'<text class="pv-tick" x="{left - 8}" y="{y(percent) + 4:.1f}" text-anchor="end">{percent}%</text>')
    out.append(f'<line class="pv-axis" x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    for tick in _tick_dates(start, end):
        out.append(f'<text class="pv-tick" x="{x(tick):.1f}" y="{height - 12}" text-anchor="middle">{short_date(tick)}</text>')

    # reference markers: today and the target date, labelled at the top so they never sit on the lines
    markers = [("Today", now)] + ([(f"Target {short_date(target_end)}", target_end)] if target_end else [])
    for index, (text, moment) in enumerate(markers):
        px = x(moment)
        close = len(markers) == 2 and abs(x(markers[0][1]) - x(markers[1][1])) < 90
        anchor = "end" if (close and index == 0) or px > width - right - 40 else ("start" if close else "middle")
        dx = -5 if anchor == "end" else (5 if anchor == "start" else 0)
        out.append(f'<line class="pv-marker" x1="{px:.1f}" x2="{px:.1f}" y1="{top - 6}" y2="{y(0):.1f}"/>')
        out.append(f'<text class="pv-marker-label" x="{px + dx:.1f}" y="{top - 12}" text-anchor="{anchor}">{escape(text)}</text>')

    today_percent = actual[-1][1]
    even = [(start, 0.0), (target_end, 100.0)] if target_end else []
    projection = combined_projection(snap, end) if today_percent < 100 else []
    if len(projection) < 2 or projection[-1][0] <= now:
        projection = []
    ahead = not even or today_percent >= _at(even, now)       # is the line above the even pace today?

    def label(css: str, text: str, moment: datetime, percent: float, above: bool) -> None:
        """Put a label beside the point, on the side given: above and to the left, or below and to the right."""
        px, py = x(moment), y(percent)
        est = 7.2 * len(text)                                  # rough text width at 12-13px
        fits_above = py - 10 >= top + 12 and px - 10 - est >= left
        fits_below = py + 18 <= y(0) - 4 and px + 9 + est <= width - right
        if (above and fits_above) or (not above and not fits_below and fits_above):
            out.append(f'<text class="{css}" x="{px - 9:.1f}" y="{py - 9:.1f}" text-anchor="end">{escape(text)}</text>')
        elif fits_below:
            out.append(f'<text class="{css}" x="{px + 9:.1f}" y="{py + 18:.1f}" text-anchor="start">{escape(text)}</text>')
        # otherwise there is no clear room: the legend above the chart still names the line

    if even:
        out.append(f'<polyline class="pv-needed" points="{_points((x(m), y(p)) for m, p in even)}"/>')
    if projection:
        out.append(f'<polyline class="pv-projected" points="{_points((x(m), y(p)) for m, p in projection)}"/>')
    out.append(f'<polyline class="pv-actual" points="{_points((x(m), y(p)) for m, p in actual)}"/>')
    out.append(f'<circle class="pv-dot" cx="{x(now):.1f}" cy="{y(today_percent):.1f}" r="5"/>')

    # Direct labels. Ahead of the even pace, the blue lines run above the grey one, so blue labels
    # go above their line and the grey label below its line; behind, the other way round.
    label("pv-value", f"{today_percent:.0f}% read", now, today_percent, above=ahead)
    future = (min(target_end, end) - now) if target_end else (end - now)
    if even and future > (end - start) * 0.2:
        at = now + future * 0.66
        below_projection = bool(projection) and _at(projection, at) >= _at(even, at)
        label("pv-label", "Even pace", at, _at(even, at), above=not (below_projection if projection else ahead))
    elif even and now - start > (end - start) * 0.3:
        at = start + (min(now, target_end) - start) * 0.5
        label("pv-label", "Even pace", at, _at(even, at), above=_at(actual, at) < _at(even, at))
    if projection and projection[-1][0] - now > (end - start) * 0.2:
        at = now + (projection[-1][0] - now) * 0.36
        label("pv-label", "At the current pace", at, _at(projection, at),
              above=not even or _at(projection, at) >= _at(even, at))

    # hover layer: a crosshair and a dot the page script moves; a transparent plate catches the pointer
    out.append(f'<line class="pv-crosshair" x1="0" x2="0" y1="{top - 6}" y2="{y(0):.1f}" visibility="hidden"/>')
    out.append('<circle class="pv-hover-dot" r="5" visibility="hidden"/>')
    out.append(f'<rect class="pv-plate" x="{left}" y="{top - 6}" width="{width - left - right}" '
               f'height="{y(0) - top + 6:.1f}" tabindex="0" aria-label="Move across the chart to read each day"/>')
    out.append("</svg>")

    rows = []
    for moment, percent in actual[1:]:
        even = None
        if target_end and target_end > start:
            even = max(0.0, min(100.0, 100.0 * (moment - start).total_seconds() / (target_end - start).total_seconds()))
        rows.append({"date": moment - timedelta(seconds=1) if moment != now else now, "actual": percent, "even": even})
    data = {
        "x0": left, "x1": width - right, "y0": y(0), "y100": y(100), "start": start.timestamp(), "end": end.timestamp(),
        "points": [{"t": m.timestamp(), "label": short_date(r["date"], year=True), "actual": round(p, 1),
                    "even": None if r["even"] is None else round(r["even"], 1)}
                   for (m, p), r in zip(actual[1:], rows)],
    }
    return {"svg": Markup("".join(out)), "rows": rows, "data": data,
            "has_target": bool(target_end), "has_projection": bool(projection)}


def _names(people: list[Person]) -> str:
    """'Ann', 'Ann and Bo', 'Ann, Bo and Cy', 'Ann, Bo, Cy and 2 more'."""
    names = [p.name for p in people[:3]]
    if len(people) > 3:
        return ", ".join(names) + f" and {len(people) - 3} more"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def chart_summary(snap: dict) -> str:
    """One or two sentences that say what the chart shows: its caption, and its text for screen readers."""
    overall, target, last = snap["overall"], snap["target"], snap["last"]
    text = f"{overall.percent:.0f}% of assigned pages read"
    if overall.status == "finished":
        return text + ": the review is complete."
    stalled = snap["not_started"] + snap["not_reading"]
    if stalled:
        level = combined_projection(snap, snap["now"] + timedelta(days=3650))[-1][1]
        who = []
        if snap["not_started"]:
            who.append(f"{_names(snap['not_started'])} {'has' if len(snap['not_started']) == 1 else 'have'} not started")
        if snap["not_reading"]:
            who.append(f"{_names(snap['not_reading'])} {'has' if len(snap['not_reading']) == 1 else 'have'} "
                       f"not read in the last {PACE_WINDOW_DAYS} days")
        return (text + f". At the current pace the review levels off at {level:.0f}% and never finishes: "
                + "; ".join(who) + ".")
    if overall.projected and target:
        late = (overall.projected - snap["target_end"]).total_seconds() / 86400
        verdict = (f"{late:.0f} day{'' if round(late) == 1 else 's'} after the target of {short_date(target, True)}"
                   if late > 0 else f"in time for the target of {short_date(target, True)}")
        return text + f". At the current pace the last reviewer, {last.name}, finishes on {short_date(overall.projected, True)}: {verdict}."
    if overall.projected:
        return text + f". At the current pace the last reviewer, {last.name}, finishes on {short_date(overall.projected, True)}."
    return text + "."


def sparkline(person: Person, snap: dict, width: int = 168, height: int = 44, css: str = "pv-spark") -> Markup:
    """A small trajectory for one reviewer: their line against the even pace to the target."""
    now, target_end = snap["now"], snap["target_end"]
    actual = _series(person, now)
    pad = 5
    if not actual:
        return Markup("")
    start = actual[0][0]
    if target_end and target_end <= start:
        target_end = None
    end = max(now, target_end or now)
    if end - start < timedelta(days=2):
        end = start + timedelta(days=2)
    span = (end - start).total_seconds()
    x = lambda moment: pad + (width - 2 * pad) * (moment - start).total_seconds() / span   # noqa: E731
    y = lambda percent: pad + (height - 2 * pad) * (1 - percent / 100.0)                    # noqa: E731
    even_now = None
    if target_end and target_end > start:
        even_now = max(0.0, min(100.0, 100.0 * (now - start).total_seconds() / (target_end - start).total_seconds()))
    summary = f"{person.name}: {person.percent:.0f}% read"
    if even_now is not None:
        summary += f"; an even pace would be at {even_now:.0f}% today"
    out = [f'<svg class="{css}" viewBox="0 0 {width} {height}" role="img"><title>{escape(summary)}</title>',
           f'<line class="pv-axis" x1="{pad}" x2="{width - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>']
    if target_end:
        out.append(f'<line class="pv-needed" x1="{x(start):.1f}" y1="{y(0):.1f}" x2="{x(target_end):.1f}" y2="{y(100):.1f}"/>')
    out.append(f'<polyline class="pv-actual" points="{_points((x(m), y(p)) for m, p in actual)}"/>')
    out.append(f'<circle class="pv-dot" cx="{x(now):.1f}" cy="{y(actual[-1][1]):.1f}" r="4"/></svg>')
    return Markup("".join(out))


def _needed_phrase(needed: float) -> str:
    """'about 6 pages a day', or for a gentle pace 'about 5 pages a week'."""
    if needed >= 1.5:
        return f"about {math.ceil(needed)} pages a day"
    per_week = max(1, math.ceil(needed * 7))
    return f"about {per_week} page{'' if per_week == 1 else 's'} a week"


def outlook(person: Person, snap: dict) -> tuple[str, str]:
    """Where a reviewer stands, said to them directly: a short heading and one or two plain sentences.

    The same figures the author sees, worded as guidance: what their current pace leads to, and
    what pace would finish in time.
    """
    target, target_end, left = snap["target"], snap["target_end"], person.remaining
    pages = f"{left} page{'' if left == 1 else 's'}"
    if person.status == "finished":
        return "You have finished", "You have read every page of every chapter shared with you. Thank you."
    if person.status == "overdue":
        return "The target date has passed", f"The target was {short_date(target, True)} and {pages} are still to read."
    if target is None:
        detail = f"{pages} left to read."
        if person.projected:
            detail += f" At your current pace you would finish on {short_date(person.projected, True)}."
        return ("Not started yet" if person.read == 0 else "In progress"), detail
    by = f"by {short_date(target, True)}"
    if person.status == "not_started":
        return "Not started yet", f"{pages} to read. {_needed_phrase(person.needed).capitalize()} from today would finish {by}."
    if person.projected is None:
        return "A faster pace is needed", (f"You have not read in the last {PACE_WINDOW_DAYS} days and have {pages} left. "
                                           f"{_needed_phrase(person.needed).capitalize()} from today would finish {by}.")
    days = (person.projected - target_end).total_seconds() / 86400
    if person.status == "on_track":
        spare = math.floor(-days)
        margin = f", {spare} day{'' if spare == 1 else 's'} before the target" if spare >= 1 else ", just in time for the target"
        return "On course", f"At your current pace you finish on {short_date(person.projected, True)}{margin} of {short_date(target, True)}."
    late = max(1, math.ceil(days))
    return "A faster pace is needed", (
        f"At your current pace you would finish on {short_date(person.projected, True)}, {late} day{'' if late == 1 else 's'} "
        f"after the target. {_needed_phrase(person.needed).capitalize()} would finish {by}.")


def status_icon(tone: str) -> Markup:
    """A small shape per state, so on track / at risk / behind differ by more than colour."""
    shapes = {
        "good": '<circle cx="6" cy="6" r="5"/>',
        "warning": '<path d="M6 1 11.5 11H.5z"/>',
        "critical": '<rect x="1" y="1" width="10" height="10" rx="1.5"/>',
        "done": '<path d="M1.5 6.4 4.6 9.5 10.5 2.8" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
        "idle": '<circle cx="6" cy="6" r="4.2" fill="none" stroke-width="1.8"/>',
        "neutral": '<circle cx="6" cy="6" r="5"/>',
    }
    return Markup(f'<svg class="pv-icon pv-{tone}" viewBox="0 0 12 12" aria-hidden="true">{shapes[tone]}</svg>')
