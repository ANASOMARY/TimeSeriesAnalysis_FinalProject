"""Recognising a program, rather than a person, behind a reviewer's account.

Only tools that say what they are can be recognised here: AI agents and crawlers that announce
themselves, scripting libraries, and browsers driven by test software. An assistant built into an
ordinary browser, or an agent steering one, looks exactly like the person and is not caught; nor
is a reviewer who pastes screenshots into a chatbot. This is a tripwire, not a guarantee.
"""
from __future__ import annotations

from datetime import timedelta

from flask import abort, g, request

from . import audit, db

# Lower-case fragments of the User-Agent header, with the name shown to the author.
DECLARED = (
    # AI assistants and agents fetching pages on a user's behalf
    ("chatgpt-user", "ChatGPT"), ("chatgpt agent", "ChatGPT agent"),
    ("claude-user", "Claude"), ("claude-web", "Claude"), ("perplexity-user", "Perplexity"),
    ("mistralai-user", "Mistral Le Chat"), ("gemini-deep-research", "Gemini Deep Research"),
    ("google-agent", "Google agent"), ("duckassistbot", "DuckAssist"), ("novaact", "Amazon Nova Act"),
    ("meta-externalfetcher", "Meta AI"), ("browser-use", "browser-use"),
    # AI crawlers
    ("gptbot", "OpenAI crawler"), ("oai-searchbot", "OpenAI crawler"), ("claudebot", "Anthropic crawler"),
    ("claude-searchbot", "Anthropic crawler"), ("anthropic-ai", "Anthropic crawler"),
    ("perplexitybot", "Perplexity crawler"), ("bytespider", "ByteDance crawler"), ("ccbot", "Common Crawl"),
    ("amazonbot", "Amazon crawler"), ("meta-externalagent", "Meta crawler"), ("cohere-ai", "Cohere crawler"),
    ("googleother", "Google crawler"), ("google-cloudvertexbot", "Google crawler"), ("diffbot", "Diffbot"),
    ("youbot", "You.com crawler"), ("ai2bot", "AI2 crawler"), ("timpibot", "Timpi crawler"),
    # Browsers driven by a program
    ("headlesschrome", "a headless browser"), ("phantomjs", "a headless browser"), ("puppeteer", "Puppeteer"),
    ("playwright", "Playwright"), ("selenium", "Selenium"),
    # Scripts and download tools
    ("python-requests", "a Python script"), ("python-urllib", "a Python script"), ("python-httpx", "a Python script"),
    ("aiohttp", "a Python script"), ("scrapy", "a scraper"), ("curl/", "curl"), ("wget/", "wget"),
    ("go-http-client", "a script"), ("node-fetch", "a script"), ("axios/", "a script"), ("undici", "a script"),
    ("okhttp", "a script"), ("libwww-perl", "a script"), ("java/", "a script"), ("httrack", "a site copier"),
)

MESSAGE = ("Chapters can only be opened by a person using an ordinary web browser. Automated tools, "
           "scripts and AI agents are not allowed, and this attempt has been recorded for the author.")


def declared(headers) -> str | None:
    """What the request says it comes from, if that is a program rather than a browser."""
    agent = headers.get("User-Agent", "").lower()
    if not agent.strip():
        return "a tool that does not identify itself"
    for fragment, name in DECLARED:
        if fragment in agent:
            return name
    # Agents that sign their requests (the "Web Bot Auth" convention) name themselves here.
    signer = headers.get("Signature-Agent", "").strip().strip('"')
    if signer:
        return f"a signed agent ({signer[:60]})"
    return None


def note(user, what: str) -> None:
    """Record a blocked attempt, once per quarter of an hour per reviewer and tool."""
    since = db.iso(db.utcnow() - timedelta(minutes=15))
    if not db.scalar("SELECT 1 FROM events WHERE kind = 'automation_blocked' AND user_id = ? AND detail = ? AND ts > ?",
                     user["id"], what, since):
        audit.log("automation_blocked", user=user, detail=what)


def guard() -> None:
    """Refuse a signed-in reviewer's request when it comes from a declared tool. The author is
    not restricted, and neither are the public pages: only chapters are being protected."""
    if g.user is None or g.user["role"] != "reviewer":
        return
    what = declared(request.headers)
    if what:
        note(g.user, what)
        abort(403, description=MESSAGE)
