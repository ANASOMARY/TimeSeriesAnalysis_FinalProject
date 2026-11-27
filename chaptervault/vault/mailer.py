"""Sending email through a mail account the author configures in Settings.

Any standard SMTP service works: the author's own mailbox, Amazon SES, or a transactional
provider. Connections are always encrypted; there is no plain-text option.
"""
from __future__ import annotations

import base64
import re
import smtplib
import socket
import ssl
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid

from . import security

SECURITY_MODES = {"starttls": "STARTTLS (usually port 587)", "ssl": "SSL/TLS (usually port 465)"}
ADDRESS = re.compile(r"^[^@\s<>\"',;:\\]+@[^@\s<>\"',;:\\]+\.[^@\s<>\"',;:\\]+$")
TIMEOUT = 20

DEFAULT_INVITE_SUBJECT = "Chapters for your review"
DEFAULT_INVITE_MESSAGE = (
    "Hello {name},\n\n"
    "I would like to share some chapters of {book} with you for review.\n\n"
    "Use this private link to choose a password and start reading:\n{link}\n\n"
    "The link is personal to you, works once, and expires on {expires}. Please do not forward it.\n\n"
    "Thank you,\n{author}"
)
_PLACEHOLDER = re.compile(r"\{(name|link|book|author|expires)\}")


class MailError(Exception):
    """A delivery problem, worded so it can be shown to the author as it is."""


def valid_address(value: str) -> bool:
    return bool(ADDRESS.match(value or "")) and len(value) <= 200


def seal_password(key: bytes, password: str) -> str:
    """Encrypt the mail password for storage in the settings table."""
    return base64.urlsafe_b64encode(security.encrypt(key, password.encode(), b"smtp")).decode()


def _open_password(key: bytes, sealed: str) -> str:
    if not sealed:
        return ""
    try:
        return security.decrypt(key, base64.urlsafe_b64decode(sealed), b"smtp").decode()
    except (ValueError, TypeError):
        return ""


def configuration(settings: dict, key: bytes) -> dict | None:
    """The mail account from Settings, or None when email has not been set up."""
    host, sender = settings["smtp_host"].strip(), settings["mail_from_address"].strip()
    if not host or not valid_address(sender):
        return None
    port = settings["smtp_port"].strip()
    return {
        "host": host,
        "port": int(port) if port.isdigit() else 587,
        "security": settings["smtp_security"] if settings["smtp_security"] in SECURITY_MODES else "starttls",
        "username": settings["smtp_username"].strip(),
        "password": _open_password(key, settings["smtp_password_enc"]),
        "from_name": settings["mail_from_name"].strip() or settings["author_name"].strip(),
        "from_address": sender,
        # Where replies should go when that is not the sending address, for example when mail is
        # sent from a dedicated sending domain that has no mailbox behind it.
        "reply_to": settings["mail_reply_to"].strip() if valid_address(settings["mail_reply_to"].strip()) else "",
    }


def _one_line(text: str) -> str:
    """Collapse whitespace so nothing typed into a form can add its own mail headers."""
    return " ".join(str(text).split())


def _reply(error: smtplib.SMTPResponseException) -> str:
    text = error.smtp_error.decode("utf-8", "replace") if isinstance(error.smtp_error, bytes) else str(error.smtp_error)
    return _one_line(text)[:240]


def send(config: dict, to_name: str, to_address: str, subject: str, body: str) -> None:
    """Send one plain-text message. Raises MailError with a reason the author can act on."""
    if not valid_address(to_address):
        raise MailError(f"“{to_address}” is not a valid email address.")
    message = EmailMessage()
    message["Subject"] = _one_line(subject)[:200] or "(no subject)"
    message["From"] = formataddr((_one_line(config["from_name"]), config["from_address"]))
    message["To"] = formataddr((_one_line(to_name), to_address))
    if config.get("reply_to"):
        message["Reply-To"] = config["reply_to"]
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = make_msgid(domain=config["from_address"].rsplit("@", 1)[1])
    message.set_content(body.replace("\r\n", "\n").strip() + "\n")

    where = f"{config['host']} on port {config['port']}"
    context = ssl.create_default_context()
    try:
        if config["security"] == "ssl":
            server = smtplib.SMTP_SSL(config["host"], config["port"], timeout=TIMEOUT, context=context)
        else:
            server = smtplib.SMTP(config["host"], config["port"], timeout=TIMEOUT)
        with server:
            if config["security"] != "ssl":
                server.starttls(context=context)
            if config["username"]:
                server.login(config["username"], config["password"])
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        raise MailError(f"The mail server rejected the username or password ({_reply(exc)}).") from exc
    except smtplib.SMTPNotSupportedError as exc:
        raise MailError(f"The mail server at {where} did not offer what was asked of it: {_one_line(exc)}. "
                        "Check the port and the connection type.") from exc
    except smtplib.SMTPRecipientsRefused as exc:
        reason = next(iter(exc.recipients.values()), (0, b""))[1]
        reason = reason.decode("utf-8", "replace") if isinstance(reason, bytes) else str(reason)
        raise MailError(f"The mail server refused to deliver to {to_address}: {_one_line(reason)[:240]}") from exc
    except smtplib.SMTPSenderRefused as exc:
        raise MailError(f"The mail server refused the sender address {config['from_address']}: {_reply(exc)}") from exc
    except smtplib.SMTPResponseException as exc:
        raise MailError(f"The mail server refused the message: {_reply(exc)}") from exc
    except ssl.SSLError as exc:
        raise MailError(f"A secure connection to {where} could not be made. "
                        "Check that the connection type matches the port.") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise MailError(f"The mail server at {where} did not answer in time.") from exc
    except smtplib.SMTPException as exc:
        raise MailError(f"The mail server at {where} could not be used: {_one_line(exc)[:240]}") from exc
    except OSError as exc:
        raise MailError(f"Could not connect to the mail server at {where}. Check the server name and port.") from exc


def invitation(settings: dict, person, link: str, expires: str, fallback_author: str) -> tuple[str, str]:
    """Subject and text of an invitation, from the wording the author set (or the default)."""
    values = {
        "name": _one_line(person["name"]),
        "link": link,
        "book": settings["book_title"].strip() or "my manuscript",
        "author": settings["author_name"].strip() or fallback_author,
        "expires": expires,
    }
    fill = lambda template: _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)  # noqa: E731
    template = settings["invite_message"].strip() or DEFAULT_INVITE_MESSAGE
    body = fill(template)
    if "{link}" not in template:      # an invitation without its link would be useless
        body += f"\n\n{link}"
    return fill(settings["invite_subject"].strip() or DEFAULT_INVITE_SUBJECT), body
