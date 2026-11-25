"""Cryptographic helpers: encryption at rest, tokens and keyed hashes."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import Config

NONCE_LEN = 12


def load_master_key(cfg: Config) -> bytes:
    """Return the 256-bit master key.

    Order of precedence: the CHAPTERVAULT_MASTER_KEY environment variable
    (url-safe base64), then ``data/master.key``. A new key file is generated on
    first start. Losing this key makes every stored chapter unreadable.
    """
    env = os.environ.get("CHAPTERVAULT_MASTER_KEY")
    if env:
        key = base64.urlsafe_b64decode(env + "=" * (-len(env) % 4))
        if len(key) != 32:
            raise RuntimeError("CHAPTERVAULT_MASTER_KEY must decode to exactly 32 bytes")
        return key
    if cfg.key_path.exists():
        key = base64.urlsafe_b64decode(cfg.key_path.read_text().strip())
        if len(key) != 32:
            raise RuntimeError(f"{cfg.key_path} is corrupt")
        return key
    key = AESGCM.generate_key(bit_length=256)
    fd = os.open(cfg.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(base64.urlsafe_b64encode(key).decode())
    return key


def derive(master: bytes, label: str) -> bytes:
    """Derive an independent 256-bit sub-key for one purpose."""
    return hmac.new(master, b"chaptervault/" + label.encode(), hashlib.sha256).digest()


def encrypt(key: bytes, data: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(NONCE_LEN)
    return nonce + AESGCM(key).encrypt(nonce, data, aad)


def decrypt(key: bytes, blob: bytes, aad: bytes) -> bytes:
    try:
        return AESGCM(key).decrypt(blob[:NONCE_LEN], blob[NONCE_LEN:], aad)
    except InvalidTag as exc:
        raise ValueError("Stored data failed its integrity check (wrong key or tampered file)") from exc


def new_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:
    """Tokens are stored only as SHA-256 digests, so a database leak does not expose them."""
    return hashlib.sha256(token.encode()).hexdigest()


def mac(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode(), hashlib.sha256).hexdigest()


def safe_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
