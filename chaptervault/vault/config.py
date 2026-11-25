"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from pathlib import Path


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    def __init__(self, data_dir: str | os.PathLike | None = None):
        self.data_dir = Path(data_dir or os.environ.get("CHAPTERVAULT_DATA", "data")).resolve()
        self.blob_dir = self.data_dir / "blobs"
        self.db_path = self.data_dir / "vault.db"
        self.key_path = self.data_dir / "master.key"
        # Set when running behind a TLS-terminating reverse proxy (Caddy, nginx, a PaaS router).
        self.behind_proxy = _flag("CHAPTERVAULT_BEHIND_PROXY")
        self.max_upload_mb = int(os.environ.get("CHAPTERVAULT_MAX_UPLOAD_MB", "200"))
        # Public base URL used when building invite links (falls back to the request host).
        self.base_url = os.environ.get("CHAPTERVAULT_BASE_URL", "").rstrip("/")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.data_dir, 0o700)
        except OSError:
            pass
