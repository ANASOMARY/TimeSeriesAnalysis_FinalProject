#!/usr/bin/env python3
"""Start ChapterVault with a production-grade WSGI server (waitress)."""
from __future__ import annotations

import argparse
import os

from waitress import serve

from vault import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ChapterVault server")
    parser.add_argument("--host", default=os.environ.get("CHAPTERVAULT_HOST", "127.0.0.1"),
                        help="address to listen on (use 0.0.0.0 inside a container or behind a proxy)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--data", default=None, help="data directory (default: ./data or $CHAPTERVAULT_DATA)")
    args = parser.parse_args()

    app = create_app(args.data)
    cfg = app.config["VAULT"]
    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    base = cfg.base_url or f"http://{shown_host}:{args.port}"
    print(f"\n  ChapterVault is running at {base}")
    print(f"  Data directory: {cfg.data_dir}")
    if app.config["SETUP_TOKEN"]:
        print("\n  No author account exists yet. Open this one-time link to create it:")
        print(f"    {base}/setup?token={app.config['SETUP_TOKEN']}\n")
    serve(app, host=args.host, port=args.port, threads=8, ident=None,
          max_request_body_size=cfg.max_upload_mb * 1024 * 1024 + 1024 * 1024)


if __name__ == "__main__":
    main()
