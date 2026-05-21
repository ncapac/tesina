#!/usr/bin/env python3
"""
Pick the newest tesina export bundle from your local Downloads folder, move
it into ``results/exports/``, and restore its contents into the repo.

Default usage (after clicking the download button in notebook 03):

    python scripts/ingest_downloaded_bundle.py

Options:
    --downloads-dir   Override the Downloads folder (default: $XDG_DOWNLOAD_DIR
                      or ~/Downloads).
    --prefix          Bundle filename prefix to match (default: ddpm_baseline).
    --no-overwrite    Keep existing restored files instead of overwriting.
    --keep-original   Copy the archive into results/exports/ instead of moving.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.runtime_paths import restore_export_bundle  # noqa: E402


def default_downloads_dir() -> Path:
    env = os.environ.get("XDG_DOWNLOAD_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Downloads"


def find_latest_bundle(downloads_dir: Path, prefix: str) -> Path | None:
    if not downloads_dir.exists():
        return None
    candidates = sorted(
        downloads_dir.glob(f"{prefix}*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", type=Path, default=default_downloads_dir())
    parser.add_argument("--prefix", default="ddpm_baseline")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--keep-original", action="store_true")
    args = parser.parse_args()

    bundle = find_latest_bundle(args.downloads_dir, args.prefix)
    if bundle is None:
        print(
            f"No '{args.prefix}*.tar.gz' archive found in {args.downloads_dir}.",
            file=sys.stderr,
        )
        return 1

    exports_dir = REPO_ROOT / "results" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    destination = exports_dir / bundle.name

    if args.keep_original:
        shutil.copy2(bundle, destination)
        action = "Copied"
    else:
        shutil.move(str(bundle), destination)
        action = "Moved"
    print(f"{action} {bundle}  ->  {destination}")

    restored = restore_export_bundle(
        destination,
        REPO_ROOT,
        overwrite=not args.no_overwrite,
    )
    print(f"\nRestored {len(restored)} file(s) into {REPO_ROOT}")
    for path in restored:
        print(f"  - {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
