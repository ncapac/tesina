#!/usr/bin/env python3
"""Restore a downloaded tesina artifact bundle into the local repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.runtime_paths import restore_export_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore a DDPM or RF export bundle into the current tesina repo.",
    )
    parser.add_argument(
        "archive",
        type=Path,
        help="Path to the downloaded .tar.gz export bundle.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to restore into. Defaults to the current tesina repo.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing files instead of overwriting them.",
    )
    args = parser.parse_args()

    restored_paths = restore_export_bundle(
        args.archive,
        args.repo_root,
        overwrite=not args.no_overwrite,
    )

    print(f"Restored {len(restored_paths)} file(s) into {args.repo_root.resolve()}")
    for path in restored_paths:
        print(f"  - {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())