"""
runtime_paths.py
----------------
Helpers for resolving artifact directories inside the tesina repository.

Default behavior keeps checkpoints and generated results inside the repo
working tree so notebooks share a single, consistent location both locally
and on remote GPU runtimes.
"""

from __future__ import annotations

import shutil
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    repo_root: Path
    data_dir: Path
    checkpoints_dir: Path
    results_root: Path
    run_results_dir: Path


def _is_within_directory(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_dir(repo_root: Path, env_var: str, default: Path) -> Path:
    raw_value = os.environ.get(env_var)
    if not raw_value:
        return default

    resolved = Path(raw_value).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved


def prepare_artifact_dirs(
    repo_root: str | Path,
    experiment: str | None = None,
    *,
    output_subdir: str | None = None,
) -> ArtifactPaths:
    """
    Resolve and create repo-local artifact directories.

    Parameters
    ----------
    output_subdir:
        Optional sub-path relative to *repo_root* that acts as the base for
        checkpoints and results (e.g. ``'output/03'`` or ``'output/03b'``).
        When *None* the legacy layout (``repo_root/checkpoints/`` and
        ``repo_root/results/``) is used.

    Environment overrides are optional:
      - TESINA_CHECKPOINT_DIR
      - TESINA_RESULTS_DIR

    Relative override paths are interpreted relative to repo_root.
    """
    root = Path(repo_root).expanduser().resolve()
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected data directory under {root}")

    base = root / output_subdir if output_subdir else root
    checkpoints_dir = _resolve_dir(root, "TESINA_CHECKPOINT_DIR", base / "checkpoints")
    results_root = _resolve_dir(root, "TESINA_RESULTS_DIR", base / "results")
    run_results_dir = results_root if not experiment else results_root / experiment

    for directory in (checkpoints_dir, results_root, run_results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ArtifactPaths(
        repo_root=root,
        data_dir=data_dir,
        checkpoints_dir=checkpoints_dir,
        results_root=results_root,
        run_results_dir=run_results_dir,
    )


def find_latest_export_bundle(
    repo_root: str | Path,
    prefix: str | None = None,
    *,
    search_dir: str | Path | None = None,
) -> Path | None:
    """
    Return the newest export bundle, if one exists.

    Parameters
    ----------
    search_dir:
        Directory to search for ``*.tar.gz`` bundles.  May be absolute or
        relative to *repo_root*.  Defaults to ``repo_root/results/exports``
        when *None*.
    prefix:
        If provided, only archive names starting with this prefix are
        considered.
    """
    root = Path(repo_root).expanduser().resolve()
    if search_dir is not None:
        exports_dir = Path(search_dir)
        if not exports_dir.is_absolute():
            exports_dir = root / search_dir
    else:
        exports_dir = root / "results" / "exports"
    if not exports_dir.exists():
        return None

    pattern = "*.tar.gz" if prefix is None else f"{prefix}*.tar.gz"
    bundles = sorted(exports_dir.glob(pattern))
    if not bundles:
        return None
    return bundles[-1].resolve()


def restore_export_bundle(
    archive_path: str | Path,
    repo_root: str | Path,
    *,
    overwrite: bool = True,
) -> list[Path]:
    """
    Restore a packaged artifact bundle into the repository working tree.

    Bundle members are expected to be stored relative to ``repo_root``.
    Extraction is done member-by-member so path traversal is rejected.
    """
    root = Path(repo_root).expanduser().resolve()
    archive = Path(archive_path).expanduser().resolve()
    if not archive.exists():
        raise FileNotFoundError(f"Artifact bundle not found: {archive}")

    prepare_artifact_dirs(root)
    restored_paths: list[Path] = []

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute():
                raise ValueError(f"Archive contains absolute path: {member.name}")

            destination = (root / member_path).resolve()
            if not _is_within_directory(root, destination):
                raise ValueError(f"Archive member escapes repo root: {member.name}")

            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            with extracted, open(destination, "wb") as output_file:
                shutil.copyfileobj(extracted, output_file)
            restored_paths.append(destination)

    return restored_paths


def restore_latest_export_bundle(
    repo_root: str | Path,
    prefix: str | None = None,
    *,
    overwrite: bool = True,
    search_dir: str | Path | None = None,
) -> tuple[Path, list[Path]] | None:
    """
    Restore the most recent export bundle.

    Parameters
    ----------
    search_dir:
        Passed through to :func:`find_latest_export_bundle`.  See that
        function for details.

    Returns the archive path and restored file paths, or ``None`` if no bundle
    is available.
    """
    latest_bundle = find_latest_export_bundle(repo_root, prefix=prefix, search_dir=search_dir)
    if latest_bundle is None:
        return None
    restored_paths = restore_export_bundle(latest_bundle, repo_root, overwrite=overwrite)
    return latest_bundle, restored_paths