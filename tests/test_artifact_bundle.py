"""tests for artifact bundle restore helpers"""

from __future__ import annotations

import io
import tarfile


def test_find_latest_export_bundle_returns_newest_matching_archive(tmp_path):
    from src.runtime_paths import find_latest_export_bundle

    repo_root = tmp_path / "tesina"
    exports_dir = repo_root / "results" / "exports"
    exports_dir.mkdir(parents=True)

    older = exports_dir / "ddpm_baseline_20260408T100000Z.tar.gz"
    newer = exports_dir / "ddpm_baseline_20260408T120000Z.tar.gz"
    other = exports_dir / "rf_baseline_20260408T130000Z.tar.gz"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    other.write_bytes(b"rf")

    latest = find_latest_export_bundle(repo_root, prefix="ddpm_baseline_")
    assert latest == newer.resolve()


def test_restore_export_bundle_extracts_repo_relative_files(tmp_path):
    from src.runtime_paths import restore_export_bundle

    repo_root = tmp_path / "tesina"
    (repo_root / "data").mkdir(parents=True)
    archive_path = tmp_path / "bundle.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        payload = b"checkpoint-bytes"
        info = tarfile.TarInfo("checkpoints/best_model.pkl")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

        metrics = b"condition,acf_l2\ncluster0_weekday,0.1\n"
        info = tarfile.TarInfo("results/evaluation/evaluation_metrics.csv")
        info.size = len(metrics)
        tar.addfile(info, io.BytesIO(metrics))

    restored = restore_export_bundle(archive_path, repo_root)

    checkpoint_path = repo_root / "checkpoints" / "best_model.pkl"
    metrics_path = repo_root / "results" / "evaluation" / "evaluation_metrics.csv"
    assert checkpoint_path in restored
    assert metrics_path in restored
    assert checkpoint_path.read_bytes() == b"checkpoint-bytes"
    assert metrics_path.read_text() == "condition,acf_l2\ncluster0_weekday,0.1\n"


def test_restore_export_bundle_rejects_path_traversal(tmp_path):
    from src.runtime_paths import restore_export_bundle

    repo_root = tmp_path / "tesina"
    (repo_root / "data").mkdir(parents=True)
    archive_path = tmp_path / "bundle.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        payload = b"bad"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    try:
        restore_export_bundle(archive_path, repo_root)
    except ValueError as exc:
        assert "escapes repo root" in str(exc)
    else:
        raise AssertionError("Expected ValueError for path traversal archive member")