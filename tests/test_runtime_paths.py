"""tests for src/runtime_paths.py"""


def test_prepare_artifact_dirs_creates_repo_local_structure(tmp_path):
    from src.runtime_paths import prepare_artifact_dirs

    repo_root = tmp_path / "tesina"
    (repo_root / "data").mkdir(parents=True)

    paths = prepare_artifact_dirs(repo_root, experiment="diffusion")

    assert paths.repo_root == repo_root.resolve()
    assert paths.data_dir == (repo_root / "data").resolve()
    assert paths.checkpoints_dir.exists()
    assert paths.results_root.exists()
    assert paths.run_results_dir == (repo_root / "results" / "diffusion").resolve()
    assert paths.run_results_dir.exists()