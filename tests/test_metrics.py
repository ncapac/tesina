"""tests for src/evaluation/metrics.py

Updated for new API:
  - discriminative_score removed; spectral_frechet_distance added.
  - sample_condition_batch accepts (N, >=2) arrays.
  - compare_models records spectral_frechet in summary_df.
"""

import numpy as np
import pytest


def test_sample_condition_batch_resamples_empirical_rows():
    from src.evaluation.metrics import sample_condition_batch

    # New: 3-column conditioning (cluster, day_type, season)
    condition_rows = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
        ],
        dtype=np.int32,
    )

    batch = sample_condition_batch(condition_rows, n_samples=12, seed=7)
    assert batch.shape == (12, 3)
    allowed = {tuple(row) for row in condition_rows}
    assert {tuple(row) for row in batch}.issubset(allowed)


def test_sample_condition_batch_accepts_4_cols():
    """Legacy 4-col arrays are still accepted (>=2 cols)."""
    from src.evaluation.metrics import sample_condition_batch

    condition_rows = np.zeros((5, 4), dtype=np.int32)
    batch = sample_condition_batch(condition_rows, n_samples=6, seed=0)
    assert batch.shape == (6, 4)


def test_spectral_frechet_distance_same_data():
    """SFD between a distribution and itself should be near 0."""
    from src.evaluation.metrics import spectral_frechet_distance

    rng = np.random.default_rng(42)
    x = rng.standard_normal((200, 24)).astype(np.float32)
    sfd = spectral_frechet_distance(x, x)
    assert sfd >= 0.0
    assert sfd < 1.0   # identical distributions → near 0


def test_spectral_frechet_distance_different_distributions():
    """SFD between different distributions should be clearly > 0."""
    from src.evaluation.metrics import spectral_frechet_distance

    rng = np.random.default_rng(0)
    real = rng.standard_normal((200, 24)).astype(np.float32)
    synth = (rng.standard_normal((200, 24)) * 5 + 3).astype(np.float32)
    sfd = spectral_frechet_distance(real, synth)
    assert sfd > 0.5   # clearly different distributions


def test_compare_models_uses_empirical_full_condition_mix(monkeypatch):
    import src.evaluation.metrics as metrics

    real_data  = np.random.randn(12, 24).astype(np.float32)
    # 3-column conditioning: [cluster_id, day_type, season]
    conditions = np.array(
        [[0, 0, 0]] * 6 + [[0, 0, 1]] * 6,
        dtype=np.int32,
    )
    observed_batches = []

    monkeypatch.setattr(metrics, "acf_compare",                lambda real, synthetic: 0.1)
    monkeypatch.setattr(metrics, "crps_score",                 lambda real, synthetic: 0.2)
    monkeypatch.setattr(metrics, "spectral_frechet_distance",  lambda real, synthetic: 0.3)
    monkeypatch.setattr(metrics, "marginal_wasserstein",       lambda real, synthetic: 0.4)

    def generate(c_batch, seed):
        observed_batches.append(np.array(c_batch, copy=True))
        return np.zeros((len(c_batch), 24), dtype=np.float32)

    summary_df, _ = metrics.compare_models(
        models_dict={"stub": generate},
        real_data=real_data,
        conditions=conditions,
        n_samples=8,
        unique_conditions=[(0, 0)],
        verbose=False,
    )

    assert len(observed_batches) == 1
    sampled_rows = {tuple(row) for row in observed_batches[0]}
    assert sampled_rows == {(0, 0, 0), (0, 0, 1)}
    assert "spectral_frechet" in summary_df.columns
    assert summary_df.loc[0, "n_empirical_meta"] == 2
