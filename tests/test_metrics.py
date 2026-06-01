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


def test_bootstrap_aggregate_metrics_is_deterministic_and_ordered():
    from src.evaluation.metrics import bootstrap_aggregate_metrics
    import pandas as pd

    rows = []
    for condition_idx in range(4):
        for model_name, offset in [("A", 0.0), ("B", 10.0)]:
            rows.append({
                "model": model_name,
                "condition": f"c{condition_idx}",
                "n_real": 10 + condition_idx,
                "acf_l2": offset + condition_idx,
                "wasserstein": offset + 2 * condition_idx,
                "crps": offset + 3 * condition_idx,
                "spectral_frechet": offset + 4 * condition_idx,
            })
    summary_df = pd.DataFrame(rows)

    ci1 = bootstrap_aggregate_metrics(summary_df, n_bootstrap=50, seed=123, confidence=0.9)
    ci2 = bootstrap_aggregate_metrics(summary_df, n_bootstrap=50, seed=123, confidence=0.9)

    assert ci1.equals(ci2)
    assert set(ci1["model"]) == {"A", "B"}
    assert set(ci1["metric"]) == {"acf_l2", "wasserstein", "crps", "spectral_frechet"}
    assert (ci1["n_conditions"] == 4).all()
    assert (ci1["ci_lower"] <= ci1["mean"]).all()
    assert (ci1["mean"] <= ci1["ci_upper"]).all()


def test_bootstrap_aggregate_metrics_supports_weighted_means():
    from src.evaluation.metrics import bootstrap_aggregate_metrics
    import pandas as pd

    summary_df = pd.DataFrame([
        {"model": "A", "condition": "small", "n_real": 1, "acf_l2": 0.0},
        {"model": "A", "condition": "large", "n_real": 9, "acf_l2": 10.0},
    ])

    unweighted = bootstrap_aggregate_metrics(
        summary_df, n_bootstrap=10, seed=0, metric_cols=["acf_l2"]
    )
    weighted = bootstrap_aggregate_metrics(
        summary_df, n_bootstrap=10, seed=0, metric_cols=["acf_l2"], weight_col="n_real"
    )

    assert unweighted.loc[0, "mean"] == 5.0
    assert weighted.loc[0, "mean"] == 9.0
    assert weighted.loc[0, "weight_col"] == "n_real"
