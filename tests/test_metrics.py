"""tests for src/evaluation/metrics.py"""

import numpy as np


def test_sample_condition_batch_resamples_empirical_rows():
    from src.evaluation.metrics import sample_condition_batch

    condition_rows = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 1, 3],
        ],
        dtype=np.int32,
    )

    batch = sample_condition_batch(condition_rows, n_samples=12, seed=7)
    assert batch.shape == (12, 4)
    allowed = {tuple(row) for row in condition_rows}
    assert {tuple(row) for row in batch}.issubset(allowed)


def test_compare_models_uses_empirical_full_condition_mix(monkeypatch):
    import src.evaluation.metrics as metrics

    real_data = np.random.randn(12, 24).astype(np.float32)
    conditions = np.array(
        [[0, 0, 0, 1]] * 6 + [[0, 0, 1, 2]] * 6,
        dtype=np.int32,
    )
    observed_batches = []

    monkeypatch.setattr(metrics, "acf_compare", lambda real, synthetic: 0.1)
    monkeypatch.setattr(metrics, "crps_score", lambda real, synthetic: 0.2)
    monkeypatch.setattr(metrics, "discriminative_score", lambda real, synthetic: 0.5)
    monkeypatch.setattr(metrics, "marginal_wasserstein", lambda real, synthetic: 0.3)

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
    assert sampled_rows == {(0, 0, 0, 1), (0, 0, 1, 2)}
    assert summary_df.loc[0, "n_empirical_meta"] == 2