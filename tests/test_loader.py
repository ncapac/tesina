"""
tests/test_loader.py
Unit tests for src/data/loader.py
"""
import numpy as np
import pandas as pd
import pickle
import pytest
import tempfile
import os


def _write_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ─── load_raw ─────────────────────────────────────────────────────────────────

class TestLoadRaw:
    def test_dataframe_input(self, tmp_path):
        from src.data.loader import load_raw

        idx = pd.date_range("2020-01-01", periods=48, freq="h")
        df_in = pd.DataFrame(np.random.rand(48, 5), index=idx).astype("float32")
        p = tmp_path / "power.pk"
        _write_pickle(df_in, p)

        df = load_raw(p)
        assert df.shape == (48, 5)
        assert df.dtypes.unique()[0] == np.float32
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_ndarray_input(self, tmp_path):
        from src.data.loader import load_raw

        arr = np.ones((24, 3), dtype=np.float32)
        p = tmp_path / "power.pk"
        _write_pickle(arr, p)

        df = load_raw(p)
        assert df.shape == (24, 3)

    def test_ndarray_transposed(self, tmp_path):
        """If N > T the loader should transpose so rows = time"""
        from src.data.loader import load_raw

        arr = np.ones((10, 200), dtype=np.float32)  # N > T
        p = tmp_path / "power.pk"
        _write_pickle(arr, p)

        df = load_raw(p)
        assert df.shape[0] > df.shape[1]

    def test_dict_input(self, tmp_path):
        from src.data.loader import load_raw

        idx = pd.date_range("2020-01-01", periods=24, freq="h")
        obj = {
            "data": np.random.rand(24, 4).astype(np.float32),
            "timestamps": idx,
            "meter_ids": list(range(4)),
        }
        p = tmp_path / "power.pk"
        _write_pickle(obj, p)

        df = load_raw(p)
        assert df.shape == (24, 4)
        assert isinstance(df.index, pd.DatetimeIndex)


# ─── compute_stats ────────────────────────────────────────────────────────────

class TestComputeStats:
    def _make_df(self, T=100, N=6):
        rng = np.random.default_rng(0)
        return pd.DataFrame(rng.random((T, N)).astype(np.float32))

    def test_returns_all_cluster_ids(self):
        from src.data.loader import compute_stats

        df = self._make_df()
        labels = np.array([0, 0, 1, 1, 2, 2])
        stats = compute_stats(df, labels)
        assert set(stats.keys()) == {0, 1, 2}

    def test_std_positive(self):
        from src.data.loader import compute_stats

        df = self._make_df()
        labels = np.zeros(6, dtype=int)
        stats = compute_stats(df, labels)
        assert stats[0]["std"] > 0

    def test_no_labels_gives_single_cluster(self):
        from src.data.loader import compute_stats

        df = self._make_df()
        stats = compute_stats(df, None)
        assert 0 in stats


# ─── normalize / denormalize ──────────────────────────────────────────────────

class TestNormalize:
    def test_roundtrip(self):
        from src.data.loader import compute_stats, normalize, denormalize

        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.random((50, 4)).astype(np.float32) * 100)
        labels = np.array([0, 0, 1, 1])

        stats = compute_stats(df, labels)
        df_norm = normalize(df, stats, labels)

        # cluster-0 values should be ~z-scored
        vals = df_norm.iloc[:, :2].values
        assert abs(vals.mean()) < 1.0  # mean near 0

        # denormalize should recover original values
        orig = denormalize(vals, 0, stats)
        np.testing.assert_allclose(orig, df.iloc[:, :2].values, rtol=1e-4)
