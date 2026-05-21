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

    def test_returns_one_entry_per_meter(self):
        from src.data.loader import compute_stats

        df = self._make_df(T=100, N=6)
        stats = compute_stats(df)
        assert set(stats.keys()) == set(range(6))

    def test_scale_positive(self):
        from src.data.loader import compute_stats

        df = self._make_df()
        stats = compute_stats(df)
        for i in stats:
            assert stats[i]["scale"] > 0

    def test_cluster_labels_ignored_for_backcompat(self):
        """Old call signature with cluster_labels must still work."""
        from src.data.loader import compute_stats

        df = self._make_df()
        labels = np.array([0, 0, 1, 1, 2, 2])
        stats_a = compute_stats(df, labels)
        stats_b = compute_stats(df)
        # Same per-meter scales regardless of (now unused) labels
        for i in stats_a:
            assert stats_a[i]["scale"] == stats_b[i]["scale"]

    def test_scale_equals_column_mean(self):
        from src.data.loader import compute_stats

        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.random((200, 4)).astype(np.float32) * 50 + 1.0)
        stats = compute_stats(df)
        expected = df.values.mean(axis=0)
        for i in range(4):
            assert abs(stats[i]["scale"] - expected[i]) < 1e-4

    def test_zero_meter_uses_floor(self):
        from src.data.loader import compute_stats

        df = pd.DataFrame(np.zeros((50, 3), dtype=np.float32))
        stats = compute_stats(df)
        for i in range(3):
            assert stats[i]["scale"] >= 1e-8


# ─── normalize / denormalize ──────────────────────────────────────────────────

class TestNormalize:
    def test_roundtrip(self):
        from src.data.loader import compute_stats, normalize, denormalize

        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.random((50, 4)).astype(np.float32) * 100 + 1.0)

        stats = compute_stats(df)
        df_norm = normalize(df, stats)

        # Each column has mean ≈ 1 after dividing by its own mean
        col_means = df_norm.values.mean(axis=0)
        np.testing.assert_allclose(col_means, np.ones(4), rtol=1e-4)

        # denormalize per meter should recover original values
        for i in range(4):
            recovered = denormalize(df_norm.iloc[:, i].values, i, stats)
            np.testing.assert_allclose(
                recovered, df.iloc[:, i].values, rtol=1e-4
            )

    def test_normalize_preserves_shape(self):
        from src.data.loader import compute_stats, normalize

        df = pd.DataFrame(np.random.rand(20, 5).astype(np.float32))
        stats = compute_stats(df)
        df_norm = normalize(df, stats)
        assert df_norm.shape == df.shape
        assert df_norm.values.dtype == np.float32

    def test_denormalize_batch_vector(self):
        from src.data.loader import denormalize_batch

        arr = np.ones((4, 24), dtype=np.float32)
        scales = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = denormalize_batch(arr, scales)
        assert out.shape == arr.shape
        # Each row scaled by corresponding scale
        for i, s in enumerate(scales):
            np.testing.assert_allclose(out[i], np.full(24, s))

    def test_denormalize_batch_scalar(self):
        from src.data.loader import denormalize_batch

        arr = np.ones((4, 24), dtype=np.float32)
        out = denormalize_batch(arr, 2.5)
        np.testing.assert_allclose(out, np.full((4, 24), 2.5))

    def test_scales_array_lookup(self):
        from src.data.loader import compute_stats, scales_array

        df = pd.DataFrame(
            np.tile(np.arange(1, 5, dtype=np.float32), (10, 1))
        )  # column i has mean i+1
        stats = compute_stats(df)
        mids = np.array([0, 1, 2, 3, 0, 3], dtype=np.int32)
        out = scales_array(stats, mids)
        expected = np.array([1, 2, 3, 4, 1, 4], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5)


# ─── filter_outlier_meters ────────────────────────────────────────────────────

class TestFilterOutlierMeters:
    def test_drops_meters_above_threshold(self):
        from src.data.loader import filter_outlier_meters

        # 5 meters with means [1, 1, 1, 1, 100]; median = 1; factor=10 keeps
        # only those <= 10.  Last column should be dropped.
        cols = [
            np.ones(24, dtype=np.float32) * v for v in (1.0, 1.0, 1.0, 1.0, 100.0)
        ]
        df = pd.DataFrame(np.stack(cols, axis=1))
        cluster_labels = np.array([0, 0, 1, 1, 2])

        df_kept, cl_kept, mask = filter_outlier_meters(df, cluster_labels, factor=10.0)
        assert df_kept.shape == (24, 4)
        assert mask.tolist() == [True, True, True, True, False]
        np.testing.assert_array_equal(cl_kept, np.array([0, 0, 1, 1]))
        # surviving columns are reindexed 0..3
        assert list(df_kept.columns) == [0, 1, 2, 3]

    def test_no_outliers_when_factor_large(self):
        from src.data.loader import filter_outlier_meters

        df = pd.DataFrame(np.ones((10, 3), dtype=np.float32))
        cl = np.array([0, 1, 2])
        df_kept, cl_kept, mask = filter_outlier_meters(df, cl, factor=100.0)
        assert df_kept.shape == df.shape
        assert mask.all()
        np.testing.assert_array_equal(cl_kept, cl)

    def test_mismatched_cluster_labels_raises(self):
        from src.data.loader import filter_outlier_meters

        df = pd.DataFrame(np.ones((5, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="cluster_labels"):
            filter_outlier_meters(df, np.array([0, 1]))
