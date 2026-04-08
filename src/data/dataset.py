"""
dataset.py
----------
Slice the normalised time series into daily windows and build batches
suitable for training the diffusion model.

Window layout
-------------
  - Each window is 24 timesteps  (24h × 1 step/h, hourly resolution).
  - Each sample carries a conditioning vector c = [cluster_id, day_type]
      cluster_id : int in {0, …, K-1}
      day_type   : 0 = weekday, 1 = weekend

Public API
----------
make_windows(df, cluster_labels, timestamps=None)
    -> xs: np.ndarray  (N_windows, 24)  float32
    -> cs: np.ndarray  (N_windows, 4)   int32    [cluster_id, day_type, month, dow]

train_val_split(xs, cs, meter_ids, val_fraction=0.15)
    -> (xs_tr, cs_tr), (xs_va, cs_va)

numpy_dataloader(xs, cs, batch_size, shuffle=True, rng=0)
    -> generator of (x_batch, c_batch) float32 arrays
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

STEPS_PER_DAY = 24  # hourly resolution


def _day_type(day_of_week: int) -> int:
    """0=weekday, 1=weekend"""
    return int(day_of_week >= 5)


def make_windows(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice a (T, N_meters) DataFrame into non-overlapping daily windows.

    Parameters
    ----------
    df              : normalised (T, N_meters) DataFrame
    cluster_labels  : (N_meters,) int array
    timestamps      : DatetimeIndex of length T; if None, infers weekday
                      from positional index (day = t // STEPS_PER_DAY)

    Returns
    -------
    xs  : (N_windows, 24) float32 — normalised consumption windows
    cs  : (N_windows, 4)  int32   — [cluster_id, day_type, month, dow]
    mid : (N_windows,)    int32   — meter column index (for splitting)
    """
    T, N = df.shape
    n_complete_days = T // STEPS_PER_DAY

    xs_list, cs_list, mid_list = [], [], []

    for meter_idx in range(N):
        cid = int(cluster_labels[meter_idx])
        series = df.iloc[: n_complete_days * STEPS_PER_DAY, meter_idx].values

        for day in range(n_complete_days):
            start = day * STEPS_PER_DAY
            window = series[start : start + STEPS_PER_DAY].astype(np.float32)

            # Skip windows with too many NaNs (>10%)
            if np.isnan(window).mean() > 0.10:
                continue

            # Linearly interpolate remaining NaNs
            if np.isnan(window).any():
                nans = np.isnan(window)
                xs_coord = np.where(~nans)[0]
                window[nans] = np.interp(np.where(nans)[0], xs_coord, window[~nans])

            # Day type, month, day-of-week
            if timestamps is not None:
                step_ts = timestamps[start]
                dow   = step_ts.dayofweek   # 0=Mon … 6=Sun
                month = step_ts.month - 1   # 0-indexed: 0=Jan … 11=Dec
            else:
                # Assume start of data is Monday
                dow   = day % 7
                month = (day // 30) % 12    # rough month from day index
            dt = _day_type(dow)

            xs_list.append(window)
            cs_list.append([cid, dt, month, dow])
            mid_list.append(meter_idx)

    xs = np.stack(xs_list, axis=0)                  # (N, 24)
    cs = np.array(cs_list, dtype=np.int32)          # (N, 4)
    mid = np.array(mid_list, dtype=np.int32)        # (N,)
    return xs, cs, mid


def train_val_split(
    xs: np.ndarray,
    cs: np.ndarray,
    mid: np.ndarray,
    n_meters: int,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified split by meter — val meters are held out entirely.
    This avoids leakage between train and val.
    """
    rng = np.random.default_rng(seed)
    all_meters = np.arange(n_meters)
    rng.shuffle(all_meters)
    n_val = max(1, int(np.round(n_meters * val_fraction)))
    val_meters = set(all_meters[:n_val].tolist())

    val_mask = np.array([m in val_meters for m in mid], dtype=bool)
    tr_mask = ~val_mask

    return xs[tr_mask], cs[tr_mask], xs[val_mask], cs[val_mask]


class _InfiniteLoader:
    """
    Infinite iterator of (x_batch, c_batch) pairs. Reshuffles every pass.

    Attributes
    ----------
    epoch_len : int
        Number of complete batches per pass, used by ``Trainer.fit`` to bound
        one training epoch to a single pass through the data.
    """

    def __init__(
        self,
        xs: np.ndarray,
        cs: np.ndarray,
        batch_size: int,
        shuffle: bool,
        rng: np.random.Generator,
    ):
        n = len(xs)
        self.epoch_len: int = max(1, n // batch_size)
        self._gen = self._make_gen(xs, cs, batch_size, shuffle, rng)

    @staticmethod
    def _make_gen(xs, cs, batch_size, shuffle, rng):
        n = len(xs)
        idx = np.arange(n)
        while True:
            if shuffle:
                rng.shuffle(idx)
            for start in range(0, n, batch_size):
                b = idx[start : start + batch_size]
                if len(b) < batch_size:
                    break
                yield xs[b], cs[b]

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)


class _BalancedInfiniteLoader:
    """
    Infinite iterator that approximately balances batches across selected
    conditioning groups by resampling each group uniformly.
    """

    def __init__(
        self,
        xs: np.ndarray,
        cs: np.ndarray,
        batch_size: int,
        balance_condition_cols: tuple[int, ...],
        shuffle: bool,
        rng: np.random.Generator,
    ):
        if len(balance_condition_cols) == 0:
            raise ValueError("balance_condition_cols must not be empty")

        grouped_indices: dict[tuple[int, ...], list[int]] = {}
        for idx, row in enumerate(cs):
            key = tuple(int(row[col]) for col in balance_condition_cols)
            grouped_indices.setdefault(key, []).append(idx)

        self.group_keys = sorted(grouped_indices)
        self.grouped_indices = {
            key: np.asarray(indices, dtype=np.int32)
            for key, indices in grouped_indices.items()
        }
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        self.xs = xs
        self.cs = cs
        self.epoch_len = max(1, len(xs) // batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        n_groups = len(self.group_keys)
        base = self.batch_size // n_groups
        remainder = self.batch_size % n_groups

        batch_indices = []
        for group_idx, key in enumerate(self.group_keys):
            n_take = base + int(group_idx < remainder)
            if n_take == 0:
                continue

            pool = self.grouped_indices[key]
            sampled = self.rng.choice(pool, size=n_take, replace=len(pool) < n_take)
            batch_indices.append(sampled)

        batch_indices = np.concatenate(batch_indices, axis=0)
        if self.shuffle:
            self.rng.shuffle(batch_indices)

        return self.xs[batch_indices], self.cs[batch_indices]


def numpy_dataloader(
    xs: np.ndarray,
    cs: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    balance_condition_cols: tuple[int, ...] | None = None,
    rng: int | np.random.Generator = 0,
) -> "_InfiniteLoader | _BalancedInfiniteLoader":
    """
    Infinite iterator of (x_batch float32, c_batch int32) pairs.
    Reshuffles every pass through the data.

    If ``balance_condition_cols`` is provided, batches are built by uniform
    resampling over the unique groups defined by those conditioning columns.
    In this project ``(0, 1)`` corresponds to ``(cluster_id, day_type)``.

    The returned object exposes ``epoch_len = len(xs) // batch_size`` so that
    ``Trainer.fit`` can automatically bound each epoch to one data pass.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    if balance_condition_cols is not None:
        return _BalancedInfiniteLoader(
            xs,
            cs,
            batch_size,
            tuple(balance_condition_cols),
            shuffle,
            rng,
        )
    return _InfiniteLoader(xs, cs, batch_size, shuffle, rng)
