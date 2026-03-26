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
    -> cs: np.ndarray  (N_windows, 2)   int32    [cluster_id, day_type]

train_val_split(xs, cs, meter_ids, val_fraction=0.15)
    -> (xs_tr, cs_tr), (xs_va, cs_va)

numpy_dataloader(xs, cs, batch_size, shuffle=True, rng=0)
    -> generator of (x_batch, c_batch) float32 arrays
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Generator

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
    cs  : (N_windows, 2)  int32   — [cluster_id, day_type]
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

            # Day type
            if timestamps is not None:
                step_ts = timestamps[start]
                dow = step_ts.dayofweek
            else:
                # Assume start of data is Monday
                dow = (day % 7)
            dt = _day_type(dow)

            xs_list.append(window)
            cs_list.append([cid, dt])
            mid_list.append(meter_idx)

    xs = np.stack(xs_list, axis=0)                  # (N, 24)
    cs = np.array(cs_list, dtype=np.int32)           # (N, 2)
    mid = np.array(mid_list, dtype=np.int32)         # (N,)
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

    return (xs[tr_mask], cs[tr_mask]), (xs[val_mask], cs[val_mask])


def numpy_dataloader(
    xs: np.ndarray,
    cs: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: int | np.random.Generator = 0,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Infinite generator of (x_batch float32, c_batch int32) pairs.
    Reshuffles every epoch.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    n = len(xs)
    idx = np.arange(n)
    while True:
        if shuffle:
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            if len(b) < batch_size:
                # drop last incomplete batch
                break
            yield xs[b], cs[b]
