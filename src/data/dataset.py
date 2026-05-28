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
    -> mid: np.ndarray (N_windows,)     int32    meter column index

make_daily_instances(df, temp_daily_normed, cluster_labels=None)
    -> xs           : (N, 24)  float32
    -> c_discrete   : (N, 3)   int32    [cluster_id, day_type, season]
    -> c_continuous : (N, 1)   float32  [daily_mean_temp_normed]
    -> dates        : (N,)     object   calendar date for each instance
    -> meter_ids    : (N,)     int32    column index in df

train_val_split(xs, cs, mid, n_meters, val_fraction=0.15)
    -> (xs_tr, cs_tr, mid_tr, xs_va, cs_va, mid_va)

numpy_dataloader(xs, c_discrete, batch_size, c_continuous=None, shuffle=True, rng=0)
    -> generator of (x_batch, c_disc_batch) or (x_batch, c_disc_batch, c_cont_batch)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

STEPS_PER_DAY = 24  # hourly resolution
SHAPE_NORM_EPS = 1e-3  # floor used when log-transforming per-instance mean/std


def shape_normalize(
    xs: np.ndarray,
    eps: float = SHAPE_NORM_EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-instance shape normalisation: z-score each daily profile to itself.

    For each row ``x_i``:
        mu_i    = x_i.mean()
        sigma_i = max(x_i.std(), eps)
        x_i_norm = (x_i - mu_i) / sigma_i              (shape, mean 0 std 1)
        log_mean_i = log(max(mu_i, eps))               (raw, NOT z-scored)
        log_std_i  = log(sigma_i)                      (raw, NOT z-scored)

    The returned ``log_mean`` / ``log_std`` are the per-instance scales needed
    to invert the normalisation (see :func:`shape_denormalize`). They match
    the quantities used in notebook 02 to build the global ``log_mean_z`` /
    ``log_std_z`` conditioning channels — the z-score statistics needed to
    map between the two spaces are produced by
    :func:`src.data.loader.compute_scale_stats`.

    Parameters
    ----------
    xs : (N, L) float array of raw daily profiles (e.g. Watts).
    eps: floor applied before taking logs and dividing.

    Returns
    -------
    xs_norm  : (N, L) float32 — per-instance z-scored profiles.
    log_mean : (N,)   float32 — per-instance log of the daily mean.
    log_std  : (N,)   float32 — per-instance log of the daily std.
    """
    xs = np.asarray(xs, dtype=np.float32)
    mu = xs.mean(axis=1, keepdims=True)
    sigma = np.clip(xs.std(axis=1, keepdims=True), eps, None)
    xs_norm = ((xs - mu) / sigma).astype(np.float32)
    log_mean = np.log(np.clip(mu[:, 0], eps, None)).astype(np.float32)
    log_std = np.log(sigma[:, 0]).astype(np.float32)
    return xs_norm, log_mean, log_std


def shape_denormalize(
    xs_norm: np.ndarray,
    log_mean: np.ndarray,
    log_std: np.ndarray,
) -> np.ndarray:
    """
    Inverse of :func:`shape_normalize`.

    Given per-instance ``log_mean`` and ``log_std`` (the raw, un-z-scored
    quantities), restore raw-unit profiles:

        x_i = x_i_norm * exp(log_std_i) + exp(log_mean_i)

    Broadcasting: ``log_mean`` / ``log_std`` must be scalars or 1-D of length
    matching the leading axis of ``xs_norm``.
    """
    xs_norm = np.asarray(xs_norm, dtype=np.float32)
    log_mean = np.asarray(log_mean, dtype=np.float32)
    log_std = np.asarray(log_std, dtype=np.float32)
    mu = np.exp(log_mean)
    sigma = np.exp(log_std)
    if xs_norm.ndim == 2 and mu.ndim == 1:
        return xs_norm * sigma[:, None] + mu[:, None]
    return xs_norm * sigma + mu


def split_mask_by_meter(
    meter_ids: np.ndarray,
    n_meters: int,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """
    Produce the boolean validation mask used by :func:`train_val_split_instances`.

    Useful when an additional per-instance array (e.g. ``log_mean``,
    ``log_std``) needs to be split with the *exact same* partition without
    re-invoking the full split helper.

    Returns
    -------
    val_mask : (N,) bool — True for validation instances.
    """
    rng = np.random.default_rng(seed)
    all_meters = np.arange(n_meters)
    rng.shuffle(all_meters)
    n_val = max(1, int(np.round(n_meters * val_fraction)))
    val_meters = set(all_meters[:n_val].tolist())
    return np.array([int(m) in val_meters for m in meter_ids], dtype=bool)


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


# ---------------------------------------------------------------------------
# Season helper
# ---------------------------------------------------------------------------

def _month_to_season(month_0indexed: int) -> int:
    """
    Map 0-indexed month (0=Jan … 11=Dec) to meteorological season.
      0 = winter  (Dec, Jan, Feb)
      1 = spring  (Mar, Apr, May)
      2 = summer  (Jun, Jul, Aug)
      3 = autumn  (Sep, Oct, Nov)
    """
    return [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0][month_0indexed]


# ---------------------------------------------------------------------------
# Instance-level windowing for the Rolle dataset
# ---------------------------------------------------------------------------

def make_daily_instances(
    df: pd.DataFrame,
    temp_daily_normed: pd.Series,
    cluster_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training instances where each instance = one meter × one calendar day.

    This is the primary data-preparation function for the Rolle dataset. Unlike
    ``make_windows`` (which strides over the raw time axis), this function
    aligns windows to calendar-day boundaries and enriches each instance with
    continuous weather conditioning from NWP forecasts.

    Parameters
    ----------
    df : pd.DataFrame, shape (T_hourly, N_meters)
        Per-meter z-score normalised power, hourly resolution, with a
        tz-aware DatetimeIndex.  Values with >10 % NaN in a day are discarded;
        remaining NaNs are linearly interpolated.
    temp_daily_normed : pd.Series
        Daily mean temperature, z-score normalised globally (output of
        ``normalize_temp()`` in loader.py).  Index must be tz-aware and
        aligned to calendar-day midnight (``resample('1D')`` convention).
        Days not present in ``temp_daily_normed`` are filled with 0.0
        (global mean after z-scoring).
    cluster_labels : ndarray | dict | None
        Optional per-instance cluster ids.
        * If an ndarray, must have shape ``(N_instances,)`` and be in the
          iteration order produced by this function for the *same*
          ``(df, temp_daily_normed)`` pair — i.e. labels obtained from a
          prior call without ``cluster_labels``.
        * If a dict, keys must be ``(meter_id:int, date)`` where ``date`` is
          either a ``datetime.date`` or an ISO string ``"YYYY-MM-DD"``;
          missing keys default to ``cluster_id = 0`` with a warning.
        * If None, ``cluster_id`` is set to 0 (useful before clustering
          labels are available).

    Returns
    -------
    xs            : (N, 24) float32  normalised 24h profiles
    c_discrete    : (N, 3)  int32    [cluster_id, day_type, season]
                    day_type: 0=weekday, 1=weekend
                    season:   0=winter, 1=spring, 2=summer, 3=autumn
    c_continuous  : (N, 1)  float32  [daily_mean_temp_normalised]
    dates         : (N,)    object   calendar date (datetime.date) per instance
    meter_ids     : (N,)    int32    meter column index in ``df``
    """
    T, N_meters = df.shape

    # Align temp_daily to date (normalize to midnight)
    if hasattr(temp_daily_normed.index, "normalize"):
        temp_idx = temp_daily_normed.index.normalize()
    else:
        temp_idx = pd.DatetimeIndex(temp_daily_normed.index).normalize()
    temp_lookup: dict = dict(zip(temp_idx.date, temp_daily_normed.values))

    # Group power index by calendar date
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex")

    all_dates = sorted(set(df.index.date))

    xs_list: list[np.ndarray] = []
    cd_list: list[list[int]] = []
    cc_list: list[list[float]] = []
    date_list: list = []
    mid_list: list[int] = []

    instance_count = 0
    for date in all_dates:
        day_mask = df.index.date == date
        day_df = df.loc[day_mask]

        if len(day_df) != STEPS_PER_DAY:
            # Incomplete day (e.g. first or last partial day) — skip
            continue

        # Calendar features from the date
        ts = pd.Timestamp(date)
        dow = ts.dayofweek                   # 0=Mon … 6=Sun
        month_0idx = ts.month - 1            # 0=Jan … 11=Dec
        dt = _day_type(dow)
        season = _month_to_season(month_0idx)

        # Daily mean temperature (fallback to 0.0 = global mean after z-score)
        temp_val = float(temp_lookup.get(date, 0.0))

        for meter_idx in range(N_meters):
            window = day_df.iloc[:, meter_idx].values.astype(np.float32)

            # Skip days with >10 % missing values
            if np.isnan(window).mean() > 0.10:
                continue

            # Linearly interpolate remaining NaN gaps
            if np.isnan(window).any():
                nans = np.isnan(window)
                ok = np.where(~nans)[0]
                window[nans] = np.interp(np.where(nans)[0], ok, window[~nans])

            xs_list.append(window)
            cd_list.append([0, dt, season])   # cluster_id placeholder = 0
            cc_list.append([temp_val])
            date_list.append(date)
            mid_list.append(meter_idx)
            instance_count += 1

    xs = np.stack(xs_list, axis=0).astype(np.float32)             # (N, 24)
    c_disc = np.array(cd_list, dtype=np.int32)                     # (N, 3)
    c_cont = np.array(cc_list, dtype=np.float32)                   # (N, 1)
    dates_arr = np.array(date_list, dtype=object)                  # (N,)
    meter_ids = np.array(mid_list, dtype=np.int32)                 # (N,)

    # Assign cluster labels if provided.
    # Two accepted forms:
    #   (a) ndarray-like of shape (N_instances,) in iteration order
    #   (b) dict mapping (meter_id:int, date) -> cluster_id, where `date`
    #       may be either a datetime.date or an ISO string ("YYYY-MM-DD").
    if cluster_labels is not None:
        if isinstance(cluster_labels, dict):
            resolved = np.empty(len(xs), dtype=np.int32)
            missing = 0
            for i, (mid, d) in enumerate(zip(mid_list, date_list)):
                if (mid, d) in cluster_labels:
                    resolved[i] = int(cluster_labels[(mid, d)])
                elif (mid, str(d)) in cluster_labels:
                    resolved[i] = int(cluster_labels[(mid, str(d))])
                else:
                    resolved[i] = 0
                    missing += 1
            if missing > 0:
                import warnings
                warnings.warn(
                    f"cluster_labels dict is missing {missing}/{len(xs)} "
                    f"(meter_id, date) keys; defaulting to cluster_id=0 for those.",
                    RuntimeWarning,
                )
            c_disc[:, 0] = resolved
        else:
            arr = np.asarray(cluster_labels, dtype=np.int32)
            if arr.ndim != 1 or len(arr) != len(xs):
                raise ValueError(
                    f"cluster_labels array must have shape ({len(xs)},); "
                    f"got shape {arr.shape}"
                )
            c_disc[:, 0] = arr

    return xs, c_disc, c_cont, dates_arr, meter_ids


def train_val_split_instances(
    xs: np.ndarray,
    c_discrete: np.ndarray,
    c_continuous: np.ndarray,
    meter_ids: np.ndarray,
    n_meters: int,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, ...]:
    """
    Stratified train/val split for daily instances, held out by meter.

    Splits by meter (not by day) to avoid temporal leakage: all days from
    a held-out meter go exclusively into the validation set.

    Returns
    -------
    x_tr, cd_tr, cc_tr, mid_tr, x_val, cd_val, cc_val, mid_val  (8-tuple)
    """
    rng = np.random.default_rng(seed)
    all_meters = np.arange(n_meters)
    rng.shuffle(all_meters)
    n_val = max(1, int(np.round(n_meters * val_fraction)))
    val_meters = set(all_meters[:n_val].tolist())

    val_mask = np.array([int(m) in val_meters for m in meter_ids], dtype=bool)
    tr_mask = ~val_mask

    return (
        xs[tr_mask], c_discrete[tr_mask], c_continuous[tr_mask], meter_ids[tr_mask],
        xs[val_mask], c_discrete[val_mask], c_continuous[val_mask], meter_ids[val_mask],
    )


def train_val_split(
    xs: np.ndarray,
    cs: np.ndarray,
    mid: np.ndarray,
    n_meters: int,
    val_fraction: float = 0.15,
    seed: int = 42,
    return_mid: bool = False,
) -> tuple[np.ndarray, ...]:
    """
    Stratified split by meter — val meters are held out entirely.
    This avoids leakage between train and val.

    Always returns ``x_tr, c_tr, mid_tr, x_val, c_val, mid_val``.
    The ``return_mid`` parameter is accepted for API compatibility.
    """
    rng = np.random.default_rng(seed)
    all_meters = np.arange(n_meters)
    rng.shuffle(all_meters)
    n_val = max(1, int(np.round(n_meters * val_fraction)))
    val_meters = set(all_meters[:n_val].tolist())

    val_mask = np.array([m in val_meters for m in mid], dtype=bool)
    tr_mask = ~val_mask

    return (
        xs[tr_mask],
        cs[tr_mask],
        mid[tr_mask],
        xs[val_mask],
        cs[val_mask],
        mid[val_mask],
    )


class _InfiniteLoader:
    """
    Infinite iterator of (x_batch, c_batch) pairs, or 3-tuple
    (x_batch, c_disc_batch, c_cont_batch) when ``cc`` is provided.

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
        cc: Optional[np.ndarray] = None,
    ):
        n = len(xs)
        self.epoch_len: int = max(1, n // batch_size)
        self._gen = self._make_gen(xs, cs, batch_size, shuffle, rng, cc)

    @staticmethod
    def _make_gen(xs, cs, batch_size, shuffle, rng, cc):
        n = len(xs)
        idx = np.arange(n)
        while True:
            if shuffle:
                rng.shuffle(idx)
            for start in range(0, n, batch_size):
                b = idx[start : start + batch_size]
                if len(b) < batch_size:
                    break
                if cc is None:
                    yield xs[b], cs[b]
                else:
                    yield xs[b], cs[b], cc[b]

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
        self.cc: Optional[np.ndarray] = None  # set after construction when needed
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

        if self.cc is None:
            return self.xs[batch_indices], self.cs[batch_indices]
        return self.xs[batch_indices], self.cs[batch_indices], self.cc[batch_indices]


def numpy_dataloader(
    xs: np.ndarray,
    cs: np.ndarray,
    batch_size: int,
    c_continuous: Optional[np.ndarray] = None,
    shuffle: bool = True,
    balance_condition_cols: tuple[int, ...] | None = None,
    rng: int | np.random.Generator = 0,
) -> "_InfiniteLoader | _BalancedInfiniteLoader":
    """
    Infinite data iterator for training.

    Yields 2-tuples ``(x_batch, c_disc_batch)`` when ``c_continuous`` is
    None, or 3-tuples ``(x_batch, c_disc_batch, c_cont_batch)`` when
    ``c_continuous`` is provided.

    Parameters
    ----------
    xs              : (N, 24) float32  normalised power profiles
    cs              : (N, n_disc) int32  discrete conditioning
    batch_size      : int
    c_continuous    : optional (N, n_cont) float32  continuous conditioning
    shuffle         : whether to shuffle each epoch
    balance_condition_cols : tuple of column indices in ``cs`` to balance
                             over (e.g. ``(0, 1)`` for cluster × day_type)
    rng             : seed or pre-built Generator

    The returned object exposes ``epoch_len = len(xs) // batch_size`` so that
    ``Trainer.fit`` can automatically bound each epoch to one data pass.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    if balance_condition_cols is not None:
        loader = _BalancedInfiniteLoader(
            xs,
            cs,
            batch_size,
            tuple(balance_condition_cols),
            shuffle,
            rng,
        )
        loader.cc = c_continuous
        return loader
    return _InfiniteLoader(xs, cs, batch_size, shuffle, rng, cc=c_continuous)
