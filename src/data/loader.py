"""
loader.py
---------
Load the raw power.pk dataset, inspect it, and produce a clean
pandas DataFrame with a DatetimeIndex and one column per smart-meter.

Data facts (Portuguese smart-meter dataset, 2012–2014)
------------------------------------------------------
  T × N  : ~26 304 timesteps × 321 meters  (hourly, 24 steps/day)
  Units  : Wh per hour (equivalent to average W per hour);
           values in [0, ~764 000] across all meters.
  Outliers: 22 meters have mean > 10× dataset median (max ×339);
            they are kept in training. Per-meter scale normalisation
            (division by the meter's annual mean) makes the diffusion
            task purely about shape — scale is recovered at inference
            time by multiplying by the target meter's known annual mean
            (a scalar available in practice from billing data).

Expected pickle structure (will be confirmed during EDA):
  - Either a pandas DataFrame  (meters as columns, DatetimeIndex)
  - Or a dict {'data': np.ndarray, 'timestamps': ..., 'meter_ids': ...}

Public API
----------
load_raw(path)                    -> pd.DataFrame  shape (T, N_meters)
compute_stats(df)                 -> dict {meter_idx: {'scale': float}}
normalize(df, stats)              -> pd.DataFrame  consumption / annual mean
denormalize(arr, meter_id, stats) -> np.ndarray    arr * stats[meter_id]['scale']
denormalize_batch(arr, scales)    -> np.ndarray    vectorised per-sample rescale
scales_array(stats, meter_ids)    -> np.ndarray    gather per-sample scales
filter_outlier_meters(df, ...)    -> (df, labels, mask)
compute_meter_stats(df)           -> dict  per-meter mean/std arrays
normalize_by_meter(df, stats)     -> pd.DataFrame  z-scored per meter
denormalize_by_meter(arr, ...)    -> np.ndarray    invert per-meter z-score
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# Columns in power_data['P_mean'] that are hierarchical aggregations, not real meters.
_AGG_COLS: frozenset[str] = frozenset({"S1", "S2", "S11", "S12", "S21", "S22", "all"})


def load_raw(path: str | Path = "data/power.pk") -> pd.DataFrame:
    """Load the raw pickle file and return a (T, N_meters) DataFrame."""
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # --- handle common serialisation formats --------------------------------
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, dict):
        # Try common key patterns
        data_key = next(
            (k for k in obj if k in ("data", "values", "X", "consumption")), None
        )
        time_key = next(
            (k for k in obj if k in ("timestamps", "index", "time", "dates")), None
        )
        meter_key = next(
            (k for k in obj if k in ("meter_ids", "columns", "ids", "meters")), None
        )
        if data_key is None:
            raise ValueError(
                f"Cannot find data array in dict with keys: {list(obj.keys())}"
            )
        data = np.asarray(obj[data_key])
        if data.ndim == 1:
            data = data[:, None]
        # data should be (T, N) after this
        if data.shape[0] < data.shape[1]:
            data = data.T

        index = pd.to_datetime(obj[time_key]) if time_key else pd.RangeIndex(len(data))
        columns = np.asarray(obj[meter_key]) if meter_key else np.arange(data.shape[1])
        df = pd.DataFrame(data, index=index, columns=columns)
    elif isinstance(obj, np.ndarray):
        # bare array — assume (T, N) or (N, T)
        arr = obj
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        df = pd.DataFrame(arr)
    else:
        raise TypeError(f"Unsupported pickle type: {type(obj)}")

    # Ensure float32
    df = df.astype(np.float32)
    return df


def compute_stats(
    df: pd.DataFrame,
    cluster_labels: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute per-meter normalisation scale = annual mean consumption.

    The diffusion model is trained on profiles divided by their meter's
    annual mean, so the network only has to learn shape. Scale is
    reintroduced at inference time by multiplying by a known target mean.

    Parameters
    ----------
    df : (T, N_meters) DataFrame
    cluster_labels : ignored — kept only for backward compatibility with
                     older notebook code; per-meter scaling does not need
                     cluster labels.

    Returns
    -------
    stats : dict mapping meter_idx (int, column position in df) ->
            {'scale': float}.  Scale is floored to 1e-8 to protect against
            all-zero meters.
    """
    del cluster_labels  # accepted for back-compat; no longer used

    means = np.nanmean(df.values.astype(np.float64), axis=0)
    scales = np.maximum(means, 1e-8)
    return {int(i): {"scale": float(scales[i])} for i in range(df.shape[1])}


def normalize(
    df: pd.DataFrame,
    stats: dict,
    cluster_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Divide each meter's consumption by its annual-mean scale.

    Parameters
    ----------
    df             : (T, N_meters) DataFrame
    stats          : output of compute_stats — per-meter scales
    cluster_labels : ignored — kept for backward compatibility.

    Returns
    -------
    Normalised DataFrame, same shape as df. Each column has empirical
    mean ≈ 1.0 (exactly 1.0 in the no-NaN case).
    """
    del cluster_labels  # back-compat

    n = df.shape[1]
    scales = np.array(
        [stats[int(i)]["scale"] for i in range(n)], dtype=np.float32
    )
    arr = df.values.astype(np.float32) / scales[None, :]
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def denormalize(
    arr: np.ndarray,
    meter_id: int,
    stats: dict,
) -> np.ndarray:
    """
    Invert per-meter normalisation for a batch from a single meter.

    Parameters
    ----------
    arr      : array of any shape (last axis = time)
    meter_id : column index of the meter in the original DataFrame
    stats    : output of compute_stats

    Returns
    -------
    arr * stats[meter_id]['scale']
    """
    s = stats[int(meter_id)]["scale"]
    return arr * np.float32(s)


def denormalize_batch(
    arr: np.ndarray,
    scales: np.ndarray | float,
) -> np.ndarray:
    """
    Vectorised denormalisation when each sample in ``arr`` has its own
    scale (e.g. after the generator produces N profiles, each tagged
    with a target annual mean drawn from the empirical distribution).

    Parameters
    ----------
    arr    : shape (B, L) or (B,)
    scales : scalar, or array broadcastable along the leading axis of arr.

    Returns
    -------
    arr scaled element-wise along the batch dimension.
    """
    s = np.asarray(scales, dtype=np.float32)
    if arr.ndim == 2 and s.ndim == 1:
        return arr * s[:, None]
    return arr * s


def filter_outlier_meters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    factor: float = 10.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Drop meters whose annual mean exceeds ``factor`` × the dataset-wide
    median annual mean.

    With ``factor=10`` on the Portuguese dataset this removes 22 of 321
    meters (top offender: meter 313, ×339 the median).

    Parameters
    ----------
    df             : (T, N_meters) DataFrame.
    cluster_labels : (N_meters,) array of cluster ids, aligned with df columns.
    factor         : multiplier on the median annual mean.

    Returns
    -------
    (df_kept, cluster_labels_kept, kept_mask) where ``kept_mask`` is a
    boolean array of shape (N_meters,) marking surviving columns.
    """
    cluster_labels = np.asarray(cluster_labels)
    if cluster_labels.shape[0] != df.shape[1]:
        raise ValueError(
            f"cluster_labels has {cluster_labels.shape[0]} entries but df has "
            f"{df.shape[1]} meters"
        )

    meter_means = np.nanmean(df.values.astype(np.float64), axis=0)
    threshold = float(factor) * float(np.nanmedian(meter_means))
    kept_mask = meter_means <= threshold

    df_kept = df.iloc[:, kept_mask].copy()
    df_kept.columns = pd.RangeIndex(df_kept.shape[1])
    cluster_labels_kept = cluster_labels[kept_mask]
    return df_kept, cluster_labels_kept, kept_mask


def scales_array(stats: dict, meter_ids: np.ndarray) -> np.ndarray:
    """
    Lookup helper: gather per-sample scales for a vector of meter ids.

    Parameters
    ----------
    stats     : output of compute_stats
    meter_ids : (N,) int array of meter column indices (e.g. ``mid`` from
                ``make_windows``)

    Returns
    -------
    (N,) float32 array of per-sample scales.
    """
    return np.array(
        [stats[int(i)]["scale"] for i in meter_ids], dtype=np.float32
    )


def compute_meter_stats(df: pd.DataFrame) -> dict:
    """
    Compute long-period mean/std for each meter column.

    This is the preferred normalisation for generative modelling in this
    project: the model learns shape in per-household z-score space, then
    generated profiles are mapped back to Wh with the same meter's scale.

    Returns
    -------
    stats : dict with keys ``mode``, ``mean``, ``std``, ``meter_ids``.
    """
    mean = np.nanmean(df.values, axis=0).astype(np.float32)
    std = (np.nanstd(df.values, axis=0) + 1e-8).astype(np.float32)
    return {
        "mode": "meter",
        "mean": mean,
        "std": std,
        "meter_ids": list(df.columns),
    }


def normalize_by_meter(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Z-score normalise each meter with its own long-period mean/std.
    """
    if stats.get("mode") != "meter":
        raise ValueError("normalize_by_meter expects stats from compute_meter_stats()")
    out = df.copy()
    out.iloc[:, :] = (df.values - stats["mean"][None, :]) / stats["std"][None, :]
    return out


def denormalize_by_meter(
    arr: np.ndarray,
    meter_indices: np.ndarray | int,
    stats: dict,
) -> np.ndarray:
    """
    Invert per-meter z-score normalisation for generated or real windows.

    Parameters
    ----------
    arr : array, shape (N, L) or (L,)
        Normalised profile(s).
    meter_indices : array shape (N,) or int
        Meter column index used to choose the matching long-period scale.
    stats : output of compute_meter_stats
    """
    if stats.get("mode") != "meter":
        raise ValueError("denormalize_by_meter expects stats from compute_meter_stats()")

    values = np.asarray(arr)
    meter_indices = np.asarray(meter_indices, dtype=np.int32)

    if values.ndim == 1:
        mean = stats["mean"][int(meter_indices)]
        std = stats["std"][int(meter_indices)]
        return values * std + mean

    mean = stats["mean"][meter_indices][:, None]
    std = stats["std"][meter_indices][:, None]
    return values * std + mean


# ---------------------------------------------------------------------------
# Rolle (Switzerland) dataset — new primary data source
# ---------------------------------------------------------------------------

def load_rolle_data(
    data_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Rolle (Switzerland) power + NWP dataset.

    Dataset
    -------
    Source  : Zenodo 10.5281/zenodo.3463136
    Meters  : 62 IEC 61000-4-30 Class A power-quality meters, 24 real
              substations/LV cabinets in Rolle (CH).  The ``P_mean``
              DataFrame additionally contains 7 fictitious hierarchical
              aggregation columns (``S1``, ``S2``, ``S11``, ``S12``,
              ``S21``, ``S22``, ``all``) that are dropped here.
    Period  : 2018-01-13 – 2019-01-19 (~1 year)
    Native  : 10-minute resolution → resampled to 1-hour.
    NWP     : Meteoblue forecasts (temperature, GHI, GNI, RH, wind speed /
              direction) stored as 24-element arrays (24h ahead) at every
              10-minute timestamp.

    Parameters
    ----------
    data_dir : path to the directory containing ``power_data.p`` and
               ``nwp_data.h5``.

    Returns
    -------
    power_hourly : pd.DataFrame, shape (T_hourly, 24), float32
        Mean active power [W] per real meter, hourly resolution.
        Columns are meter SHA-hash identifiers.  Index is tz-aware UTC.
    temp_daily : pd.Series, shape (N_days,), float64
        Daily mean temperature [°C] for Rolle, indexed by calendar day
        (tz-aware UTC midnight).  Computed as the mean of all 10-minute
        NWP temperature forecast arrays within each calendar day.
    """
    data_dir = Path(data_dir)

    # --- Power data ----------------------------------------------------------
    with open(data_dir / "power_data.p", "rb") as f:
        power_dict = pickle.load(f)

    pm: pd.DataFrame = power_dict["P_mean"]
    # Drop hierarchical aggregation columns; keep only real meter columns.
    agg_cols = [c for c in pm.columns if str(c) in _AGG_COLS]
    pm = pm.drop(columns=agg_cols)

    # Resample 10-min → hourly (mean).
    pm = pm.resample("1h").mean()
    pm = pm.astype(np.float32)

    # --- NWP data ------------------------------------------------------------
    try:
        nwp: pd.DataFrame = pd.read_hdf(data_dir / "nwp_data.h5", "df")
    except ImportError as exc:
        raise ImportError(
            "PyTables is required to read HDF5 files: pip install tables"
        ) from exc

    # Each cell in the 'temperature' column is a 24-element numpy array
    # representing a 24-hour ahead temperature forecast [°C].
    # Compute the mean of each forecast to get a scalar "expected daily mean
    # temperature" for the 24h window starting at that timestamp.
    scalar_temp: pd.Series = nwp["temperature"].apply(
        lambda arr: float(np.asarray(arr).mean())
    )

    # Resample to calendar days (mean of all 10-min estimates in the day).
    temp_daily: pd.Series = scalar_temp.resample("1D").mean()
    # Normalise the index to midnight UTC so it aligns with power_hourly dates.
    temp_daily.index = temp_daily.index.normalize()

    return pm, temp_daily


def compute_temp_stats(temp_daily: pd.Series) -> dict:
    """
    Compute global mean and std for daily mean temperature normalisation.

    Returns
    -------
    dict with keys ``mean`` (float) and ``std`` (float, floored at 1e-8).
    """
    return {
        "mean": float(temp_daily.mean()),
        "std": float(max(temp_daily.std(), 1e-8)),
    }


def normalize_temp(temp_daily: pd.Series, temp_stats: dict) -> pd.Series:
    """Z-score normalise a daily mean temperature Series."""
    return (temp_daily - temp_stats["mean"]) / temp_stats["std"]


# ---------------------------------------------------------------------------
# Per-instance scale stats (for shape normalisation, see dataset.shape_normalize)
# ---------------------------------------------------------------------------

def compute_scale_stats(log_mean: np.ndarray, log_std: np.ndarray) -> dict:
    """
    Compute the global mean/std used to z-score per-instance ``log_mean`` and
    ``log_std`` channels in ``c_continuous``.

    These four scalars are exactly what notebook 02 §1b applies to build
    ``log_mean_z`` and ``log_std_z`` from the raw per-instance log-scales
    produced by :func:`src.data.dataset.shape_normalize`. Persisting them
    (e.g. to ``data/scale_stats.json``) lets evaluation notebooks invert the
    z-score and recover per-query ``log_mean`` / ``log_std`` from
    ``c_continuous[:, 1:3]`` — which in turn enables
    :func:`src.data.dataset.shape_denormalize` to map generated normalised
    profiles back to raw-unit Watts.
    """
    return {
        "log_mean_mean": float(np.mean(log_mean)),
        "log_mean_std": float(max(np.std(log_mean), 1e-8)),
        "log_std_mean": float(np.mean(log_std)),
        "log_std_std": float(max(np.std(log_std), 1e-8)),
    }


def invert_scale_z(
    log_mean_z: np.ndarray,
    log_std_z: np.ndarray,
    scale_stats: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse of the global z-score applied to per-instance log scales.

    Given the (z-scored) channels carried in ``c_continuous`` and the stats
    saved by :func:`compute_scale_stats`, recover the raw per-instance
    ``log_mean`` / ``log_std`` arrays needed by
    :func:`src.data.dataset.shape_denormalize`.
    """
    log_mean = np.asarray(log_mean_z, dtype=np.float32) * np.float32(
        scale_stats["log_mean_std"]
    ) + np.float32(scale_stats["log_mean_mean"])
    log_std = np.asarray(log_std_z, dtype=np.float32) * np.float32(
        scale_stats["log_std_std"]
    ) + np.float32(scale_stats["log_std_mean"])
    return log_mean, log_std
