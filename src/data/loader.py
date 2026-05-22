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
            they are kept in training but isolated into their own
            cluster naturally by the shape-normalised K-Means.
            Prefer per-meter z-score normalisation for generation so each
            household's long-period scale can be restored deterministically.

Expected pickle structure (will be confirmed during EDA):
  - Either a pandas DataFrame  (meters as columns, DatetimeIndex)
  - Or a dict {'data': np.ndarray, 'timestamps': ..., 'meter_ids': ...}

Public API
----------
load_raw(path)   -> pd.DataFrame  shape (T, N_meters)
compute_stats(df, cluster_labels) -> dict  per-cluster mean/std
normalize(df, stats)              -> pd.DataFrame  z-scored per cluster
compute_meter_stats(df)           -> dict  per-meter mean/std arrays
normalize_by_meter(df, stats)     -> pd.DataFrame  z-scored per meter
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


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
    Compute per-cluster (or global) mean and std for z-score normalisation.

    Parameters
    ----------
    df : (T, N_meters) DataFrame
    cluster_labels : (N_meters,) integer array or None
        If None, treats all meters as one group.

    Returns
    -------
    stats : dict mapping cluster_id -> {'mean': float, 'std': float}
    """
    if cluster_labels is None:
        cluster_labels = np.zeros(df.shape[1], dtype=int)

    stats: dict[int, dict] = {}
    for cid in np.unique(cluster_labels):
        mask = cluster_labels == cid
        vals = df.iloc[:, mask].values.ravel()
        stats[int(cid)] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals) + 1e-8),
        }
    return stats


def normalize(
    df: pd.DataFrame,
    stats: dict,
    cluster_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Z-score normalise each meter using its cluster statistics.

    Parameters
    ----------
    df            : (T, N_meters)
    stats         : output of compute_stats
    cluster_labels: (N_meters,) integer array or None (global normalisation)

    Returns
    -------
    normalised DataFrame, same shape as df
    """
    if cluster_labels is None:
        cluster_labels = np.zeros(df.shape[1], dtype=int)

    out = df.copy()
    for cid, s in stats.items():
        mask = cluster_labels == cid
        out.iloc[:, mask] = (df.iloc[:, mask].values - s["mean"]) / s["std"]
    return out


def denormalize(
    arr: np.ndarray,
    cluster_id: int,
    stats: dict,
) -> np.ndarray:
    """Invert z-score for a batch of samples from a given cluster."""
    s = stats[cluster_id]
    return arr * s["std"] + s["mean"]


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
