"""
loader.py
---------
Load the raw power.pk dataset, inspect it, and produce a clean
pandas DataFrame with a DatetimeIndex and one column per smart-meter.

Expected pickle structure (will be confirmed during EDA):
  - Either a pandas DataFrame  (meters as columns, DatetimeIndex)
  - Or a dict {'data': np.ndarray, 'timestamps': ..., 'meter_ids': ...}

Public API
----------
load_raw(path)   -> pd.DataFrame  shape (T, N_meters)
compute_stats(df, cluster_labels) -> dict  per-cluster mean/std
normalize(df, stats)              -> pd.DataFrame  z-scored per cluster
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
