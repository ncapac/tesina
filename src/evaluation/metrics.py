"""
metrics.py
----------
Evaluation metrics for comparing real vs. synthetic energy load profiles.

Metrics
-------
  acf_compare          — ACF/PACF comparison (plots + scalar L2 distance)
  marginal_kde         — KDE overlay per time bin (peak / shoulder / night)
  crps_score           — Continuous Ranked Probability Score (probabilistic quality)
  discriminative_score — Train a 1D-conv real/fake classifier; accuracy ≈ 50% = ideal
  envelope_plot        — Mean ± std envelope comparison

References
----------
  CRPS:  eq. 57 in arXiv 2507.14507 (Su et al., 2025)
  Discriminative score: popularised by TimeGAN (Yoon et al., 2019)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# ACF / PACF
# ---------------------------------------------------------------------------

def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample ACF for a 1-D series."""
    x = x - x.mean()
    n = len(x)
    acfs = []
    for lag in range(nlags + 1):
        num = (x[: n - lag] * x[lag:]).sum()
        denom = (x ** 2).sum()
        acfs.append(num / denom if denom != 0 else 0.0)
    return np.array(acfs)


def acf_compare(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    nlags: int = 23,        # default: full 24-step window (0..23)
    ax: Optional[plt.Axes] = None,
    label: str = "",
) -> float:
    # Clamp nlags to the actual sequence length to avoid empty-slice artifacts
    nlags = min(nlags, real.shape[1] - 1)
    """
    Overlay mean ACF of real vs synthetic samples.
    Returns L2 distance between the two mean ACF vectors.
    """
    real_acfs = np.array([_acf(s, nlags) for s in real])    # (N, nlags+1)
    syn_acfs  = np.array([_acf(s, nlags) for s in synthetic])

    mean_real = real_acfs.mean(0)
    mean_syn  = syn_acfs.mean(0)

    if ax is not None:
        lags = np.arange(nlags + 1)
        ax.fill_between(lags, real_acfs.min(0), real_acfs.max(0), alpha=0.2, color="steelblue", label="real range")
        ax.fill_between(lags, syn_acfs.min(0),  syn_acfs.max(0),  alpha=0.2, color="coral",     label="syn range")
        ax.plot(lags, mean_real, color="steelblue", linewidth=2, label="real mean")
        ax.plot(lags, mean_syn,  color="coral",     linewidth=2, label="syn mean")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("ACF")
        ax.set_title(f"ACF comparison {label}")
        ax.legend(fontsize=8)

    return float(np.linalg.norm(mean_real - mean_syn))


# ---------------------------------------------------------------------------
# Marginal KDE
# ---------------------------------------------------------------------------

def marginal_kde(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    time_bins: int = 4,
    ax: Optional[plt.Axes] = None,
    label: str = "",
) -> None:
    """
    Plot KDE of value distributions for equally-spaced time bins.
    """
    L = real.shape[1]
    bin_size = L // time_bins
    if ax is None:
        _, ax = plt.subplots(1, time_bins, figsize=(4 * time_bins, 3))
    elif not hasattr(ax, "__len__"):
        ax = [ax] * time_bins

    for i in range(time_bins):
        start = i * bin_size
        end   = start + bin_size
        r_vals = real[:, start:end].ravel()
        s_vals = synthetic[:, start:end].ravel()

        common = np.linspace(
            min(r_vals.min(), s_vals.min()),
            max(r_vals.max(), s_vals.max()), 200
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax[i].plot(common, gaussian_kde(r_vals)(common), color="steelblue",
                       linewidth=2, label="real")
            ax[i].plot(common, gaussian_kde(s_vals)(common), color="coral",
                       linewidth=2, label="synthetic")
        ax[i].set_title(f"Steps {start}–{end}")
        ax[i].set_xlabel("Normalised consumption")
        if i == 0:
            ax[i].set_ylabel("Density")
        ax[i].legend(fontsize=8)


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def crps_score(
    real: np.ndarray,     # (N_real,  L)
    samples: np.ndarray,  # (N_samples, L)  — ensemble of synthetic draws
) -> float:
    """
    Mean CRPS over all timestep positions.

    Approximates CRPS using an ensemble of N_samples predictions vs each
    real observation (one randomly selected real per CRPS call here we
    use the mean over all reals).

    CRPS(F, x) ≈ E_F[|Y - x|] - 0.5 · E_F[|Y - Y'|]
    """
    N_s = samples.shape[0]
    # Mean absolute error term E[|Y - x|]
    # Broadcast: (N_real, 1, L) vs (1, N_samples, L)
    real_exp = real[:, None, :]       # (N_r, 1, L)
    samp_exp = samples[None, :, :]    # (1, N_s, L)
    mae_term = np.abs(real_exp - samp_exp).mean(axis=1)  # (N_r, L)

    # Dispersion term E[|Y - Y'|]
    # Compute pairwise sample distances cheaply via sorted trick:
    #   E|Y - Y'| = (2/N^2) sum_{i<j} |s_i - s_j| ≈ 2 * std(samples) (rough)
    sorted_s = np.sort(samples, axis=0)                  # (N_s, L)
    ranks = (2 * np.arange(N_s) - N_s + 1)[:, None]     # (N_s, 1)
    disp_term = (ranks * sorted_s).sum(0) / N_s ** 2     # (L,)  == E|Y−Y'|/2

    crps_per_real = mae_term - disp_term[None, :]        # (N_r, L)
    return float(crps_per_real.mean())


# ---------------------------------------------------------------------------
# Discriminative score (real/fake classifier)
# ---------------------------------------------------------------------------

def discriminative_score(
    real: np.ndarray,       # (N_real,  L)
    synthetic: np.ndarray,  # (N_syn,   L)
    n_epochs: int = 30,
    seed: int = 0,
) -> float:
    """
    Train a lightweight 1-D conv classifier to distinguish real from synthetic.
    Returns test accuracy; ideal = 0.5 (indistinguishable distributions).

    Uses scikit-learn's MLPClassifier as a simple baseline discriminator.
    The two classes are balanced by subsampling the majority class so that
    the score is not dominated by class imbalance.
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn required for discriminative_score")

    # Balance classes by subsampling the larger set
    n_min = min(len(real), len(synthetic))
    rng_bal = np.random.default_rng(seed)
    real_sub = real[rng_bal.choice(len(real), n_min, replace=False)]
    syn_sub  = synthetic[rng_bal.choice(len(synthetic), n_min, replace=False)]

    X = np.concatenate([real_sub, syn_sub], axis=0)
    y = np.array([0] * n_min + [1] * n_min, dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=n_epochs,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=5,
    )
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    return acc


# ---------------------------------------------------------------------------
# Envelope plot
# ---------------------------------------------------------------------------

def envelope_plot(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    ax: Optional[plt.Axes] = None,
    label: str = "",
    steps_per_hour: int = 1,   # hourly resolution (24-step windows)
) -> None:
    """
    Plot mean ± 1 std envelope for real and synthetic side by side.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    L = real.shape[1]
    hours = np.arange(L) / steps_per_hour

    r_mean, r_std = real.mean(0), real.std(0)
    s_mean, s_std = synthetic.mean(0), synthetic.std(0)

    ax.fill_between(hours, r_mean - r_std, r_mean + r_std, alpha=0.25, color="steelblue")
    ax.fill_between(hours, s_mean - s_std, s_mean + s_std, alpha=0.25, color="coral")
    ax.plot(hours, r_mean, color="steelblue", linewidth=2, label="real mean")
    ax.plot(hours, s_mean, color="coral",     linewidth=2, label="synthetic mean")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised consumption")
    ax.set_title(f"Envelope comparison {label}")
    ax.legend()


# ---------------------------------------------------------------------------
# Convenience: run all metrics for a (cluster, day_type) group
# ---------------------------------------------------------------------------

def run_all_metrics(
    real: np.ndarray,
    synthetic: np.ndarray,
    label: str = "",
    figsize: Tuple[int, int] = (18, 12),
) -> dict:
    """
    Run all metrics and produce a summary figure.
    Returns dict with scalar values.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax_acf  = fig.add_subplot(gs[0, 0])
    ax_env  = fig.add_subplot(gs[0, 1:])
    ax_kde  = [fig.add_subplot(gs[1, i]) for i in range(3)]

    acf_dist = acf_compare(real, synthetic, nlags=real.shape[1] - 1, ax=ax_acf, label=label)
    envelope_plot(real, synthetic, ax=ax_env, label=label)
    marginal_kde(real, synthetic, time_bins=3, ax=ax_kde, label=label)

    crps = crps_score(real, synthetic)
    disc = discriminative_score(real, synthetic)

    fig.suptitle(
        f"{label} | ACF L2={acf_dist:.3f} | CRPS={crps:.4f} | Discriminative acc={disc:.3f}",
        fontsize=12,
    )
    plt.tight_layout()

    return {"acf_l2": acf_dist, "crps": crps, "discriminative_acc": disc}
