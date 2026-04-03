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
    Includes ±1.96/√N Bartlett confidence bands (95% CI for white noise).
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

        # Bartlett 95% confidence bands (white-noise null)
        n_obs = real.shape[1]
        ci = 1.96 / np.sqrt(n_obs)
        ax.axhline( ci, color="gray", linewidth=0.8, linestyle=":", label=f"±1.96/√n")
        ax.axhline(-ci, color="gray", linewidth=0.8, linestyle=":")

        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("ACF")
        ax.set_title(f"ACF {label}")
        ax.legend(fontsize=7)

    return float(np.linalg.norm(mean_real - mean_syn))


# ---------------------------------------------------------------------------
# Marginal KDE
# ---------------------------------------------------------------------------

def marginal_kde(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    ax: Optional[plt.Axes] = None,
    label: str = "",
) -> None:
    """
    Plot KDE of value distributions for 4 meaningful time-of-day bins.
    When L==24 (hourly): Night 00-05, Morning 06-11, Afternoon 12-17, Evening 18-23.
    Otherwise falls back to 4 equal-width bins.
    """
    L = real.shape[1]

    if L == 24:
        bins = [
            ("Night\n(00-05)",   0,  6),
            ("Morning\n(06-11)", 6,  12),
            ("Afternoon\n(12-17)", 12, 18),
            ("Evening\n(18-23)", 18, 24),
        ]
    else:
        bw = L // 4
        bins = [(f"Steps {i*bw}-{(i+1)*bw}", i*bw, (i+1)*bw) for i in range(4)]

    n_bins = len(bins)
    if ax is None:
        _, ax = plt.subplots(1, n_bins, figsize=(4 * n_bins, 3))
    elif not hasattr(ax, "__len__"):
        ax = [ax] * n_bins

    for i, (bin_label, start, end) in enumerate(bins):
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
        ax[i].set_title(bin_label, fontsize=9)
        ax[i].set_xlabel("Normalised consumption")
        if i == 0:
            ax[i].set_ylabel("Density")
        ax[i].legend(fontsize=8)


# ---------------------------------------------------------------------------
# Sample diversity plot
# ---------------------------------------------------------------------------

def sample_diversity_plot(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    ax: Optional[plt.Axes] = None,
    label: str = "",
    n_traces: int = 20,
) -> None:
    """
    Mean ±1σ envelope + individual sample traces.
    Reveals whether the model is diverse or collapsing to the mean.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    L   = real.shape[1]
    hrs = np.arange(L)
    rng = np.random.default_rng(0)

    # Thin sample traces
    idx_r = rng.choice(len(real),      min(n_traces, len(real)),      replace=False)
    idx_s = rng.choice(len(synthetic), min(n_traces, len(synthetic)), replace=False)
    for tr in real[idx_r]:
        ax.plot(hrs, tr, color="steelblue", alpha=0.12, linewidth=0.8)
    for tr in synthetic[idx_s]:
        ax.plot(hrs, tr, color="coral",     alpha=0.12, linewidth=0.8)

    # Mean ±σ envelopes
    r_mean, r_std = real.mean(0), real.std(0)
    s_mean, s_std = synthetic.mean(0), synthetic.std(0)
    ax.fill_between(hrs, r_mean - r_std, r_mean + r_std, alpha=0.25, color="steelblue")
    ax.fill_between(hrs, s_mean - s_std, s_mean + s_std, alpha=0.25, color="coral")
    ax.plot(hrs, r_mean, color="steelblue", linewidth=2,   label="real mean")
    ax.plot(hrs, s_mean, color="coral",     linewidth=2,   linestyle="--", label="synth mean")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised consumption")
    ax.set_title(f"Sample diversity {label}")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Pairwise hour correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
    label: str = "",
) -> None:
    """
    Side-by-side Pearson correlation matrices (L×L) for real and synthetic.
    Differences reveal whether temporal correlations are preserved.
    """
    import matplotlib.colors as mcolors
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))

    corr_r = np.corrcoef(real.T)        # (L, L)
    corr_s = np.corrcoef(synthetic.T)   # (L, L)

    vmin, vmax = -1.0, 1.0
    kw = dict(vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto")
    im0 = axes[0].imshow(corr_r, **kw)
    axes[0].set_title(f"Real corr {label}",      fontsize=9)
    im1 = axes[1].imshow(corr_s, **kw)
    axes[1].set_title(f"Synthetic corr {label}", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Hour"); ax.set_ylabel("Hour")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)


# ---------------------------------------------------------------------------
# Per-timestep standard deviation comparison
# ---------------------------------------------------------------------------

def per_timestep_stddev_plot(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    ax: Optional[plt.Axes] = None,
    label: str = "",
) -> None:
    """
    Plot real σ(t) vs synthetic σ(t) for every timestep.
    More intuitive than ACF for detecting heteroskedasticity mismatch:
    if the model is too flat it shows as suppressed σ during peak hours.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    hrs = np.arange(real.shape[1])
    ax.plot(hrs, real.std(0),      color="steelblue", linewidth=2, label="real σ")
    ax.plot(hrs, synthetic.std(0), color="coral",     linewidth=2, linestyle="--", label="synth σ")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("σ (normalised)")
    ax.set_title(f"Per-hour std deviation {label}")
    ax.legend(fontsize=8)


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
    figsize: Tuple[int, int] = (20, 16),
    show: bool = True,
    return_fig: bool = False,
) -> dict:
    """
    Run all metrics and produce a 4-row summary figure.

    Layout
    ------
    Row 0 : ACF (with 95% CI bands)  |  Sample diversity (mean±σ + traces)
    Row 1 : Marginal KDE — 4 meaningful hour-of-day bins
    Row 2 : Per-hour std deviation comparison  |  Pairwise correlation heatmaps
    Row 3 : (reserved for future extension)

    Parameters
    ----------
    show       : if False the figure is not displayed (batch/script mode).
    return_fig : if True the matplotlib Figure is returned alongside the
                 scalar dict, as a (dict, fig) tuple. Useful for notebook 05
                 where many figures need to be saved without being shown.

    Returns
    -------
    dict  with keys: acf_l2, crps, discriminative_acc
        or (dict, fig) if return_fig=True
    """
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

    ax_acf  = fig.add_subplot(gs[0, 0])
    ax_div  = fig.add_subplot(gs[0, 1:])
    ax_kde  = [fig.add_subplot(gs[1, i]) for i in range(4)]
    ax_std  = fig.add_subplot(gs[2, :2])
    ax_corr = [fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3])]

    acf_dist = acf_compare(real, synthetic, nlags=real.shape[1] - 1, ax=ax_acf, label=label)
    sample_diversity_plot(real, synthetic, ax=ax_div, label=label)
    marginal_kde(real, synthetic, ax=ax_kde, label=label)
    per_timestep_stddev_plot(real, synthetic, ax=ax_std, label=label)
    correlation_heatmap(real, synthetic, axes=ax_corr, label=label)

    crps = crps_score(real, synthetic)
    disc = discriminative_score(real, synthetic)

    fig.suptitle(
        f"{label}  |  ACF L2={acf_dist:.3f}  |  CRPS={crps:.4f}  |  Disc. acc={disc:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if show:
        plt.show()
    elif not return_fig:
        plt.close(fig)

    scalars = {"acf_l2": acf_dist, "crps": crps, "discriminative_acc": disc}
    return (scalars, fig) if return_fig else scalars
