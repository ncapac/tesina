"""
metrics.py
----------
Evaluation metrics for comparing real vs. synthetic energy load profiles.

Metrics
-------
  acf_compare              — ACF/PACF comparison (plots + scalar L2 distance)
  marginal_kde             — KDE overlay per time bin (peak / shoulder / night)
  crps_score               — Continuous Ranked Probability Score (probabilistic quality)
  spectral_frechet_distance — Fréchet distance in FFT magnitude space (12 harmonics)
  envelope_plot            — Mean ± std envelope comparison

References
----------
  CRPS:  eq. 57 in arXiv 2507.14507 (Su et al., 2025)
  Spectral Fréchet: Fréchet inception distance applied to FFT spectra
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    import pandas as pd


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
# Spectral Fréchet Distance
# ---------------------------------------------------------------------------

def spectral_frechet_distance(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    n_harmonics: int = 12,  # number of FFT harmonics to use (skip DC bin 0)
) -> float:
    """
    Fréchet distance computed in the FFT magnitude spectrum space.

    For each sample x of length L, compute the magnitude spectrum
    S(x) = |FFT(x)|[1 : n_harmonics+1]  (skipping the DC component).
    Fit a multivariate Gaussian N(μ, Σ) on the real spectra and on the
    synthetic spectra, then return the Fréchet distance:

        FD = ||μ_r - μ_s||² + Tr(Σ_r + Σ_s - 2 * sqrt(Σ_r @ Σ_s))

    A value near 0 means the two distributions overlap in spectrum space.
    Replacing the discriminative score with this metric avoids training
    a classifier and gives a differentiable geometry-aware distance.

    Requires scipy.
    """
    from scipy.linalg import sqrtm

    def _spectra(x: np.ndarray) -> np.ndarray:
        """(N, n_harmonics) magnitude spectra, DC excluded."""
        mag = np.abs(np.fft.rfft(x, axis=-1))
        return mag[:, 1 : n_harmonics + 1].astype(np.float64)

    sr = _spectra(real)       # (N_r, H)
    ss = _spectra(synthetic)  # (N_s, H)

    mu_r, mu_s = sr.mean(0), ss.mean(0)                       # (H,)
    cov_r = np.cov(sr.T) + 1e-6 * np.eye(n_harmonics)        # (H, H)
    cov_s = np.cov(ss.T) + 1e-6 * np.eye(n_harmonics)

    diff   = mu_r - mu_s
    sq     = sqrtm(cov_r @ cov_s)
    if np.iscomplexobj(sq):
        sq = sq.real

    frechet = float(diff @ diff + np.trace(cov_r + cov_s - 2.0 * sq))
    return max(frechet, 0.0)   # numerical safety: clamp to ≥0


def spectral_power_plot(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
    ax: Optional[plt.Axes] = None,
    label: str = "",
) -> None:
    """
    Plot mean FFT magnitude spectrum for real vs. synthetic.
    Frequency axis in cycles-per-day (for 24h hourly sequences: 0.5, 1, 1.5, …).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    L = real.shape[1]
    freqs = np.fft.rfftfreq(L, d=1.0 / L)   # cycles per day for hourly data

    mean_r = np.abs(np.fft.rfft(real,      axis=-1)).mean(0)  # (L//2+1,)
    mean_s = np.abs(np.fft.rfft(synthetic, axis=-1)).mean(0)

    ax.plot(freqs, mean_r, color="steelblue", linewidth=2, label="real")
    ax.plot(freqs, mean_s, color="coral",     linewidth=2, linestyle="--", label="synthetic")
    ax.set_xlabel("Frequency (cycles / day)")
    ax.set_ylabel("|FFT| (mean)")
    ax.set_title(f"Spectral power {label}")
    ax.legend(fontsize=8)


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
    dict  with keys: acf_l2, crps, spectral_frechet
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
    spectral_fd = spectral_frechet_distance(real, synthetic)

    fig.suptitle(
        f"{label}  |  ACF L2={acf_dist:.3f}  |  CRPS={crps:.4f}  |  Spectral FD={spectral_fd:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if show:
        plt.show()
    elif not return_fig:
        plt.close(fig)

    scalars = {"acf_l2": acf_dist, "crps": crps, "spectral_frechet": spectral_fd}
    return (scalars, fig) if return_fig else scalars


# ---------------------------------------------------------------------------
# Marginal Wasserstein distance (1-D, per timestep, then averaged)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Diversity metrics (novelty, coverage, intra-diversity)
# ---------------------------------------------------------------------------

def _global_epsilon(x_train: np.ndarray, subsample: int = 500, percentile: float = 25.0) -> float:
    """
    Compute a scale-adaptive distance threshold from the training set.

    Draws ``subsample`` rows from *x_train* and returns the ``percentile``-th
    percentile of the within-sample nearest-neighbour distances.  Using a
    small percentile (e.g. 25) produces a tight threshold: two profiles must
    be very similar for one to be considered a "copy" of the other.

    Parameters
    ----------
    x_train   : (N_train, L) array in any consistent unit (e.g. Watts).
    subsample : cap on the number of rows used for the computation to keep
                it O(subsample²) regardless of training-set size.
    percentile: which percentile of within-sample NND to return (0–100).
    """
    from scipy.spatial.distance import cdist

    n = len(x_train)
    if n < 2:
        return 1e-6
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(subsample, n), replace=False)
    sample = x_train[idx].astype(np.float64)
    D = cdist(sample, sample)
    np.fill_diagonal(D, np.inf)
    nnd = np.min(D, axis=1)
    return float(np.percentile(nnd, percentile))


def novelty_score(
    syn: np.ndarray,      # (N_syn, L)
    x_train: np.ndarray,  # (N_train, L)
    epsilon: float,
) -> float:
    """
    Fraction of synthetic samples that are not a near-copy of any training sample.

    ``novelty = mean(min_dist(syn → x_train) > epsilon)``

    A value near 1 means the generator is producing genuinely new profiles;
    near 0 means it is memorising the training set (as the historical baseline
    always does by construction).

    Parameters
    ----------
    syn     : (N_syn, L) synthetic profiles.
    x_train : (N_train, L) training profiles used as the reference set.
    epsilon : distance threshold; use :func:`_global_epsilon` for a
              data-adaptive value.
    """
    from scipy.spatial.distance import cdist

    D = cdist(np.asarray(syn, dtype=np.float64),
              np.asarray(x_train, dtype=np.float64))  # (N_syn, N_train)
    return float(np.mean(np.min(D, axis=1) > epsilon))


def coverage_score(
    real: np.ndarray,   # (N_real, L)
    syn: np.ndarray,    # (N_syn,  L)
    epsilon: float,
) -> float:
    """
    Fraction of real profiles that are covered by at least one synthetic sample.

    ``coverage = mean(min_dist(real → syn) < epsilon)``

    Complements novelty: high coverage + high novelty means the generator
    both covers the real distribution and produces samples beyond it.

    Parameters
    ----------
    real    : (N_real, L) validation profiles for this condition.
    syn     : (N_syn, L) synthetic profiles for this condition.
    epsilon : same threshold as in :func:`novelty_score`.
    """
    from scipy.spatial.distance import cdist

    D = cdist(np.asarray(real, dtype=np.float64),
              np.asarray(syn, dtype=np.float64))       # (N_real, N_syn)
    return float(np.mean(np.min(D, axis=1) < epsilon))


def intra_diversity_score(syn: np.ndarray) -> float:
    """
    Mean nearest-neighbour distance within the synthetic set.

    Higher values indicate a more spread-out synthetic ensemble; near zero
    signals mode collapse (all generated samples are identical or very similar).
    Requires no epsilon and is independent of the training or validation set.

    Parameters
    ----------
    syn : (N_syn, L) synthetic profiles.
    """
    from scipy.spatial.distance import cdist

    n = len(syn)
    if n < 2:
        return 0.0
    D = cdist(np.asarray(syn, dtype=np.float64),
              np.asarray(syn, dtype=np.float64))       # (N, N)
    np.fill_diagonal(D, np.inf)
    return float(np.mean(np.min(D, axis=1)))


def marginal_wasserstein(
    real: np.ndarray,       # (N_real, L)
    synthetic: np.ndarray,  # (N_syn,  L)
) -> float:
    """
    Mean 1-D Wasserstein-1 distance averaged over all L timesteps.

    At each timestep h, W1(real[:, h], syn[:, h]) is computed exactly via
    the sorted-ranks formula (O(N log N), no binning):
        W1 = mean |sort(real_h) - sort(syn_h)|
    where both sorted sequences are interpolated to the same length.

    A value near 0 means the marginal distributions match at every hour.
    Complementary to ACF L2 (temporal structure) and CRPS (sharpness).
    """
    L = real.shape[1]
    w1_per_step = []
    for h in range(L):
        r = np.sort(real[:, h])
        s = np.sort(synthetic[:, h])
        # Interpolate the shorter one to match the longer
        if len(r) != len(s):
            n_out = max(len(r), len(s))
            r = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(r)), r)
            s = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(s)), s)
        w1_per_step.append(float(np.mean(np.abs(r - s))))
    return float(np.mean(w1_per_step))


def sample_condition_batch(
    condition_rows: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Resample full conditioning rows from the empirical validation distribution.

    This preserves the month/day-of-week mixture inside a broader
    (cluster_id, day_type) slice, so evaluation compares like with like.
    """
    rows = np.asarray(condition_rows, dtype=np.int32)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError(f"Expected conditioning array of shape (N, ≥2); got {rows.shape}")
    if len(rows) == 0:
        raise ValueError("Cannot sample from an empty conditioning array")

    rng = np.random.default_rng(seed)
    replace = len(rows) < n_samples
    indices = rng.choice(len(rows), size=n_samples, replace=replace)
    batch = rows[indices].astype(np.int32, copy=False)
    assert batch.shape[0] == n_samples
    return batch


# ---------------------------------------------------------------------------
# Multi-model comparison framework
# ---------------------------------------------------------------------------

def compare_models(
    models_dict: dict,          # {model_name: sample_generator_fn}
    real_data: np.ndarray,      # (N_real, L)  real windows (all conditions pooled or per-condition)
    conditions: np.ndarray,     # (N_real, ≥2) int32 conditioning vectors (first two cols: cluster_id, day_type)
    n_samples: int = 200,
    unique_conditions: Optional[list] = None,
    guidance_scale: float = 1.5,
    n_ddim_steps: int = 50,
    seed: int = 0,
    show_figs: bool = False,
    verbose: bool = True,
    c_continuous: Optional[np.ndarray] = None,
    train_data: Optional[np.ndarray] = None,
) -> Tuple["pd.DataFrame", dict]:
    """
    Unified multi-model comparison across all (cluster × day_type) conditions.

    Condition groups with fewer than 10 real validation profiles are skipped.
    This prevents unstable distributional metrics on tiny empirical pools; all
    models are then compared on the same retained condition groups.

    Parameters
    ----------
    models_dict : dict mapping model name → callable with signature:
                    generate(c_batch: np.ndarray, key) -> np.ndarray (N, L)
                  where c_batch is (N, ≥2) int32 conditioning array.
    real_data   : (N_real, L) array of real windows in Watts.
    conditions  : (N_real, ≥2) int32 conditioning vectors matching real_data rows.
                  First column = cluster_id, second = day_type.
    n_samples   : number of synthetic samples to generate per condition per model.
    unique_conditions : list of (cluster_id, day_type) tuples to evaluate.
                        If None, all unique combinations in `conditions` are used.
    guidance_scale : CFG guidance scale passed to generator functions.
    n_ddim_steps   : DDIM/Euler steps for samplers.
    seed           : base random seed.
    show_figs      : if True, display a per-condition metric figure for each model.
    verbose        : print progress.
    train_data  : (N_train, L) training profiles in Watts.  When provided,
                  three diversity metrics are added to every row:
                  - ``novelty``       — fraction of syn samples not near any training sample.
                  - ``coverage``      — fraction of real val samples covered by syn.
                  - ``intra_diversity`` — mean nearest-neighbour distance within syn.
                  A global epsilon is computed once from *train_data* using
                  :func:`_global_epsilon` (25th-percentile within-training NND on a
                  500-sample draw) and reused for every condition.

    Returns
    -------
    summary_df : pd.DataFrame  rows = (model, cluster, day_type),
                                cols = acf_l2, crps, spectral_frechet, wasserstein
                                [+ novelty, coverage, intra_diversity when train_data given]
    figs_dict  : {f"{model}_{condition_label}": matplotlib.Figure}
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for compare_models")

    if unique_conditions is None:
        # Extract unique (cluster_id, day_type) pairs
        unique_conditions = list({
            (int(c[0]), int(c[1])) for c in conditions
        })
        unique_conditions.sort()

    # Pre-compute global epsilon from the full training set so the novelty /
    # coverage threshold is consistent across all conditions and models.
    epsilon: Optional[float] = None
    if train_data is not None:
        epsilon = _global_epsilon(np.asarray(train_data, dtype=np.float64))
        if verbose:
            print(f"[diversity] global epsilon = {epsilon:.2f}  "
                  f"(25th-pct NND on {min(500, len(train_data))}-sample draw from train_data)")

    rows = []
    figs_dict = {}
    rng = np.random.default_rng(seed)

    for model_name, generate_fn in models_dict.items():
        if verbose:
            print(f"\n── Model: {model_name} ──")
        for cid, dt in unique_conditions:
            cond_label = f"cluster{cid}_{'weekday' if dt == 0 else 'weekend'}"
            if verbose:
                print(f"  {cond_label} ... ", end="", flush=True)

            # Real windows for this condition
            mask = (conditions[:, 0] == cid) & (conditions[:, 1] == dt)
            real_cond = real_data[mask]
            cond_rows = conditions[mask]
            if len(real_cond) < 10:
                if verbose:
                    print(f"skipped (only {len(real_cond)} real samples)")
                continue

            c_batch = sample_condition_batch(
                cond_rows,
                n_samples,
                seed=int(rng.integers(0, 2**31)),
            )
            key = rng.integers(0, 2**31)

            if c_continuous is not None:
                cc_pool = c_continuous[mask]
                # Resample c_continuous rows in the same way (independent draw is fine
                # since the empirical mix is already represented by cond_rows)
                idx = np.random.default_rng(int(rng.integers(0, 2**31))).choice(
                    len(cc_pool), size=n_samples, replace=len(cc_pool) < n_samples
                )
                cc_batch = np.asarray(cc_pool[idx], dtype=np.float32)
                synth_cond = generate_fn(c_batch, cc_batch, key)
            else:
                synth_cond = generate_fn(c_batch, key)          # (n_samples, L)
            synth_cond = np.array(synth_cond, dtype=np.float32)

            # Fidelity metrics
            acf_l2    = acf_compare(real_cond, synth_cond)
            crps      = crps_score(real_cond, synth_cond)
            spectral  = spectral_frechet_distance(real_cond, synth_cond)
            wass      = marginal_wasserstein(real_cond, synth_cond)

            row: dict = {
                "model":            model_name,
                "cluster":          cid,
                "day_type":         "weekday" if dt == 0 else "weekend",
                "condition":        cond_label,
                "n_real":           len(real_cond),
                "n_synthetic":      n_samples,
                "n_empirical_meta": int(len(np.unique(cond_rows[:, 2:], axis=0))),
                "acf_l2":           acf_l2,
                "crps":             crps,
                "spectral_frechet": spectral,
                "wasserstein":      wass,
            }

            # Diversity metrics (only when training reference set is available)
            if train_data is not None and epsilon is not None:
                nov  = novelty_score(synth_cond, train_data, epsilon)
                cov  = coverage_score(real_cond, synth_cond, epsilon)
                idiv = intra_diversity_score(synth_cond)
                row["novelty"]          = nov
                row["coverage"]         = cov
                row["intra_diversity"]  = idiv

            if verbose:
                msg = (f"spectral_fd={spectral:.3f}  crps={crps:.4f}  "
                       f"acf_l2={acf_l2:.4f}  wass={wass:.4f}")
                if train_data is not None:
                    msg += (f"  novelty={row['novelty']:.3f}  "
                            f"coverage={row['coverage']:.3f}  "
                            f"intra_div={row['intra_diversity']:.1f}")
                print(msg)

            rows.append(row)

            if show_figs:
                scalars, fig = run_all_metrics(
                    real_cond, synth_cond,
                    label=f"{model_name} | {cond_label}",
                    show=True, return_fig=True,
                )
                figs_dict[f"{model_name}_{cond_label}"] = fig

    import pandas as pd
    summary_df = pd.DataFrame(rows)
    return summary_df, figs_dict


def bootstrap_aggregate_metrics(
    summary_df: "pd.DataFrame",
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    metric_cols: Optional[list[str]] = None,
    weight_col: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Bootstrap aggregate model metrics over paired condition groups.

    Each bootstrap draw samples condition groups with replacement and keeps all
    model rows for each sampled condition together. This preserves the paired
    comparison structure used by notebook 05: every model is re-aggregated over
    the same sampled set of conditions.

    Parameters
    ----------
    summary_df : DataFrame
        Long comparison table from :func:`compare_models` with at least
        ``model`` plus either ``condition`` or ``cluster``/``day_type``.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for deterministic resampling.
    confidence : float
        Central confidence mass, e.g. 0.95 for 2.5/97.5 percentiles.
    metric_cols : list[str] | None
        Metric columns to aggregate. Defaults to the standard comparison
        metrics present in ``summary_df``.
    weight_col : str | None
        Optional non-negative weight column such as ``n_real``. Leave as None
        to reproduce the unweighted scorecard definition.

    Returns
    -------
    pd.DataFrame
        Columns: model, metric, mean, ci_lower, ci_upper, n_conditions,
        n_bootstrap, confidence, weight_col.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for bootstrap_aggregate_metrics")

    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    if "model" not in summary_df.columns:
        raise ValueError("summary_df must contain a 'model' column")

    standard_metrics = ["acf_l2", "wasserstein", "crps", "spectral_frechet"]
    if metric_cols is None:
        metric_cols = [col for col in standard_metrics if col in summary_df.columns]
    if not metric_cols:
        raise ValueError("No metric columns available for bootstrap aggregation")

    missing_metrics = [col for col in metric_cols if col not in summary_df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")
    if weight_col is not None and weight_col not in summary_df.columns:
        raise ValueError(f"Missing weight column: {weight_col}")

    df = summary_df.copy()
    condition_col = "condition"
    if condition_col not in df.columns:
        if {"cluster", "day_type"}.issubset(df.columns):
            condition_col = "__condition__"
            df[condition_col] = list(zip(df["cluster"], df["day_type"]))
        else:
            raise ValueError("summary_df must contain 'condition' or both 'cluster' and 'day_type'")

    agg_cols = ["model", condition_col, *metric_cols]
    if weight_col is not None:
        agg_cols.append(weight_col)
    df = df[agg_cols].copy()
    df = df.dropna(subset=metric_cols)
    if df.empty:
        raise ValueError("summary_df has no finite metric rows to aggregate")

    models = sorted(df["model"].unique())
    conditions = sorted(df[condition_col].unique())
    by_condition = {
        condition: df.loc[df[condition_col] == condition]
        for condition in conditions
    }

    def aggregate(sampled_conditions: list) -> dict[tuple[str, str], float]:
        sampled = pd.concat([by_condition[condition] for condition in sampled_conditions], ignore_index=True)
        values: dict[tuple[str, str], float] = {}
        for model_name, group in sampled.groupby("model"):
            for metric in metric_cols:
                if weight_col is None:
                    values[(model_name, metric)] = float(group[metric].mean())
                else:
                    weights = np.asarray(group[weight_col], dtype=np.float64)
                    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
                        raise ValueError(f"weight_col '{weight_col}' must contain finite non-negative weights")
                    values[(model_name, metric)] = float(np.average(group[metric], weights=weights))
        return values

    point = aggregate(conditions)
    rng = np.random.default_rng(seed)
    boot_values = {key: [] for key in point}
    for _ in range(n_bootstrap):
        sampled_conditions = rng.choice(conditions, size=len(conditions), replace=True).tolist()
        boot = aggregate(sampled_conditions)
        for key in boot_values:
            boot_values[key].append(boot[key])

    alpha = (1.0 - confidence) / 2.0
    rows = []
    for model_name in models:
        for metric in metric_cols:
            key = (model_name, metric)
            if key not in point:
                continue
            values = np.asarray(boot_values[key], dtype=np.float64)
            rows.append({
                "model": model_name,
                "metric": metric,
                "mean": point[key],
                "ci_lower": float(np.quantile(values, alpha)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                "n_conditions": len(conditions),
                "n_bootstrap": n_bootstrap,
                "confidence": confidence,
                "weight_col": "unweighted" if weight_col is None else weight_col,
            })

    return pd.DataFrame(rows)
