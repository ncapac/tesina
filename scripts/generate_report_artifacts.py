from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import make_daily_instances, shape_normalize, split_mask_by_meter
from src.data.loader import compute_temp_stats, load_rolle_data, normalize_temp
from src.runtime_paths import DATA_DIR, OUTPUT_DIR


REPORT_DIR = OUTPUT_DIR / "report"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

HOURS = np.arange(24)
SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
DAY_TYPE_NAMES = {0: "Weekday", 1: "Weekend"}
MODEL_NAMES = {
    "historical": "Historical",
    "diffusion": "Diffusion",
    "rf": "Rectified flow",
    "cvae": "CVAE",
}
MODEL_ORDER = ["cvae", "historical", "diffusion", "rf"]
COLORS = {
    "cvae": "#2a9d8f",
    "historical": "#5f6c7b",
    "diffusion": "#457b9d",
    "rf": "#e76f51",
}


def savefig(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path.relative_to(REPO_ROOT)}")


def tidy(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#d8dee4", linewidth=0.6, alpha=0.65)


def load_report_instances() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    power_hourly, temp_daily = load_rolle_data(DATA_DIR)
    temp_normed = normalize_temp(temp_daily, compute_temp_stats(temp_daily))
    profiles, c_disc, c_cont, dates, meter_ids = make_daily_instances(power_hourly, temp_normed)
    profiles_norm, log_mean, log_std = shape_normalize(profiles)

    meta = pd.DataFrame(
        {
            "meter_id": meter_ids.astype(int),
            "date": pd.to_datetime(dates),
            "day_type": c_disc[:, 1].astype(int),
            "season": c_disc[:, 2].astype(int),
            "temp_normed": c_cont[:, 0].astype(float),
            "log_mean": log_mean.astype(float),
            "log_std": log_std.astype(float),
        }
    )
    clusters = pd.read_csv(DATA_DIR / "clusters.csv", parse_dates=["date"])
    meta = meta.merge(clusters, on=["meter_id", "date"], how="left")
    if meta["cluster_id"].isna().any():
        missing = int(meta["cluster_id"].isna().sum())
        raise RuntimeError(f"Missing cluster assignments for {missing} instances")
    meta["cluster_id"] = meta["cluster_id"].astype(int)
    return profiles, profiles_norm, meta


def plot_data_overview(profiles: np.ndarray, meta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))

    ax = axes[0, 0]
    for day_type, group in meta.groupby("day_type"):
        idx = group.index.to_numpy()
        mean = profiles[idx].mean(axis=0)
        std = profiles[idx].std(axis=0)
        ax.plot(HOURS, mean, linewidth=2.2, label=DAY_TYPE_NAMES[int(day_type)])
        ax.fill_between(HOURS, mean - std, mean + std, alpha=0.18)
    ax.set(title="Mean daily profile by day type", xlabel="Hour", ylabel="Power [W]")
    ax.legend(frameon=False)
    tidy(ax)

    ax = axes[0, 1]
    for season, group in meta.groupby("season"):
        idx = group.index.to_numpy()
        mean = profiles[idx].mean(axis=0)
        std = profiles[idx].std(axis=0)
        ax.plot(HOURS, mean, linewidth=2.0, label=SEASON_NAMES[int(season)])
        ax.fill_between(HOURS, mean - std, mean + std, alpha=0.14)
    ax.set(title="Mean daily profile by season", xlabel="Hour", ylabel="Power [W]")
    ax.legend(frameon=False, ncol=2)
    tidy(ax)

    ax = axes[1, 0]
    daily_mean = profiles.mean(axis=1)
    ax.hist(daily_mean, bins=40, color="#457b9d", alpha=0.82)
    ax.set(title="Daily mean load spread", xlabel="Daily mean power [W]", ylabel="Daily instances")
    tidy(ax)

    ax = axes[1, 1]
    daily_total = profiles.sum(axis=1)
    sample_idx = meta.sample(min(1800, len(meta)), random_state=7).index.to_numpy()
    ax.scatter(
        meta.loc[sample_idx, "temp_normed"],
        daily_total[sample_idx],
        s=10,
        alpha=0.22,
        color="#2a9d8f",
        linewidth=0,
    )
    ax.set(
        title="Temperature covariate and daily total",
        xlabel="Normalized daily mean temperature",
        ylabel="Daily total [Wh]",
    )
    tidy(ax)

    fig.suptitle("Rolle smart-meter daily profiles", fontsize=16, y=1.02)
    fig.tight_layout()
    savefig(fig, "01_data_overview.png")


def plot_shape_normalization(profiles: np.ndarray, profiles_norm: np.ndarray) -> None:
    idx = np.random.default_rng(42).choice(len(profiles), size=90, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)

    for row in profiles[idx]:
        axes[0].plot(HOURS, row, color="#5f6c7b", alpha=0.14, linewidth=0.9)
    axes[0].plot(HOURS, profiles[idx].mean(axis=0), color="#111827", linewidth=2.5, label="sample mean")
    axes[0].set(title="Raw profiles mix scale and shape", xlabel="Hour", ylabel="Power [W]")
    axes[0].legend(frameon=False)
    tidy(axes[0])

    for row in profiles_norm[idx]:
        axes[1].plot(HOURS, row, color="#2a9d8f", alpha=0.16, linewidth=0.9)
    axes[1].plot(HOURS, profiles_norm[idx].mean(axis=0), color="#0f766e", linewidth=2.5, label="sample mean")
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set(title="Per-instance normalization exposes shape", xlabel="Hour", ylabel="Profile z-score")
    axes[1].legend(frameon=False)
    tidy(axes[1])

    fig.tight_layout()
    savefig(fig, "02_shape_normalization.png")


def plot_cluster_profiles(profiles: np.ndarray, profiles_norm: np.ndarray, meta: pd.DataFrame) -> None:
    clusters = sorted(meta["cluster_id"].unique())
    fig, axes = plt.subplots(2, len(clusters), figsize=(15, 6), sharex=True)

    for col, cluster_id in enumerate(clusters):
        idx = meta.index[meta["cluster_id"] == cluster_id].to_numpy()
        for row, data, ylabel, color, ylim in [
            (0, profiles_norm, "Shape z-score", "#2a9d8f", (-2.6, 2.6)),
            (1, profiles, "Power [W]", "#457b9d", None),
        ]:
            ax = axes[row, col]
            mean = data[idx].mean(axis=0)
            std = data[idx].std(axis=0)
            ax.plot(HOURS, mean, color=color, linewidth=2.1)
            ax.fill_between(HOURS, mean - std, mean + std, color=color, alpha=0.2)
            if row == 0:
                ax.axhline(0, color="#111827", linewidth=0.7)
                ax.set_title(f"Cluster {cluster_id}\nn={len(idx)}")
            else:
                ax.set_xlabel("Hour")
            if col == 0:
                ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            tidy(ax)

    fig.suptitle("Shape clusters used as discrete conditioning", fontsize=16, y=1.04)
    fig.tight_layout()
    savefig(fig, "03_cluster_profiles.png")


def plot_cluster_method(profiles_norm: np.ndarray, meta: pd.DataFrame) -> None:
    pca = PCA(n_components=20, random_state=42)
    embeddings = pca.fit_transform(profiles_norm)
    ks = list(range(2, 11))
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings).inertia_ for k in ks]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    ax.plot(ks, inertias, marker="o", color="#457b9d", linewidth=2)
    ax.axvline(5, color="#e76f51", linestyle="--", linewidth=1.6, label="chosen k=5")
    ax.set(title="K-Means elbow on PCA embeddings", xlabel="Number of clusters", ylabel="Inertia")
    ax.legend(frameon=False)
    tidy(ax)

    ax = axes[0, 1]
    sample_idx = meta.sample(min(3500, len(meta)), random_state=3).index.to_numpy()
    scatter = ax.scatter(
        embeddings[sample_idx, 0],
        embeddings[sample_idx, 1],
        c=meta.loc[sample_idx, "cluster_id"],
        cmap="tab10",
        s=8,
        alpha=0.38,
        linewidth=0,
    )
    ax.legend(*scatter.legend_elements(), title="Cluster", frameon=False, fontsize=8)
    ax.set(title="PCA projection colored by cluster", xlabel="PC1", ylabel="PC2")
    tidy(ax)

    day_cross = pd.crosstab(meta["cluster_id"], meta["day_type"], normalize="index").rename(columns=DAY_TYPE_NAMES)
    season_cross = pd.crosstab(meta["cluster_id"], meta["season"], normalize="index").rename(columns=SEASON_NAMES)
    sns.heatmap(day_cross, annot=True, fmt=".0%", cmap="Blues", cbar=False, ax=axes[1, 0])
    axes[1, 0].set(title="Cluster composition by day type", xlabel="", ylabel="Cluster")
    sns.heatmap(season_cross, annot=True, fmt=".0%", cmap="Greens", cbar=False, ax=axes[1, 1])
    axes[1, 1].set(title="Cluster composition by season", xlabel="", ylabel="Cluster")

    fig.tight_layout()
    savefig(fig, "04_cluster_method.png")


def plot_lda_support(profiles_norm: np.ndarray, meta: pd.DataFrame) -> None:
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_projection = lda.fit_transform(profiles_norm, meta["cluster_id"].to_numpy())
    val_mask = split_mask_by_meter(
        meta["meter_id"].to_numpy(),
        n_meters=int(meta["meter_id"].nunique()),
        val_fraction=0.15,
        seed=42,
    )
    val = meta.loc[val_mask].copy()
    support = pd.crosstab(val["cluster_id"], val["day_type"])
    support = support.reindex(index=sorted(meta["cluster_id"].unique()), columns=[0, 1], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    sample_idx = meta.sample(min(4500, len(meta)), random_state=11).index.to_numpy()
    scatter = ax.scatter(
        lda_projection[sample_idx, 0],
        lda_projection[sample_idx, 1],
        c=meta.loc[sample_idx, "cluster_id"],
        cmap="tab10",
        s=8,
        alpha=0.35,
        linewidth=0,
    )
    centroids = (
        pd.DataFrame(lda_projection, columns=["LD1", "LD2"])
        .join(meta["cluster_id"])
        .groupby("cluster_id")
        .mean()
    )
    ax.scatter(centroids["LD1"], centroids["LD2"], marker="x", s=110, color="#111827", linewidth=2.5)
    for cluster_id, row in centroids.iterrows():
        ax.text(row["LD1"], row["LD2"], f" {cluster_id}", fontsize=9, weight="bold")
    ax.legend(*scatter.legend_elements(), title="Cluster", frameon=False, fontsize=8)
    ax.set(title="LDA projection separates cluster modes", xlabel="LD1", ylabel="LD2")
    tidy(ax)

    ax = axes[1]
    x = np.arange(len(support.index))
    width = 0.38
    ax.bar(x - width / 2, support[0], width=width, label="weekday", color="#457b9d")
    ax.bar(x + width / 2, support[1], width=width, label="weekend", color="#e76f51")
    ax.axhline(10, color="#111827", linestyle="--", linewidth=1.2, label="minimum support")
    ax.set_xticks(x, [str(i) for i in support.index])
    ax.set(title="Seed-42 validation support by cluster", xlabel="Cluster", ylabel="Validation profiles")
    ax.legend(frameon=False)
    tidy(ax)

    fig.tight_layout()
    savefig(fig, "05_lda_and_validation_support.png")


def plot_model_results() -> None:
    scorecard = pd.read_csv(OUTPUT_DIR / "05" / "results" / "model_scorecard.csv")
    scorecard["model_label"] = scorecard["model"].map(MODEL_NAMES)
    scorecard["model"] = pd.Categorical(scorecard["model"], categories=MODEL_ORDER, ordered=True)
    scorecard = scorecard.sort_values("model")
    metrics = ["acf_l2", "wasserstein", "crps", "spectral_frechet"]
    titles = ["ACF L2", "Wasserstein", "CRPS", "Spectral Frechet"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        bars = ax.bar(
            scorecard["model_label"],
            scorecard[metric],
            color=[COLORS[m] for m in scorecard["model"].astype(str)],
        )
        best = scorecard[metric].min()
        for bar, value in zip(bars, scorecard[metric]):
            label = f"{value:.2f}" if value >= 1 else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + best * 0.04, label, ha="center", fontsize=8)
        ax.set(title=title, ylabel="Lower is better")
        ax.tick_params(axis="x", rotation=18)
        tidy(ax)
    fig.suptitle("Seed-42 aggregate fidelity scorecard", fontsize=16, y=1.02)
    fig.tight_layout()
    savefig(fig, "06_model_scorecard.png")


def plot_efficiency_robustness() -> None:
    ratio = pd.read_csv(OUTPUT_DIR / "05" / "results" / "metric_ratio_to_historical.csv")
    ratio = ratio[ratio["model"].isin(MODEL_ORDER)].copy()
    ratio["model_label"] = ratio["model"].map(MODEL_NAMES)
    ratio["model"] = pd.Categorical(ratio["model"], categories=MODEL_ORDER, ordered=True)
    ratio = ratio.sort_values("model")
    ratio_table = ratio.set_index("model_label")[["acf_l2", "wasserstein", "crps", "spectral_frechet"]]

    training = pd.read_csv(OUTPUT_DIR / "05" / "results" / "training_summary_table.csv")
    scorecard = pd.read_csv(OUTPUT_DIR / "05" / "results" / "model_scorecard.csv")
    efficiency = scorecard.merge(training[["model", "params_M"]], on="model", how="left")
    efficiency["model_label"] = efficiency["model"].map(MODEL_NAMES)

    robustness = pd.read_csv(OUTPUT_DIR / "robustness" / "results" / "model_rank_stability.csv")
    robustness["model_label"] = robustness["model"].map(MODEL_NAMES)
    robustness["model"] = pd.Categorical(robustness["model"], categories=MODEL_ORDER, ordered=True)
    robustness = robustness.sort_values("model")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    sns.heatmap(ratio_table, annot=True, fmt=".2f", cmap="RdYlGn_r", center=1.0, ax=axes[0])
    axes[0].set(title="Metric ratio to historical", xlabel="", ylabel="")

    ax = axes[1]
    for _, row in efficiency.iterrows():
        ax.scatter(row["params_M"], row["mean_rank"], s=150, color=COLORS.get(row["model"], "#5f6c7b"))
        ax.text(row["params_M"] + 0.025, row["mean_rank"], row["model_label"], va="center", fontsize=9)
    ax.set(title="Quality versus model size", xlabel="Parameters [M]", ylabel="Mean rank")
    ax.invert_yaxis()
    tidy(ax)

    ax = axes[2]
    ax.bar(
        robustness["model_label"],
        robustness["mean_rank_mean"],
        yerr=robustness["mean_rank_std"].fillna(0),
        capsize=4,
        color=[COLORS[m] for m in robustness["model"].astype(str)],
    )
    ax.set(title="Two-split rank stability", ylabel="Mean rank +/- std")
    ax.tick_params(axis="x", rotation=18)
    tidy(ax)

    fig.tight_layout()
    savefig(fig, "07_ratios_efficiency_robustness.png")


def plot_temperature_dependence() -> None:
    path = OUTPUT_DIR / "04" / "results" / "partial_dependence_temp.csv"
    if not path.exists():
        print("missing partial_dependence_temp.csv; skipping temperature figure")
        return
    pd_temp = pd.read_csv(path)
    syn = pd_temp[pd_temp["quantile"].astype(str).str.startswith("0")].copy()
    real = pd_temp[pd_temp["quantile"].astype(str).str.startswith("real_bin")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.errorbar(
        real["temp_normed"],
        real["syn_mean_total_Wh"],
        yerr=real["syn_std_total_Wh"],
        marker="o",
        color="#457b9d",
        linewidth=2,
        capsize=3,
        label="real bins",
    )
    ax.errorbar(
        syn["temp_normed"],
        syn["syn_mean_total_Wh"],
        yerr=syn["syn_std_total_Wh"],
        marker="s",
        color="#e76f51",
        linewidth=2,
        capsize=3,
        label="synthetic sweep",
    )
    ax.set(title="Daily-total response to temperature", xlabel="Normalized daily mean temperature", ylabel="Daily total [Wh]")
    ax.legend(frameon=False)
    tidy(ax)

    ax = axes[1]
    real_slope = np.polyfit(real["temp_normed"], real["syn_mean_total_Wh"], 1)[0]
    syn_slope = np.polyfit(syn["temp_normed"], syn["syn_mean_total_Wh"], 1)[0]
    ax.bar(["Real bins", "Synthetic sweep"], [real_slope, syn_slope], color=["#457b9d", "#e76f51"])
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set(title="Estimated slope", ylabel="Wh per normalized temp unit")
    tidy(ax)

    fig.tight_layout()
    savefig(fig, "08_temperature_dependence.png")


def write_report_tables(meta: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    meter_quality = pd.read_csv(OUTPUT_DIR / "01" / "results" / "meter_quality_summary.csv")
    data_summary = pd.DataFrame(
        [
            {"item": "Meters retained", "value": int(meter_quality["retain_decision"].eq("retain").sum())},
            {"item": "Daily instances", "value": len(meta)},
            {"item": "Calendar days per meter", "value": int(meter_quality["daily_instances"].median())},
            {"item": "Mean load range [W]", "value": f"{meter_quality['mean_w'].min():.1f}-{meter_quality['mean_w'].max():.1f}"},
            {"item": "Max zero fraction", "value": f"{meter_quality['zero_fraction'].max():.4f}"},
        ]
    )
    data_summary.to_csv(TABLE_DIR / "data_summary.csv", index=False)
    print(f"saved {(TABLE_DIR / 'data_summary.csv').relative_to(REPO_ROOT)}")

    for relative in [
        Path("05/results/model_scorecard.csv"),
        Path("05/results/training_summary_table.csv"),
        Path("05/results/skipped_conditions.csv"),
        Path("robustness/results/model_rank_stability.csv"),
    ]:
        src = OUTPUT_DIR / relative
        if src.exists():
            dst = TABLE_DIR / relative.name
            pd.read_csv(src).to_csv(dst, index=False)
            print(f"saved {dst.relative_to(REPO_ROOT)}")


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    profiles, profiles_norm, meta = load_report_instances()
    write_report_tables(meta)
    plot_data_overview(profiles, meta)
    plot_shape_normalization(profiles, profiles_norm)
    plot_cluster_profiles(profiles, profiles_norm, meta)
    plot_cluster_method(profiles_norm, meta)
    plot_lda_support(profiles_norm, meta)
    plot_model_results()
    plot_efficiency_robustness()
    plot_temperature_dependence()
    print(f"Wrote report artifacts under {REPORT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
