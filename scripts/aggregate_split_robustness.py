#!/usr/bin/env python3
"""Aggregate meter-split robustness results across baseline and seed runs.

The script is intentionally tolerant of incomplete runs: it always includes the
existing seed-42 baseline under output/05/results when present, then adds every
completed output/robustness/seedXXX/05/results directory it can find.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RESULT_FILES = {
    "comparison_long": "comparison_long.csv",
    "model_scorecard": "model_scorecard.csv",
    "model_scorecard_with_temperature_response": "model_scorecard_with_temperature_response.csv",
    "metric_ratio_to_historical": "metric_ratio_to_historical.csv",
    "temperature_response_scorecard": "temperature_response_scorecard.csv",
    "training_summary_table": "training_summary_table.csv",
    "bootstrap_ci_aggregate": "bootstrap_ci_aggregate.csv",
    "bootstrap_ci_aggregate_weighted_n_real": "bootstrap_ci_aggregate_weighted_n_real.csv",
    "skipped_conditions": "skipped_conditions.csv",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _seed_from_tag(run_tag: str) -> int | None:
    match = re.fullmatch(r"seed(\d+)", run_tag)
    return int(match.group(1)) if match else None


def discover_result_dirs(output_dir: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    baseline = output_dir / "05" / "results"
    if baseline.exists():
        runs.append(
            {
                "meter_split_seed": 42,
                "run_tag": "seed042",
                "robustness_run": False,
                "result_dir": baseline,
            }
        )

    robustness_root = output_dir / "robustness"
    if robustness_root.exists():
        for result_dir in sorted(robustness_root.glob("seed*/05/results")):
            run_tag = result_dir.parents[1].name
            seed = _seed_from_tag(run_tag)
            if seed is None:
                continue
            runs.append(
                {
                    "meter_split_seed": seed,
                    "run_tag": run_tag,
                    "robustness_run": True,
                    "result_dir": result_dir,
                }
            )
    return runs


def _read_result(run: dict[str, object], filename: str) -> pd.DataFrame | None:
    path = Path(run["result_dir"]) / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.insert(0, "result_dir", str(run["result_dir"]))
    df.insert(0, "robustness_run", bool(run["robustness_run"]))
    df.insert(0, "run_tag", str(run["run_tag"]))
    df.insert(0, "meter_split_seed", int(run["meter_split_seed"]))
    return df


def aggregate(output_dir: Path) -> dict[str, Path]:
    runs = discover_result_dirs(output_dir)
    destination = output_dir / "robustness" / "results"
    destination.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    run_index = pd.DataFrame(runs)
    if not run_index.empty:
        run_index = run_index.assign(result_dir=run_index["result_dir"].astype(str))
    run_index_path = destination / "split_run_index.csv"
    run_index.to_csv(run_index_path, index=False)
    written["split_run_index"] = run_index_path

    collected: dict[str, pd.DataFrame] = {}
    for name, filename in RESULT_FILES.items():
        frames = [df for run in runs if (df := _read_result(run, filename)) is not None]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        out_path = destination / f"split_{filename}"
        combined.to_csv(out_path, index=False)
        collected[name] = combined
        written[name] = out_path

    scorecard = collected.get("model_scorecard")
    if scorecard is not None and not scorecard.empty:
        metrics = [
            "acf_l2", "wasserstein", "crps", "spectral_frechet",
            "novelty", "coverage", "intra_diversity",
        ]
        available_metrics = [metric for metric in metrics if metric in scorecard.columns]
        per_split_winners = (
            scorecard.loc[scorecard.groupby("run_tag")["mean_rank"].idxmin(), ["run_tag", "model"]]
            .rename(columns={"model": "split_winner"})
        )
        scorecard_with_winner = scorecard.merge(per_split_winners, on="run_tag", how="left")
        scorecard_with_winner["is_split_winner"] = (
            scorecard_with_winner["model"] == scorecard_with_winner["split_winner"]
        )
        stability = (
            scorecard_with_winner.groupby("model")
            .agg(
                n_splits=("run_tag", "nunique"),
                mean_rank_mean=("mean_rank", "mean"),
                mean_rank_std=("mean_rank", "std"),
                mean_rank_median=("mean_rank", "median"),
                first_place_splits=("is_split_winner", "sum"),
                metric_wins_total=("wins", "sum"),
            )
            .reset_index()
        )
        stability["first_place_splits"] = stability["first_place_splits"].astype(int)
        for metric in available_metrics:
            metric_stats = (
                scorecard.groupby("model")[metric]
                .agg(["mean", "std", "median"])
                .rename(
                    columns={
                        "mean": f"{metric}_mean",
                        "std": f"{metric}_std",
                        "median": f"{metric}_median",
                    }
                )
                .reset_index()
            )
            stability = stability.merge(metric_stats, on="model", how="left")
        stability = stability.sort_values(["mean_rank_mean", "mean_rank_median", "model"])
        stability_path = destination / "model_rank_stability.csv"
        stability.to_csv(stability_path, index=False)
        written["model_rank_stability"] = stability_path

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "output",
        help="Project output directory. Defaults to <repo>/output.",
    )
    args = parser.parse_args()

    written = aggregate(args.output_dir)
    print("Aggregated split robustness files:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
