# True Meter-Split Robustness

This document records the method for the alternative meter-split retraining study. The goal is to separate two questions that were previously entangled:

1. how good the generators are under the original seed-42 meter split;
2. whether that conclusion is stable when the held-out meters change.

The seed-42 split remains the thesis baseline. Alternative splits are not allowed to overwrite it.

## Study Design

- Split unit: whole meters, using the same `train_val_split_instances(...)` and `split_mask_by_meter(...)` functions as the main notebooks.
- Validation fraction: `0.15`, matching the baseline.
- Model seeds: unchanged inside each model family (`0` for diffusion, `1` or trainer default for rectified flow, `2` for CVAE construction), so the intervention is the meter split rather than a broad random-seed sweep.
- Conditioning schema: unchanged and locked to:
  - `c_discrete = [cluster_id, day_type, season]`
  - `c_continuous = [temp_normed, log_mean_z, log_std_z]`
- Metrics: same main comparison families as the baseline: four functional distances (ACF L2, marginal Wasserstein, CRPS, spectral Frechet), shape-space diversity metrics (novelty, coverage, intra-diversity), and the notebook-04 temperature-response diagnostic when available.
- Condition coverage: notebook 05 keeps the same rule of skipping `(cluster, day_type)` groups with fewer than 10 real validation profiles, and writes `skipped_conditions.csv` for every split.

Recommended alternative meter split seeds:

| Seed | Reason |
|------|--------|
| 5 | Full 10/10 condition coverage in the pre-check and good cluster-1 support. |
| 4 | Full 10/10 condition coverage; useful independent alternative. |
| 101 | Full 10/10 condition coverage; third sensitivity check if time allows. |

## Output Layout

Default mode is unchanged:

```text
output/03a/...
output/03b/...
output/03c/...
output/05/results/...
```

Robustness mode writes to seed-specific folders:

```text
output/robustness/seed005/03a/checkpoints/best_model.pkl
output/robustness/seed005/03a/results/diffusion/training_summary.json
output/robustness/seed005/03b/checkpoints/best_model.pkl
output/robustness/seed005/03b/results/rectified_flow/training_summary.json
output/robustness/seed005/03c/checkpoints/best_model.pkl
output/robustness/seed005/03c/results/cvae/training_summary.json
output/robustness/seed005/04/results/evaluation_metrics.csv
output/robustness/seed005/04/results/partial_dependence_temp.csv
output/robustness/seed005/04/results/partial_dependence_temp_multimodel.csv
output/robustness/seed005/04/results/temperature_response_metrics.csv
output/robustness/seed005/04/results/condition_audit.csv
output/robustness/seed005/04/results/skipped_conditions.csv
output/robustness/seed005/05/results/comparison_long.csv
output/robustness/seed005/05/results/model_scorecard.csv
output/robustness/seed005/05/results/model_scorecard_with_temperature_response.csv
output/robustness/seed005/05/results/temperature_response_scorecard.csv
output/robustness/seed005/05/results/bootstrap_ci_aggregate.csv
output/robustness/seed005/05/results/bootstrap_ci_aggregate_weighted_n_real.csv
```

This means downloading or pulling a robustness run does not overwrite the seed-42 baseline. RF and CVAE upload cells also create seed-named archives such as `tesina_03b_seed005_results.tar.gz`; the archive contains the full `output/robustness/seed005/...` paths.

For the completed seed005 Colab run, the full checkpoint archive is:

```text
/content/tesina/output/robustness/tesina_seed005_results.tar.gz
```

It is about 147.8 MB and contains paths rooted at `output/robustness/seed005`, so extracting it into the repository will not overwrite baseline `output/03a`, `output/04`, or `output/05` files.

VS Code connected to the Colab runtime could not reliably download that full archive directly. For local aggregation, notebook 05 therefore created and transferred a compact results-only archive:

```text
/content/tesina_transfer/tesina_seed005_results_only.tar.gz
```

That archive contains the seed005 `03a/results`, `03b/results`, `03c/results`, `04/results`, and `05/results` folders, excluding checkpoints. It was extracted locally on 2026-06-01, and `python scripts/aggregate_split_robustness.py` now includes both seed042 and seed005 in `output/robustness/results/split_run_index.csv`.

## Colab Run Procedure

Run each seed all the way through 03a, 03b, 03c, 04, then 05 with the same environment variables:

```python
import os
os.environ["TESINA_GPU"] = "1"
os.environ["TESINA_ROBUSTNESS"] = "1"
os.environ["TESINA_METER_SPLIT_SEED"] = "5"
```

For the next split, change only `TESINA_METER_SPLIT_SEED` to `4` or `101` and rerun the five robustness notebooks.

Expected notebook order per seed:

1. `notebooks/03a_diffusion_training.ipynb`
2. `notebooks/03b_rectified_flow_training.ipynb`
3. `notebooks/03c_cvae_training.ipynb`
4. `notebooks/04_evaluation.ipynb`
5. `notebooks/05_comparison.ipynb`

After downloading/pulling each seed's results, aggregate everything locally:

```bash
python scripts/aggregate_split_robustness.py
```

The aggregate files are written to `output/robustness/results/`.

## Aggregate Files

The aggregation script writes one split-stacked CSV per result type:

- `split_run_index.csv`
- `split_comparison_long.csv`
- `split_model_scorecard.csv`
- `split_metric_ratio_to_historical.csv`
- `split_training_summary_table.csv`
- `split_bootstrap_ci_aggregate.csv`
- `split_bootstrap_ci_aggregate_weighted_n_real.csv`
- `split_skipped_conditions.csv`
- `split_temperature_response_scorecard.csv` when split notebooks write it
- `split_model_scorecard_with_temperature_response.csv` when split notebooks write it
- `model_rank_stability.csv`

Every row receives:

- `meter_split_seed`
- `run_tag`
- `robustness_run`
- `result_dir`

`model_rank_stability.csv` is the thesis-facing summary table for whether the model ordering is stable across splits.

## Results Log

| Run tag | Split seed | Status | Validation meters | Conditions used | Winner | Notes |
|---------|------------|--------|-------------------|-----------------|--------|-------|
| seed042 | 42 | baseline complete | 15, 16, 18, 19 | 8/10 | CVAE | Cluster 1 skipped because validation support is too small. |
| seed005 | 5 | complete and aggregated locally | 3, 11, 18, 23 | 10/10 in notebook 05; 34/40 cluster-day-season groups in notebook 04 | CVAE / Historical tie | CVAE wins Wasserstein and spectral Frechet; historical wins ACF L2 and CRPS. Result CSVs are locally under `output/robustness/seed005/...`; full checkpoint archive remains at `/content/tesina/output/robustness/tesina_seed005_results.tar.gz`. |
| seed004 | 4 | planned | TBD | TBD | TBD | Full coverage expected from pre-check. |
| seed101 | 101 | optional/planned | TBD | TBD | TBD | Full coverage expected from pre-check. |

### Seed005 Run Details

Execution order completed in the Colab/GPU runtime: 01, 02, 03, 03a, 03b, 03c, 04, 05. Notebooks 01-03 regenerate shared artifacts and the historical benchmark; notebooks 03a/03b/03c/04/05 were run with `TESINA_ROBUSTNESS=1` and `TESINA_METER_SPLIT_SEED=5`.

Training summaries:

| Model | GPU | Steps | Parameters | Final train loss | Final val loss |
|-------|-----|------:|-----------:|-----------------:|---------------:|
| Diffusion | true | 5,800 | 1.106M | 0.2435 | 0.2959 |
| Rectified flow | true | 5,800 | 1.106M | 0.6328 | 0.7515 |
| CVAE | true | 5,800 | 0.286M | 0.1317 | 0.2066 |

Diffusion-only notebook 04 notes:

- Validation meters are `[3, 11, 18, 23]`, giving 1,488 validation instances.
- All 40 `(cluster, day_type, season)` groups exist in validation, but notebook 04 evaluates only groups with at least 10 real profiles, so 34/40 are used.
- Skipped 04 groups are `c0_wke_win` (3), `c0_wke_sum` (4), `c2_wkd_win` (2), `c2_wke_win` (1), `c2_wke_spr` (6), and `c2_wke_aut` (8).
- Hardest diffusion cluster on this split is cluster 3, with the largest Wasserstein, CRPS, and spectral Frechet distances.
- Temperature partial dependence for the largest pool (`cluster=4`, weekday, spring) shows a weak synthetic daily-total response: synthetic totals stay near 446-449 Wh across the temperature sweep, while real bin means fall from about 598 Wh to 353 Wh. This supports reporting continuous-conditioning weakness.

Notebook 05 scorecard:

| Model | ACF L2 | Wasserstein | CRPS | Spectral Frechet | Mean rank | Wins |
|-------|-------:|------------:|-----:|-----------------:|----------:|-----:|
| CVAE | 0.3704 | 1.6453 | 3.0291 | 598.1004 | 1.50 | 2 |
| Historical | 0.3373 | 1.6626 | 3.0254 | 672.6911 | 1.50 | 2 |
| Rectified flow | 0.4457 | 1.9684 | 3.1697 | 912.4842 | 3.25 | 0 |
| Diffusion | 0.4863 | 2.0210 | 3.1921 | 779.1853 | 3.75 | 0 |

Interpretation for seed005: the baseline seed-42 conclusion that CVAE is clearly strongest is not fully invariant to changing held-out meters. On seed005, CVAE remains the best learned model and has the best quality-per-parameter tradeoff, but historical retrieval ties it on mean rank and is slightly better on ACF L2 and CRPS. This should be described as a nuanced robustness result rather than a simple replication of the seed-42 ranking.

Bootstrap closure for seed005:

- `model_scorecard.csv` records `n_conditions_used=10` and `n_conditions_possible=10`.
- Unweighted 95% bootstrap intervals over condition groups are wide enough that CVAE and historical are not cleanly separated on aggregate means.
- `bootstrap_ci_aggregate_weighted_n_real.csv` provides the `n_real`-weighted sensitivity table for thesis reporting.

Two-split aggregate after local extraction:

| Model | Splits | Mean rank mean | First-place splits | Total metric wins |
|-------|-------:|---------------:|-------------------:|------------------:|
| CVAE | 2 | 1.25 | 2 | 6 |
| Historical | 2 | 1.75 | 0 | 2 |
| Diffusion | 2 | 3.38 | 0 | 0 |
| Rectified flow | 2 | 3.62 | 0 | 0 |

This aggregate preserves the main thesis result more strongly than the seed005 scorecard alone: CVAE remains the best learned model and the best overall model across locally aggregated splits, but the small gap to historical retrieval should still be reported.

## Interpretation Rules

- Do not compare a seed-specific model against checkpoints trained on another seed. Notebook 05 now reads checkpoints from the same `RUN_OUTPUT_DIR` used by 03a/03b/03c.
- Treat seed 42 as the original thesis result and the alternative seeds as robustness checks, not replacements.
- If the same model wins the mean-rank scorecard across alternative splits, the conclusion is stable.
- If the winner changes, report the range of ranks and metrics from `model_rank_stability.csv` and discuss whether the baseline seed was unusually favorable or unfavorable.
- Always report condition coverage alongside the scorecard because different meter splits can change which `(cluster, day_type)` groups are statistically usable.
