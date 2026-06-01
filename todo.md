# Project notes & open TODOs

Last updated: 2026-06-01

---

## Pipeline fixes shipped 2026-05-29 (pre-GPU review pass)

1. **Colab bootstrap cell** added at the top of 03a / 03b / 03c. Clones
   `ncapac/tesina`, pip-installs `requirements.txt`, `cd`s into
   `notebooks/`, and warns about missing raw-data files. No-op locally.
   (Previously cell 1 on Colab failed with `ModuleNotFoundError: equinox`.)
2. **LR-schedule budget alignment.** The 03a/b/c train cells now derive
   `TOTAL_STEPS = N_EPOCHS × train_loader.epoch_len` instead of passing
   100 000 / 50 000. Previously the optax cosine schedule decayed over a
   100 k budget while training stopped at ~5.8 k steps (GPU) / ~36 steps
   (smoke), so the LR never left the warmup-end plateau. Warmup now scales
   as 5 % of `TOTAL_STEPS` (min 20). README §6 table updated to match.
3. **`val_every=5` → `val_every=1`** in 03a/b/c so validation, early
   stopping, and `best_model.pkl` (saved by the early-stop branch) all
   actually run — previously the smoke profile (`n_epochs=3`) never
   triggered a single val pass, leaving `final_val_loss=null` and the
   `checkpoints/` folder empty.
4. **README param count** corrected from `~845 k` to `~1.11 M` to match
   the locked `n_continuous=3` conditioning surface.

---

## Current state

| Notebook / artefact | Status |
|---------------------|--------|
| 01 — EDA | ✅ Done |
| 02 — Clustering | ✅ Done (k=5, shape-normalised PCA → K-Means) — locks conditioning schema |
| 03 — Benchmark (historical retrieval) | ✅ Done — reference floor for functional-distance metrics |
| 03a — Diffusion training | ✅ Full GPU run recovered locally (`gpu_profile=true`, 5 800 steps, final val loss 0.2729) |
| 03b — Rectified flow training | ✅ Full GPU run recovered locally (`gpu_profile=true`, 5 800 steps, final val loss 0.7380) |
| 03c — CVAE training | ✅ Full GPU run recovered locally (`gpu_profile=true`, 5 800 steps, final val loss 0.1466) |
| 04 — Evaluation | ✅ Re-run on Colab GPU with fresh 03a checkpoint; outputs restored locally |
| 05 — Comparison | ✅ Re-run on Colab GPU; outputs restored locally; scorecard/complexity plots improved; bootstrap/skipped-condition robustness outputs added |
| 01 — Meter-quality closure | ✅ All 24 meters retained; compact diagnostic written to `output/01/results/meter_quality_summary.csv` |
| True meter-split robustness | 🚧 Seed005 completed in Colab/GPU runtime and compact result tables extracted locally; seed004 still pending |
| README.md | ✅ Updated with final results, robustness closure, meter-retention decision, seed005 aggregate status, and current roadmap |
| Engineering cleanup | ✅ `_epoch_len` now fails loudly; CFG null conditioning is shared/tested; paired bootstrap helper tested |

### Final output files now present

- `output/04/results/evaluation_metrics.csv` — 26 per-condition diffusion rows; all finite; `n_syn=200`.
- `output/04/results/partial_dependence_temp.csv` — temperature sweep for the largest-pool condition.
- `output/05/results/comparison_long.csv` — 32 rows = 4 models × 8 condition groups; all finite.
- `output/05/results/training_summary_table.csv` — compact training facts for 03a/b/c.
- `output/05/results/model_scorecard.csv` — aggregate means, mean rank, metric wins, and 8/10 condition coverage.
- `output/05/results/metric_ratio_to_historical.csv` — ratios vs historical baseline.
- `output/05/results/bootstrap_ci_aggregate.csv` — 2 000-draw paired-condition bootstrap CIs for unweighted aggregate means.
- `output/05/results/bootstrap_ci_aggregate_weighted_n_real.csv` — `n_real`-weighted sensitivity table.
- `output/05/results/skipped_conditions.csv` — all possible `(cluster, day_type)` groups with included/skipped reason.
- `output/01/results/meter_quality_summary.csv` — per-meter quality diagnostic; all 24 meters retained.
- `output/robustness/results/*.csv` — split-stacked aggregate tables produced by `scripts/aggregate_split_robustness.py`; currently includes seed042 and seed005.

### Final scorecard (lower is better)

| Model | ACF L2 | Wasserstein | CRPS | Spectral Fréchet | Mean rank | Wins |
|-------|-------:|------------:|-----:|-----------------:|----------:|-----:|
| CVAE | 0.2595 | 1.8883 | 5.6889 | 398.7623 | 1.0 | 4 |
| Historical | 0.2763 | 2.0411 | 5.6916 | 626.5811 | 2.0 | 0 |
| Diffusion | 0.3721 | 2.9079 | 5.9683 | 806.8116 | 3.0 | 0 |
| Rectified flow | 0.3807 | 2.9187 | 5.9779 | 997.9370 | 4.0 | 0 |

**Working thesis conclusion:** CVAE is the strongest generator for the
current Rolle daily-load experiment. It wins all four aggregate
functional-distance metrics, is slightly better than historical retrieval
on CRPS, clearly better on spectral structure, and uses far fewer
parameters than diffusion/RF.

---

## Open TODOs

### Thesis writing / presentation
- [ ] Write final results chapter around the CVAE scorecard win, the
      historical baseline comparison, and the quality-vs-parameter result.
- [ ] Discuss why cluster 1 is omitted from 04/05 metric tables: only two
      validation examples, so metrics would be dominated by sampling noise.
- [ ] Discuss the bootstrap CIs as uncertainty over the eight retained
      condition groups, not as a full alternative-split robustness study.
- [ ] Discuss the diffusion temperature partial-dependence finding: the
      synthetic daily total is much flatter than the empirical binned
      response, so continuous conditioning is weaker than desired.
- [ ] Decide which figures/tables from 04/05 enter the final document:
      scorecard, bootstrap CI table, ratio-to-historical plot,
      quality-vs-size plot, diffusion partial-dependence plot, and
      representative envelope gallery.

### Robustness / cleanup shipped 2026-06-01
- [x] Added paired-condition bootstrap CIs for the 05 aggregate metrics.
- [x] Added `skipped_conditions.csv` and coverage columns to `model_scorecard.csv`.
- [x] Confirmed cluster 1 is skipped because validation support is 2 weekday
      examples and 0 weekend examples under the seed-42 meter split.
- [x] Inspected the 24 Rolle meters; retained all meters because each has
      complete daily coverage, zero raw missingness, and no all-zero pattern.
- [x] Confirmed/tracked CFG null consistency via a shared helper and unit tests.
- [x] Made `_epoch_len(loader)` fail loudly when `epoch_len` is absent.

### True meter-split robustness in progress
- [x] Parameterize 03a/03b/03c/04/05 with `TESINA_METER_SPLIT_SEED` and `TESINA_ROBUSTNESS`.
- [x] Keep default seed-42 outputs unchanged while routing robustness runs to `output/robustness/seedXXX/...`.
- [x] Add seed-specific RF/CVAE transfer archives so downloaded artifacts preserve their intended output paths.
- [x] Add `scripts/aggregate_split_robustness.py` and aggregate outputs under `output/robustness/results/`.
- [x] Create `ROBUSTNESS.md` with method, run procedure, output layout, and results log.
- [x] Run Colab GPU retraining/evaluation for seed 5.
- [ ] Run Colab GPU retraining/evaluation for seed 4.
- [ ] Optionally run Colab GPU retraining/evaluation for seed 101.
- [x] Download/extract the completed seed005 result tables into the local repo.
- [x] Re-run `python scripts/aggregate_split_robustness.py` after extracting seed005 results.
- [x] Update `ROBUSTNESS.md` aggregate interpretation and thesis text from `model_rank_stability.csv` once seed005 is present locally.

Seed 5 note: completed in the Colab/GPU runtime with validation meters
`[3, 11, 18, 23]`. Notebook 05 uses all 10/10 `(cluster, day_type)` groups;
CVAE and historical tie on mean rank (`1.50`, two metric wins each), while RF
and diffusion trail. The compact results-only archive was extracted locally
and aggregated with seed042; the full checkpoint archive remains at
`/content/tesina/output/robustness/tesina_seed005_results.tar.gz` if needed.

### Post-thesis engineering polish (deferred)
- [ ] Move per-cluster training-loss diagnostics out of the inner training
      loop or default `log_cluster_losses=False` for GPU runs; the current
      implementation adds extra forward passes per batch.
- [ ] Save wall-clock runtime metadata in future training summaries if the
      thesis needs an efficiency comparison beyond parameter count.

### Optional extensions (post-thesis, from Lorenzo's note)
- [ ] Generate a full year of synthetic data and compare annual consumption,
      frequency content, and distance to nearest real day.

---

## Design notes (for thesis)

**Conditioning schema** (locked, consistent across all models):
- `c_discrete   = [cluster_id, day_type, season]` — AdaLN
- `c_continuous = [temp_normed, log_mean_z, log_std_z]` — cross-attention
- The CVAE constructor accepts `n_continuous` as a parameter; notebooks 03c and 05 instantiate it with `n_continuous=3` so all four models share the same conditioning surface.

**Weekday vs weekend signal on the Rolle dataset** — weaker than on residential-heavy datasets because the LV cabinets aggregate commercial + residential load. Expect significant KDE overlap on the `day_type` axis; report it explicitly.

**Small meter count (24)** — the meter-based train/val split leaves whole meters on one side, so per-cluster val pools can be tiny. When reporting per-cluster metrics, always show `n_real` alongside the metric value and flag rows where it is small.

**Cluster assignment is meter-day specific** — generated under the assumption that the cluster id of a future query is known. Generalisation to brand-new meters with unknown cluster membership is out of scope.

---

## Design history & key findings

### Dataset (from 01 — EDA)

- **Source**: Rolle (CH) power-quality + NWP, Zenodo `10.5281/zenodo.3463136`.
- **Files**: `power_data.p` (pickle, holds `P_mean` DataFrame at 10-min) and `nwp_data.h5` (Meteoblue 24h-ahead arrays).
- **24 meter columns** after dropping the 7 hierarchical-aggregation columns (`S1, S2, S11, S12, S21, S22, all`); resampled to hourly.
- **Period**: 2018-01-13 → 2019-01-19, **372 calendar days** = **8 928 daily instances** (24 × 372).
- **Hourly resolution, 24 steps/day** — the brief mentioned 15-min/96-step; the actual data is hourly. Codebase uses `seq_len=24`.
- **Mean Rolle temperature** over the record ≈ 12 °C.

### Clustering (from 02 — Clustering)

- **Per-instance shape normalisation** (not per-meter): each daily profile is divided by its own daily mean before PCA + K-Means. This decouples shape from scale at the instance granularity — the same meter can land in different clusters on different days.
- **k=5** chosen on silhouette + interpretability.
- The per-instance log-mean and log-std are *not* discarded — they become the `log_mean_z` / `log_std_z` continuous conditioning channels, with the global z-score scalars persisted in `data/scale_stats.json`.

### Conditioning schema decisions

- `day_type` kept despite the (likely weak) signal on Rolle — cheap to include and may matter more on other datasets.
- `season` (0=Winter, 1=Spring, 2=Summer, 3=Autumn) encodes temperature seasonality more coarsely but more robustly than `month` for this dataset size.
- `log_mean_z` and `log_std_z` (log of per-instance daily mean/std, globally z-scored using `scale_stats.json`) added to the continuous vector — they let the model condition on the *scale* of the target instance while learning shape only.
- Absolute units are recovered at inference by inverting the z-score with `scale_stats.json` and multiplying the generated shape by `exp(log_mean)`.

### Normalisation design (Lorenzo review, 2026-04)

Per-cluster z-score was replaced with **per-instance shape normalisation**: `x_norm = x / daily_mean_instance`. The generative model learns only the *shape*; scale is recovered at inference from `log_mean_z` (de-z-scored via `data/scale_stats.json`, then exponentiated). This avoids the model having to learn the wide spread between low-consumption residential cabinets and high-consumption commercial ones.

### Evaluation metric bugs fixed

| Bug | Fix |
|-----|-----|
| `acf_compare(nlags=48)` on 24-step sequences → artifact zeros for lags ≥ 24 | Auto-clamp `nlags` to `min(nlags, L-1)` |
| `envelope_plot(steps_per_hour=4)` — 15-min design assumption on hourly data | Default changed to `steps_per_hour=1` |
| `discriminative_score`: 30k real vs 100 synthetic → trivial classifier from class imbalance | Balance classes by subsampling real to `min(N_real, N_syn)` |
| `FFT(eps_pred) vs FFT(noise)` frequency loss in DDPM — white-spectrum bias, redundant with MSE by Parseval | Removed; optional `data_freq_loss_weight` targets `FFT(x̂₀) vs FFT(x₀)` in data space (default 0.0) |

### Model architecture (03a — Diffusion)

- `DiffusionTransformer1D`: seq_len=24, d_model=128, n_heads=4, n_layers=4, d_ff=256, locked `n_continuous=3` → **~1.106M parameters**
- Discrete conditioning via AdaLN; continuous via cross-attention.
- CFG null token: `[-1, -1, -1]` (discrete), zeros (continuous). Dropout rate `p_uncond=0.15`.
- Inference: DDIM 50 steps, guidance scale 2.0.
- Training: AdamW + cosine LR schedule, gradient clip 1.0, warmup 500 steps.
- **Training profile**: CPU smoke uses the 1 000-instance local profile; the completed GPU run used the full dataset for 5 800 derived schedule steps.
