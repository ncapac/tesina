# SUPSI DAS Tesina — Conditional Generative Models for Smart-Meter Load Profiles

DAS thesis project: **conditional generation of synthetic daily
smart-meter load profiles** with three modern generative families — DDPM,
Rectified Flow, and Conditional β-VAE — benchmarked against a **historical
nearest-neighbour retriever**, all in JAX + Equinox.

The thesis asks for a comparison of generative time-series tools on a
smart-meter dataset with explicit attention to autocorrelation,
noise-to-signal characteristics, and partial dependences on exogenous
inputs (temperature). The project uses the **Rolle (Switzerland)
power-quality + NWP dataset** (Zenodo `10.5281/zenodo.3463136`).

---

## 1. What's in this repository

| Layer | What it does | Source |
|-------|-------------|--------|
| Data | Load the Rolle power + NWP dataset; daily-instance windowing aligned to calendar days; per-instance shape normalisation; meter-based train/val split | [src/data/](src/data/) |
| Models | 1-D Transformer backbone shared by DDPM and rectified flow; conditional β-VAE; nearest-neighbour historical baseline | [src/models/](src/models/) |
| Training | Trainers for all three learned models with cosine-with-warmup schedule, classifier-free guidance dropout, early stopping, checkpointing | [src/training/](src/training/) |
| Evaluation | Four functional-distance metrics (ACF L2, marginal Wasserstein-1, CRPS, spectral Fréchet), a `compare_models` driver, and paired-condition bootstrap CIs for aggregate metrics | [src/evaluation/metrics.py](src/evaluation/metrics.py) |
| Notebooks | EDA → clustering → benchmark → three training notebooks → evaluation → cross-model comparison | [notebooks/](notebooks/) |
| Tests | 60+ unit tests covering loader, dataset, models, training, metrics, runtime paths, export bundles | [tests/](tests/) |

---

## 2. Data

### 2.1 Raw inputs (must be placed in `data/`)

| File | Provenance | Content |
|------|-----------|---------|
| `data/power_data.p` | Rolle (CH) power-quality archive, Zenodo `10.5281/zenodo.3463136` | Pickled dict; the `P_mean` DataFrame holds active-power readings at 10-min resolution; the loader drops 7 hierarchical-aggregation columns (`S1, S2, S11, S12, S21, S22, all`) and resamples to **hourly** |
| `data/nwp_data.h5` | Meteoblue NWP archive (same Zenodo record) | 24-element hourly forecast arrays (temperature, GHI, GNI, RH, wind speed/direction) at every 10-min timestamp; the loader collapses temperature to a **daily mean** series |

Both files are gitignored. The original `3463137.zip` Zenodo archive can
also be placed in `data/` (it is what the two files are extracted from).

**Concrete dataset facts** (verified from the loader and the derived
artefacts in `data/`):

- **24 meter columns** (24 real LV substations / cabinets in Rolle, after dropping the 7 aggregation columns)
- **Period**: 2018-01-13 → 2019-01-19, **372 calendar days**
- **Hourly resolution**, 24 steps/day → `seq_len=24` throughout the codebase
- **8 928 daily instances** = 24 meters × 372 days (1:1 because the loader's 10 % NaN gate is not triggered on this dataset)
- Mean Rolle temperature over the record ≈ **12 °C**

### 2.2 Derived artefacts (tracked)

| File | Built by | Content |
|------|----------|---------|
| `data/clusters.csv` | [notebooks/02_clustering.ipynb](notebooks/02_clustering.ipynb) | `(meter_id, date, cluster_id)` — per-instance cluster assignment, k=5, shape-normalised K-Means on PCA embeddings. 8 928 rows. |
| `data/daily_covariates.csv` | [notebooks/02_clustering.ipynb](notebooks/02_clustering.ipynb) §1b | `(meter_id, date, temp_normed, log_mean_z, log_std_z)` — the three continuous conditioning channels, one row per instance |
| `data/scale_stats.json` | [notebooks/02_clustering.ipynb](notebooks/02_clustering.ipynb) §1b | The four scalars `{log_mean_mean, log_mean_std, log_std_mean, log_std_std}` used to z-score the per-instance log-scale channels. Evaluation notebooks read this to invert the z-score and recover absolute units (Watts) via `src.data.dataset.shape_denormalize`. |

### 2.3 Conditioning schema (LOCKED across all models)

```
c_discrete   = [cluster_id, day_type, season]            # 3 channels, AdaLN
c_continuous = [temp_normed, log_mean_z, log_std_z]      # 3 channels, cross-attention
```

| Channel | Meaning | Range |
|---------|---------|-------|
| `cluster_id` | K-Means cluster on shape-normalised daily profile | `0..4` |
| `day_type` | 0=weekday, 1=weekend | `{0,1}` |
| `season` | 0=winter, 1=spring, 2=summer, 3=autumn | `{0..3}` |
| `temp_normed` | Daily mean Rolle temperature, z-scored over the record | float |
| `log_mean_z` | $\log$ of each instance's daily-mean load, then globally z-scored using `scale_stats.json` | float |
| `log_std_z`  | $\log$ of each instance's daily-std of load, then globally z-scored using `scale_stats.json` | float |

Note that `src.data.dataset.make_daily_instances` returns only **1**
continuous channel (`temp_normed`). The two log-scale channels are
computed in **notebook 02 §1b** via `shape_normalize(...)` +
`compute_scale_stats(...)` and persisted to `daily_covariates.csv` +
`scale_stats.json`. All downstream notebooks (03 / 03a / 03b / 03c / 04 /
05) load both CSVs and stitch them into the 3-channel `c_continuous`
vector before training or sampling.

Conditioning the model on the per-instance scale lets it learn *shape*
only (the network sees `xs / mean`); the absolute Watts are recovered at
inference time by reading the requested `log_mean_z` / `log_std_z`,
inverting the z-score with `scale_stats.json`, and multiplying the
generated shape by `exp(log_mean)`.

### 2.4 Known dataset quirks (recorded in [todo.md](todo.md))

Detailed analysis lives in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).
The headline points to remember when reading evaluation results:

- **24 meters is small**: meter-based train/val splits leave individual meters entirely on one side of the split, so some clusters may be over- or under-represented at evaluation time. Per-cluster metrics with tiny val pools must be flagged in the discussion.
- **One full annual cycle**: 372 days covers exactly one yearly cycle, so the season channel is informative but each (cluster, day_type, season) bucket has at most ~24 × 90 / 4 ≈ 540 instances before filtering.
- **Hourly resolution** (not 15 min as the original brief mentions): `seq_len=24` everywhere.
- **Per-instance shape normalisation** (rather than per-meter or per-cluster): chosen so that the same model handles low-consumption residential cabinets and high-consumption commercial ones without scale dominating the loss.
- **Meter-quality decision**: all 24 meters are retained. [output/01/results/meter_quality_summary.csv](output/01/results/meter_quality_summary.csv) shows 372 daily instances for every meter, zero raw missingness, max zero fraction `0.0017`, and max absolute robust mean-load z-score `2.29`; no meter is structurally broken or exclusion-worthy.

---

## 3. Notebooks (intended run order)

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | [01_eda.ipynb](notebooks/01_eda.ipynb) | Exploratory analysis: outliers, seasonal heatmaps, weekday/weekend, diurnal shape |
| 02 | [02_clustering.ipynb](notebooks/02_clustering.ipynb) | Shape-normalised K-Means (k=5); writes `clusters.csv` and `daily_covariates.csv`; **locks the conditioning schema** |
| 03 | [03_benchmark.ipynb](notebooks/03_benchmark.ipynb) | Historical nearest-neighbour baseline; reference floor for all functional-distance metrics; val MAE ≈ 15.43 kW |
| 03a | [03a_diffusion_training.ipynb](notebooks/03a_diffusion_training.ipynb) | Conditional DDPM training (Trainer, cosine LR, CFG p_uncond=0.15, DDIM-50 inference) |
| 03b | [03b_rectified_flow_training.ipynb](notebooks/03b_rectified_flow_training.ipynb) | Rectified-flow training (linear path, velocity loss, Euler sampler) |
| 03c | [03c_cvae_training.ipynb](notebooks/03c_cvae_training.ipynb) | Conditional β-VAE training (β-ELBO with separate recon/KL logging) |
| 04 | [04_evaluation.ipynb](notebooks/04_evaluation.ipynb) | Single-model deep dive (diffusion): per-condition functional distances, per-hour W1 heatmap, **partial dependence on temperature**, visual diagnostic |
| 05 | [05_comparison.ipynb](notebooks/05_comparison.ipynb) | Cross-model comparison: all 4 generators × every condition with ≥10 val instances; failure-mode→metric-signature interpretation key |

All training notebooks share an identical structure (§1 data → §2 split →
§3 model → §4 trainer → §5 fit → §6 curves → §7 sanity panel →
§8 summary JSON) so they can be reviewed side-by-side.

---

## 4. Models

### 4.1 Diffusion (DDPM + DDIM, CFG)

- File: [src/models/diffusion.py](src/models/diffusion.py), [src/models/transformer1d.py](src/models/transformer1d.py)
- Architecture: `DiffusionTransformer1D` — `seq_len=24, d_model=128, n_heads=4, n_layers=4, d_ff=256` → **~1.11 M parameters** with the locked `n_continuous=3` conditioning (the earlier ~845 k figure was for the single-continuous-channel prototype)
- Conditioning: AdaLN on the 3 discrete channels; cross-attention on the 3 continuous channels
- Schedule: cosine, T=1000
- Classifier-free guidance: discrete null `[-1,-1,-1]`, continuous null zeros, dropout `p_uncond=0.15`
- Inference: DDIM 50 steps, default guidance scale 2.0

### 4.2 Rectified flow

- File: [src/models/rectified_flow.py](src/models/rectified_flow.py)
- Same Transformer backbone as the diffusion model (shares weights / architecture)
- Linear interpolant path $x_t = (1-t)\,x_0 + t\,\epsilon$, velocity-matching loss
- Inference: Euler ODE solver, 50 steps, same CFG mechanism

### 4.3 Conditional β-VAE

- File: [src/models/cvae.py](src/models/cvae.py)
- 1-D convolutional encoder/decoder with conditioning concatenated to the latent
- Trained with β-ELBO; `train_recon` / `train_kl` exposed separately for diagnostics

### 4.4 Historical baseline

- File: [src/models/historical_baseline.py](src/models/historical_baseline.py)
- `.fit(xs, c_discrete, c_continuous, dates, meter_ids)` — indexes the training set by discrete condition and stores the continuous covariates
- `.generate(c_batch, key, c_continuous_batch, temp_tol)` — exact match on `(cluster, day_type, season)` ∩ Euclidean ball on the continuous vector (adaptive pool growth: `min_pool=5`, doubles `temp_tol` up to 3 times before falling back to discrete-only)
- No learnable parameters; sets the *irreducible noise floor* for the four functional-distance metrics. Any learned model should at minimum match it.

---

## 5. Evaluation framework — functional distance

Generating daily load profiles is fundamentally a problem of comparing
**distributions of curves in $\mathbb{R}^{24}$**, not scalar errors. The
evaluation framework in [src/evaluation/metrics.py](src/evaluation/metrics.py)
provides four complementary scalar distances, plus visual diagnostics.

| Metric | Function | Captures | Failure mode it catches |
|--------|----------|----------|------------------------|
| **ACF L2** | `acf_compare(real, syn, nlags=23)` | Temporal autocorrelation $\rho(h)$ | Right level but wrong temporal memory (e.g. white noise around the mean) |
| **Marginal Wasserstein-1** | `marginal_wasserstein(real, syn)` | Per-hour level distribution | Mean-curve generator (zero spread), level offsets |
| **CRPS** | `crps_score(real, syn)` | Probabilistic sharpness | Overconfidence / under-coverage of the synthetic ensemble |
| **Spectral Fréchet** | `spectral_frechet_distance(real, syn)` | FID-style distance in 12-harmonic FFT space | Peaks shifted in time (correct marginals but wrong phase) |

A summary 4-row diagnostic figure can be produced for any
`(real, syn)` ensemble with `run_all_metrics(real, syn, label=...)`, and
cross-model comparison is one call to `compare_models(models_dict, real,
conditions, c_continuous=..., n_samples=200, ...)`. The comparison driver
skips condition groups with fewer than 10 real validation examples; notebook
05 writes [output/05/results/skipped_conditions.csv](output/05/results/skipped_conditions.csv)
so that omitted groups are visible.

For aggregate uncertainty, `bootstrap_aggregate_metrics(...)` resamples
condition groups as paired units, preserving all model rows for each sampled
condition. Notebook 05 writes both the unweighted thesis scorecard CIs and an
`n_real`-weighted sensitivity table.

Notebook 04 also produces:

- A **per-hour Wasserstein heatmap** (`24 × n_conditions`) — diagnoses *when* during the day the model deviates from real.
- A **partial-dependence sweep** in temperature — for the largest-pool condition, holds discrete channels + `log_mean/std_z` fixed and sweeps `temp_normed` across empirical quantiles, then overlays synthetic vs real-bin mean profiles and a scatter of daily totals. This addresses the "partial dependences" requirement from the original thesis brief.

---

## 6. Runtime profiles

Every training notebook and 04/05 honour the `TESINA_GPU` environment
variable, picked up at notebook startup:

```python
GPU = bool(int(os.environ.get("TESINA_GPU", "0")))
```

The variable controls a coarse hyperparameter switch:

| Setting | CPU smoke (`TESINA_GPU=0`) | GPU run (`TESINA_GPU=1`) |
|---------|---------------------------|--------------------------|
| Epoch cap | 3 | 100 |
| LR warmup | 5 % of derived `total_steps` (min 20) | same |
| `total_steps` (LR-schedule budget) | derived as `n_epochs × train_loader.epoch_len` so the cosine LR matches actual training duration | same |
| Early-stop patience | 3 | 20 |
| Validation cadence | every epoch | every epoch |
| Batch size | 64 | 128 |
| Training subsample | 1 000 instances | Full dataset |

If `jax.devices()[0].platform in ("gpu","tpu")`, the GPU flag is
auto-upgraded even when the env var is unset. This lets a notebook run
end-to-end on a laptop CPU in a few minutes as a smoke test, and produce
publication-quality results when re-executed on a GPU.

---

## 7. Final thesis results

The full GPU workflow has been completed for the three learned models,
then evaluated with notebooks 04 and 05. All three learned models used
the same split (`7 440` train instances, `1 488` validation instances),
the same locked conditioning schema, and `5 800` GPU-profile training
steps.

### 7.1 Training summaries

| Model | Parameters | Final train loss | Final val loss | GPU profile |
|-------|-----------:|-----------------:|---------------:|:-----------:|
| Diffusion | 1.106 M | 0.2450 | 0.2729 | yes |
| Rectified flow | 1.106 M | 0.6364 | 0.7380 | yes |
| CVAE | 0.286 M | 0.1317 | 0.1466 | yes |

These losses are useful convergence checks, but they are not directly
comparable across families because DDPM/RF and CVAE optimize different
objectives. The final ranking is therefore based on the functional
distances below.

### 7.2 Cross-model scorecard

Notebook 05 compares historical retrieval, diffusion, rectified flow,
and CVAE on the same validation condition groups. Lower is better for all
four metrics.

| Model | ACF L2 | Wasserstein | CRPS | Spectral Frechet | Mean rank | Metric wins | Conditions |
|-------|-------:|------------:|-----:|-----------------:|----------:|------------:|-----------:|
| CVAE | 0.2595 | 1.8883 | 5.6889 | 398.7623 | 1.0 | 4 | 8 / 10 |
| Historical | 0.2763 | 2.0411 | 5.6916 | 626.5811 | 2.0 | 0 | 8 / 10 |
| Diffusion | 0.3721 | 2.9079 | 5.9683 | 806.8116 | 3.0 | 0 | 8 / 10 |
| Rectified flow | 0.3807 | 2.9187 | 5.9779 | 997.9370 | 4.0 | 0 | 8 / 10 |

Relative to the historical baseline (`1.0 = historical`), CVAE is better
on ACF L2 (`0.939`), Wasserstein (`0.925`), and spectral Frechet
(`0.636`), and essentially tied on CRPS (`0.9995`). Diffusion and
rectified flow are worse than the historical baseline on all four
aggregate metrics in this run.

**Thesis verdict:** the conditional CVAE is the strongest generator for
this Rolle daily-load setting. It wins the aggregate functional-distance
scorecard and does so with about one quarter of the parameters used by
diffusion/RF. Diffusion and rectified flow remain useful references, but
the recovered GPU results do not justify their extra complexity on this
dataset.

### 7.3 Diffusion deep dive

Notebook 04 evaluates the diffusion model alone at full condition
resolution: `26` `(cluster, day_type, season)` rows with at least `10`
real validation profiles, each compared against `200` synthetic samples.
Cluster 1 is intentionally absent because the validation split contains
only two real examples for that cluster; reporting a metric there would
be statistically fragile.

The temperature partial-dependence diagnostic is important for the
discussion: diffusion's synthetic daily total is nearly flat across the
temperature sweep, whereas the real validation bins show a much stronger
temperature response. That is evidence that plausible-looking generated
profiles do not necessarily imply a faithful continuous-conditioning
response.

### 7.4 Robustness closure

Notebook 05 now writes bootstrap confidence intervals for the aggregate
metrics:

- [output/05/results/bootstrap_ci_aggregate.csv](output/05/results/bootstrap_ci_aggregate.csv) — unweighted, matching the thesis scorecard definition.
- [output/05/results/bootstrap_ci_aggregate_weighted_n_real.csv](output/05/results/bootstrap_ci_aggregate_weighted_n_real.csv) — `n_real`-weighted sensitivity table.
- [output/05/results/skipped_conditions.csv](output/05/results/skipped_conditions.csv) — exact support for all 10 possible `(cluster, day_type)` groups.

The bootstrap resamples the 8 retained condition groups as paired units
(`2 000` draws, 95 % intervals). It should be read as an uncertainty band
over condition mix, not as a substitute for retraining on alternative meter
splits. The point ranking remains CVAE → historical → diffusion → rectified
flow, and the `n_real`-weighted sensitivity table still places CVAE first on
all four aggregate means. Cluster 1 is excluded because the validation split
contains only 2 weekday examples and 0 weekend examples for that cluster.

---

## 8. Output layout

All run artefacts go under `output/` (gitignored, except summary JSONs
which are small enough to track if desired):

```
output/
  01/results/meter_quality_summary.csv                 # meter-retention diagnostic
  03a/checkpoints/best_model.pkl                       # diffusion
       results/diffusion/training_summary.json
  03b/checkpoints/best_model.pkl                       # rectified flow
       results/rectified_flow/training_summary.json
  03c/checkpoints/best_model.pkl                       # cvae
       results/cvae/training_summary.json
  04/results/evaluation_metrics.csv                    # per-condition metrics
       results/partial_dependence_temp.csv             # PD sweep
  05/results/comparison_long.csv                       # 4-model × N-cond table
      results/training_summary_table.csv              # compact 03a/b/c run facts
      results/model_scorecard.csv                     # mean metrics + rank/wins + condition coverage
      results/metric_ratio_to_historical.csv          # learned models vs baseline
      results/bootstrap_ci_aggregate.csv              # paired-condition bootstrap CIs
      results/bootstrap_ci_aggregate_weighted_n_real.csv
      results/skipped_conditions.csv                  # included/skipped condition support
```

Every `training_summary.json` carries an identical schema:

```json
{
  "model": "diffusion|rectified_flow|cvae",
  "parametric": true,
  "train_instances": 7440,
  "val_instances": 1488,
  "n_clusters": 5,
  "gpu_profile": true,
  "total_steps": 5800,
  "n_epochs_cap": 100,
  "c_discrete":   ["cluster_id", "day_type", "season"],
  "c_continuous": ["temp_normed", "log_mean_z", "log_std_z"],
  "n_parameters": 1106274,
  "batch_size": 128,
  "final_train_loss": ...,
  "final_val_loss":   ...,
  "random_seed": 42,
  "val_fraction": 0.15
}
```

---

## 9. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Confirm with the test suite (no GPU required):

```bash
pytest -q
```

Recommended development environment: JAX CPU build on Linux for smoke
tests, JAX GPU build on Colab/A100 for full training runs. The
`requirements.txt` pins NumPy 2.1.x and SciPy 1.14.x to match the
versions used during development.

---

## 10. Reproducing the thesis results

```bash
source .venv/bin/activate

# 1. EDA + clustering (writes clusters.csv and daily_covariates.csv)
jupyter execute notebooks/01_eda.ipynb
jupyter execute notebooks/02_clustering.ipynb

# 2. Reference floor
jupyter execute notebooks/03_benchmark.ipynb

# 3. Train the three learned models on GPU
TESINA_GPU=1 jupyter execute notebooks/03a_diffusion_training.ipynb
TESINA_GPU=1 jupyter execute notebooks/03b_rectified_flow_training.ipynb
TESINA_GPU=1 jupyter execute notebooks/03c_cvae_training.ipynb

# 4. Evaluation + comparison
jupyter execute notebooks/04_evaluation.ipynb
jupyter execute notebooks/05_comparison.ipynb
```

For a local CPU smoke test of any training notebook, omit `TESINA_GPU`
(or set it to `0`) — the notebook will run end-to-end on a 1 000-instance
subsample in a few minutes and write a (small) checkpoint plus
`training_summary.json` with the expected schema.

---

## 11. Repository layout

```
src/
  runtime_paths.py            # DATA_DIR / OUTPUT_DIR resolution, repo-root aware
  data/
    loader.py                 # load_rolle_data, compute_temp_stats, normalize_temp, compute_scale_stats
    dataset.py                # make_daily_instances, shape_normalize, shape_denormalize, train_val_split_instances
  models/
    transformer1d.py          # Shared backbone (DDPM + RF)
    diffusion.py              # DDPM + DDIM + CFG
    rectified_flow.py         # Linear-path RF + CFG
    cvae.py                   # Conditional β-VAE
    historical_baseline.py    # Nearest-neighbour retriever
  training/
    train.py                  # Trainer (diffusion)
    train_rf.py               # RFTrainer
    train_cvae.py             # CVAETrainer
  evaluation/
    metrics.py                # All distance metrics, run_all_metrics, compare_models, HistoricalBaseline helpers
notebooks/                    # See §3
tests/                        # pytest, 60+ tests
scripts/
  ingest_downloaded_bundle.py # Pull a Colab export bundle into ./output
  restore_export_bundle.py    # Restore checkpoints from a tar.gz bundle
data/                         # Raw (power_data.p, nwp_data.h5) + derived (clusters.csv, daily_covariates.csv, scale_stats.json) — see §2
output/                       # All run artefacts (see §7)
biblio/                       # Project notes, initial brief, todo
```

---

## 12. Roadmap / open work

See [todo.md](todo.md) for the up-to-date task list. The remaining work,
in order, is:

1. Write the final thesis discussion around the CVAE-vs-historical result,
  the weak diffusion temperature partial dependence, the bootstrap condition
  uncertainty bands, and the omission of cluster 1 from metric tables because
  of tiny validation support.
2. Decide which figures/tables from 04/05 enter the final document:
  scorecard, bootstrap CI table, ratio-to-historical plot, quality-vs-size
  plot, diffusion partial-dependence plot, and representative envelope gallery.
3. Record wall-clock runtimes manually only if the thesis needs an efficiency
  comparison beyond parameter count.
4. Optional extension: generate a full synthetic year and compare annual
  consumption, frequency content, and distance to the nearest real day.

The core modelling, evaluation, robustness checks, and lightweight
engineering cleanup are complete; the remaining work is thesis writing and
optional extensions.
