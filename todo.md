# Project notes & open TODOs

Last updated: 2026-05-29

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

| Notebook | Status |
|----------|--------|
| 01 — EDA | ✅ Done |
| 02 — Clustering | ✅ Done (k=5, shape-normalised PCA → K-Means) — locks conditioning schema |
| 03 — Benchmark (historical retrieval) | ✅ Done — val MAE 15.43 kW reference floor |
| 03a — Diffusion training | ✅ Reworked (GPU profile + 6-channel conditioning + verbose md + summary JSON) · ⏳ GPU run pending |
| 03b — Rectified flow training | ✅ Reworked (same structure as 03a) · ⏳ GPU run pending |
| 03c — CVAE training | ✅ Reworked (same structure as 03a) · ⏳ GPU run pending |
| 04 — Evaluation | ✅ Reworked: 4 functional-distance metrics, per-hour W1 heatmap, **partial dependence on temperature** · ⏳ Re-run after GPU training |
| 05 — Comparison | ✅ Reworked: 4-way comparison (historical + diffusion + rf + cvae) via `compare_models`, failure-mode → metric-signature interpretation key · ⏳ Re-run after GPU training |
| README.md | ✅ Rewritten (2026-05-28) — full project description, models, evaluation framework, runtime profiles, output layout, roadmap |

---

## Open TODOs

### Verification of the 2026-05-29 fixes
- [x] CPU smoke of 03a end-to-end with the patched train cell —
      confirm `best_model.pkl` is written and `final_val_loss` is non-null
- [ ] Same CPU smoke of 03b and 03c
- [ ] Re-run 04 + 05 on the smoke checkpoints to confirm the load path
      still works

### Residual ML pitfalls worth a second pass (not blocking)
- [ ] **Per-cluster training-loss diagnostic is in the inner loop.**
      `train.py` / `train_rf.py` call an extra `eval_step` *per cluster
      slice in every batch*. On a 5-cluster dataset this is ~5× the
      forward-pass cost of the actual training step. Either gate behind
      `log_cluster_losses=False` for GPU runs, or move it to a once-per-
      epoch sweep over a held-out batch.
- [ ] **`_epoch_len` fallback (`200` / `10_000`) is silent.** If the
      loader ever loses its `epoch_len` attribute (e.g. a wrapped
      iterator) the trainer will run a wildly wrong number of steps
      with no warning. Make it an assertion or surface a printed
      warning.
- [ ] **24 meters, 15 % val → ≈4 val meters.** Per-cluster val pools
      can be empty for some `(cluster, day_type, season)` cells. Flag
      `n_real` next to every reported metric in 04/05 and consider a
      bootstrap-CI or meter-shuffle robustness check before publishing
      a ranking.
- [ ] Confirm DDIM sampler null token (`[-1,-1,-1]`, zeros) is byte-for-
      byte the same as the CFG-dropout null used in `train_step`. (Spot
      check; looks consistent but worth a unit test.)

### GPU training (blocker for everything below)
- [ ] Run 03a (diffusion) on GPU — `TESINA_GPU=1`, ~100k steps
- [ ] Run 03b (rectified flow) on GPU — same
- [ ] Run 03c (CVAE) on GPU — same
- [ ] After each run: verify `output/03*/results/*/training_summary.json` is saved

### Evaluation & comparison
- [ ] Re-run `04_evaluation.ipynb` end-to-end with the trained checkpoints
- [ ] Re-run `05_comparison.ipynb` — pivoted metrics table, per-condition envelope gallery, training-curve overlay
- [ ] Record per-cluster CRPS, ACF L2, marginal Wasserstein, spectral Fréchet for all four models (historical + diffusion + rf + cvae)
- [ ] Inspect partial-dependence plot in 04 §7 — does the diffusion model's daily-total response to temperature match the empirical binned response?
- [ ] Bootstrap confidence intervals on comparison metrics *(deferred — post GPU run)*
- [ ] Wall-clock training time comparison (record manually during GPU runs)

### Data / design decisions
- [ ] Inspect the 24 Rolle meters in 01_eda and confirm whether any need exclusion (per-instance shape normalisation already absorbs scale, but check for sparse / pathological traces)

### Optional extensions (post-thesis, from Lorenzo's note)
- [ ] Generate a full year of synthetic data and compare against real on: annual consumption, frequency content, distance-to-nearest-real-day

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

- `DiffusionTransformer1D`: seq_len=24, d_model=128, n_heads=4, n_layers=4, d_ff=256 → **~845k parameters**
- Discrete conditioning via AdaLN; continuous via cross-attention.
- CFG null token: `[-1, -1, -1]` (discrete), zeros (continuous). Dropout rate `p_uncond=0.15`.
- Inference: DDIM 50 steps, guidance scale 2.0.
- Training: AdamW + cosine LR schedule, gradient clip 1.0, warmup 500 steps.
- **CPU smoke test**: ~0.13 s/step after JIT warm-up; full 100k-step run requires GPU (~2–3 h on Colab A100).
