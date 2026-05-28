# Project notes & open TODOs

Last updated: 2026-05-28

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

### Normalisation fix (2026-05-28) — verification
- [ ] Run notebooks 01 → 05 end-to-end (CPU smoke profile) to confirm the per-instance shape-normalisation refactor executes cleanly and metrics now report in Watts

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
