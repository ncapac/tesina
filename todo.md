# Notebook Validation — Observations & TODO

Running date: 2026-03-26

---

## Notebook 01 — EDA (`01_eda.ipynb`)

### ✅ Status: validated & refined

### Key findings

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | **Data is hourly (24 steps/day)**, not 15-min/96-step as stated in project docs / README | Low | Doc fix: update README/mode instructions to say "hourly, 24-step windows" |
| 2 | **321 meters**, not 320 (one extra compared to project description) | Low | Verify: probably harmless, just update docs |
| 3 | **22 outlier meters** (mean > 10× median). Top: meter 313 (×339 median), meter 155 (×133), meter 236 (×78) | High | See §Downstream below |
| 4 | **Weekday vs weekend difference is ~1.4%** — CFG `day_type` conditioning signal is very weak in this dataset | Medium | Note in discussion/results chapter |
| 5 | §4 "Average daily profile" was misleading: flat shape + negative lower std band due to mixing all meters at raw scale | Fixed | ✅ Notebook updated (§4b added with log-scale) |
| 6 | §5 Seasonal heatmap was completely washed out by outlier meters | Fixed | ✅ Added side-by-side raw / log-scale heatmap |
| 7 | §6 Distribution plots bins all compressed at 0 by extreme outliers | Fixed | ✅ Switched to log₁₀ x-axis |
| 8 | §7 Notes was a blank placeholder | Fixed | ✅ Filled with real observations |
| 9 | New §2b added: outlier meter identification with log-rank plot | Added | ✅ |
| 10 | New §4b added: weekday vs weekend comparison (normal meters only) | Added | ✅ |

### Diurnal profile (normal meters)
- Minimum at ~03–04 h, rising from 06 h, peak at 19–20 h.
- Consumption is 0-based (min=0, no negatives). Units likely Wh/hour (equivalent to average W).

### Downstream implications for notebooks 02–04

**→ Notebook 02 (Clustering)**
- [ ] Use log-transformed values for K-Means or use StandardScaler on log1p(x) to avoid outliers dominating centroid placement
- [ ] Explicitly check which cluster(s) the 22 outlier meters fall into — they should ideally form their own cluster(s)
- [ ] Evaluate whether K=3 is sufficient or whether K=4–5 better separates outliers from bulk consumers
- [ ] Print per-cluster sample count, mean, and std after clustering
- [ ] Validate silhouette score / elbow curve

**→ Notebook 03 (Diffusion training)**
- [ ] Confirm `STEPS_PER_DAY = 24` everywhere (matches data)
- [ ] CFG `day_type` signal is weak — consider dropping or replacing with more informative features (e.g. month, season) if results don't separate
- [ ] Verify that `compute_stats` / `normalize` in `loader.py` is called per-cluster to handle the 3-order-of-magnitude scale range
- [ ] Monitor training loss split by cluster to detect if large-scale clusters are harder to learn

**→ Notebook 04 (Evaluation)**
- [ ] When reporting metrics (CRPS, discriminative score), report them split by cluster
- [ ] Weekday vs weekend KDE overlap will likely be high given weak day_type signal — note this in discussion
- [ ] Envelope plot: generate separate envelopes per cluster, not just one global envelope
- [ ] Discriminative score ~0.5 is the target; expect it may be harder to achieve for outlier clusters

---

## Notebook 02 — Clustering (`02_clustering.ipynb`)

### ✅ Status: validated & refined

### Key findings

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | **Original clustering was degenerate**: StandardScaler + raw mean/std → n=310/1/10. Purely scale-based, not shape-based. | Critical | Fixed ✅ |
| 2 | Switched to **shape-normalised 24-dim profiles** (unit-mean per meter) → PCA (6D, 95% var) → K-Means k=3 | Core fix | ✅ |
| 3 | Results: **n=183/34/104** — balanced, interpretable shape clusters | Good | ✅ |
| 4 | Three distinct shapes: C0=daytime bell, C1=midday plateau (industrial), C2=evening-peak (residential) | Good | ✅ |
| 5 | Silhouette scores are moderate (0.47 at k=3) — typical for load-profile clustering | Expected | Note in thesis |
| 6 | Outlier meters distributed across all clusters (6%/21%/4%) — per-cluster normalisation handles scale | OK | ✅ |
| 7 | Raw-scale centroid plots showed negative std bands — replaced with dual-view (raw + normalised) | Fixed | ✅ |

### Downstream implications for notebooks 03–04

**→ Notebook 03 (Diffusion training)**
- [ ] Use the new `clusters.csv` (183/34/104 split); ensure `make_windows` in dataset.py loads it correctly
- [ ] Verify `compute_stats` / `normalize` groups meters by these cluster_ids
- [ ] Cluster 1 (n=34 meters) = 37 k daily windows — assess if enough for training
- [ ] Consider printing per-cluster training window count before training starts

---

## Notebook 03 — Diffusion Training (`03_diffusion_training.ipynb`)

### ✅ Status: fully edited; background training running (PIDs 719740 + 721269, both CPU)

### Key findings & fixes

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | `CHECKPOINT_NAME` variable missing `CHECKPOINT_DIR`; save cell called `trainer.save_checkpoint()` (wrong method) | Bug | Fixed ✅ `trainer.save(CHECKPOINT_NAME)` |
| 2 | Sample inspection cell showed normalised values only — no physical scale comparison | Incomplete | Fixed ✅ added `denormalize()` + real-sample overlay |
| 3 | Loss-curve cell showed only raw loss — no convergence indicator | Incomplete | Fixed ✅ added % relative-improvement subplot |
| 4 | Training duration on CPU: ~0.13 s/step × 4675 steps/epoch × 50 epochs ≈ 8 hours | Blocker | Added `QUICK_RUN=True` flag (5 epochs) + note for GPU |
| 5 | `c_batch` in sample cell lacked `dtype=jnp.int32` — could silently produce wrong conditioning | Bug | Fixed ✅ in related code |
| 6 | No per-cluster window count at data-loading stage | Completeness | Fixed ✅ added to §1 cell |
| 7 | Model checkpoint load-on-resume logic missing | Completeness | Fixed ✅ cell 7 loads `CHECKPOINT_NAME` if it exists |
| 8 | `log_every_steps` was hardcoded; `save_every` was over-aggressive | Style | Fixed ✅ computed dynamically |

### Pipeline validation (terminal run)

| Check | Result |
|-------|--------|
| Cluster layout | C0=169,880 train / C1=31,784 / C2=97,544 |
| z-score stats | mean≈+0.025 (slight bias expected from train/val split), std≈1.077 |
| Model params | 845,890 ✓ |
| Denormalize | cluster-0 zero→2321 Wh ✓ |
| Step 1168 loss | 0.9734 (training confirmed working) |
- 845,890 parameters, seq_len=24, d_model=128, n_heads=4, n_layers=4
- CFG with null conditioning `c=[-1,-1]`, guidance_scale=1.5 at inference
- AdamW + cosine warmup schedule, gradient clip=1.0
- DDIM 50 steps for fast inference, DDPM for training

### CPU performance (observed)
- First JIT step: ~4 s | subsequent steps: ~0.13 s
- Steps per epoch: 4,675 | estimated epoch time: ~606 s (~10 min)
- Full 5-epoch quick run: ~50 min on CPU
- Full training (200 ep): ~33 h CPU → **must use GPU for real training**

### Downstream implications

**→ Notebook 04 (Evaluation)**
- [ ] After full GPU training, reload best checkpoint and rerun all metrics
- [ ] After only 5 epochs, discriminative score will be high (model not converged) — expected
- [ ] Sample quality at 5 epochs: shapes may have rough temporal structure, not yet sharp peaks

---

## Notebook 04 — Evaluation (`04_evaluation.ipynb`)

### ✅ Status: fully edited, ready to run once checkpoint is available

### Key findings & fixes

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | `acf_compare` called with `nlags=48` but sequence length is 24 — lags ≥ 24 produce artifact zeros | Bug | Fixed ✅ in `metrics.py`: auto-clamp to `min(nlags, L-1)`, default changed to 23 |
| 2 | `envelope_plot` default `steps_per_hour=4` (15-min design) but data is hourly 24-step | Bug | Fixed ✅ default `steps_per_hour=1` |
| 3 | `discriminative_score` used all real samples (up to 30,688) vs 100 synthetic → class imbalance → trivial classifier | Bug | Fixed ✅ class balancing via subsampling to `min(N_real, N_syn)` |
| 4 | `c_batch` dtype in generation cell lacked `dtype=jnp.int32` | Bug | Fixed ✅ |
| 5 | Data cell discarded `x_train`/`c_train` (underscore) | Completeness | Fixed ✅ |
| 6 | No final observations cell | Completeness | Added ✅ §8 with per-condition quality printout |

### Metric validation (random normals)
- ACF L2 (random vs random, 24 lags): 0.077 — correct for finite-sample noise
- Discriminative acc (200 real, 100 synth; balanced): 0.617 — expected slight overfit with tiny samples
- CRPS: 0.555 — expected for random normal ensembles

### Expected results (5-epoch quick-run checkpoint)
- Discriminative accuracy: likely 0.7–0.9 (model not converged → easy to distinguish)
- CRPS: hard to benchmark without baseline; will improve with training
- ACF L2: possibly 0.1–0.3 for early-training samples
- After full training (200 ep): discriminative acc target ≤ 0.55

---

## Global / Cross-cutting TODOs

- [ ] Update README / project description: data is **hourly (24 steps/day)** not 15-min/96-step
- [ ] Decide definitively whether to include the 22 high-consumption outlier meters in training or exclude them
- [ ] Add a `units` note to `loader.py` docstring (values appear to be in Wh or average W; min=0, max=764,000)
- [ ] Consider adding temperature/meteorological conditioning (mentioned in plan.txt) as future work if not feasible now
- [x] Fix `metrics.py`: `acf_compare` nlags safeguard, `envelope_plot` steps_per_hour, `discriminative_score` class balance
- [ ] Run `04_evaluation.ipynb` once background training completes (checkpoints/best_model.pkl)  
      Training status: step 2336/23375, loss 0.9218 (both CPU procs competing; ETA several hours)
- [ ] For thesis-quality results: re-train on GPU (~200 epochs); expected discriminative acc ≤ 0.55
