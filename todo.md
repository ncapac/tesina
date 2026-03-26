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
- [x] Fix `metrics.py`: `acf_compare` nlags safeguard, `envelope_plot` steps_per_hour, `discriminative_score` class balance
- [ ] Run `04_evaluation.ipynb` once background training completes (checkpoints/best_model.pkl)  
      Training status: step 2336/23375, loss 0.9218 (both CPU procs competing; ETA several hours)
- [ ] For thesis-quality results: re-train on GPU (~200 epochs); expected discriminative acc ≤ 0.55

---

## ⚠ MAJOR GAP — Comparative Study & Conditioning (added 2026-03-26)

### Problem statement

The project scope (see `biblio/initial_docs/plan.txt` and `todo.txt`) explicitly requires:

1. **"Compare different time series generative tools"** — flow matching, rectified flows, etc.
2. **"Conditional to metadata and exogenous inputs (e.g. external temperature)"**
3. **"Compare … in terms of generative expressivity while respecting in-sample time series characteristics"**

Currently the codebase implements **only DDPM/DDIM with a Transformer backbone**. There is no alternative generative method, no comparison framework, and only minimal conditioning (cluster_id + weekday/weekend binary). This is the single biggest missing piece in the thesis.

---

### PLAN — Phase A: Richer conditioning on the existing DDPM model

The current conditioning vector `c = [cluster_id, day_type]` is thin. Day-type is almost useless (only 1.4% difference weekday vs weekend). Before adding new models, make the existing one properly conditioned.

**A1. Expand the conditioning vector** `c = [cluster_id, day_type, month, day_of_week]`

- [ ] `dataset.py` → `make_windows()`: extract `month` (0–11) and `day_of_week` (0–6) from the DatetimeIndex and append to `cs`
- [ ] `transformer1d.py` → add `month_emb: eqx.nn.Embedding(12, d)` and `dow_emb: eqx.nn.Embedding(7, d)`, merge into the conditioning projection
- [ ] `diffusion.py` → CFG null token becomes `[-1, -1, -1, -1]`
- [ ] `train.py` → `train_step` already broadcasts null mask; just verify shapes
- [ ] Update notebook 03 to pass the wider `c` vector
- [ ] Re-run training (quick sanity) and confirm loss drops faster with richer conditioning

**A2. Temperature / exogenous conditioning — DEFERRED**

Temperature data will arrive later with a different dataset. For now, document as future work.
When the data arrives, the architecture is ready to accept it:
- Add a continuous conditioning channel (temperature → small MLP → merge with discrete embeddings)
- This is an extension, not a blocker for the current comparative study

---

### PLAN — Phase B: Alternative generative method(s)

The plan.txt+todo.txt explicitly list **flow matching** and **rectified flows** as alternatives to explore. We need at least **one alternative method** trained on the same data with the same conditioning and evaluated on the same metrics to make a meaningful comparison. With a **2-month timeline** and **cloud GPU** (uploaded git repo), we target **Rectified Flow as the primary alternative** and **TimeGAN as a stretch goal** if time permits.

**Candidate methods** (all implementable in JAX/Equinox, reusing the same Transformer backbone):

| # | Method | Key idea | Implementation effort | Notes |
|---|--------|----------|----------------------|-------|
| 1 | **Rectified Flow (RF)** | ODE-based, linear interpolation `x_t = (1-t)·x_0 + t·ε`, velocity prediction `v_θ(x_t, t, c)` | **Low** — same backbone, replace noise schedule with linear path, replace `p_losses` with velocity MSE, replace sampler with Euler ODE solver | **Primary alternative.** Liu et al. 2023. Clean, modern, directly comparable to DDPM. |
| 2 | **Conditional Flow Matching (CFM)** | Generalisation of RF with optimal-transport paths | **Low-Medium** — same as RF but with OT-conditioned paths; can start with simple RF and optionally add OT mini-batch coupling | Lipman et al. 2023. Practically a superset of RF. Could be a variant within the RF notebook. |
| 3 | **TimeGAN-style GAN** | Adversarial + supervised + reconstruction losses | **Medium** — needs discriminator + different training loop, but useful as non-diffusion baseline | Yoon et al. 2019. **Stretch goal** — only if Phase B1 + C are done with time to spare. |

**Decision**: Implement **Rectified Flow** (B1). Decide on TimeGAN (B2) after B1+C are working.

**B1. Implement Rectified Flow**

- [ ] `src/models/rectified_flow.py` — new `RectifiedFlowProcess` class:
  - Linear interpolation forward: `x_t = (1-t)·x_0 + t·ε` for `t ∈ [0,1]`
  - Loss: `MSE(v_θ(x_t, t, c), ε - x_0)` (velocity matching)
  - Sampler: Euler ODE solver, N steps (e.g. 50–100)
  - CFG: same formula as DDPM — `v_guided = (1+s)·v_cond - s·v_uncond`
- [ ] **Reuse the existing `DiffusionTransformer1D` backbone** — the denoiser architecture is agnostic to the noise process. Only `__call__` signature needs `t` to be continuous [0,1] instead of discrete [0,T]. Add a flag or normalise `t` internally.
- [ ] `src/training/train_rf.py` — training step for RF (same structure as `train.py`, different loss)
- [ ] Notebook `03b_rectified_flow_training.ipynb` — parallel to 03, same data pipeline

**B2. (Stretch goal) Implement TimeGAN baseline**

Only pursue after B1 + Phase C are complete and working. Decision point: ~4 weeks before submission.

- [ ] `src/models/timegan.py` — encoder, recovery, generator, discriminator (small RNNs or 1D-conv)
- [ ] `src/training/train_gan.py` — adversarial training loop
- [ ] Notebook `03c_timegan_training.ipynb`

---

### PLAN — Phase C: Comparison framework & evaluation

Currently `04_evaluation.ipynb` is wired to a single model. We need a unified comparison.

**C1. Standardise the evaluation interface**

- [ ] `src/evaluation/metrics.py` → add `compare_models(models_dict, real_data, conditions, ...)` that:
  - Takes a dict of `{model_name: sample_generator_fn}`
  - Generates N samples per (cluster, day_type) for each model
  - Runs all metrics (ACF L2, CRPS, discriminative score, marginal KDE, envelope)
  - Returns a summary DataFrame and composite figure
- [ ] Add a **context-FID** or **marginal Wasserstein distance** metric (common in time-series generation literature) for an additional comparison axis

**C2. Notebook `05_comparison.ipynb`** — the core comparative analysis

- [ ] Load best checkpoints for DDPM and RF (and TimeGAN if available)
- [ ] Generate matched sample sets (same conditions, same sample count)
- [ ] Side-by-side metric table: rows = metrics, columns = models
- [ ] Per-cluster comparison plots
- [ ] Statistical significance: bootstrap confidence intervals on metrics
- [ ] Ablation: effect of conditioning features (cluster only vs cluster+day_type vs cluster+day_type+month+dow)
- [ ] Ablation: guidance scale sweep (s = 0, 0.5, 1.0, 1.5, 2.0, 3.0)

**C3. Thesis figures & tables**

- [ ] Summary table of all models × all metrics × all clusters
- [ ] Training convergence comparison (loss curves overlaid)
- [ ] Sample quality gallery: grid of real vs DDPM vs RF (vs TimeGAN) for each cluster
- [ ] Wall-clock training time comparison

---

### PLAN — Phase D: Documentation & write-up support

All comparison results will be in a **single thesis chapter** (not per-method chapters).

- [ ] Update `README.md` to reflect the comparative scope, list all models
- [ ] Thesis structure: single "Results & Comparison" chapter with subsections per metric
- [ ] Method descriptions: DDPM (current), Rectified Flow — theory + implementation differences
- [ ] Related work section: position these methods in the generative time-series literature
- [ ] Discussion: why RF may outperform/underperform DDPM on this specific dataset (small L=24, discrete conditioning, moderate dataset size)
- [ ] Future work: temperature conditioning (data arriving later), TimeGAN if not completed, continuous-time extensions

---

### Decisions log (2026-03-26)

| Question | Answer |
|----------|--------|
| Number of alternative methods | TBD — at least RF; TimeGAN only if time permits |
| Temperature data | Deferred — will arrive later with different data |
| GPU access | Cloud GPU (git repo uploaded online) |
| Timeline | ~2 months to submission |
| Conditioning depth | `[cluster, day_type, month, dow]` sufficient for now |
| Thesis chapter structure | Single comparison chapter |

### Rough timeline (8 weeks)

| Week | Phase | Milestone |
|------|-------|-----------|
| 1 | A1 | Expanded conditioning implemented & tested |
| 2 | A1 | DDPM retrained with new conditioning (cloud GPU) |
| 3 | B1 | Rectified Flow process + training loop implemented |
| 4 | B1 | RF trained on cloud GPU, quick evaluation |
| 5 | C1–C2 | Comparison framework + notebook 05 |
| 6 | C2–C3 | Ablations, figures, tables |
| 7 | D | Thesis writing, discussion, future work |
| 8 | D | Buffer / polish / B2 stretch goal |
