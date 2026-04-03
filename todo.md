# Notebook Validation — Observations & TODO

Running date: 2026-03-26  
Last updated: 2026-04-03

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
- [x] Use log-transformed values for K-Means or use StandardScaler on log1p(x) to avoid outliers dominating centroid placement
- [x] Explicitly check which cluster(s) the 22 outlier meters fall into — they should ideally form their own cluster(s)
- [x] Evaluate whether K=3 is sufficient or whether K=4–5 better separates outliers from bulk consumers
- [x] Print per-cluster sample count, mean, and std after clustering
- [x] Validate silhouette score / elbow curve

**→ Notebook 03 (Diffusion training)**
- [x] Confirm `STEPS_PER_DAY = 24` everywhere (matches data)
- [x] CFG `day_type` signal is weak — replaced with richer conditioning: `[cluster_id, day_type, month, dow]` (Phase A1)
- [x] Verify that `compute_stats` / `normalize` in `loader.py` is called per-cluster to handle the 3-order-of-magnitude scale range
- [ ] Monitor training loss split by cluster to detect if large-scale clusters are harder to learn

**→ Notebook 04 (Evaluation)**
- [x] When reporting metrics (CRPS, discriminative score), report them split by cluster
- [ ] Weekday vs weekend KDE overlap will likely be high given weak day_type signal — note this in discussion
- [x] Envelope plot: replaced with `per_timestep_stddev_plot` per cluster
- [x] Discriminative score bar chart: quality bands added (green ≤0.52, yellow ≤0.60, red >0.60)

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
- [x] Use the new `clusters.csv` (183/34/104 split); ensure `make_windows` in dataset.py loads it correctly
- [x] Verify `compute_stats` / `normalize` groups meters by these cluster_ids
- [x] Cluster 1 (n=34 meters) = 37 k daily windows — assess if enough for training
- [x] Consider printing per-cluster training window count before training starts (added §1 cell)

---

## Notebook 03 — Diffusion Training (`03_diffusion_training.ipynb`)

### ✅ Status: fully edited; quick-run (5 epochs) completed; **full GPU training pending**

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

### ✅ Status: fully edited; validated on 5-epoch checkpoint; **re-run pending after full GPU training**

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

- [x] Update README / project description: data is **hourly (24 steps/day)** not 15-min/96-step — README fully rewritten
- [ ] Decide definitively whether to include the 22 high-consumption outlier meters in training or exclude them
- [x] Add a `units` note to `loader.py` docstring (values in Wh/h, range 0–764,000; outlier decision rationale documented)
- [x] Fix `metrics.py`: `acf_compare` nlags safeguard, `envelope_plot` steps_per_hour, `discriminative_score` class balance
- [x] Fix `metrics.py` additional improvements: ACF Bartlett 95% CI bands, meaningful KDE hourly bins (Night/Morning/Afternoon/Evening), `sample_diversity_plot()`, `per_timestep_stddev_plot()`, `correlation_heatmap()`, `run_all_metrics()` 3-row layout
- [x] Fix `diffusion.py` freq loss bug: target was `FFT(x0)`, corrected to `FFT(noise)`
- [x] Git: rebased local commits on top of devcontainer commits; pushed to `origin/master` (HEAD: `ad92b4e`)
- [x] Run `04_evaluation.ipynb` with 5-epoch checkpoint — discriminative acc 0.85–0.98 (expected; model not converged)
- [ ] **Full GPU training**: set `QUICK_RUN = False` in nb 03, push to Colab, run ~200 epochs; expected discriminative acc ≤ 0.55
- [ ] **Re-run `04_evaluation.ipynb`** after full training to get thesis-quality metrics

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

- [x] `dataset.py` → `make_windows()`: extract `month` (0–11) and `day_of_week` (0–6) from the DatetimeIndex and append to `cs`
- [x] `transformer1d.py` → add `month_emb: eqx.nn.Embedding(12, d)` and `dow_emb: eqx.nn.Embedding(7, d)`, merge into the conditioning projection
- [x] `diffusion.py` → CFG null token becomes `[-1, -1, -1, -1]`
- [x] `train.py` → `train_step` shapes verified; null mask broadcasts correctly
- [x] Update notebooks 03 & 04 to pass the wider `c` vector (`n_months=12, n_dow=7`; 4-dim `c_batch`)
- [x] Quick sanity run (5 epochs) completed — loss decreasing correctly with richer conditioning
- [x] All 38 tests updated and passing (c shape `(4,)`, null `[-1,-1,-1,-1]`)

**A2. Temperature / exogenous conditioning — DEFERRED**

Temperature data will arrive later with a different dataset. For now, document as future work.
When the data arrives, the architecture is ready to accept it:
- Add a continuous conditioning channel (temperature → small MLP → merge with discrete embeddings)
- This is an extension, not a blocker for the current comparative study

---

---

## ⛔ GATE — Full DDPM run & evaluation (required before Phase B)

**Status: pending (waiting for GPU)**

Before starting Phase B or the comparison framework, the following must be completed and satisfactory:

| Step | Action | Done? |
|------|--------|-------|
| B-gate 1 | Set `QUICK_RUN = False` in `03_diffusion_training.ipynb`, push to Colab/GPU | ⬜ |
| B-gate 2 | Run full ~200-epoch training; confirm loss converges (relative improvement < 2%/epoch) | ⬜ |
| B-gate 3 | Save best checkpoint (`ckpt_epoch_best.pk` or highest-epoch file) | ⬜ |
| B-gate 4 | Run `04_evaluation.ipynb` end-to-end with the full checkpoint | ⬜ |
| B-gate 5 | Check discriminative accuracy ≤ 0.60 on all cluster×day_type conditions | ⬜ |
| B-gate 6 | Inspect per-cluster loss curves (`trainer.cluster_losses`) — flag if any cluster diverges | ⬜ |
| B-gate 7 | Visually inspect denormalised sample profiles vs real (§6 in nb 04) — shapes look plausible | ⬜ |
| B-gate 8 | Export `evaluation_metrics.csv` and record baseline numbers in this todo | ⬜ |

**Baseline numbers to record here after B-gate 8** (fill in after run):

| Condition | Disc. acc | CRPS | ACF L2 |
|-----------|-----------|------|--------|
| cluster0_weekday | — | — | — |
| cluster0_weekend | — | — | — |
| cluster1_weekday | — | — | — |
| cluster1_weekend | — | — | — |
| cluster2_weekday | — | — | — |
| cluster2_weekend | — | — | — |

**Only proceed to Phase B once all B-gate checks pass.**

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

- [x] `src/models/rectified_flow.py` — `RectifiedFlowProcess`: linear interpolation, velocity loss, Euler sampler (50 steps), CFG
- [x] **Reused `DiffusionTransformer1D` backbone** — t scaled via `round(t * 999)` → pseudo-integer for sinusoidal embedding
- [x] `src/training/train_rf.py` — `RFTrainer` + `train_step_rf` (t ~ Uniform[0,1]), per-cluster loss logging, checkpoint save/load
- [x] `notebooks/03b_rectified_flow_training.ipynb` — full training notebook parallel to 03

**B2. (Stretch goal) Implement TimeGAN baseline**

Only pursue after B1 + Phase C are complete and working. Decision point: ~4 weeks before submission.

- [ ] `src/models/timegan.py` — encoder, recovery, generator, discriminator (small RNNs or 1D-conv)
- [ ] `src/training/train_gan.py` — adversarial training loop
- [ ] Notebook `03c_timegan_training.ipynb`

---

### PLAN — Phase C: Comparison framework & evaluation

Currently `04_evaluation.ipynb` is wired to a single model. We need a unified comparison.

**C1. Standardise the evaluation interface**

- [x] `src/evaluation/metrics.py` → `compare_models(models_dict, real_data, conditions, ...)`: takes `{name: generate_fn}`, runs all metrics per condition, returns summary DataFrame
- [x] `marginal_wasserstein()` added — mean 1-D W1 distance over all 24 timesteps

**C2. Notebook `05_comparison.ipynb`** — the core comparative analysis

- [x] Load best checkpoints for DDPM and RF
- [x] Generator wrapper functions (DDIM for DDPM, Euler for RF)
- [x] `compare_models()` call → summary table + CSV export to `data/comparison_metrics.csv`
- [x] Discriminative accuracy bar chart + all-metrics grouped bar chart
- [x] Per-condition mean±σ profile gallery (Real vs DDPM vs RF)
- [x] Training convergence comparison (overlaid loss curves from checkpoints)
- [x] CFG guidance scale ablation (s = 0, 0.5, 1.0, 1.5, 2.5, 4.0)
- [x] Conditioning ablation (full → no dow/month → cluster-only → unconditional)
- [ ] Bootstrap confidence intervals on metrics *(deferred — post GPU run)*

**C3. Thesis figures & tables**

- [x] Summary table (models × metrics × clusters) — `05_comparison.ipynb` §4
- [x] Training convergence comparison — `05_comparison.ipynb` §6
- [x] Sample quality gallery (real vs DDPM vs RF per cluster) — `05_comparison.ipynb` §5
- [ ] Wall-clock training time comparison *(record manually after GPU runs)*

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

| Week | Phase | Milestone | Status |
|------|-------|-----------|--------|
| 1 | A1 | Expanded conditioning implemented & tested | ✅ Done |
| 2 | A1 + Gate | DDPM retrained with new conditioning (cloud GPU) + nb 04 full evaluation | ⏳ Pending (BLOCKER) |
| 3 | B1 | Rectified Flow process + training loop implemented | ✅ Done (code ready; training blocked by gate) |
| 4 | B1 | RF trained on cloud GPU, quick evaluation | ⬜ Not started |
| 5 | C1–C2 | Comparison framework + notebook 05 | ⬜ Not started |
| 6 | C2–C3 | Ablations, figures, tables | ⬜ Not started |
| 7 | D | Thesis writing, discussion, future work | ⬜ Not started |
| 8 | D | Buffer / polish / B2 stretch goal | ⬜ Not started |
