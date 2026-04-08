# SUPSI DAS Tesina — Diffusion Model for Smart-Meter Load Profile Generation

DAS thesis project: **conditional generation of synthetic smart-meter electricity load profiles** using a DDPM/DDIM Transformer model with Classifier-Free Guidance, implemented in JAX + Equinox.

## Goal

Train a **Diffusion-TS** style model on a real Portuguese smart-meter dataset (3 years, 321 meters) to generate realistic daily load curves conditioned on meter cluster and calendar features. Evaluate against a **Rectified Flow** baseline using standard generative metrics (CRPS, discriminative score, ACF L2, correlation heatmaps).

## Data

- **Source**: `data/power.pk` — 321 Portuguese smart-meters, hourly timesteps, ~3 years (2012–2014)
- **Resolution**: **hourly, 24 steps/day** (`STEPS_PER_DAY = 24`)
- **Units**: Wh per hour (equivalent to average watts); values in [0, 764 000]
- **Windows**: non-overlapping daily windows of 24 steps
- **Outliers**: 22 meters with mean consumption > 10× dataset median (max ×339) remain in training; cluster-level normalisation handles the 3-order-of-magnitude scale range
- `data/clusters.csv` (tracked): 3-cluster K-Means result — 183 daytime-bell, 34 industrial, 104 evening-peak meters

> `power.pk` is excluded from git (large file). Place it in `data/` before running.

## Model

| Component | Detail |
| --- | --- |
| Architecture | `DiffusionTransformer1D` — Transformer with AdaLN conditioning, trend+seasonality heads |
| Diffusion | DDPM, cosine schedule, T=1000 |
| Fast sampler | DDIM 50 steps |
| Guidance | Classifier-Free Guidance, `guidance_scale=1.5` |
| Conditioning | `c = [cluster_id, day_type, month, dow]` (shape 4, all int32) |
| Null token | `c = [-1, -1, -1, -1]` (CFG unconditional pass) |
| Loss | MSE noise + λ·FFT(pred vs noise), λ=0.05 |
| Parameters | ~846 k (d_model=128, 4 heads, 4 layers) |

## Project Structure

```text
src/
  data/
    loader.py      # load_raw(), compute_stats(), normalize(), denormalize()
    dataset.py     # make_windows() → (N,24) xs, (N,4) cs; train_val_split()
  models/
    transformer1d.py   # DiffusionTransformer1D (Equinox)
    diffusion.py       # DiffusionProcess: schedule, q_sample, p_losses, DDIM, CFG
  training/
    train.py       # Trainer, JIT train_step, checkpoint save/load
  models/
    rectified_flow.py  # RectifiedFlowProcess: linear path, velocity loss, Euler sampler, CFG
  training/
    train_rf.py    # RFTrainer, JIT train_step_rf, checkpoint save/load
  evaluation/
    metrics.py     # acf_compare, marginal_kde, sample_diversity_plot,
                   # correlation_heatmap, crps_score, discriminative_score,
                   # marginal_wasserstein, compare_models, run_all_metrics
notebooks/
  01_eda.ipynb              # EDA, outlier analysis, seasonal heatmaps
  02_clustering.ipynb       # Shape-normalised K-Means, clusters.csv, outlier flagging
  03_diffusion_training.ipynb
  03b_rectified_flow_training.ipynb  # RF training (mirrors 03)
  04_evaluation.ipynb       # Generate samples, full metric suite
  05_comparison.ipynb       # DDPM vs RF: metrics, ablations, thesis figures
tests/                      # pytest — 53 tests covering all public APIs
data/
checkpoints/
biblio/
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Artifact Policy

- Checkpoints are written to `checkpoints/`
- Notebook outputs and metric exports are written to `results/`
- By default, notebooks do not redirect artifacts to Google Drive

If you attach VS Code to a Colab GPU kernel, the repo-local path is typically
`/content/tesina/...`. Those files live inside the runtime filesystem and must
be copied out manually if you want to keep them after the session ends.

If you download an exported bundle from Colab, place it under
`results/exports/` in the local repo and restore it with:

```bash
python scripts/restore_export_bundle.py results/exports/<bundle-name>.tar.gz
```

Current bundle prefixes:

- `ddpm_baseline_*.tar.gz` from `notebooks/03_diffusion_training.ipynb`
- `rf_baseline_*.tar.gz` from `notebooks/03b_rectified_flow_training.ipynb`

Notebook behavior after download:

- `notebooks/04_evaluation.ipynb` auto-restores the latest DDPM bundle if `checkpoints/best_model.pkl` is missing
- `notebooks/05_comparison.ipynb` auto-restores the latest DDPM and RF bundles if either checkpoint is missing

### GPU training (recommended)

Use Google Colab with GPU runtime. Open `notebooks/00_colab_remote_kernel_setup.ipynb` **in Colab**, run it there to start a Jupyter server, then attach VS Code to the printed URL.

For the DDPM baseline, run these notebooks in order:

1. `notebooks/03_diffusion_training.ipynb` with `QUICK_RUN = False`
2. `notebooks/04_evaluation.ipynb`

For the RF baseline and side-by-side comparison:

1. `notebooks/03b_rectified_flow_training.ipynb` with `QUICK_RUN = False`
2. `notebooks/05_comparison.ipynb`

Expected repo-local outputs:

- `checkpoints/best_model.pkl`
- `checkpoints/rf_best_model.pkl`
- `results/diffusion/training_summary.json`
- `results/rectified_flow/training_summary.json`
- `results/evaluation/evaluation_metrics.csv`
- `results/comparison/comparison_metrics.csv`
- `results/exports/ddpm_baseline_*.tar.gz`
- `results/exports/rf_baseline_*.tar.gz`

### Local CPU run (sanity check only)

```bash
source .venv/bin/activate
python -m pytest tests/          # 38 tests, ~20 s
```

Full training on CPU: ~33 h for 200 epochs. Set `QUICK_RUN = True` in notebook 03 for a 5-epoch validation pass (~10 min CPU / ~5 min GPU).

## Roadmap

- [x] Phase A1 — Expand conditioning: `[cluster, day_type]` → `[cluster, day_type, month, dow]`
- [x] Phase B1 — Rectified Flow model: `src/models/rectified_flow.py`, `src/training/train_rf.py`, `notebooks/03b_rectified_flow_training.ipynb`
- [x] Phase C infra — Comparison framework: `compare_models()`, `marginal_wasserstein()` in `metrics.py`, `notebooks/05_comparison.ipynb`
- [ ] **GATE** — Full DDPM GPU training (~200 epochs) + `04_evaluation.ipynb` validation ← *in progress*
- [ ] Full RF GPU training (`03b`) + run `05_comparison.ipynb`
- [ ] Phase D — Temperature conditioning (data arriving later)
