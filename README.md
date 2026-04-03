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
|---|---|
| Architecture | `DiffusionTransformer1D` — Transformer with AdaLN conditioning, trend+seasonality heads |
| Diffusion | DDPM, cosine schedule, T=1000 |
| Fast sampler | DDIM 50 steps |
| Guidance | Classifier-Free Guidance, `guidance_scale=1.5` |
| Conditioning | `c = [cluster_id, day_type, month, dow]` (shape 4, all int32) |
| Null token | `c = [-1, -1, -1, -1]` (CFG unconditional pass) |
| Loss | MSE noise + λ·FFT(pred vs noise), λ=0.05 |
| Parameters | ~846 k (d_model=128, 4 heads, 4 layers) |

## Project Structure

```
src/
  data/
    loader.py      # load_raw(), compute_stats(), normalize(), denormalize()
    dataset.py     # make_windows() → (N,24) xs, (N,4) cs; train_val_split()
  models/
    transformer1d.py   # DiffusionTransformer1D (Equinox)
    diffusion.py       # DiffusionProcess: schedule, q_sample, p_losses, DDIM, CFG
  training/
    train.py       # Trainer, JIT train_step, checkpoint save/load
  evaluation/
    metrics.py     # acf_compare, marginal_kde, sample_diversity_plot,
                   # correlation_heatmap, crps_score, discriminative_score,
                   # run_all_metrics
notebooks/
  01_eda.ipynb              # EDA, outlier analysis, seasonal heatmaps
  02_clustering.ipynb       # Shape-normalised K-Means, clusters.csv
  03_diffusion_training.ipynb
  04_evaluation.ipynb       # Generate samples, full metric suite
  05_comparison.ipynb        # (planned) DDPM vs Rectified Flow
tests/                      # pytest — 38 tests covering all public APIs
data/
checkpoints/
biblio/
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### GPU training (recommended)

Use Google Colab with GPU runtime. Open `notebooks/00_colab_remote_kernel_setup.ipynb` **in Colab**, run it there to start a Jupyter server, then attach VS Code to the printed URL.

### Local CPU run (sanity check only)

```bash
source .venv/bin/activate
python -m pytest tests/          # 38 tests, ~20 s
```

Full training on CPU: ~33 h for 200 epochs. Set `QUICK_RUN = True` in notebook 03 for a 5-epoch validation pass (~10 min CPU / ~5 min GPU).

## Roadmap

- [x] Phase A1 — Expand conditioning: `[cluster, day_type]` → `[cluster, day_type, month, dow]`
- [ ] Phase B1 — Rectified Flow alternative model (`src/models/rectified_flow.py`)
- [ ] Phase C — Comparison notebook 05 (DDPM vs RF, ablations, thesis figures)
- [ ] Phase D — Temperature conditioning (data arriving later)
