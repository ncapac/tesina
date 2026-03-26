# SUPSI DAS Tesina — Diffusion Model for Energy Load Forecasting

DAS thesis project for the SUPSI DAS 'Big Data Analytics e Machine Learning' programme.

## Goal

Train a **Diffusion-TS** model (DDPM/DDIM with Transformer backbone) on smart-meter
energy load data to generate realistic probabilistic time-series forecasts.

## Project Structure

```
src/
  data/         # Dataset loading & preprocessing
  models/       # Diffusion model + Transformer1D backbone
  training/     # Training loop
  evaluation/   # Metrics (CRPS, discriminative score, …)
notebooks/
  01_eda.ipynb              # Exploratory data analysis
  02_clustering.ipynb       # Smart-meter clustering
  03_diffusion_training.ipynb
  04_evaluation.ipynb
data/                       # Raw data (large files excluded via .gitignore)
checkpoints/                # Saved model weights (excluded via .gitignore)
biblio/                     # Reference papers
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place `power.pk` inside `data/` (not tracked by git due to size).
The CSV metadata file is included in the repo.
