"""
train.py
--------
JIT-compiled training step and training loop for the diffusion model.

Key features
------------
  • optax.adamw + cosine LR schedule with linear warmup
  • CFG dropout: conditioning randomly zeroed with p_uncond during training
  • Checkpoint saving / loading (numpy pickle)
  • Validation loss tracked each epoch

Usage
-----
    from src.models.transformer1d import DiffusionTransformer1D
    from src.models.diffusion import DiffusionProcess
    from src.training.train import Trainer

    model = DiffusionTransformer1D(...)
    diffusion = DiffusionProcess(T=1000)
    trainer = Trainer(model, diffusion, lr=1e-3, warmup_steps=500,
                      total_steps=50_000, checkpoint_dir='checkpoints/')
    trainer.fit(train_loader, val_loader, n_epochs=100)
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx


# ---------------------------------------------------------------------------
# Training step (pure function, JIT-compiled)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def train_step(
    model: eqx.Module,
    diffusion,                    # DiffusionProcess
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x0: jax.Array,               # (B, L) float32
    c: jax.Array,                # (B, 2) int32
    key: jax.Array,
    p_uncond: float = 0.15,
) -> Tuple[eqx.Module, optax.OptState, jax.Array]:
    """
    One JIT-compiled training step.

    Returns updated model, updated opt_state, and scalar loss.
    """
    key_cfg, key_t, key_noise = jax.random.split(key, 3)

    B = x0.shape[0]

    # CFG: randomly null out conditioning
    null_mask = jax.random.bernoulli(key_cfg, p=p_uncond, shape=(B,))   # (B,)
    c_train = jnp.where(null_mask[:, None], jnp.full_like(c, -1), c)

    # Sample random diffusion timesteps
    t = jax.random.randint(key_t, shape=(B,), minval=0, maxval=diffusion.T)

    def loss_fn(m):
        return diffusion.p_losses(m, x0, c_train, t, key_noise)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


@eqx.filter_jit
def eval_step(
    model: eqx.Module,
    diffusion,
    x0: jax.Array,
    c: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Validation loss (no gradient)."""
    B = x0.shape[0]
    t = jax.random.randint(key, shape=(B,), minval=0, maxval=diffusion.T)
    return diffusion.p_losses(model, x0, c, t, key)


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: eqx.Module,
        diffusion,
        lr: float = 1e-3,
        warmup_steps: int = 500,
        total_steps: int = 100_000,
        checkpoint_dir: str = "checkpoints",
        p_uncond: float = 0.15,
        seed: int = 0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.p_uncond = p_uncond
        self.key = jax.random.PRNGKey(seed)

        # Optimizer: AdamW + cosine LR with linear warmup
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=lr * 0.01,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # gradient clipping
            optax.adamw(schedule, weight_decay=1e-4),
        )
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        self.step = 0
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.cluster_losses: dict[int, list[float]] = {}  # {cluster_id: [mean loss per epoch]}

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterator,
        val_loader: Optional[Iterator],
        n_epochs: int = 100,
        val_every: int = 1,
        save_every: int = 10,
        log_every_steps: int = 50,
        val_batches: int = 20,
        log_cluster_losses: bool = True,
        patience: int = 20,
        min_delta: float = 1e-4,
    ):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            epoch_losses = []
            # per-cluster loss accumulation: {cluster_id: [losses]}
            cluster_loss_accum: dict[int, list[float]] = {}

            for x0_np, c_np in train_loader:
                self.key, subkey = jax.random.split(self.key)
                x0 = jnp.array(x0_np)
                c  = jnp.array(c_np)

                self.model, self.opt_state, loss = train_step(
                    self.model, self.diffusion,
                    self.opt_state, self.optimizer,
                    x0, c, subkey, self.p_uncond,
                )
                self.step += 1
                loss_val = float(loss)
                epoch_losses.append(loss_val)

                # Accumulate per-cluster loss (best-effort: compute per cluster subset)
                if log_cluster_losses:
                    cluster_ids = np.array(c_np[:, 0])
                    for cid in np.unique(cluster_ids):
                        mask = cluster_ids == cid
                        x0_c = jnp.array(x0_np[mask])
                        c_c  = jnp.array(c_np[mask])
                        self.key, sk2 = jax.random.split(self.key)
                        cl_loss = eval_step(self.model, self.diffusion, x0_c, c_c, sk2)
                        cluster_loss_accum.setdefault(int(cid), []).append(float(cl_loss))

                if self.step % log_every_steps == 0:
                    print(f"  step {self.step:6d}  loss {loss_val:.4f}")

                # One epoch = one pass through data — break after N batches
                # (train_loader is infinite; we bound epoch length externally)
                if len(epoch_losses) >= _epoch_len(train_loader):
                    break

            mean_tr = float(np.mean(epoch_losses))
            self.train_losses.append(mean_tr)

            val_str = ""
            if val_loader is not None and epoch % val_every == 0:
                val_loss_vals = []
                for i, (xv_np, cv_np) in enumerate(val_loader):
                    if i >= val_batches:
                        break
                    self.key, subkey = jax.random.split(self.key)
                    xv = jnp.array(xv_np)
                    cv = jnp.array(cv_np)
                    vl = eval_step(self.model, self.diffusion, xv, cv, subkey)
                    val_loss_vals.append(float(vl))
                mean_val = float(np.mean(val_loss_vals))
                self.val_losses.append(mean_val)
                val_str = f"  val_loss {mean_val:.4f}"

            elapsed = time.time() - t0
            cluster_str = ""
            if log_cluster_losses and cluster_loss_accum:
                parts = [
                    f"C{cid}={np.mean(losses):.4f}"
                    for cid, losses in sorted(cluster_loss_accum.items())
                ]
                cluster_str = "  [" + "  ".join(parts) + "]"
                # store per-epoch mean per cluster
                for cid, losses in cluster_loss_accum.items():
                    self.cluster_losses.setdefault(cid, []).append(float(np.mean(losses)))
            print(
                f"Epoch {epoch:3d}/{n_epochs}  train_loss {mean_tr:.4f}"
                f"{val_str}  [{elapsed:.1f}s]{cluster_str}"
            )

            if epoch % save_every == 0:
                self.save(f"ckpt_epoch{epoch:04d}.pk")

            # Early stopping on validation loss
            if self.val_losses:
                current_val = self.val_losses[-1]
                if current_val < best_val_loss - min_delta:
                    best_val_loss = current_val
                    epochs_without_improvement = 0
                    self.save("best_model.pkl")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"\nEarly stopping at epoch {epoch} "
                              f"(no improvement for {patience} epochs). "
                              f"Best val loss: {best_val_loss:.4f}")
                        break

    # ------------------------------------------------------------------
    def save(self, filename: str):
        path = self.checkpoint_dir / filename
        with open(path, "wb") as f:
            pickle.dump({
                "model":     self.model,
                "opt_state": self.opt_state,
                "step":      self.step,
                "train_losses": self.train_losses,
                "val_losses":   self.val_losses,
                "cluster_losses": self.cluster_losses,
            }, f)
        print(f"  ✓ checkpoint saved → {path}")

    def load(self, filename: str):
        path = self.checkpoint_dir / filename
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.model      = ckpt["model"]
        self.opt_state  = ckpt["opt_state"]
        self.step       = ckpt["step"]
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses   = ckpt.get("val_losses", [])
        self.cluster_losses = ckpt.get("cluster_losses", {})
        print(f"  ✓ checkpoint loaded ← {path}  (step {self.step})")


def _epoch_len(loader) -> int:
    """Try to infer epoch length from loader attribute, else default."""
    return getattr(loader, "epoch_len", 200)
