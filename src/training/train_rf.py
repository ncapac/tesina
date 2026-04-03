"""
train_rf.py
-----------
JIT-compiled training step and training loop for the Rectified Flow model.

Mirrors the interface of train.py; the only substantive difference is that
diffusion timesteps are sampled from Uniform[0, 1] (continuous) rather than
Uniform{0, …, T-1} (discrete).

Usage
-----
    from src.models.transformer1d import DiffusionTransformer1D
    from src.models.rectified_flow import RectifiedFlowProcess
    from src.training.train_rf import RFTrainer

    model = DiffusionTransformer1D(...)
    rf    = RectifiedFlowProcess()
    trainer = RFTrainer(model, rf, lr=1e-3, warmup_steps=500,
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
def train_step_rf(
    model: eqx.Module,
    rf,                           # RectifiedFlowProcess
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x0: jax.Array,               # (B, L) float32
    c: jax.Array,                # (B, 4) int32
    key: jax.Array,
    p_uncond: float = 0.15,
) -> Tuple[eqx.Module, optax.OptState, jax.Array]:
    """
    One JIT-compiled RF training step.

    t is sampled from Uniform[0, 1] (continuous RF convention).
    Returns updated model, updated opt_state, and scalar loss.
    """
    key_cfg, key_t, key_noise = jax.random.split(key, 3)

    B = x0.shape[0]

    # CFG: randomly null out conditioning
    null_mask = jax.random.bernoulli(key_cfg, p=p_uncond, shape=(B,))
    c_train = jnp.where(null_mask[:, None], jnp.full_like(c, -1), c)

    # Sample continuous t ~ Uniform[0, 1]
    t = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=1.0)

    def loss_fn(m):
        return rf.p_losses(m, x0, c_train, t, key_noise)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


@eqx.filter_jit
def eval_step_rf(
    model: eqx.Module,
    rf,
    x0: jax.Array,
    c: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Validation loss (no gradient)."""
    B = x0.shape[0]
    key_t, key_n = jax.random.split(key)
    t = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=1.0)
    return rf.p_losses(model, x0, c, t, key_n)


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class RFTrainer:
    def __init__(
        self,
        model: eqx.Module,
        rf,                          # RectifiedFlowProcess
        lr: float = 1e-3,
        warmup_steps: int = 500,
        total_steps: int = 100_000,
        checkpoint_dir: str = "checkpoints",
        p_uncond: float = 0.15,
        seed: int = 0,
    ):
        self.model = model
        self.rf    = rf
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.p_uncond = p_uncond
        self.key = jax.random.PRNGKey(seed)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=lr * 0.01,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=1e-4),
        )
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        self.step = 0
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.cluster_losses: dict[int, list[float]] = {}

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
    ):
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            epoch_losses = []
            cluster_loss_accum: dict[int, list[float]] = {}

            for x0_np, c_np in train_loader:
                self.key, subkey = jax.random.split(self.key)
                x0 = jnp.array(x0_np)
                c  = jnp.array(c_np)

                self.model, self.opt_state, loss = train_step_rf(
                    self.model, self.rf,
                    self.opt_state, self.optimizer,
                    x0, c, subkey, self.p_uncond,
                )
                self.step += 1
                loss_val = float(loss)
                epoch_losses.append(loss_val)

                if log_cluster_losses:
                    cluster_ids = np.array(c_np[:, 0])
                    for cid in np.unique(cluster_ids):
                        mask = cluster_ids == cid
                        x0_c = jnp.array(x0_np[mask])
                        c_c  = jnp.array(c_np[mask])
                        self.key, sk2 = jax.random.split(self.key)
                        cl_loss = eval_step_rf(self.model, self.rf, x0_c, c_c, sk2)
                        cluster_loss_accum.setdefault(int(cid), []).append(float(cl_loss))

                if self.step % log_every_steps == 0:
                    print(f"  step {self.step:6d}  loss {loss_val:.4f}")

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
                    vl = eval_step_rf(self.model, self.rf, xv, cv, subkey)
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
                for cid, losses in cluster_loss_accum.items():
                    self.cluster_losses.setdefault(cid, []).append(float(np.mean(losses)))
            print(
                f"Epoch {epoch:3d}/{n_epochs}  train_loss {mean_tr:.4f}"
                f"{val_str}  [{elapsed:.1f}s]{cluster_str}"
            )

            if epoch % save_every == 0:
                self.save(f"rf_ckpt_epoch{epoch:04d}.pk")

    # ------------------------------------------------------------------
    def save(self, filename: str):
        path = self.checkpoint_dir / filename
        with open(path, "wb") as f:
            pickle.dump({
                "model":          self.model,
                "opt_state":      self.opt_state,
                "step":           self.step,
                "train_losses":   self.train_losses,
                "val_losses":     self.val_losses,
                "cluster_losses": self.cluster_losses,
            }, f)
        print(f"  ✓ checkpoint saved → {path}")

    def load(self, filename: str):
        path = self.checkpoint_dir / filename
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.model          = ckpt["model"]
        self.opt_state      = ckpt["opt_state"]
        self.step           = ckpt["step"]
        self.train_losses   = ckpt.get("train_losses", [])
        self.val_losses     = ckpt.get("val_losses", [])
        self.cluster_losses = ckpt.get("cluster_losses", {})
        print(f"  ✓ checkpoint loaded ← {path}  (step {self.step})")


def _epoch_len(loader) -> int:
    return getattr(loader, "epoch_len", 200)
