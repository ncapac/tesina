"""
train_cvae.py
-------------
Training loop for the CVAE model.

Mirrors the interface of Trainer in train.py.  Key differences:
  • Loss is β-ELBO = MSE(recon) + β·KL (no diffusion timestep).
  • No classifier-free guidance at training time.
  • Validation tracks both total loss and per-component recon / KL.

Usage
-----
    from src.models.cvae import CVAETransformer1D
    from src.training.train_cvae import CVAETrainer

    model = CVAETransformer1D(...)
    trainer = CVAETrainer(model, lr=1e-3, checkpoint_dir='output/03c/checkpoints')
    trainer.fit(train_loader, val_loader, n_epochs=100)
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from src.models.cvae import CVAETransformer1D


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

@eqx.filter_jit
def train_step_cvae(
    model: CVAETransformer1D,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x0: jax.Array,               # (B, L) float32
    c_discrete: jax.Array,       # (B, 3) int32
    c_continuous: jax.Array,     # (B, n_cont) float32
    key: jax.Array,
) -> Tuple[CVAETransformer1D, optax.OptState, jax.Array, jax.Array, jax.Array]:
    """
    One JIT-compiled CVAE training step.

    Returns (new_model, new_opt_state, total_loss, recon_loss, kl_loss).
    """
    keys = jax.random.split(key, x0.shape[0])

    def loss_fn(m):
        # vmap over batch
        totals, recons, kls = jax.vmap(m.loss)(x0, c_discrete, c_continuous, keys)
        return totals.mean(), (recons.mean(), kls.mean())

    (total, (recon, kl)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, total, recon, kl


@eqx.filter_jit
def eval_step_cvae(
    model: CVAETransformer1D,
    x0: jax.Array,
    c_discrete: jax.Array,
    c_continuous: jax.Array,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Validation step. Returns (total_loss, recon_loss, kl_loss)."""
    keys = jax.random.split(key, x0.shape[0])
    totals, recons, kls = jax.vmap(model.loss)(x0, c_discrete, c_continuous, keys)
    return totals.mean(), recons.mean(), kls.mean()


def _epoch_len(loader) -> int:
    """Return the explicit loader epoch length, failing loudly if absent."""
    if not hasattr(loader, "epoch_len"):
        raise AttributeError(
            "Training loaders must expose an integer 'epoch_len' attribute. "
            "Use src.data.dataset.numpy_dataloader(...) or set loader.epoch_len explicitly."
        )
    epoch_len = int(getattr(loader, "epoch_len"))
    if epoch_len <= 0:
        raise ValueError(f"loader.epoch_len must be positive, got {epoch_len}")
    return epoch_len


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class CVAETrainer:
    def __init__(
        self,
        model: CVAETransformer1D,
        lr: float = 1e-3,
        warmup_steps: int = 200,
        total_steps: int = 50_000,
        checkpoint_dir: str = "checkpoints",
        seed: int = 0,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
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
        self.train_recon:  list[float] = []
        self.train_kl:     list[float] = []
        self.val_recon:    list[float] = []
        self.val_kl:       list[float] = []

    def fit(
        self,
        train_loader: Iterator,
        val_loader: Optional[Iterator],
        n_epochs: int = 100,
        val_every: int = 1,
        save_every: int = 10,
        log_every_steps: int = 50,
        val_batches: int = 20,
        patience: int = 20,
        min_delta: float = 1e-4,
    ):
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            ep_total, ep_recon, ep_kl = [], [], []

            for x0_np, c_disc_np, c_cont_np in train_loader:
                self.key, subkey = jax.random.split(self.key)
                x0         = jnp.array(x0_np)
                c_discrete = jnp.array(c_disc_np)
                c_continuous = jnp.array(c_cont_np)

                self.model, self.opt_state, tot, rec, kl = train_step_cvae(
                    self.model, self.opt_state, self.optimizer,
                    x0, c_discrete, c_continuous, subkey,
                )
                self.step += 1
                ep_total.append(float(tot))
                ep_recon.append(float(rec))
                ep_kl.append(float(kl))

                if self.step % log_every_steps == 0:
                    print(
                        f"  step {self.step:6d}  "
                        f"loss {float(tot):.4f}  "
                        f"recon {float(rec):.4f}  "
                        f"kl {float(kl):.4f}"
                    )

                if len(ep_total) >= _epoch_len(train_loader):
                    break

            mean_total = float(np.mean(ep_total))
            self.train_losses.append(mean_total)
            self.train_recon.append(float(np.mean(ep_recon)))
            self.train_kl.append(float(np.mean(ep_kl)))

            val_str = ""
            if val_loader is not None and epoch % val_every == 0:
                vt, vr, vk = [], [], []
                for i, (xv_np, cdv_np, ccv_np) in enumerate(val_loader):
                    if i >= val_batches:
                        break
                    self.key, subkey = jax.random.split(self.key)
                    tot_v, rec_v, kl_v = eval_step_cvae(
                        self.model,
                        jnp.array(xv_np),
                        jnp.array(cdv_np),
                        jnp.array(ccv_np),
                        subkey,
                    )
                    vt.append(float(tot_v))
                    vr.append(float(rec_v))
                    vk.append(float(kl_v))
                mean_val = float(np.mean(vt))
                self.val_losses.append(mean_val)
                self.val_recon.append(float(np.mean(vr)))
                self.val_kl.append(float(np.mean(vk)))
                val_str = (
                    f"  val_loss {mean_val:.4f} "
                    f"(recon {float(np.mean(vr)):.4f}  kl {float(np.mean(vk)):.4f})"
                )

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{n_epochs}  "
                f"train {mean_total:.4f} "
                f"(recon {float(np.mean(ep_recon)):.4f}  kl {float(np.mean(ep_kl)):.4f})"
                f"{val_str}  [{elapsed:.1f}s]"
            )

            if epoch % save_every == 0:
                self.save(f"cvae_epoch{epoch:04d}.pkl")

            if self.val_losses:
                if self.val_losses[-1] < best_val - min_delta:
                    best_val = self.val_losses[-1]
                    no_improve = 0
                    self.save("cvae_best.pkl")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(
                            f"\nEarly stopping at epoch {epoch} "
                            f"(best val {best_val:.4f})"
                        )
                        break

    def save(self, filename: str):
        path = self.checkpoint_dir / filename
        with open(path, "wb") as f:
            pickle.dump({
                "model":        self.model,
                "opt_state":    self.opt_state,
                "step":         self.step,
                "train_losses": self.train_losses,
                "val_losses":   self.val_losses,
            }, f)
        print(f"  Saved checkpoint → {path}")

    @classmethod
    def load(cls, path: str, **trainer_kwargs) -> "CVAETrainer":
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        model = ckpt["model"]
        trainer = cls(model, **trainer_kwargs)
        trainer.opt_state    = ckpt["opt_state"]
        trainer.step         = ckpt["step"]
        trainer.train_losses = ckpt.get("train_losses", [])
        trainer.val_losses   = ckpt.get("val_losses",   [])
        return trainer
