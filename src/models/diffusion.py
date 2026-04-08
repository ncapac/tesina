"""
diffusion.py
------------
DDPM / DDIM diffusion process + Classifier-Free Guidance (CFG).

References
----------
  Ho et al., 2020  — DDPM      (cosine schedule: Nichol & Dhariwal 2021)
  Song et al., 2020 — DDIM     (non-Markovian deterministic sampler)
  Ho & Salimans 2022 — CFG

  CFG formula (eq. 15 from arXiv 2507.14507):
    ε_guided = (1+s) · ε_θ(x_t, c, t) - s · ε_θ(x_t, ∅, t)

Public API
----------
DiffusionProcess(T, schedule)
    .q_sample(x0, t, noise)         — forward noising
    .p_losses(model, x0, c, t, key) — training loss (noise + freq)
    .ddpm_sample(model, c, L, key, guidance_scale)  — full DDPM reverse
    .ddim_sample(model, c, L, key, n_steps, guidance_scale) — DDIM sampler
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).
    Returns beta array of shape (T,).
    """
    steps = T + 1
    t = np.linspace(0, T, steps)
    alphas_cumprod = np.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999).astype(np.float32)


# ---------------------------------------------------------------------------
# Diffusion process
# ---------------------------------------------------------------------------

class DiffusionProcess(eqx.Module):
    """
    Holds the precomputed noise schedule arrays as static buffers.
    All methods are pure JAX functions (JIT-compatible).
    """
    T: int
    betas: jax.Array           # (T,)
    alphas: jax.Array          # (T,)
    alphas_cumprod: jax.Array  # (T,)
    sqrt_acp: jax.Array        # sqrt(ᾱ_t)
    sqrt_one_minus_acp: jax.Array  # sqrt(1 - ᾱ_t)
    log_one_minus_acp: jax.Array
    sqrt_recip_acp: jax.Array
    posterior_variance: jax.Array  # β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
    freq_loss_weight: float

    def __init__(self, T: int = 1000, freq_loss_weight: float = 0.05):
        self.T = T
        self.freq_loss_weight = freq_loss_weight

        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        acp = np.cumprod(alphas)
        acp_prev = np.concatenate([[1.0], acp[:-1]])

        self.betas              = jnp.array(betas)
        self.alphas             = jnp.array(alphas)
        self.alphas_cumprod     = jnp.array(acp)
        self.sqrt_acp           = jnp.array(np.sqrt(acp))
        self.sqrt_one_minus_acp = jnp.array(np.sqrt(1 - acp))
        self.log_one_minus_acp  = jnp.array(np.log(1 - acp + 1e-20))
        self.sqrt_recip_acp     = jnp.array(np.sqrt(1.0 / acp))
        self.posterior_variance = jnp.array(
            betas * (1 - acp_prev) / np.clip(1 - acp, 1e-20, None)
        )

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: jax.Array,   # (..., L)
        t: jax.Array,    # (...,) int
        noise: jax.Array,
    ) -> jax.Array:
        """
        Sample x_t ~ q(x_t | x_0) in one step:
            x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
        """
        sqrt_acp_t    = self.sqrt_acp[t][..., None]
        sqrt_1macp_t  = self.sqrt_one_minus_acp[t][..., None]
        return sqrt_acp_t * x0 + sqrt_1macp_t * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        model: eqx.Module,
        x0: jax.Array,          # (B, L)
        c: jax.Array,           # (B, 2) int; may be null [-1,-1] via CFG
        t: jax.Array,           # (B,)   int diffusion steps
        key: jax.Array,
    ) -> jax.Array:
        """
        Loss = MSE(ε, ε_θ) + λ · ‖FFT(ε_θ) - FFT(x0)‖²

        Returns scalar loss.
        """
        noise = jax.random.normal(key, x0.shape)
        x_t = self.q_sample(x0, t, noise)

        # Predict noise — vmap over batch
        eps_pred = jax.vmap(model)(x_t, t, c)   # (B, L)

        # MSE noise loss
        mse = jnp.mean((noise - eps_pred) ** 2)

        # Frequency loss: compare magnitude spectra of noise prediction vs actual noise
        # (regularises the model to match the temporal-frequency structure of Gaussian noise)
        fft_pred  = jnp.abs(jnp.fft.rfft(eps_pred, axis=-1))
        fft_noise = jnp.abs(jnp.fft.rfft(noise,    axis=-1))
        freq_loss = jnp.mean((fft_pred - fft_noise) ** 2)

        return mse + self.freq_loss_weight * freq_loss

    # ------------------------------------------------------------------
    # CFG noise prediction
    # ------------------------------------------------------------------

    def _predict_eps_cfg(
        self,
        model: eqx.Module,
        x_t: jax.Array,        # (B, L)
        c: jax.Array,           # (B, 4) int
        t: jax.Array,           # (B,)   int
        guidance_scale: float,
    ) -> jax.Array:
        """
        Classifier-Free Guidance:
          ε_guided = (1+s) · ε_θ(x_t, c, t)  -  s · ε_θ(x_t, ∅, t)

        Null token: jnp.full_like(c, -1) works for any c shape (2 or 4 dims).
        """
        eps_cond   = jax.vmap(model)(x_t, t, c)

        null_c = jnp.full_like(c, -1)  # null conditioning token
        eps_uncond = jax.vmap(model)(x_t, t, null_c)

        return (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

    # ------------------------------------------------------------------
    # DDPM reverse sampler
    # ------------------------------------------------------------------

    def ddpm_sample(
        self,
        model: eqx.Module,
        c: jax.Array,           # (B, 4) int conditioning
        seq_len: int,
        batch_size: int,
        key: jax.Array,
        guidance_scale: float = 1.5,
    ) -> jax.Array:
        """
        Full DDPM reverse process (T steps).
        Returns (B, seq_len) samples.
        """
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (batch_size, seq_len))

        for t_int in reversed(range(self.T)):
            key, subkey = jax.random.split(key)
            t_batch = jnp.full((batch_size,), t_int, dtype=jnp.int32)

            eps = self._predict_eps_cfg(model, x, c, t_batch, guidance_scale)

            # Posterior mean
            sqrt_recip = self.sqrt_recip_acp[t_int]
            sqrt_1macp = self.sqrt_one_minus_acp[t_int]
            x0_pred = sqrt_recip * (x - sqrt_1macp * eps)
            x0_pred = jnp.clip(x0_pred, -4.0, 4.0)  # clip to prevent drift

            # Posterior mean
            acp      = self.alphas_cumprod[t_int]
            acp_prev = self.alphas_cumprod[t_int - 1] if t_int > 0 else jnp.array(1.0)
            beta_t   = self.betas[t_int]

            post_mean = (
                jnp.sqrt(acp_prev) * beta_t / (1 - acp) * x0_pred
                + jnp.sqrt(self.alphas[t_int]) * (1 - acp_prev) / (1 - acp) * x
            )
            post_var = self.posterior_variance[t_int]

            noise = jax.random.normal(subkey, x.shape)
            x = post_mean + (t_int > 0) * jnp.sqrt(post_var) * noise

        return x

    # ------------------------------------------------------------------
    # DDIM deterministic sampler
    # ------------------------------------------------------------------

    def ddim_sample(
        self,
        model: eqx.Module,
        c: jax.Array,           # (B, 4) int
        seq_len: int,
        batch_size: int,
        key: jax.Array,
        n_steps: int = 50,
        guidance_scale: float = 1.5,
        eta: float = 0.0,       # 0 = deterministic DDIM
    ) -> jax.Array:
        """
        DDIM sampler — fast (n_steps << T) deterministic generation.
        Returns (B, seq_len) samples.
        """
        # Evenly-spaced subset of diffusion steps
        step_indices = np.linspace(0, self.T - 1, n_steps, dtype=int)[::-1]

        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (batch_size, seq_len))

        for i, t_int in enumerate(step_indices):
            t_prev = int(step_indices[i + 1]) if i + 1 < len(step_indices) else 0

            t_batch = jnp.full((batch_size,), t_int, dtype=jnp.int32)
            eps = self._predict_eps_cfg(model, x, c, t_batch, guidance_scale)

            acp_t    = self.alphas_cumprod[t_int]
            acp_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else jnp.array(1.0)

            # Predicted x0
            x0_pred = (x - jnp.sqrt(1 - acp_t) * eps) / jnp.sqrt(acp_t)
            x0_pred = jnp.clip(x0_pred, -4.0, 4.0)

            # Direction pointing to x_t
            dir_xt = jnp.sqrt(1 - acp_prev - eta ** 2 * (1 - acp_t) / acp_t * (1 - acp_prev) / (1 - acp_t)) * eps

            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, x.shape) if eta > 0 else jnp.zeros_like(x)

            x = jnp.sqrt(acp_prev) * x0_pred + dir_xt + eta * jnp.sqrt(
                (1 - acp_prev) * (1 - acp_t) / acp_t
            ) * noise

        return x
