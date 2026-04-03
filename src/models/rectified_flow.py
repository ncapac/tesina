"""
rectified_flow.py
-----------------
Rectified Flow (RF) generative process for 1-D time series.

Reference
---------
  Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data
  with Rectified Flow", ICLR 2023.  arXiv:2209.03003

Design
------
  Forward path :  x_t = (1-t) · x_0 + t · ε,   t ∈ [0, 1]
  Target       :  v* = ε - x_0   (velocity from data → noise)
  Loss         :  MSE(v_θ(x_t, t, c), ε - x_0)  + λ · ‖FFT(v_θ) - FFT(v*)‖²
  Sampler      :  Euler ODE,  x_{t+Δt} = x_t + Δt · v_θ(x_t, t, c)
                  starting from x_0 ~ N(0, I) and integrating t: 1→0
  CFG          :  v_guided = (1+s) · v_cond - s · v_uncond

The denoiser backbone (DiffusionTransformer1D) is shared with the DDPM model.
The only difference is that `t` is now a *continuous* float in [0, 1] rather
than a discrete integer in [0, T).  We pass it as a float and scale into the
sinusoidal embedding range by multiplying by a large constant (1000).

Public API
----------
RectifiedFlowProcess(freq_loss_weight)
    .interpolate(x0, noise, t)       — forward linear interpolation
    .p_losses(model, x0, c, t, key)  — training loss
    .sample(model, c, seq_len, batch_size, key, n_steps, guidance_scale)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

# Scale factor: continuous t ∈ [0,1] → pseudo-timestep for sinusoidal embedding
# Must be consistent between training and sampling.
_T_SCALE: int = 1000


class RectifiedFlowProcess(eqx.Module):
    """
    Rectified Flow process.  Stateless (no precomputed schedule arrays).
    """
    freq_loss_weight: float

    def __init__(self, freq_loss_weight: float = 0.05):
        self.freq_loss_weight = freq_loss_weight

    # ------------------------------------------------------------------
    # Forward interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        x0: jax.Array,    # (..., L) clean sample
        noise: jax.Array,  # (..., L) ~ N(0,I)
        t: jax.Array,      # (...,) float in [0, 1]
    ) -> jax.Array:
        """
        x_t = (1 - t) · x_0 + t · ε
        """
        t_ = t[..., None]   # broadcast over L
        return (1.0 - t_) * x0 + t_ * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        model: eqx.Module,
        x0: jax.Array,     # (B, L)
        c: jax.Array,      # (B, 4) int32
        t: jax.Array,      # (B,)   float32 in [0, 1]
        key: jax.Array,
    ) -> jax.Array:
        """
        Loss = MSE(v_θ, v*) + λ · ‖FFT(v_θ) - FFT(v*)‖²
        where v* = ε - x_0  (target velocity).

        Returns scalar loss.
        """
        noise = jax.random.normal(key, x0.shape)
        x_t   = self.interpolate(x0, noise, t)

        # Convert continuous t → pseudo-integer timestep for the backbone's
        # sinusoidal embedding (matches DDPM convention: step ∈ [0, T))
        t_int = jnp.round(t * (_T_SCALE - 1)).astype(jnp.int32)  # (B,)

        # Target velocity
        v_target = noise - x0   # (B, L)

        # Predicted velocity
        v_pred = jax.vmap(model)(x_t, t_int, c)   # (B, L)

        # MSE loss
        mse = jnp.mean((v_target - v_pred) ** 2)

        # Frequency loss
        fft_pred   = jnp.abs(jnp.fft.rfft(v_pred,   axis=-1))
        fft_target = jnp.abs(jnp.fft.rfft(v_target, axis=-1))
        freq_loss  = jnp.mean((fft_pred - fft_target) ** 2)

        return mse + self.freq_loss_weight * freq_loss

    # ------------------------------------------------------------------
    # CFG velocity prediction
    # ------------------------------------------------------------------

    def _predict_v_cfg(
        self,
        model: eqx.Module,
        x_t: jax.Array,        # (B, L)
        c: jax.Array,           # (B, 4) int
        t_int: jax.Array,       # (B,)   int32  pseudo-timestep
        guidance_scale: float,
    ) -> jax.Array:
        """
        CFG:  v_guided = (1+s) · v_cond - s · v_uncond
        """
        v_cond   = jax.vmap(model)(x_t, t_int, c)
        null_c   = jnp.full_like(c, -1)
        v_uncond = jax.vmap(model)(x_t, t_int, null_c)
        return (1 + guidance_scale) * v_cond - guidance_scale * v_uncond

    # ------------------------------------------------------------------
    # Euler ODE sampler
    # ------------------------------------------------------------------

    def sample(
        self,
        model: eqx.Module,
        c: jax.Array,           # (B, 4) int32 conditioning
        seq_len: int,
        batch_size: int,
        key: jax.Array,
        n_steps: int = 50,
        guidance_scale: float = 1.5,
    ) -> jax.Array:
        """
        Euler ODE sampler integrating t: 1 → 0.

        x_{t - Δt} = x_t  -  Δt · v_θ(x_t, t, c)

        Returns (B, seq_len) samples.
        """
        dt = 1.0 / n_steps
        # Uniform t grid: start at t=1 (pure noise), end at t=0 (clean data)
        t_vals = np.linspace(1.0, dt, n_steps)   # n_steps values, [1.0 … dt]

        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (batch_size, seq_len))

        for t_float in t_vals:
            t_arr = jnp.full((batch_size,), t_float, dtype=jnp.float32)
            t_int = jnp.round(t_arr * (_T_SCALE - 1)).astype(jnp.int32)

            v = self._predict_v_cfg(model, x, c, t_int, guidance_scale)
            x = x - dt * v

        return x
