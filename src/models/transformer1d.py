"""
transformer1d.py
----------------
Diffusion-TS-style denoiser for 1-D time series (Equinox).

Architecture
------------
  Input  : (B, L) noisy time series  +  t (int diffusion step)  +  c (int [cluster, day_type])
  Output : (B, L) predicted noise  ε_θ(x_t, c, t)

Key design choices (from Diffusion-TS, arXiv 2403.01742 + survey 2507.14507):
  • Sinusoidal diffusion timestep embedding → linear projection to d_model
  • Discrete conditioning (cluster_id, day_type) → learnable embeddings → small MLP → (γ, β)
    applied as AdaLN inside every Transformer block
  • Trend head: moving-average decomposition, outputs slow component S_trend
  • Seasonality head: residual S_res = input - S_trend
  • Final output: combine trend + seasonality predictions
  • Positional encoding: sinusoidal fixed, added to patch tokens

Usage
-----
    model = DiffusionTransformer1D(
        seq_len=24, d_model=128, n_heads=4, n_layers=4,
        d_ff=256, n_clusters=3, n_day_types=2,
        ma_kernel=5,
    )
    # forward expects unbatched or batched:
    eps_pred = jax.vmap(model)(x_t, t, c)   # (B, seq_len)
"""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: jax.Array, dim: int) -> jax.Array:
    """
    Sinusoidal diffusion timestep embedding.

    Parameters
    ----------
    t   : scalar integer (diffusion step index)
    dim : embedding dimensionality (must be even)

    Returns
    -------
    (dim,) float32 embedding
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / (half - 1)
    )
    t_f = jnp.asarray(t, dtype=jnp.float32)
    args = t_f * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (dim,)


def fixed_positional_encoding(seq_len: int, d_model: int) -> jax.Array:
    """(seq_len, d_model) sinusoidal positional encoding."""
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    dims = jnp.arange(d_model, dtype=jnp.float32)[None, :]
    angles = positions / jnp.power(10000.0, (2 * (dims // 2)) / d_model)
    enc = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return enc  # (seq_len, d_model)


def moving_average(x: jax.Array, kernel_size: int) -> jax.Array:
    """
    1-D causal moving-average along the last axis.

    x : (L,) float
    Returns trend (L,) float
    """
    pad = kernel_size - 1
    x_padded = jnp.concatenate([jnp.zeros(pad, dtype=x.dtype), x], axis=0)
    kernel = jnp.ones(kernel_size, dtype=x.dtype) / kernel_size
    # Manual convolution (JAX does not have a 1-D conv helper for scalars)
    # Use associative scan for efficiency
    trend = jax.lax.conv_general_dilated(
        x_padded[None, None, :],          # (1, 1, L+pad)
        kernel[None, None, :],             # (1, 1, K)
        window_strides=(1,),
        padding="VALID",
    )[0, 0, :]                             # (L,)
    return trend


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AdaLN(eqx.Module):
    """
    Adaptive Layer Normalisation conditioned on a context vector c.

      AdaLN(h, c) = γ(c) ⊙ LayerNorm(h) + β(c)

    The scale/shift are produced by a tiny 2-layer MLP applied to c.
    """
    ln: eqx.nn.LayerNorm
    mlp_gamma: eqx.nn.Linear
    mlp_beta: eqx.nn.Linear

    def __init__(self, d_model: int, d_cond: int, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.ln = eqx.nn.LayerNorm(d_model)
        self.mlp_gamma = eqx.nn.Linear(d_cond, d_model, use_bias=True, key=k1)
        self.mlp_beta  = eqx.nn.Linear(d_cond, d_model, use_bias=True, key=k2)

    def __call__(self, h: jax.Array, c: jax.Array) -> jax.Array:
        """
        h : (d_model,)   intermediate feature
        c : (d_cond,)    conditioning vector
        """
        h_norm = self.ln(h)
        gamma = jax.nn.silu(self.mlp_gamma(c))  # (d_model,)
        beta  = self.mlp_beta(c)                 # (d_model,)
        return (1 + gamma) * h_norm + beta


class MultiHeadSelfAttention(eqx.Module):
    """Vanilla multi-head self-attention (Equinox)."""
    to_qkv: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    n_heads: int
    d_head: int

    def __init__(self, d_model: int, n_heads: int, *, key: jax.Array):
        assert d_model % n_heads == 0
        k1, k2 = jax.random.split(key)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.to_qkv  = eqx.nn.Linear(d_model, 3 * d_model, use_bias=False, key=k1)
        self.proj_out = eqx.nn.Linear(d_model, d_model, use_bias=True, key=k2)

    def __call__(self, x: jax.Array) -> jax.Array:
        """x : (L, d_model)  →  (L, d_model)"""
        L, D = x.shape
        qkv = jax.vmap(self.to_qkv)(x)              # (L, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)        # each (L, D)

        # Reshape to (n_heads, L, d_head)
        q = q.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)
        k = k.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)
        v = v.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)

        scale = math.sqrt(self.d_head)
        attn = jax.nn.softmax(jnp.einsum("hid,hjd->hij", q, k) / scale, axis=-1)
        out  = jnp.einsum("hij,hjd->hid", attn, v)           # (H, L, d_head)
        out  = out.transpose(1, 0, 2).reshape(L, D)           # (L, D)
        return jax.vmap(self.proj_out)(out)


class TransformerBlock(eqx.Module):
    """One Transformer block with AdaLN conditioning."""
    attn: MultiHeadSelfAttention
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear
    adaln1: AdaLN
    adaln2: AdaLN

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_cond: int, *, key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.attn   = MultiHeadSelfAttention(d_model, n_heads, key=k1)
        self.ff1    = eqx.nn.Linear(d_model, d_ff, use_bias=True, key=k2)
        self.ff2    = eqx.nn.Linear(d_ff, d_model, use_bias=True, key=k3)
        self.adaln1 = AdaLN(d_model, d_cond, key=k4)
        self.adaln2 = AdaLN(d_model, d_cond, key=k5)

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        """
        x    : (L, d_model)
        cond : (d_cond,)
        """
        # Self-attention with AdaLN pre-norm
        x_norm = jax.vmap(lambda h: self.adaln1(h, cond))(x)
        x = x + self.attn(x_norm)

        # Feed-forward with AdaLN pre-norm
        x_norm = jax.vmap(lambda h: self.adaln2(h, cond))(x)
        ff_out = jax.vmap(lambda h: self.ff2(jax.nn.gelu(self.ff1(h))))(x_norm)
        x = x + ff_out
        return x


# ---------------------------------------------------------------------------
# Full denoiser
# ---------------------------------------------------------------------------

class DiffusionTransformer1D(eqx.Module):
    """
    Diffusion-TS inspired 1-D Transformer denoiser.

    Input  shape : (seq_len,)  — single un-batched sample
                   call with jax.vmap for batches
    Output shape : (seq_len,)  — predicted noise

    Conditioning
    ------------
    c = [cluster_id, day_type, month, dow]  (int32, shape (4,))
    Null conditioning: c = [-1, -1, -1, -1]  (CFG unconditional pass)

    t_emb_proj  : sinusoidal(t) → d_model
    cond_embed  : [cluster ⊕ day_type ⊕ month ⊕ dow] → d_cond  (for AdaLN)
    """
    # Embeddings
    t_proj: eqx.nn.Linear
    cluster_emb: eqx.nn.Embedding
    daytype_emb: eqx.nn.Embedding
    month_emb: eqx.nn.Embedding
    dow_emb: eqx.nn.Embedding
    cond_proj: eqx.nn.Linear   # merge t + all discrete cond → d_cond

    # Input projection
    in_proj: eqx.nn.Linear     # scalar → d_model per timestep

    # Core Transformer layers
    layers: list

    # Output heads
    trend_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    # Config (stored as static data)
    seq_len: int
    d_model: int
    t_emb_dim: int
    ma_kernel: int

    pos_enc: jax.Array   # (seq_len, d_model) — fixed, not a parameter

    def __init__(
        self,
        seq_len: int = 24,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        n_clusters: int = 3,
        n_day_types: int = 2,
        n_months: int = 12,
        n_dow: int = 7,
        ma_kernel: int = 5,
        t_emb_dim: int = 128,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 12 + n_layers)  # 2 extra for month_emb + dow_emb
        ki = iter(keys)

        self.seq_len   = seq_len
        self.d_model   = d_model
        self.t_emb_dim = t_emb_dim
        self.ma_kernel = ma_kernel

        # Timestep embedding: sinusoidal(t) has size t_emb_dim → project to d_model
        self.t_proj = eqx.nn.Linear(t_emb_dim, d_model, key=next(ki))

        # Discrete conditioning embeddings — equal share of d_model each
        # Total emb dim = 4 × (d_model//4) = d_model, so cond_proj input stays 2×d_model
        d_cluster = d_model // 4
        d_daytype = d_model // 4
        d_month   = d_model // 4
        d_dow     = d_model // 4
        self.cluster_emb = eqx.nn.Embedding(n_clusters,  d_cluster, key=next(ki))
        self.daytype_emb = eqx.nn.Embedding(n_day_types, d_daytype, key=next(ki))
        self.month_emb   = eqx.nn.Embedding(n_months,    d_month,   key=next(ki))
        self.dow_emb     = eqx.nn.Embedding(n_dow,       d_dow,     key=next(ki))

        # Fuse t_emb + all discrete embeddings → d_cond for AdaLN
        # input size = d_model + 4×(d_model//4) = 2×d_model = 256 (same as before)
        d_cond = d_model
        self.cond_proj = eqx.nn.Linear(
            d_model + d_cluster + d_daytype + d_month + d_dow, d_cond, key=next(ki)
        )

        # Input projection: each scalar timestep value → d_model token
        self.in_proj = eqx.nn.Linear(1, d_model, key=next(ki))

        # Transformer blocks
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff, d_cond, key=next(ki))
            for _ in range(n_layers)
        ]

        # Decomposition output heads
        self.trend_proj = eqx.nn.Linear(d_model, 1, key=next(ki))
        self.out_proj   = eqx.nn.Linear(d_model, 1, key=next(ki))

        # Fixed positional encoding (not a parameter)
        self.pos_enc = fixed_positional_encoding(seq_len, d_model)

    def __call__(
        self,
        x_t: jax.Array,    # (seq_len,)  noisy input
        t: jax.Array,       # ()          scalar int diffusion step
        c: jax.Array,       # (4,)        int [cluster_id, day_type, month, dow]; ∅ = [-1,-1,-1,-1]
    ) -> jax.Array:
        """
        Returns predicted noise ε_θ(x_t, c, t) of shape (seq_len,).

        When c = [-1, -1, -1, -1] (null conditioning for CFG unconditional pass),
        all discrete embeddings are zeroed out.
        """
        L = self.seq_len

        # 1. Timestep embedding
        t_sinusoid = sinusoidal_embedding(t, self.t_emb_dim)   # (t_emb_dim,)
        t_emb = jax.nn.silu(self.t_proj(t_sinusoid))          # (d_model,)

        # 2. Discrete conditioning  (null conditioning: ids = -1 → zero vector)
        null    = (c[0] < 0)
        safe_c0 = jnp.where(null, 0, c[0])
        safe_c1 = jnp.where(null, 0, c[1])
        safe_c2 = jnp.where(null, 0, c[2])
        safe_c3 = jnp.where(null, 0, c[3])
        cl_emb = self.cluster_emb(safe_c0)   # (d_cluster,)
        dt_emb = self.daytype_emb(safe_c1)   # (d_daytype,)
        mo_emb = self.month_emb(safe_c2)     # (d_month,)
        dw_emb = self.dow_emb(safe_c3)       # (d_dow,)
        cl_emb = jnp.where(null, jnp.zeros_like(cl_emb), cl_emb)
        dt_emb = jnp.where(null, jnp.zeros_like(dt_emb), dt_emb)
        mo_emb = jnp.where(null, jnp.zeros_like(mo_emb), mo_emb)
        dw_emb = jnp.where(null, jnp.zeros_like(dw_emb), dw_emb)

        # 3. Fuse into conditioning vector for AdaLN
        cond = jax.nn.silu(
            self.cond_proj(jnp.concatenate([t_emb, cl_emb, dt_emb, mo_emb, dw_emb]))
        )  # (d_cond,)

        # 4. Token embedding: each of the L scalar values → d_model
        tokens = jax.vmap(lambda v: self.in_proj(v[None]))(x_t)  # (L, d_model)
        tokens = tokens + self.pos_enc                             # add positional enc

        # 5. Transformer blocks
        for layer in self.layers:
            tokens = layer(tokens, cond)

        # 6. Decomposition heads
        #    Trend: moving-average of the token projections back to scalar space
        raw = jax.vmap(lambda h: self.out_proj(h)[0])(tokens)  # (L,)

        # trend via MA on the raw output
        trend = moving_average(raw, self.ma_kernel)   # (L,)
        seasonality = raw - trend                      # (L,)

        # predicted noise = trend + seasonality (identity, but keeps decomposition explicit)
        eps_pred = trend + seasonality                 # (L,)
        return eps_pred
