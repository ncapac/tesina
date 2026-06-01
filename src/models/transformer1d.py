"""
transformer1d.py
----------------
Diffusion-TS-style denoiser for 1-D time series (Equinox).

Architecture
------------
  Input  : (B, L) noisy time series  +  t (int diffusion step)
           +  c_discrete   (int   [cluster_id, day_type, season])
           +  c_continuous (float [daily_mean_temp_normed, …])
  Output : (B, L) predicted noise  ε_θ(x_t, c_discrete, c_continuous, t)

Key design choices:
  • Sinusoidal diffusion timestep embedding → linear projection to d_model
  • Discrete conditioning (cluster_id, day_type, season) → learnable embeddings
    → small MLP → (γ, β) applied as AdaLN inside every Transformer block
  • Continuous conditioning (daily mean temperature, …) → linear projection →
    context tokens passed through cross-attention in every Transformer block
  • Trend head: moving-average decomposition, outputs slow component S_trend
  • Seasonality head: residual S_res = input - S_trend
  • Positional encoding: sinusoidal fixed, added to patch tokens

Usage
-----
    model = DiffusionTransformer1D(
        seq_len=24, d_model=128, n_heads=4, n_layers=4,
        d_ff=256, n_clusters=5, n_day_types=2, n_seasons=4,
        n_continuous=1, ma_kernel=5,
    )
    # forward (un-batched):
    eps_pred = model(x_t, t, c_discrete, c_continuous)   # (seq_len,)
    # batched:
    eps_pred = jax.vmap(model)(x_t, t, c_discrete, c_continuous)  # (B, seq_len)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CFG_NULL_DISCRETE: int = -1


def make_cfg_null_conditioning(
    c_discrete: jax.Array,
    c_continuous: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the shared null conditioning used by classifier-free guidance."""
    return (
        jnp.full_like(c_discrete, CFG_NULL_DISCRETE),
        jnp.zeros_like(c_continuous),
    )


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
    trend = jax.lax.conv_general_dilated(
        x_padded[None, None, :],
        kernel[None, None, :],
        window_strides=(1,),
        padding="VALID",
    )[0, 0, :]
    return trend


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AdaLN(eqx.Module):
    """
    Adaptive Layer Normalisation conditioned on a context vector c.

      AdaLN(h, c) = γ(c) ⊙ LayerNorm(h) + β(c)
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
        """h : (d_model,)   c : (d_cond,)"""
        h_norm = self.ln(h)
        gamma = jax.nn.silu(self.mlp_gamma(c))
        beta  = self.mlp_beta(c)
        return (1 + gamma) * h_norm + beta


class MultiHeadSelfAttention(eqx.Module):
    """Vanilla multi-head self-attention."""
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
        qkv = jax.vmap(self.to_qkv)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)
        k = k.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)
        v = v.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)
        scale = math.sqrt(self.d_head)
        attn = jax.nn.softmax(jnp.einsum("hid,hjd->hij", q, k) / scale, axis=-1)
        out  = jnp.einsum("hij,hjd->hid", attn, v)
        out  = out.transpose(1, 0, 2).reshape(L, D)
        return jax.vmap(self.proj_out)(out)


class MultiHeadCrossAttention(eqx.Module):
    """
    Multi-head cross-attention: Q from sequence, K and V from context.

    Used to inject continuous conditioning tokens (e.g. daily temperature)
    into each Transformer block.
    """
    to_q: eqx.nn.Linear
    to_kv: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    n_heads: int
    d_head: int

    def __init__(self, d_model: int, n_heads: int, *, key: jax.Array):
        assert d_model % n_heads == 0
        k1, k2, k3 = jax.random.split(key, 3)
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.to_q    = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k1)
        self.to_kv   = eqx.nn.Linear(d_model, 2 * d_model, use_bias=False, key=k2)
        self.proj_out = eqx.nn.Linear(d_model, d_model, use_bias=True, key=k3)

    def __call__(self, x: jax.Array, context: jax.Array) -> jax.Array:
        """
        x       : (L, d_model)   sequence (queries)
        context : (M, d_model)   context tokens (keys/values)
        Returns : (L, d_model)
        """
        L, D = x.shape
        M = context.shape[0]
        q  = jax.vmap(self.to_q)(x)                   # (L, D)
        kv = jax.vmap(self.to_kv)(context)             # (M, 2D)
        k, v = jnp.split(kv, 2, axis=-1)              # each (M, D)

        q = q.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)  # (H, L, dh)
        k = k.reshape(M, self.n_heads, self.d_head).transpose(1, 0, 2)  # (H, M, dh)
        v = v.reshape(M, self.n_heads, self.d_head).transpose(1, 0, 2)  # (H, M, dh)

        scale = math.sqrt(self.d_head)
        attn = jax.nn.softmax(jnp.einsum("hid,hjd->hij", q, k) / scale, axis=-1)
        out  = jnp.einsum("hij,hjd->hid", attn, v)    # (H, L, dh)
        out  = out.transpose(1, 0, 2).reshape(L, D)   # (L, D)
        return jax.vmap(self.proj_out)(out)


class TransformerBlock(eqx.Module):
    """
    Transformer block with AdaLN discrete conditioning and cross-attention
    for continuous conditioning.

    Order of sublayers:
      1. Self-attention   with AdaLN pre-norm   (discrete + timestep cond)
      2. Cross-attention  with LayerNorm pre-norm (continuous cond tokens)
      3. Feed-forward     with AdaLN pre-norm   (discrete + timestep cond)
    """
    attn: MultiHeadSelfAttention
    cross_attn: MultiHeadCrossAttention
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear
    adaln1: AdaLN
    cross_ln: eqx.nn.LayerNorm
    adaln2: AdaLN

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_cond: int, *, key: jax.Array):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.attn       = MultiHeadSelfAttention(d_model, n_heads, key=k1)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, key=k2)
        self.ff1        = eqx.nn.Linear(d_model, d_ff, use_bias=True, key=k3)
        self.ff2        = eqx.nn.Linear(d_ff, d_model, use_bias=True, key=k4)
        self.adaln1     = AdaLN(d_model, d_cond, key=k5)
        self.cross_ln   = eqx.nn.LayerNorm(d_model)
        self.adaln2     = AdaLN(d_model, d_cond, key=k6)

    def __call__(
        self,
        x: jax.Array,       # (L, d_model)
        cond: jax.Array,    # (d_cond,)
        context: jax.Array, # (M, d_model)  continuous conditioning tokens
    ) -> jax.Array:
        # 1. Self-attention with AdaLN
        x_norm = jax.vmap(lambda h: self.adaln1(h, cond))(x)
        x = x + self.attn(x_norm)

        # 2. Cross-attention with continuous context tokens
        x_norm = jax.vmap(self.cross_ln)(x)
        x = x + self.cross_attn(x_norm, context)

        # 3. Feed-forward with AdaLN
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
    c_discrete   : (3,) int32  [cluster_id, day_type, season]
                   Null conditioning: c_discrete[0] < 0  (CFG unconditional)
    c_continuous : (n_continuous,) float32  [daily_mean_temp_normed, …]
                   Each variable is projected to a (d_model,) context token;
                   tokens are passed through cross-attention in every block.
                   Zeroed out when c_discrete[0] < 0 (null conditioning).
    """
    # Embeddings — discrete conditioning
    t_proj: eqx.nn.Linear
    cluster_emb: eqx.nn.Embedding
    daytype_emb: eqx.nn.Embedding
    season_emb: eqx.nn.Embedding
    cond_proj: eqx.nn.Linear

    # Continuous conditioning projection(s)
    cont_projs: list   # list of eqx.nn.Linear(1, d_model), len = n_continuous

    # Input projection
    in_proj: eqx.nn.Linear

    # Core Transformer layers
    layers: list

    # Output heads
    trend_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    # Config
    seq_len: int
    d_model: int
    t_emb_dim: int
    ma_kernel: int
    n_continuous: int

    pos_enc: jax.Array   # (seq_len, d_model) — fixed, not a parameter

    def __init__(
        self,
        seq_len: int = 24,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        n_clusters: int = 5,
        n_day_types: int = 2,
        n_seasons: int = 4,
        n_continuous: int = 1,
        ma_kernel: int = 5,
        t_emb_dim: int = 128,
        *,
        key: jax.Array,
    ):
        # Count keys needed:
        # t_proj, cluster_emb, daytype_emb, season_emb, cond_proj,
        # n_continuous × cont_proj, in_proj, n_layers × block, trend_proj, out_proj
        n_keys = 8 + n_continuous + n_layers
        keys = jax.random.split(key, n_keys)
        ki = iter(keys)

        self.seq_len      = seq_len
        self.d_model      = d_model
        self.t_emb_dim    = t_emb_dim
        self.ma_kernel    = ma_kernel
        self.n_continuous = n_continuous

        # Timestep embedding
        self.t_proj = eqx.nn.Linear(t_emb_dim, d_model, key=next(ki))

        # Discrete conditioning embeddings (d_model // 4 each = 32 for d_model=128)
        d_emb = d_model // 4
        self.cluster_emb = eqx.nn.Embedding(n_clusters,  d_emb, key=next(ki))
        self.daytype_emb = eqx.nn.Embedding(n_day_types, d_emb, key=next(ki))
        self.season_emb  = eqx.nn.Embedding(n_seasons,   d_emb, key=next(ki))

        # Fuse t_emb (d_model) + 3 discrete embs (d_emb each) → d_cond
        d_cond = d_model
        self.cond_proj = eqx.nn.Linear(
            d_model + 3 * d_emb, d_cond, key=next(ki)
        )

        # Continuous conditioning: one Linear(1 → d_model) per variable
        self.cont_projs = [
            eqx.nn.Linear(1, d_model, key=next(ki))
            for _ in range(n_continuous)
        ]

        # Input projection
        self.in_proj = eqx.nn.Linear(1, d_model, key=next(ki))

        # Transformer blocks
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff, d_cond, key=next(ki))
            for _ in range(n_layers)
        ]

        # Output heads
        self.trend_proj = eqx.nn.Linear(d_model, 1, key=next(ki))
        self.out_proj   = eqx.nn.Linear(d_model, 1, key=next(ki))

        # Fixed positional encoding
        self.pos_enc = fixed_positional_encoding(seq_len, d_model)

    def __call__(
        self,
        x_t: jax.Array,          # (seq_len,)
        t: jax.Array,             # ()  scalar int diffusion step
        c_discrete: jax.Array,   # (3,)  int32  [cluster_id, day_type, season]
        c_continuous: jax.Array, # (n_continuous,)  float32
    ) -> jax.Array:
        """
        Returns predicted noise ε_θ(x_t, t, c_discrete, c_continuous),
        shape (seq_len,).

        Null conditioning (CFG unconditional pass):
          c_discrete[0] < 0  →  all discrete embeddings zeroed,
                                 all continuous context tokens zeroed.
        """
        # 1. Timestep embedding
        t_sinusoid = sinusoidal_embedding(t, self.t_emb_dim)
        t_emb = jax.nn.silu(self.t_proj(t_sinusoid))   # (d_model,)

        # 2. Discrete conditioning (null: c_discrete[0] < 0)
        null = (c_discrete[0] < 0)
        safe = jnp.where(null, jnp.zeros_like(c_discrete), c_discrete)
        cl_emb = self.cluster_emb(safe[0])
        dt_emb = self.daytype_emb(safe[1])
        se_emb = self.season_emb(safe[2])
        cl_emb = jnp.where(null, jnp.zeros_like(cl_emb), cl_emb)
        dt_emb = jnp.where(null, jnp.zeros_like(dt_emb), dt_emb)
        se_emb = jnp.where(null, jnp.zeros_like(se_emb), se_emb)

        cond = jax.nn.silu(
            self.cond_proj(jnp.concatenate([t_emb, cl_emb, dt_emb, se_emb]))
        )  # (d_cond,)

        # 3. Continuous conditioning tokens
        # When null, zero out the context so cross-attention has no signal.
        safe_cont = jnp.where(null, jnp.zeros_like(c_continuous), c_continuous)
        context_tokens = jnp.stack(
            [self.cont_projs[i](safe_cont[i : i + 1]) for i in range(self.n_continuous)],
            axis=0,
        )  # (n_continuous, d_model)

        # 4. Token embedding
        tokens = jax.vmap(lambda v: self.in_proj(v[None]))(x_t)  # (L, d_model)
        tokens = tokens + self.pos_enc

        # 5. Transformer blocks (self-attn + cross-attn + FF)
        for layer in self.layers:
            tokens = layer(tokens, cond, context_tokens)

        # 6. Decomposition output heads
        raw    = jax.vmap(lambda h: self.out_proj(h)[0])(tokens)  # (L,)
        trend  = moving_average(raw, self.ma_kernel)
        eps_pred = trend + (raw - trend)                           # identity decomposition
        return eps_pred

