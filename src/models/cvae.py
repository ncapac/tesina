"""
cvae.py
-------
Conditional Variational Autoencoder (CVAE) for daily load profile generation.

Architecture
------------
Encoder : x_24 + [cluster_emb, daytype_emb, season_emb, cont_proj] → μ_z, log σ_z  (z_dim=64)
Decoder : z + [cluster_emb, daytype_emb, season_emb, cont_proj]   → x̂_24

Both encoder and decoder share the same AdaLN-like conditioning mechanism used
in the Transformer denoiser:
  • Discrete conditioning:  [cluster_id, day_type, season] → learnable embeddings
  • Continuous conditioning: [daily_mean_temp_normed, …]   → linear projection

The ELBO loss is:
  L_ELBO = E_q[log p(x|z, c)] - β · KL(q(z|x, c) || p(z))

where β is a weighting coefficient (β=0.5 by default, as per β-VAE).

Usage
-----
    model = CVAETransformer1D(seq_len=24, d_model=128, z_dim=64, ...)
    loss, recon, kl = model.loss(x, c_discrete, c_continuous, key)
    z = model.encode(x, c_discrete, c_continuous, key, deterministic=True)
    x_hat = model.decode(z, c_discrete, c_continuous)
    x_gen = model.generate(c_discrete_batch, c_continuous_batch, n_samples, key)
"""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from src.models.transformer1d import sinusoidal_embedding


# ---------------------------------------------------------------------------
# Conditioning utilities (shared between encoder and decoder)
# ---------------------------------------------------------------------------

class _ConditionEmbedder(eqx.Module):
    """
    Fuse discrete + continuous conditioning into a single conditioning vector.
    Same conditioning strategy as DiffusionTransformer1D but without timestep.
    """
    cluster_emb: eqx.nn.Embedding
    daytype_emb: eqx.nn.Embedding
    season_emb:  eqx.nn.Embedding
    cont_projs:  list   # list of Linear(1, d_model)
    cond_proj:   eqx.nn.Linear
    n_continuous: int

    def __init__(
        self,
        d_model: int,
        n_clusters: int,
        n_day_types: int,
        n_seasons: int,
        n_continuous: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 4 + n_continuous)
        ki = iter(keys)
        d_emb = d_model // 4

        self.cluster_emb = eqx.nn.Embedding(n_clusters,  d_emb, key=next(ki))
        self.daytype_emb = eqx.nn.Embedding(n_day_types, d_emb, key=next(ki))
        self.season_emb  = eqx.nn.Embedding(n_seasons,   d_emb, key=next(ki))
        self.cont_projs  = [eqx.nn.Linear(1, d_model, key=next(ki)) for _ in range(n_continuous)]
        self.cond_proj   = eqx.nn.Linear(3 * d_emb, d_model, key=next(ki))
        self.n_continuous = n_continuous

    def __call__(
        self,
        c_discrete: jax.Array,   # (3,) int32
        c_continuous: jax.Array, # (n_continuous,) float32
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Returns:
          cond         : (d_model,)             — for MLP conditioning
          context_tokens : (n_continuous, d_model) — for cross-attention
        """
        null = (c_discrete[0] < 0)
        safe = jnp.where(null, jnp.zeros_like(c_discrete), c_discrete)
        cl   = self.cluster_emb(safe[0])
        dt   = self.daytype_emb(safe[1])
        se   = self.season_emb(safe[2])
        cl   = jnp.where(null, jnp.zeros_like(cl), cl)
        dt   = jnp.where(null, jnp.zeros_like(dt), dt)
        se   = jnp.where(null, jnp.zeros_like(se), se)
        cond = jax.nn.silu(self.cond_proj(jnp.concatenate([cl, dt, se])))  # (d_model,)

        safe_cont = jnp.where(null, jnp.zeros_like(c_continuous), c_continuous)
        ctx = jnp.stack(
            [self.cont_projs[i](safe_cont[i : i + 1]) for i in range(self.n_continuous)],
            axis=0,
        )  # (n_cont, d_model)
        return cond, ctx


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CVAEEncoder(eqx.Module):
    """
    Encoder: q(z | x, c) = N(μ_z(x,c), diag(σ²_z(x,c)))

    Architecture:  x_24 is flattened and concatenated with cond → MLP → (μ, log σ)
    """
    conditioner: _ConditionEmbedder
    fc1:  eqx.nn.Linear
    fc2:  eqx.nn.Linear
    fc_mu:    eqx.nn.Linear
    fc_logsg: eqx.nn.Linear

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        z_dim: int,
        n_clusters: int,
        n_day_types: int,
        n_seasons: int,
        n_continuous: int,
        *,
        key: jax.Array,
    ):
        k0, k1, k2, k3, k4 = jax.random.split(key, 5)
        self.conditioner = _ConditionEmbedder(
            d_model, n_clusters, n_day_types, n_seasons, n_continuous, key=k0
        )
        inp_dim = seq_len + d_model
        hidden  = d_model * 2
        self.fc1  = eqx.nn.Linear(inp_dim, hidden, key=k1)
        self.fc2  = eqx.nn.Linear(hidden,  hidden, key=k2)
        self.fc_mu    = eqx.nn.Linear(hidden, z_dim, key=k3)
        self.fc_logsg = eqx.nn.Linear(hidden, z_dim, key=k4)

    def __call__(
        self,
        x: jax.Array,             # (seq_len,) float32
        c_discrete: jax.Array,    # (3,) int32
        c_continuous: jax.Array,  # (n_continuous,) float32
    ) -> Tuple[jax.Array, jax.Array]:
        """Returns (mu_z, logsg_z), each shape (z_dim,)."""
        cond, _ctx = self.conditioner(c_discrete, c_continuous)
        h = jnp.concatenate([x, cond])
        h = jax.nn.gelu(self.fc1(h))
        h = jax.nn.gelu(self.fc2(h))
        mu     = self.fc_mu(h)
        logsg  = jnp.clip(self.fc_logsg(h), -5.0, 5.0)
        return mu, logsg


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CVAEDecoder(eqx.Module):
    """
    Decoder: p(x | z, c) = N(μ_x(z, c), I)
    """
    conditioner: _ConditionEmbedder
    fc1:  eqx.nn.Linear
    fc2:  eqx.nn.Linear
    fc_out: eqx.nn.Linear

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        z_dim: int,
        n_clusters: int,
        n_day_types: int,
        n_seasons: int,
        n_continuous: int,
        *,
        key: jax.Array,
    ):
        k0, k1, k2, k3 = jax.random.split(key, 4)
        self.conditioner = _ConditionEmbedder(
            d_model, n_clusters, n_day_types, n_seasons, n_continuous, key=k0
        )
        inp_dim = z_dim + d_model
        hidden  = d_model * 2
        self.fc1    = eqx.nn.Linear(inp_dim, hidden,  key=k1)
        self.fc2    = eqx.nn.Linear(hidden,  hidden,  key=k2)
        self.fc_out = eqx.nn.Linear(hidden,  seq_len, key=k3)

    def __call__(
        self,
        z: jax.Array,             # (z_dim,) float32
        c_discrete: jax.Array,    # (3,) int32
        c_continuous: jax.Array,  # (n_continuous,) float32
    ) -> jax.Array:
        """Returns reconstructed profile x̂, shape (seq_len,)."""
        cond, _ctx = self.conditioner(c_discrete, c_continuous)
        h = jnp.concatenate([z, cond])
        h = jax.nn.gelu(self.fc1(h))
        h = jax.nn.gelu(self.fc2(h))
        return self.fc_out(h)


# ---------------------------------------------------------------------------
# Full CVAE
# ---------------------------------------------------------------------------

class CVAETransformer1D(eqx.Module):
    """
    Conditional VAE for daily (24h) load profile generation.

    This is a conceptually simple MLP-based CVAE.  The architecture is
    intentionally shallower than the diffusion Transformer so it serves as
    a fast, competent baseline rather than a competitor.

    Parameters
    ----------
    seq_len      : number of timesteps (24 for hourly)
    d_model      : hidden dimension for conditioning and MLP layers
    z_dim        : latent space dimensionality
    n_clusters   : number of cluster labels
    n_day_types  : number of day-type labels (2: weekday/weekend)
    n_seasons    : number of season labels (4)
    n_continuous : number of continuous conditioning variables (1 for temperature)
    beta         : β-VAE KL weight (1.0 = standard ELBO)
    """
    encoder: CVAEEncoder
    decoder: CVAEDecoder
    beta: float
    z_dim: int
    seq_len: int

    def __init__(
        self,
        seq_len: int = 24,
        d_model: int = 128,
        z_dim: int = 64,
        n_clusters: int = 5,
        n_day_types: int = 2,
        n_seasons: int = 4,
        n_continuous: int = 1,
        beta: float = 0.5,
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        self.encoder = CVAEEncoder(
            seq_len, d_model, z_dim,
            n_clusters, n_day_types, n_seasons, n_continuous,
            key=k1,
        )
        self.decoder = CVAEDecoder(
            seq_len, d_model, z_dim,
            n_clusters, n_day_types, n_seasons, n_continuous,
            key=k2,
        )
        self.beta    = beta
        self.z_dim   = z_dim
        self.seq_len = seq_len

    def encode(
        self,
        x: jax.Array,
        c_discrete: jax.Array,
        c_continuous: jax.Array,
        key: jax.Array,
        deterministic: bool = False,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Encode x and c → (z_sample, mu_z, logsg_z).
        If deterministic=True, returns mu_z without sampling.
        """
        mu, logsg = self.encoder(x, c_discrete, c_continuous)
        if deterministic:
            return mu, mu, logsg
        eps = jax.random.normal(key, mu.shape)
        z   = mu + jnp.exp(logsg) * eps
        return z, mu, logsg

    def decode(
        self,
        z: jax.Array,
        c_discrete: jax.Array,
        c_continuous: jax.Array,
    ) -> jax.Array:
        """Decode latent z + conditioning → x̂ of shape (seq_len,)."""
        return self.decoder(z, c_discrete, c_continuous)

    def loss(
        self,
        x: jax.Array,             # (seq_len,)
        c_discrete: jax.Array,    # (3,) int32
        c_continuous: jax.Array,  # (n_continuous,) float32
        key: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute β-ELBO loss for a single sample.

        Returns
        -------
        total_loss : scalar
        recon_loss : scalar  (MSE reconstruction)
        kl_loss    : scalar  (KL divergence)
        """
        z, mu, logsg = self.encode(x, c_discrete, c_continuous, key)
        x_hat = self.decode(z, c_discrete, c_continuous)

        recon = jnp.mean((x - x_hat) ** 2)

        # KL(N(μ, σ²) || N(0, 1))  = -0.5 * sum(1 + log σ² - μ² - σ²)
        kl = -0.5 * jnp.mean(1.0 + 2.0 * logsg - mu ** 2 - jnp.exp(2.0 * logsg))

        total = recon + self.beta * kl
        return total, recon, kl

    def generate(
        self,
        c_discrete_batch: jax.Array,    # (B, 3)  int32
        c_continuous_batch: jax.Array,  # (B, n_cont) float32
        key: jax.Array,
    ) -> jax.Array:
        """
        Prior sampling: z ~ N(0, I), then decode.
        Returns (B, seq_len) float32 profiles.
        """
        B = c_discrete_batch.shape[0]
        z = jax.random.normal(key, (B, self.z_dim))
        return jax.vmap(self.decoder)(z, c_discrete_batch, c_continuous_batch)
