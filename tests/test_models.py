"""
tests/test_models.py
Unit tests for src/models/transformer1d.py and src/models/diffusion.py
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _tiny_model(seq_len=24, d_model=32, n_heads=2, n_layers=2, n_clusters=3):
    from src.models.transformer1d import DiffusionTransformer1D
    return DiffusionTransformer1D(
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=64,
        n_clusters=n_clusters,
        n_day_types=2,
        n_months=12,
        n_dow=7,
        key=jax.random.PRNGKey(0),
    )


def _tiny_diffusion(T=10):
    from src.models.diffusion import DiffusionProcess
    return DiffusionProcess(T=T, freq_loss_weight=0.05)


# ─── Transformer ──────────────────────────────────────────────────────────────

class TestDiffusionTransformer1D:
    def test_forward_output_shape(self):
        """Unbatched forward pass must produce (seq_len,) output."""
        model = _tiny_model()
        x_t = jax.random.normal(jax.random.PRNGKey(1), (24,))
        t = jnp.array(5, dtype=jnp.int32)
        c = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        out = model(x_t, t, c)
        assert out.shape == (24,)

    def test_batched_vmap(self):
        """jax.vmap over batch dimension must work."""
        model = _tiny_model()
        B = 8
        x_t = jax.random.normal(jax.random.PRNGKey(2), (B, 24))
        t = jnp.ones(B, dtype=jnp.int32) * 5
        c = jnp.zeros((B, 4), dtype=jnp.int32)
        out = jax.vmap(model)(x_t, t, c)
        assert out.shape == (B, 24)

    def test_null_conditioning(self):
        """Null conditioning c=[-1,-1,-1,-1] must run without error."""
        model = _tiny_model()
        x_t = jax.random.normal(jax.random.PRNGKey(3), (24,))
        t = jnp.array(0, dtype=jnp.int32)
        c_null = jnp.array([-1, -1, -1, -1], dtype=jnp.int32)
        out = model(x_t, t, c_null)
        assert out.shape == (24,)
        # null conditioning must zero out all embeddings → output should be finite
        assert jnp.all(jnp.isfinite(out))

    def test_output_finite(self):
        """Output must not contain NaN or Inf."""
        model = _tiny_model()
        x_t = jax.random.normal(jax.random.PRNGKey(4), (24,))
        for cid in range(3):
            for dt in range(2):
                for mo in [0, 5, 11]:
                    t = jnp.array(1, dtype=jnp.int32)
                    c = jnp.array([cid, dt, mo, dt * 5], dtype=jnp.int32)
                    out = model(x_t, t, c)
                    assert jnp.all(jnp.isfinite(out)), f"Non-finite output for c=[{cid},{dt},{mo}]"

    def test_is_equinox_module(self):
        model = _tiny_model()
        assert isinstance(model, eqx.Module)

    def test_has_trainable_params(self):
        model = _tiny_model()
        params = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        total = sum(p.size for p in params)
        assert total > 0


# ─── DiffusionProcess ─────────────────────────────────────────────────────────

class TestDiffusionProcess:
    def test_cosine_schedule_range(self):
        """alphas_cumprod must start near 1 and decrease monotonically."""
        dp = _tiny_diffusion(T=20)
        acp = np.array(dp.alphas_cumprod)
        assert acp[0] < 1.0
        assert acp[-1] > 0.0
        assert np.all(np.diff(acp) < 0), "alphas_cumprod must be monotonically decreasing"

    def test_q_sample_shape(self):
        dp = _tiny_diffusion(T=10)
        model = _tiny_model()
        key = jax.random.PRNGKey(0)
        x0 = jax.random.normal(key, (24,))
        t = jnp.array(5, dtype=jnp.int32)
        noise = jax.random.normal(jax.random.PRNGKey(1), (24,))
        x_t = dp.q_sample(x0, t, noise)
        assert x_t.shape == (24,)

    def test_p_losses_scalar(self):
        """p_losses must return a scalar loss."""
        dp = _tiny_diffusion(T=10)
        model = _tiny_model()
        key = jax.random.PRNGKey(42)
        B = 4
        x0 = jax.random.normal(key, (B, 24))
        c = jnp.zeros((B, 4), dtype=jnp.int32)
        t = jax.random.randint(key, (B,), 0, 10, dtype=jnp.int32)
        loss = dp.p_losses(model, x0, c, t, key)
        assert loss.shape == ()
        assert float(loss) >= 0

    def test_ddpm_sample_shape(self):
        dp = _tiny_diffusion(T=10)
        model = _tiny_model()
        key = jax.random.PRNGKey(5)
        B = 3
        c = jnp.zeros((B, 4), dtype=jnp.int32)
        samples = dp.ddpm_sample(model, c, seq_len=24, batch_size=B, key=key, guidance_scale=1.0)
        assert samples.shape == (B, 24)
        assert jnp.all(jnp.isfinite(samples))

    def test_ddim_sample_shape(self):
        dp = _tiny_diffusion(T=10)
        model = _tiny_model()
        key = jax.random.PRNGKey(6)
        B = 3
        c = jnp.zeros((B, 4), dtype=jnp.int32)
        samples = dp.ddim_sample(model, c, seq_len=24, batch_size=B, key=key, n_steps=5, guidance_scale=1.0)
        assert samples.shape == (B, 24)
        assert jnp.all(jnp.isfinite(samples))

    def test_ddim_deterministic(self):
        """Same key + eta=0 must give identical samples."""
        dp = _tiny_diffusion(T=10)
        model = _tiny_model()
        c = jnp.zeros((2, 4), dtype=jnp.int32)
        key = jax.random.PRNGKey(7)
        s1 = dp.ddim_sample(model, c, seq_len=24, batch_size=2, key=key, n_steps=5, eta=0.0)
        s2 = dp.ddim_sample(model, c, seq_len=24, batch_size=2, key=key, n_steps=5, eta=0.0)
        np.testing.assert_array_equal(np.array(s1), np.array(s2))
