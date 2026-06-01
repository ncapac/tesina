"""
tests/test_models.py
Unit tests for src/models/transformer1d.py and src/models/diffusion.py

Updated for new API:
  - DiffusionTransformer1D uses c_discrete (3,) + c_continuous (n_cont,)
  - DiffusionProcess.p_losses / samplers use c_discrete + c_continuous
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
        n_seasons=4,
        n_continuous=1,
        key=jax.random.PRNGKey(0),
    )


def _tiny_diffusion(T=10, data_freq_loss_weight=0.05):
    from src.models.diffusion import DiffusionProcess
    return DiffusionProcess(T=T, data_freq_loss_weight=data_freq_loss_weight)


def _c_discrete(B=None, cid=0, dt=0, season=1):
    """Helper to build discrete conditioning arrays."""
    arr = np.array([cid, dt, season], dtype=np.int32)
    if B is not None:
        return jnp.tile(arr[None], (B, 1))
    return jnp.array(arr)


def _c_continuous(B=None, temp=0.0):
    """Helper to build continuous conditioning arrays."""
    arr = np.array([temp], dtype=np.float32)
    if B is not None:
        return jnp.tile(arr[None], (B, 1))
    return jnp.array(arr)


# ─── Transformer ──────────────────────────────────────────────────────────────

class TestDiffusionTransformer1D:
    def test_forward_output_shape(self):
        """Unbatched forward pass must produce (seq_len,) output."""
        model = _tiny_model()
        x_t = jax.random.normal(jax.random.PRNGKey(1), (24,))
        t   = jnp.array(5, dtype=jnp.int32)
        out = model(x_t, t, _c_discrete(), _c_continuous())
        assert out.shape == (24,)

    def test_batched_vmap(self):
        """jax.vmap over batch dimension must work."""
        model = _tiny_model()
        B = 8
        x_t = jax.random.normal(jax.random.PRNGKey(2), (B, 24))
        t   = jnp.ones(B, dtype=jnp.int32) * 5
        out = jax.vmap(model)(x_t, t, _c_discrete(B), _c_continuous(B))
        assert out.shape == (B, 24)

    def test_null_conditioning(self):
        """Null conditioning c_discrete[0] < 0 must run without error."""
        from src.models.transformer1d import CFG_NULL_DISCRETE

        model = _tiny_model()
        x_t   = jax.random.normal(jax.random.PRNGKey(3), (24,))
        t     = jnp.array(0, dtype=jnp.int32)
        c_null = jnp.array([CFG_NULL_DISCRETE, CFG_NULL_DISCRETE, CFG_NULL_DISCRETE], dtype=jnp.int32)
        c_zero = jnp.array([0.0], dtype=jnp.float32)
        out = model(x_t, t, c_null, c_zero)
        assert out.shape == (24,)
        assert jnp.all(jnp.isfinite(out))

    def test_cfg_null_helper_matches_model_null_representation(self):
        from src.models.transformer1d import CFG_NULL_DISCRETE, make_cfg_null_conditioning

        c_disc = _c_discrete(B=3, cid=2, dt=1, season=3)
        c_cont = _c_continuous(B=3, temp=0.7)
        null_disc, null_cont = make_cfg_null_conditioning(c_disc, c_cont)
        assert null_disc.shape == c_disc.shape
        assert null_cont.shape == c_cont.shape
        assert jnp.all(null_disc == CFG_NULL_DISCRETE)
        assert jnp.all(null_cont == 0.0)

    def test_output_finite(self):
        """Output must not contain NaN or Inf."""
        model = _tiny_model()
        x_t = jax.random.normal(jax.random.PRNGKey(4), (24,))
        for cid in range(3):
            for dt in range(2):
                for season in range(4):
                    t   = jnp.array(1, dtype=jnp.int32)
                    out = model(
                        x_t, t,
                        jnp.array([cid, dt, season], dtype=jnp.int32),
                        jnp.array([0.5], dtype=jnp.float32),
                    )
                    assert jnp.all(jnp.isfinite(out)), \
                        f"Non-finite output for c=[{cid},{dt},{season}]"

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
        key = jax.random.PRNGKey(0)
        x0 = jax.random.normal(key, (24,))
        t  = jnp.array(5, dtype=jnp.int32)
        noise = jax.random.normal(jax.random.PRNGKey(1), (24,))
        x_t = dp.q_sample(x0, t, noise)
        assert x_t.shape == (24,)

    def test_p_losses_scalar(self):
        """p_losses must return a scalar loss."""
        dp    = _tiny_diffusion(T=10)
        model = _tiny_model()
        key   = jax.random.PRNGKey(42)
        B = 4
        x0 = jax.random.normal(key, (B, 24))
        t  = jax.random.randint(key, (B,), 0, 10, dtype=jnp.int32)
        loss = dp.p_losses(model, x0, _c_discrete(B), _c_continuous(B), t, key)
        assert loss.shape == ()
        assert float(loss) >= 0

    def test_ddpm_sample_shape(self):
        dp    = _tiny_diffusion(T=10)
        model = _tiny_model()
        key   = jax.random.PRNGKey(5)
        B = 3
        samples = dp.ddpm_sample(
            model, _c_discrete(B), _c_continuous(B),
            seq_len=24, batch_size=B, key=key, guidance_scale=1.0,
        )
        assert samples.shape == (B, 24)
        assert jnp.all(jnp.isfinite(samples))

    def test_ddim_sample_shape(self):
        dp    = _tiny_diffusion(T=10)
        model = _tiny_model()
        key   = jax.random.PRNGKey(6)
        B = 3
        samples = dp.ddim_sample(
            model, _c_discrete(B), _c_continuous(B),
            seq_len=24, batch_size=B, key=key, n_steps=5, guidance_scale=1.0,
        )
        assert samples.shape == (B, 24)
        assert jnp.all(jnp.isfinite(samples))

    def test_diffusion_cfg_uses_shared_null_conditioning(self, monkeypatch):
        import src.models.diffusion as diffusion_module
        from src.models.transformer1d import CFG_NULL_DISCRETE

        observed = {}

        def spy(c_discrete, c_continuous):
            null_disc = jnp.full_like(c_discrete, CFG_NULL_DISCRETE)
            null_cont = jnp.zeros_like(c_continuous)
            observed["disc"] = null_disc
            observed["cont"] = null_cont
            return null_disc, null_cont

        monkeypatch.setattr(diffusion_module, "make_cfg_null_conditioning", spy)
        dp = _tiny_diffusion(T=10)
        model = _tiny_model()
        B = 2
        x_t = jax.random.normal(jax.random.PRNGKey(8), (B, 24))
        t = jnp.ones(B, dtype=jnp.int32)
        out = dp._predict_eps_cfg(model, x_t, _c_discrete(B), _c_continuous(B), t, guidance_scale=1.0)
        assert out.shape == (B, 24)
        assert jnp.all(observed["disc"] == CFG_NULL_DISCRETE)
        assert jnp.all(observed["cont"] == 0.0)

    def test_ddim_deterministic(self):
        """Same key + eta=0 must give identical samples."""
        dp    = _tiny_diffusion(T=10)
        model = _tiny_model()
        B     = 2
        key   = jax.random.PRNGKey(7)
        s1 = dp.ddim_sample(
            model, _c_discrete(B), _c_continuous(B),
            seq_len=24, batch_size=B, key=key, n_steps=5, eta=0.0,
        )
        s2 = dp.ddim_sample(
            model, _c_discrete(B), _c_continuous(B),
            seq_len=24, batch_size=B, key=key, n_steps=5, eta=0.0,
        )
        np.testing.assert_array_equal(np.array(s1), np.array(s2))

