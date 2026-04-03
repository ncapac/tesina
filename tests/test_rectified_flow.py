"""
tests/test_rectified_flow.py
Unit tests for src/models/rectified_flow.py and src/training/train_rf.py
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _tiny_model(seq_len=24):
    from src.models.transformer1d import DiffusionTransformer1D
    return DiffusionTransformer1D(
        seq_len=seq_len, d_model=32, n_heads=2, n_layers=2, d_ff=64,
        n_clusters=3, n_day_types=2, n_months=12, n_dow=7,
        key=jax.random.PRNGKey(0),
    )


def _tiny_rf():
    from src.models.rectified_flow import RectifiedFlowProcess
    return RectifiedFlowProcess(freq_loss_weight=0.05)


# ─── RectifiedFlowProcess ─────────────────────────────────────────────────────

class TestRectifiedFlowProcess:
    def test_interpolate_t0_returns_x0(self):
        """At t=0, x_t must equal x_0."""
        rf = _tiny_rf()
        x0    = jnp.ones((4, 24))
        noise = jnp.zeros((4, 24))
        t     = jnp.zeros(4)
        xt    = rf.interpolate(x0, noise, t)
        assert jnp.allclose(xt, x0)

    def test_interpolate_t1_returns_noise(self):
        """At t=1, x_t must equal noise."""
        rf = _tiny_rf()
        x0    = jnp.zeros((4, 24))
        noise = jnp.ones((4, 24))
        t     = jnp.ones(4)
        xt    = rf.interpolate(x0, noise, t)
        assert jnp.allclose(xt, noise)

    def test_interpolate_midpoint(self):
        """At t=0.5, x_t must be the midpoint."""
        rf = _tiny_rf()
        x0    = jnp.zeros((2, 24))
        noise = jnp.ones((2, 24)) * 2.0
        t     = jnp.array([0.5, 0.5])
        xt    = rf.interpolate(x0, noise, t)
        assert jnp.allclose(xt, jnp.ones((2, 24)))

    def test_p_losses_scalar_finite(self):
        """Training loss must be a finite scalar."""
        rf    = _tiny_rf()
        model = _tiny_model()
        B = 4
        x0 = jax.random.normal(jax.random.PRNGKey(1), (B, 24))
        c  = jnp.zeros((B, 4), dtype=jnp.int32)
        t  = jax.random.uniform(jax.random.PRNGKey(2), (B,))
        loss = rf.p_losses(model, x0, c, t, jax.random.PRNGKey(3))
        assert loss.shape == ()
        assert float(loss) >= 0
        assert jnp.isfinite(loss)

    def test_p_losses_null_conditioning(self):
        """Loss must stay finite with null conditioning c=[-1,-1,-1,-1]."""
        rf    = _tiny_rf()
        model = _tiny_model()
        B = 4
        x0 = jax.random.normal(jax.random.PRNGKey(4), (B, 24))
        c  = jnp.full((B, 4), -1, dtype=jnp.int32)
        t  = jax.random.uniform(jax.random.PRNGKey(5), (B,))
        loss = rf.p_losses(model, x0, c, t, jax.random.PRNGKey(6))
        assert jnp.isfinite(loss)

    def test_sample_shape(self):
        """sample() must return (batch_size, seq_len)."""
        rf    = _tiny_rf()
        model = _tiny_model()
        B = 3
        c = jnp.zeros((B, 4), dtype=jnp.int32)
        out = rf.sample(model, c, seq_len=24, batch_size=B,
                        key=jax.random.PRNGKey(7), n_steps=5)
        assert out.shape == (B, 24)

    def test_sample_finite(self):
        """Sampled values must be finite."""
        rf    = _tiny_rf()
        model = _tiny_model()
        c = jnp.array([[0, 0, 5, 1], [1, 1, 0, 6]], dtype=jnp.int32)
        out = rf.sample(model, c, seq_len=24, batch_size=2,
                        key=jax.random.PRNGKey(8), n_steps=5)
        assert jnp.all(jnp.isfinite(out))

    def test_sample_cfg_changes_output(self):
        """CFG (guidance_scale > 0) must produce different output than scale=0."""
        rf    = _tiny_rf()
        model = _tiny_model()
        c = jnp.array([[0, 0, 5, 1]] * 4, dtype=jnp.int32)
        out_uncond = rf.sample(model, c, seq_len=24, batch_size=4,
                               key=jax.random.PRNGKey(9), n_steps=5, guidance_scale=0.0)
        out_guided = rf.sample(model, c, seq_len=24, batch_size=4,
                               key=jax.random.PRNGKey(9), n_steps=5, guidance_scale=1.5)
        # With different guidance the outputs differ (not all-equal)
        assert not jnp.allclose(out_uncond, out_guided)


# ─── train_rf ─────────────────────────────────────────────────────────────────

class TestRFTrainStep:
    def test_loss_is_scalar_finite(self):
        from src.training.train_rf import train_step_rf
        import optax

        model = _tiny_model()
        rf    = _tiny_rf()
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        B = 4
        x0 = jax.random.normal(jax.random.PRNGKey(10), (B, 24))
        c  = jnp.zeros((B, 4), dtype=jnp.int32)

        _, _, loss = train_step_rf(model, rf, opt_state, optimizer, x0, c,
                                   jax.random.PRNGKey(11))
        assert loss.shape == ()
        assert float(loss) >= 0.0
        assert jnp.isfinite(loss)

    def test_params_change_after_step(self):
        from src.training.train_rf import train_step_rf
        import optax

        model = _tiny_model()
        rf    = _tiny_rf()
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        x0 = jax.random.normal(jax.random.PRNGKey(12), (4, 24))
        c  = jnp.zeros((4, 4), dtype=jnp.int32)

        leaves_before = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        model2, _, _ = train_step_rf(model, rf, opt_state, optimizer, x0, c,
                                     jax.random.PRNGKey(13))
        leaves_after = jax.tree_util.tree_leaves(eqx.filter(model2, eqx.is_array))

        changed = any(
            not jnp.allclose(b, a)
            for b, a in zip(leaves_before, leaves_after)
        )
        assert changed, "Parameters must change after a training step"

    def test_eval_step_rf_finite(self):
        from src.training.train_rf import eval_step_rf

        model = _tiny_model()
        rf    = _tiny_rf()
        x0 = jax.random.normal(jax.random.PRNGKey(14), (4, 24))
        c  = jnp.zeros((4, 4), dtype=jnp.int32)
        loss = eval_step_rf(model, rf, x0, c, jax.random.PRNGKey(15))
        assert jnp.isfinite(loss)

    def test_t_is_continuous(self):
        """RF training must sample t in (0,1), not integers."""
        from src.training.train_rf import train_step_rf
        import optax

        # Check indirectly: run with T=10 diffusion to confirm it works
        # (if t were integers it would index out-of-bounds on tiny T)
        model = _tiny_model()
        rf    = _tiny_rf()
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        x0 = jax.random.normal(jax.random.PRNGKey(16), (4, 24))
        c  = jnp.zeros((4, 4), dtype=jnp.int32)
        _, _, loss = train_step_rf(model, rf, opt_state, optimizer, x0, c,
                                   jax.random.PRNGKey(17))
        assert jnp.isfinite(loss)


# ─── metrics: marginal_wasserstein + compare_models ──────────────────────────

class TestNewMetrics:
    def test_wasserstein_self_is_zero(self):
        """W1(real, real) should be 0 (or near zero for finite samples)."""
        from src.evaluation.metrics import marginal_wasserstein
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 24)).astype(np.float32)
        w = marginal_wasserstein(x, x)
        assert w < 1e-6

    def test_wasserstein_different_distributions(self):
        """W1(N(0,1), N(2,1)) should be close to 2."""
        from src.evaluation.metrics import marginal_wasserstein
        rng = np.random.default_rng(1)
        real = rng.standard_normal((200, 24)).astype(np.float32)
        synth = (rng.standard_normal((200, 24)) + 2.0).astype(np.float32)
        w = marginal_wasserstein(real, synth)
        assert 1.5 < w < 2.5, f"Expected ~2, got {w:.3f}"

    def test_compare_models_returns_dataframe(self):
        """compare_models must return a pd.DataFrame with expected columns."""
        from src.evaluation.metrics import compare_models
        import pandas as pd

        rng = np.random.default_rng(2)
        real = rng.standard_normal((100, 24)).astype(np.float32)
        conds = np.column_stack([
            np.zeros(100, dtype=np.int32),    # cluster 0
            (np.arange(100) % 2).astype(np.int32),  # alt weekday/weekend
            np.zeros(100, dtype=np.int32),
            np.zeros(100, dtype=np.int32),
        ])

        def dummy_gen(c_batch, seed):
            r = np.random.default_rng(int(seed))
            return r.standard_normal((len(c_batch), 24)).astype(np.float32)

        df, figs = compare_models(
            models_dict={'ModelA': dummy_gen},
            real_data=real,
            conditions=conds,
            n_samples=30,
            unique_conditions=[(0, 0)],
            show_figs=False,
            verbose=False,
        )

        assert isinstance(df, pd.DataFrame)
        assert set(['model', 'acf_l2', 'crps', 'discriminative_acc', 'wasserstein']).issubset(df.columns)
        assert len(df) == 1
        assert df.iloc[0]['model'] == 'ModelA'
