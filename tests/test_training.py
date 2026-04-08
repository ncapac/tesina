"""
tests/test_training.py
Unit tests for src/training/train.py
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx


def _make_model_and_diffusion():
    from src.models.transformer1d import DiffusionTransformer1D
    from src.models.diffusion import DiffusionProcess
    model = DiffusionTransformer1D(
        seq_len=24, d_model=32, n_heads=2, n_layers=2, d_ff=64,
        n_clusters=3, n_day_types=2, key=jax.random.PRNGKey(0),
    )
    diffusion = DiffusionProcess(T=10, freq_loss_weight=0.05)
    return model, diffusion


class TestTrainStep:
    def test_loss_is_scalar_and_finite(self):
        from src.training.train import train_step
        import optax

        model, diffusion = _make_model_and_diffusion()
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        B = 4
        x0 = jax.random.normal(jax.random.PRNGKey(1), (B, 24))
        c = jnp.zeros((B, 4), dtype=jnp.int32)
        key = jax.random.PRNGKey(2)

        model2, opt_state2, loss = train_step(model, diffusion, opt_state, optimizer, x0, c, key)
        assert loss.shape == ()
        assert float(loss) >= 0
        assert jnp.isfinite(loss)

    def test_parameters_change_after_step(self):
        from src.training.train import train_step
        import optax

        model, diffusion = _make_model_and_diffusion()
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        x0 = jax.random.normal(jax.random.PRNGKey(3), (4, 24))
        c = jnp.zeros((4, 4), dtype=jnp.int32)

        leaves_before = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))

        model2, opt_state2, loss = train_step(
            model, diffusion, opt_state, optimizer, x0, c, jax.random.PRNGKey(4)
        )
        leaves_after = jax.tree_util.tree_leaves(eqx.filter(model2, eqx.is_array))

        changed = any(
            not np.allclose(np.array(b), np.array(a))
            for b, a in zip(leaves_before, leaves_after)
        )
        assert changed, "At least one parameter should change after a gradient step"


class TestEvalStep:
    def test_returns_scalar(self):
        from src.training.train import eval_step

        model, diffusion = _make_model_and_diffusion()
        x0 = jax.random.normal(jax.random.PRNGKey(5), (4, 24))
        c = jnp.zeros((4, 4), dtype=jnp.int32)
        key = jax.random.PRNGKey(6)

        loss = eval_step(model, diffusion, x0, c, key)
        assert loss.shape == ()
        assert jnp.isfinite(loss)


class TestTrainer:
    def test_fit_returns_loss_lists(self, tmp_path):
        from src.training.train import Trainer
        from src.data.dataset import numpy_dataloader

        model, diffusion = _make_model_and_diffusion()
        trainer = Trainer(model, diffusion, lr=1e-3, warmup_steps=2,
                          total_steps=20, checkpoint_dir=str(tmp_path))

        N = 32
        xs = np.random.randn(N, 24).astype(np.float32)
        cs = np.zeros((N, 4), dtype=np.int32)
        train_loader = numpy_dataloader(xs, cs, batch_size=8, shuffle=True)
        val_loader   = numpy_dataloader(xs, cs, batch_size=8, shuffle=False)

        # epoch_len = 32 // 8 = 4 batches per epoch → fast test
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=2,
            val_every=1,
            save_every=100,
            log_every_steps=999,
            val_batches=4,
        )
        assert len(trainer.train_losses) == 2
        assert len(trainer.val_losses) == 2
        assert all(np.isfinite(l) for l in trainer.train_losses)

    def test_checkpoint_roundtrip(self, tmp_path):
        from src.training.train import Trainer

        model, diffusion = _make_model_and_diffusion()
        trainer = Trainer(model, diffusion)

        ckpt_path = str(tmp_path / "model.pkl")
        trainer.save(ckpt_path)

        # Load into a fresh Trainer
        model2, diffusion2 = _make_model_and_diffusion()
        trainer2 = Trainer(model2, diffusion2)
        trainer2.load(ckpt_path)

        # All parameters must match
        leaves1 = jax.tree_util.tree_leaves(eqx.filter(trainer.model,  eqx.is_array))
        leaves2 = jax.tree_util.tree_leaves(eqx.filter(trainer2.model, eqx.is_array))
        for l1, l2 in zip(leaves1, leaves2):
            np.testing.assert_array_equal(np.array(l1), np.array(l2))
