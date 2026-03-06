"""Tests for the core training loop."""

import textwrap
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from torchmill.config import TorchmillConfig
from torchmill.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(extra_yaml: str = "") -> TorchmillConfig:
    base = textwrap.dedent("""\
        model:
          class: torch.nn.Linear
          args:
            in_features: 4
            out_features: 1
        data:
          type: csv
          train_path: ./train.csv
          target_column: label
          batch_size: 8
          num_workers: 0
        training:
          epochs: 3
          seed: 0
          amp: false
          device: cpu
          task: regression
          optimizer:
            type: adamw
            lr: 0.001
          scheduler:
            type: none
        callbacks:
          checkpoint:
            save_dir: /tmp/torchmill_test_ckpts
            save_top_k: 2
            monitor: val_loss
          early_stopping:
            enabled: false
          logging:
            tensorboard: false
            console: false
    """)
    yaml_str = base + "\n" + extra_yaml

    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    tmp.write(yaml_str)
    tmp.close()
    return TorchmillConfig.from_yaml(tmp.name)


def regression_datasets(n=64, features=4):
    X = torch.randn(n, features)
    y = X.sum(dim=1, keepdim=True)
    split = n // 4 * 3
    train = TensorDataset(X[:split], y[:split])
    val = TensorDataset(X[split:], y[split:])
    return train, val


def classification_datasets(n=64, features=4, num_classes=3):
    X = torch.randn(n, features)
    y = torch.randint(0, num_classes, (n,))
    split = n // 4 * 3
    train = TensorDataset(X[:split], y[:split])
    val = TensorDataset(X[split:], y[split:])
    return train, val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainerRegression:
    def test_fit_returns_metrics(self):
        config = make_config()
        model = nn.Linear(4, 1)
        train_ds, val_ds = regression_datasets()
        trainer = Trainer(config)
        metrics = trainer.fit(model, train_ds, val_ds)
        assert "train_loss" in metrics
        assert "val_loss" in metrics

    def test_fit_no_val(self):
        config = make_config()
        model = nn.Linear(4, 1)
        train_ds, _ = regression_datasets()
        trainer = Trainer(config)
        metrics = trainer.fit(model, train_ds)
        assert "train_loss" in metrics
        assert "val_loss" not in metrics

    def test_evaluate(self):
        config = make_config()
        model = nn.Linear(4, 1)
        train_ds, val_ds = regression_datasets()
        trainer = Trainer(config)
        trainer.fit(model, train_ds)
        test_metrics = trainer.evaluate(model, val_ds, task="regression")
        assert "test_loss" in test_metrics

    def test_loss_decreases_over_epochs(self):
        """After 10 epochs the training loss should drop vs random init."""
        yaml_extra = textwrap.dedent("""\
            training:
              epochs: 10
              seed: 42
              amp: false
              device: cpu
              task: regression
              optimizer:
                type: adamw
                lr: 0.05
              scheduler:
                type: none
        """)
        config = make_config(yaml_extra)
        model = nn.Linear(4, 1)
        train_ds, _ = regression_datasets(n=128)

        # Record initial loss
        with torch.no_grad():
            x = torch.randn(128, 4)
            y = x.sum(dim=1, keepdim=True)
            initial_loss = nn.MSELoss()(model(x), y).item()

        trainer = Trainer(config)
        metrics = trainer.fit(model, train_ds)

        assert metrics["train_loss"] < initial_loss


class TestTrainerClassification:
    def test_fit_classification(self):
        yaml_extra = textwrap.dedent("""\
            training:
              epochs: 3
              seed: 0
              amp: false
              device: cpu
              task: classification
              optimizer:
                type: adam
                lr: 0.001
              scheduler:
                type: none
        """)
        config = make_config(yaml_extra)
        model = nn.Linear(4, 3)  # 3 classes
        train_ds, val_ds = classification_datasets(num_classes=3)
        trainer = Trainer(config)
        metrics = trainer.fit(model, train_ds, val_ds)
        assert "train_loss" in metrics
        assert "train_accuracy" in metrics

    def test_binary_classification(self):
        yaml_extra = textwrap.dedent("""\
            training:
              epochs: 3
              seed: 0
              amp: false
              device: cpu
              task: classification
              optimizer:
                type: adamw
                lr: 0.001
              scheduler:
                type: none
        """)
        config = make_config(yaml_extra)
        model = nn.Linear(4, 1)  # binary
        X = torch.randn(64, 4)
        y = torch.randint(0, 2, (64,)).float()
        train_ds = TensorDataset(X[:48], y[:48])
        val_ds = TensorDataset(X[48:], y[48:])
        trainer = Trainer(config)
        metrics = trainer.fit(model, train_ds, val_ds)
        assert "train_loss" in metrics


class TestTrainerSchedulers:
    @pytest.mark.parametrize("sched_type", ["cosine", "linear", "step", "exponential"])
    def test_scheduler(self, sched_type):
        yaml_extra = textwrap.dedent(f"""\
            training:
              epochs: 3
              seed: 0
              amp: false
              device: cpu
              task: regression
              optimizer:
                type: adamw
                lr: 0.001
              scheduler:
                type: {sched_type}
                warmup_steps: 2
                step_size: 1
                gamma: 0.9
        """)
        config = make_config(yaml_extra)
        model = nn.Linear(4, 1)
        train_ds, _ = regression_datasets()
        trainer = Trainer(config)
        metrics = trainer.fit(model, train_ds)
        assert "train_loss" in metrics


class TestTrainerGradientClip:
    def test_gradient_clip(self):
        yaml_extra = textwrap.dedent("""\
            training:
              epochs: 2
              seed: 0
              amp: false
              device: cpu
              task: regression
              gradient_clip: 1.0
              optimizer:
                type: sgd
                lr: 0.01
                momentum: 0.0
              scheduler:
                type: none
        """)
        config = make_config(yaml_extra)
        model = nn.Linear(4, 1)
        train_ds, _ = regression_datasets()
        trainer = Trainer(config)
        metrics = trainer.fit(model, train_ds)
        assert "train_loss" in metrics


class TestCheckpointCallback:
    def test_checkpoints_created(self, tmp_path):
        yaml_extra = textwrap.dedent(f"""\
            callbacks:
              checkpoint:
                save_dir: {tmp_path}
                save_top_k: 2
                monitor: val_loss
                mode: min
              early_stopping:
                enabled: false
              logging:
                tensorboard: false
                console: false
        """)
        config = make_config(yaml_extra)
        model = nn.Linear(4, 1)
        train_ds, val_ds = regression_datasets()
        trainer = Trainer(config)
        trainer.fit(model, train_ds, val_ds)
        checkpoints = list(tmp_path.glob("*.pt"))
        assert len(checkpoints) > 0

    def test_best_checkpoint_loadable(self, tmp_path):
        from torchmill.callbacks.checkpoint import CheckpointCallback

        yaml_extra = textwrap.dedent(f"""\
            callbacks:
              checkpoint:
                save_dir: {tmp_path}
                save_top_k: 3
                monitor: val_loss
                mode: min
              early_stopping:
                enabled: false
              logging:
                tensorboard: false
                console: false
        """)
        config = make_config(yaml_extra)
        model = nn.Linear(4, 1)
        train_ds, val_ds = regression_datasets()
        trainer = Trainer(config)
        trainer.fit(model, train_ds, val_ds)

        best_path = tmp_path / "best.pt"
        assert best_path.exists()

        new_model = nn.Linear(4, 1)
        state = CheckpointCallback.load_checkpoint(str(best_path), new_model)
        assert "epoch" in state
        assert "metrics" in state


class TestEarlyStoppingCallback:
    def test_early_stopping_triggers(self):
        from torchmill.callbacks.early_stopping import EarlyStoppingCallback

        cb = EarlyStoppingCallback(patience=3, monitor="val_loss", mode="min")
        for i in range(10):
            stop = cb.on_epoch_end(i, {"val_loss": 1.0})  # no improvement
        assert cb.should_stop

    def test_early_stopping_resets_on_improvement(self):
        from torchmill.callbacks.early_stopping import EarlyStoppingCallback

        cb = EarlyStoppingCallback(patience=3, monitor="val_loss", mode="min")
        cb.on_epoch_end(0, {"val_loss": 1.0})
        cb.on_epoch_end(1, {"val_loss": 0.9})  # improvement
        assert cb.counter == 0
        assert not cb.should_stop
