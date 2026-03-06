"""Tests for config loading and validation."""

import textwrap
import tempfile
from pathlib import Path

import pytest

from torchmill.config import (
    TorchmillConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    CallbacksConfig,
)


MINIMAL_YAML = textwrap.dedent("""\
    model:
      class: torch.nn.Linear
      args:
        in_features: 4
        out_features: 1
    data:
      type: csv
      train_path: ./train.csv
      target_column: label
""")


def write_yaml(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


class TestConfigLoading:
    def test_minimal_config_from_yaml(self):
        path = write_yaml(MINIMAL_YAML)
        cfg = TorchmillConfig.from_yaml(path)
        assert cfg.model.class_path == "torch.nn.Linear"
        assert cfg.model.args == {"in_features": 4, "out_features": 1}
        assert cfg.data.type == "csv"
        assert cfg.data.train_path == "./train.csv"

    def test_defaults_populated(self):
        path = write_yaml(MINIMAL_YAML)
        cfg = TorchmillConfig.from_yaml(path)
        assert cfg.training.epochs == 100
        assert cfg.training.optimizer.type == "adamw"
        assert cfg.training.amp is False
        assert cfg.callbacks.early_stopping.patience == 10

    def test_full_config(self):
        yaml_str = textwrap.dedent("""\
            model:
              class: torch.nn.Linear
              args:
                in_features: 10
                out_features: 2
            data:
              type: csv
              train_path: ./train.csv
              val_path: ./val.csv
              target_column: label
              batch_size: 64
              num_workers: 2
            training:
              epochs: 50
              seed: 123
              amp: false
              gradient_clip: 0.5
              optimizer:
                type: adam
                lr: 0.01
              scheduler:
                type: cosine
                warmup_steps: 10
            callbacks:
              checkpoint:
                save_dir: /tmp/ckpts
                save_top_k: 1
                monitor: val_loss
              early_stopping:
                enabled: true
                patience: 5
              logging:
                tensorboard: false
                log_every_n_steps: 5
        """)
        path = write_yaml(yaml_str)
        cfg = TorchmillConfig.from_yaml(path)
        assert cfg.training.epochs == 50
        assert cfg.training.optimizer.lr == 0.01
        assert cfg.training.scheduler.type == "cosine"
        assert cfg.callbacks.checkpoint.save_top_k == 1
        assert cfg.callbacks.early_stopping.patience == 5

    def test_csv_missing_train_path_raises(self):
        yaml_str = textwrap.dedent("""\
            model:
              class: torch.nn.Linear
              args: {in_features: 4, out_features: 1}
            data:
              type: csv
        """)
        path = write_yaml(yaml_str)
        with pytest.raises(Exception):
            TorchmillConfig.from_yaml(path)

    def test_relbench_missing_fields_raises(self):
        yaml_str = textwrap.dedent("""\
            model:
              class: torch.nn.Linear
              args: {in_features: 4, out_features: 1}
            data:
              type: relbench
        """)
        path = write_yaml(yaml_str)
        with pytest.raises(Exception):
            TorchmillConfig.from_yaml(path)

    def test_to_yaml_roundtrip(self, tmp_path):
        path = write_yaml(MINIMAL_YAML)
        cfg = TorchmillConfig.from_yaml(path)
        out_path = tmp_path / "out.yaml"
        cfg.to_yaml(out_path)
        cfg2 = TorchmillConfig.from_yaml(out_path)
        assert cfg2.model.class_path == cfg.model.class_path
        assert cfg2.training.epochs == cfg.training.epochs
