"""Pydantic config schema for torchmill."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    class_path: str = Field(..., alias="class", description="Importable path to nn.Module class")
    args: Dict[str, Any] = Field(default_factory=dict, description="kwargs for model constructor")

    model_config = {"populate_by_name": True}


class DataConfig(BaseModel):
    type: Literal["csv", "relbench", "custom"] = "csv"

    # CSV / Parquet paths
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    target_column: Optional[str] = None

    # RelBench
    dataset_name: Optional[str] = None
    task_name: Optional[str] = None

    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True

    @model_validator(mode="after")
    def _check_paths(self) -> "DataConfig":
        if self.type == "csv" and self.train_path is None:
            raise ValueError("data.train_path is required for type='csv'")
        if self.type == "relbench" and (
            self.dataset_name is None or self.task_name is None
        ):
            raise ValueError(
                "data.dataset_name and data.task_name are required for type='relbench'"
            )
        return self


class OptimizerConfig(BaseModel):
    type: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    momentum: float = 0.9  # SGD only


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "linear", "step", "exponential", "none"] = "none"
    warmup_steps: int = 0
    # step / exponential
    step_size: int = 10
    gamma: float = 0.1
    # linear / cosine: total steps are inferred from training config


class TrainingConfig(BaseModel):
    epochs: int = 100
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    gradient_clip: Optional[float] = None
    amp: bool = False  # mixed precision
    seed: int = 42
    device: str = "auto"
    task: Literal["classification", "regression", "auto"] = "auto"
    # DDP
    ddp: bool = False
    ddp_backend: str = "nccl"


class CheckpointConfig(BaseModel):
    save_dir: str = "./checkpoints"
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_every_n_epochs: Optional[int] = None


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = 10
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    min_delta: float = 0.0


class LoggingConfig(BaseModel):
    tensorboard: bool = True
    log_dir: str = "./runs"
    log_every_n_steps: int = 10
    console: bool = True


class CallbacksConfig(BaseModel):
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class TorchmillConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TorchmillConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        import json
        data = json.loads(self.model_dump_json(by_alias=True))
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
