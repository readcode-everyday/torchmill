"""Pydantic config schema for torchmill.yaml (the watcher/autopilot config).

This is separate from TorchmillConfig (trainer config in config.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class BrainConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None


class InstantKillConfig(BaseModel):
    nan_loss: bool = True
    grad_norm_above: Optional[float] = 1000
    loss_above: Optional[float] = 100


class BrainTriggerConfig(BaseModel):
    val_loss_rising_for: int = 5
    loss_plateau_for: int = 10
    training_completed: bool = True
    every_n_epochs: Optional[int] = 25


class SummaryConfig(BaseModel):
    max_history_epochs: int = 20
    include_experiment_history: bool = True
    max_tokens: int = 800


class WatcherConfig(BaseModel):
    check_interval: int = 30
    instant_kill: InstantKillConfig = Field(default_factory=InstantKillConfig)
    brain_triggers: BrainTriggerConfig = Field(default_factory=BrainTriggerConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)


class AutopilotConfig(BaseModel):
    max_rounds: int = 10
    max_total_hours: float = 24.0
    approval_required: bool = False


class HistoryConfig(BaseModel):
    dir: str = "./torchmill_runs"


class TorchmillWatcherConfig(BaseModel):
    """Top-level config for ``torchmill run`` and ``torchmill watch``."""

    # Mode 1 (run) only
    command: Optional[str] = None
    workdir: str = "."
    training_config: Optional[str] = None  # file to mutate between rounds

    brain: BrainConfig = Field(default_factory=BrainConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    autopilot: AutopilotConfig = Field(default_factory=AutopilotConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TorchmillWatcherConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw or {})
