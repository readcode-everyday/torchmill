"""Layer 2: Local rules, NO LLM — detects signals from MetricsSnapshot stream.

Produces Signal objects with severity and whether they should trigger
an instant kill or a brain (LLM) analysis call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .log_parser import MetricsSnapshot


class SignalSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class SignalType(Enum):
    NAN_LOSS = "nan_loss"
    GRAD_EXPLOSION = "grad_explosion"
    LOSS_ABOVE_THRESHOLD = "loss_above"
    CUDA_OOM = "cuda_oom"
    OVERFITTING = "overfitting"
    LOSS_PLATEAU = "loss_plateau"
    LOSS_DIVERGING = "loss_diverging"
    TRAINING_COMPLETED = "training_completed"
    PERIODIC_CHECK = "periodic_check"


@dataclass
class Signal:
    type: SignalType
    severity: SignalSeverity
    message: str
    epoch: Optional[int] = None
    is_brain_trigger: bool = False
    is_kill_trigger: bool = False


class SignalDetector:
    """Stateful detector: call .process() for each MetricsSnapshot."""

    def __init__(
        self,
        instant_kill_config: Optional[dict] = None,
        brain_trigger_config: Optional[dict] = None,
    ):
        ick = instant_kill_config or {}
        btc = brain_trigger_config or {}

        # instant-kill thresholds
        self.kill_nan_loss: bool = ick.get("nan_loss", True)
        self.kill_grad_norm_above: Optional[float] = ick.get("grad_norm_above", 1000)
        self.kill_loss_above: Optional[float] = ick.get("loss_above", 100)

        # brain-trigger thresholds
        self.brain_val_loss_rising: int = btc.get("val_loss_rising_for", 5)
        self.brain_plateau_epochs: int = btc.get("loss_plateau_for", 10)
        self.brain_every_n: Optional[int] = btc.get("every_n_epochs", 25)

        self._history: List[MetricsSnapshot] = []
        self._best_val_loss: Optional[float] = None
        self._epochs_since_improvement: int = 0
        self._last_brain_epoch: Optional[int] = None

    # ------------------------------------------------------------------ #

    def process(self, snapshot: MetricsSnapshot) -> List[Signal]:
        """Process one MetricsSnapshot; returns any triggered signals."""
        signals: List[Signal] = []
        self._history.append(snapshot)

        # --- instant-kill: NaN ---
        if snapshot.has_nan and self.kill_nan_loss:
            signals.append(
                Signal(
                    type=SignalType.NAN_LOSS,
                    severity=SignalSeverity.CRITICAL,
                    message="NaN detected in training — killing",
                    epoch=snapshot.epoch,
                    is_kill_trigger=True,
                )
            )
            return signals  # no further checks needed

        # --- instant-kill: CUDA OOM ---
        if snapshot.has_cuda_oom:
            signals.append(
                Signal(
                    type=SignalType.CUDA_OOM,
                    severity=SignalSeverity.CRITICAL,
                    message="CUDA out of memory",
                    epoch=snapshot.epoch,
                    is_kill_trigger=True,
                    is_brain_trigger=True,
                )
            )
            return signals

        # --- instant-kill: gradient explosion ---
        if (
            snapshot.grad_norm is not None
            and self.kill_grad_norm_above is not None
            and snapshot.grad_norm > self.kill_grad_norm_above
        ):
            signals.append(
                Signal(
                    type=SignalType.GRAD_EXPLOSION,
                    severity=SignalSeverity.CRITICAL,
                    message=(
                        f"Gradient norm {snapshot.grad_norm:.1f} > "
                        f"{self.kill_grad_norm_above}"
                    ),
                    epoch=snapshot.epoch,
                    is_kill_trigger=True,
                )
            )

        # --- instant-kill: loss too high ---
        if (
            snapshot.train_loss is not None
            and self.kill_loss_above is not None
            and snapshot.train_loss > self.kill_loss_above
        ):
            signals.append(
                Signal(
                    type=SignalType.LOSS_ABOVE_THRESHOLD,
                    severity=SignalSeverity.CRITICAL,
                    message=(
                        f"Train loss {snapshot.train_loss:.4f} > {self.kill_loss_above}"
                    ),
                    epoch=snapshot.epoch,
                    is_kill_trigger=True,
                )
            )

        # Return early if already killing
        if any(s.is_kill_trigger for s in signals):
            return signals

        # --- track val loss improvement ---
        if snapshot.val_loss is not None:
            if (
                self._best_val_loss is None
                or snapshot.val_loss < self._best_val_loss
            ):
                self._best_val_loss = snapshot.val_loss
                self._epochs_since_improvement = 0
            else:
                self._epochs_since_improvement += 1

        # --- brain trigger: overfitting (val loss rising for N epochs) ---
        if len(self._history) >= self.brain_val_loss_rising:
            recent = self._history[-self.brain_val_loss_rising :]
            recent_val = [s.val_loss for s in recent if s.val_loss is not None]
            if len(recent_val) >= self.brain_val_loss_rising:
                if all(
                    recent_val[i] <= recent_val[i + 1]
                    for i in range(len(recent_val) - 1)
                ):
                    signals.append(
                        Signal(
                            type=SignalType.OVERFITTING,
                            severity=SignalSeverity.WARNING,
                            message=(
                                f"Val loss rising for {self.brain_val_loss_rising} "
                                "consecutive epochs — possible overfitting"
                            ),
                            epoch=snapshot.epoch,
                            is_brain_trigger=True,
                        )
                    )

        # --- brain trigger: plateau ---
        if self._epochs_since_improvement >= self.brain_plateau_epochs:
            signals.append(
                Signal(
                    type=SignalType.LOSS_PLATEAU,
                    severity=SignalSeverity.WARNING,
                    message=(
                        f"No val loss improvement for "
                        f"{self._epochs_since_improvement} epochs"
                    ),
                    epoch=snapshot.epoch,
                    is_brain_trigger=True,
                )
            )

        # --- brain trigger: periodic check ---
        if (
            snapshot.epoch is not None
            and self.brain_every_n
            and snapshot.epoch > 0
            and snapshot.epoch % self.brain_every_n == 0
            and snapshot.epoch != self._last_brain_epoch
        ):
            self._last_brain_epoch = snapshot.epoch
            signals.append(
                Signal(
                    type=SignalType.PERIODIC_CHECK,
                    severity=SignalSeverity.INFO,
                    message=f"Periodic brain check at epoch {snapshot.epoch}",
                    epoch=snapshot.epoch,
                    is_brain_trigger=True,
                )
            )

        return signals

    def training_completed_signal(self, epoch: Optional[int] = None) -> Signal:
        return Signal(
            type=SignalType.TRAINING_COMPLETED,
            severity=SignalSeverity.INFO,
            message="Training completed",
            epoch=epoch,
            is_brain_trigger=True,
        )
