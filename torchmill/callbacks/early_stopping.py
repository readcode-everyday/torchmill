"""Early stopping callback — stop training when a metric stops improving."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """Stop training when the monitored metric has not improved for ``patience``
    consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check the metric and return True if training should stop."""
        score = metrics.get(self.monitor)
        if score is None:
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                f"({self.monitor}={score:.4f}, best={self.best_score:.4f})"
            )
            if self.counter >= self.patience:
                logger.info(
                    f"EarlyStopping triggered after epoch {epoch + 1}. "
                    f"Best {self.monitor}={self.best_score:.4f}"
                )
                self.should_stop = True

        return self.should_stop
