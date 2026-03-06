"""Logging callback — TensorBoard and console logging."""

import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LoggingCallback:
    """Log training metrics to the console and optionally TensorBoard."""

    def __init__(
        self,
        tensorboard: bool = True,
        log_dir: str = "./runs",
        log_every_n_steps: int = 10,
        console: bool = True,
    ) -> None:
        self.log_every_n_steps = log_every_n_steps
        self.console = console
        self._writer = None
        self._step = 0
        self._epoch_start: Optional[float] = None

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging enabled → {log_dir}")
            except ImportError:
                logger.warning(
                    "TensorBoard not installed. "
                    "Install it with: pip install tensorboard"
                )

    def on_epoch_start(self, epoch: int) -> None:
        self._epoch_start = time.time()

    def on_step_end(self, step: int, metrics: Dict[str, float]) -> None:
        self._step += 1
        if self._step % self.log_every_n_steps != 0:
            return

        if self._writer:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, self._step)

        if self.console:
            parts = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(f"  step {self._step}: {parts}")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        elapsed = (
            f"{time.time() - self._epoch_start:.1f}s"
            if self._epoch_start is not None
            else "?"
        )

        if self._writer:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, epoch)

        if self.console:
            parts = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {parts}  [{elapsed}]")

    def close(self) -> None:
        if self._writer:
            self._writer.close()
