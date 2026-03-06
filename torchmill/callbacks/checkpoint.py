"""Checkpoint callback — save best and/or periodic model checkpoints."""

import heapq
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """Save model checkpoints during training.

    Keeps the ``save_top_k`` best checkpoints (by monitored metric) and
    optionally saves every ``save_every_n_epochs`` epochs.
    """

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_top_k: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
        save_every_n_epochs: Optional[int] = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # min-heap: (score, filepath)  — negate score for "max" mode
        self._heap: list = []
        self.best_score: Optional[float] = None
        self.best_path: Optional[Path] = None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        return score < self.best_score if self.mode == "min" else score > self.best_score

    def _heap_score(self, score: float) -> float:
        """Lower heap value = higher priority to remove."""
        return score if self.mode == "max" else -score

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        metric_val = metrics.get(self.monitor)

        # Periodic saving
        if (
            self.save_every_n_epochs is not None
            and (epoch + 1) % self.save_every_n_epochs == 0
        ):
            periodic_path = self.save_dir / f"epoch_{epoch + 1:04d}.pt"
            self._save(periodic_path, epoch, metrics, model, optimizer)

        if metric_val is None:
            return

        path = self.save_dir / f"epoch_{epoch + 1:04d}_{self.monitor}={metric_val:.4f}.pt"
        self._save(path, epoch, metrics, model, optimizer)

        heap_score = self._heap_score(metric_val)
        heapq.heappush(self._heap, (heap_score, str(path)))

        # Evict worst checkpoint when over limit
        if self.save_top_k > 0 and len(self._heap) > self.save_top_k:
            _, worst_path = heapq.heappop(self._heap)
            try:
                os.remove(worst_path)
            except FileNotFoundError:
                pass

        # Track best
        if self._is_better(metric_val):
            self.best_score = metric_val
            self.best_path = path
            # Write a "best.pt" symlink/copy
            best_link = self.save_dir / "best.pt"
            self._save(best_link, epoch, metrics, model, optimizer)
            logger.info(
                f"New best {self.monitor}={metric_val:.4f} at epoch {epoch + 1}"
            )

    def _save(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        state = {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(state, path)
        logger.debug(f"Saved checkpoint: {path}")

    @staticmethod
    def load_checkpoint(path: str, model: nn.Module, optimizer=None) -> Dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        return state
