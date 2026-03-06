"""Core training loop — train/val/test, AMP, gradient clipping, DDP."""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .callbacks.checkpoint import CheckpointCallback
from .callbacks.early_stopping import EarlyStoppingCallback
from .callbacks.logging import LoggingCallback
from .config import TorchmillConfig
from .utils.device import get_device, seed_everything

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    opt_type = cfg.type.lower()
    params = model.parameters()
    if opt_type == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer: {cfg.type}")


def _build_scheduler(cfg, optimizer, total_steps: int):
    stype = cfg.type.lower()
    if stype == "none":
        return None

    warmup = cfg.warmup_steps

    def warmup_lambda(step):
        if warmup > 0 and step < warmup:
            return float(step) / float(max(1, warmup))
        return 1.0

    if stype == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup)
        )
        if warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_sched, cosine], milestones=[warmup]
            )
        return cosine

    if stype == "linear":
        def linear_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 1.0 - progress)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_lambda)

    if stype == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma
        )

    if stype == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    raise ValueError(f"Unknown scheduler: {cfg.type}")


def _infer_task(dataset: Dataset) -> str:
    """Try to infer whether this is classification or regression."""
    try:
        sample = dataset[0]
        if isinstance(sample, (tuple, list)):
            label = sample[1]
        else:
            return "regression"
        if isinstance(label, torch.Tensor):
            if label.dtype in (torch.long, torch.int, torch.int8, torch.int16, torch.int32):
                return "classification"
            unique = label.unique()
            if unique.numel() <= 2 and set(unique.tolist()).issubset({0, 1, 0.0, 1.0}):
                return "classification"
    except Exception:
        pass
    return "regression"


def _compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
    task: str,
) -> Dict[str, float]:
    metrics = {"loss": loss.item()}
    with torch.no_grad():
        if task == "classification":
            if logits.dim() == 1 or logits.shape[1] == 1:
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            else:
                preds = logits.argmax(dim=1).float()
            targets_flat = targets.view(-1)
            if targets_flat.dtype != preds.dtype:
                targets_flat = targets_flat.float()
            acc = (preds == targets_flat).float().mean().item()
            metrics["accuracy"] = acc
        else:
            mse = torch.mean((logits.squeeze() - targets.squeeze()) ** 2).item()
            mae = torch.mean(torch.abs(logits.squeeze() - targets.squeeze())).item()
            metrics["mse"] = mse
            metrics["mae"] = mae
    return metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Model-agnostic training loop.

    Supports:
    - Mixed precision (AMP)
    - Gradient clipping
    - LR schedulers (cosine, linear, step, exponential)
    - Early stopping
    - Checkpointing
    - TensorBoard + console logging
    - DDP multi-GPU
    """

    def __init__(self, config: TorchmillConfig) -> None:
        self.cfg = config
        self.tcfg = config.training
        self.ccfg = config.callbacks

        seed_everything(self.tcfg.seed)
        self.device = get_device(self.tcfg.device)
        logger.info(f"Using device: {self.device}")

        # DDP setup
        self.ddp = self.tcfg.ddp and torch.cuda.device_count() > 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if self.ddp:
            torch.distributed.init_process_group(backend=self.tcfg.ddp_backend)
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")

        # AMP scaler (CUDA only)
        self.use_amp = self.tcfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Callbacks
        self.checkpoint_cb = CheckpointCallback(
            save_dir=self.ccfg.checkpoint.save_dir,
            save_top_k=self.ccfg.checkpoint.save_top_k,
            monitor=self.ccfg.checkpoint.monitor,
            mode=self.ccfg.checkpoint.mode,
            save_every_n_epochs=self.ccfg.checkpoint.save_every_n_epochs,
        )
        self.early_stopping_cb = EarlyStoppingCallback(
            patience=self.ccfg.early_stopping.patience,
            monitor=self.ccfg.early_stopping.monitor,
            mode=self.ccfg.early_stopping.mode,
            min_delta=self.ccfg.early_stopping.min_delta,
        ) if self.ccfg.early_stopping.enabled else None

        self.logging_cb = LoggingCallback(
            tensorboard=self.ccfg.logging.tensorboard,
            log_dir=self.ccfg.logging.log_dir,
            log_every_n_steps=self.ccfg.logging.log_every_n_steps,
            console=self.ccfg.logging.console,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict[str, float]:
        """Train the model.

        Returns the best metrics dict from the last completed epoch.
        """
        model = model.to(self.device)

        if self.ddp:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank]
            )

        optimizer = _build_optimizer(self.tcfg.optimizer, model)

        # Infer task
        task = self.tcfg.task
        if task == "auto":
            task = _infer_task(train_dataset)
        logger.info(f"Task: {task}")

        loss_fn = self._build_loss(task)

        # Build dataloaders
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = self._make_loader(val_dataset, shuffle=False) if val_dataset else None

        total_steps = len(train_loader) * self.tcfg.epochs
        scheduler = _build_scheduler(self.tcfg.scheduler, optimizer, total_steps)

        best_metrics: Dict[str, float] = {}

        for epoch in range(self.tcfg.epochs):
            self.logging_cb.on_epoch_start(epoch)

            train_metrics = self._train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, task, epoch
            )
            metrics = {f"train_{k}": v for k, v in train_metrics.items()}

            if val_loader:
                val_metrics = self._eval_epoch(model, val_loader, loss_fn, task)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            self.logging_cb.on_epoch_end(epoch, metrics)
            self.checkpoint_cb.on_epoch_end(epoch, metrics, model, optimizer)

            if self.early_stopping_cb:
                stop = self.early_stopping_cb.on_epoch_end(epoch, metrics)
                if stop:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            best_metrics = metrics

        self.logging_cb.close()
        return best_metrics

    def evaluate(
        self,
        model: nn.Module,
        dataset: Dataset,
        task: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on a dataset and return metrics."""
        model = model.to(self.device)
        if task is None:
            task = self.tcfg.task
        if task == "auto":
            task = _infer_task(dataset)
        loss_fn = self._build_loss(task)
        loader = self._make_loader(dataset, shuffle=False)
        metrics = self._eval_epoch(model, loader, loss_fn, task)
        return {f"test_{k}": v for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_loss(self, task: str) -> nn.Module:
        if task == "classification":
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _make_loader(self, dataset: Optional[Dataset], shuffle: bool) -> Optional[DataLoader]:
        if dataset is None:
            return None
        dcfg = self.cfg.data
        sampler = None
        if self.ddp and shuffle:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=dcfg.batch_size,
            shuffle=shuffle,
            num_workers=dcfg.num_workers,
            sampler=sampler,
            pin_memory=(self.device.type == "cuda"),
        )

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        task: str,
        epoch: int,
    ) -> Dict[str, float]:
        model.train()
        total_metrics: Dict[str, float] = {}
        count = 0

        for step, batch in enumerate(loader):
            x, y = self._unpack_batch(batch)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = model(x)
                loss = self._compute_loss(loss_fn, logits, y, task)

            self.scaler.scale(loss).backward()

            if self.tcfg.gradient_clip:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.tcfg.gradient_clip)

            self.scaler.step(optimizer)
            self.scaler.update()

            if scheduler is not None:
                scheduler.step()

            step_metrics = _compute_metrics(logits.detach(), y, loss.detach(), task)
            self.logging_cb.on_step_end(step, {f"train_{k}": v for k, v in step_metrics.items()})

            for k, v in step_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            count += 1

        return {k: v / max(count, 1) for k, v in total_metrics.items()}

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        task: str,
    ) -> Dict[str, float]:
        model.eval()
        total_metrics: Dict[str, float] = {}
        count = 0

        for batch in loader:
            x, y = self._unpack_batch(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = model(x)
                loss = self._compute_loss(loss_fn, logits, y, task)
            step_metrics = _compute_metrics(logits, y, loss, task)
            for k, v in step_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            count += 1

        return {k: v / max(count, 1) for k, v in total_metrics.items()}

    def _unpack_batch(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError(
                "Batches must be (features, targets) tuples. "
                f"Got: {type(batch)}"
            )
        return x.to(self.device), y.to(self.device)

    def _compute_loss(
        self,
        loss_fn: nn.Module,
        logits: torch.Tensor,
        targets: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        if task == "classification":
            if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
                # Binary — BCEWithLogitsLoss
                return nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(), targets.float().squeeze()
                )
            return loss_fn(logits, targets.long())
        return loss_fn(logits.squeeze(), targets.squeeze().float())
