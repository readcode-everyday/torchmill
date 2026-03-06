"""Layer 1: Regex-based extraction from raw log lines → MetricsSnapshot.

99% of raw log lines are discarded here; only lines that yield useful
metrics (loss, grad norm, LR, AUROC, errors) produce a MetricsSnapshot.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricsSnapshot:
    epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    grad_norm: Optional[float] = None
    lr: Optional[float] = None
    auroc: Optional[float] = None
    has_nan: bool = False
    has_cuda_oom: bool = False
    has_error: bool = False
    raw_line: str = ""
    extra: Dict[str, float] = field(default_factory=dict)


class LogParser:
    """Parse log lines into MetricsSnapshot objects."""

    # ---------- epoch ----------
    _EPOCH_RE = [
        re.compile(r"[Ee]poch\s+(\d+)\s*/\s*\d+"),      # Epoch 3/100
        re.compile(r"[Ee]poch\s*[=:]\s*(\d+)"),          # Epoch: 3 / epoch=3
        re.compile(r"[Ee]poch\s+(\d+)"),                  # Epoch 3
    ]

    # ---------- train loss ----------
    _TRAIN_LOSS_RE = [
        # "Train Loss 0.45" / "Train Loss: 0.45" / "train_loss=0.45"
        re.compile(r"[Tt]rain(?:ing)?[_\s][Ll]oss\s*[=:\s]\s*([\d.eE+\-]+)"),
        re.compile(r"train_loss\s*[=:]\s*([\d.eE+\-]+)", re.I),
    ]
    # generic "loss" — used only if no val_loss pattern matched first
    _GENERIC_LOSS_RE = re.compile(r"\bloss\s*[=:]\s*([\d.eE+\-]+)", re.I)

    # ---------- val loss ----------
    _VAL_LOSS_RE = [
        # "Val Loss 0.51" / "Validation Loss: 0.51" / "val_loss=0.51"
        re.compile(r"[Vv]al(?:idation)?[_\s][Ll]oss\s*[=:\s]\s*([\d.eE+\-]+)"),
        re.compile(r"val_loss\s*[=:]\s*([\d.eE+\-]+)", re.I),
    ]

    # ---------- gradient norm ----------
    _GRAD_NORM_RE = [
        re.compile(r"grad(?:ient)?[_\s]norm\s*[=:]\s*([\d.eE+\-]+)", re.I),
        re.compile(r"grad_norm\s*[=:]\s*([\d.eE+\-]+)", re.I),
    ]

    # ---------- learning rate ----------
    _LR_RE = [
        re.compile(r"\blr\s*[=:]\s*([\d.eE+\-]+)", re.I),
        re.compile(r"learning[_\s]rate\s*[=:]\s*([\d.eE+\-]+)", re.I),
    ]

    # ---------- AUROC / AUC ----------
    _AUROC_RE = [
        re.compile(r"\bAUROC\s*[=:]\s*([\d.eE+\-]+)", re.I),
        re.compile(r"\bauc\s*[=:]\s*([\d.eE+\-]+)", re.I),
    ]

    # ---------- NaN / OOM / error ----------
    _NAN_RE = re.compile(r"\bnan\b", re.I)
    _OOM_RE = re.compile(r"out\s+of\s+memory|CUDA error", re.I)
    _ERR_RE = re.compile(r"(?:^|\s)(?:Error|Exception|Traceback)[:\s(]", re.I)

    # ------------------------------------------------------------------ #

    def parse_line(self, line: str) -> Optional[MetricsSnapshot]:
        """Parse one log line. Returns None if nothing useful found."""
        snap = MetricsSnapshot(raw_line=line.rstrip())
        found = False

        # --- error conditions (quick checks) ---
        if self._OOM_RE.search(line):
            snap.has_cuda_oom = True
            found = True

        if self._ERR_RE.search(line):
            snap.has_error = True
            found = True

        # NaN: check text *and* numeric values later
        if self._NAN_RE.search(line):
            snap.has_nan = True
            found = True

        # --- epoch ---
        for pat in self._EPOCH_RE:
            m = pat.search(line)
            if m:
                snap.epoch = int(m.group(1))
                found = True
                break

        # --- val loss (before train loss so generic loss fallback is for train) ---
        for pat in self._VAL_LOSS_RE:
            m = pat.search(line)
            if m:
                val = _safe_float(m.group(1))
                if val is not None and math.isnan(val):
                    snap.has_nan = True
                else:
                    snap.val_loss = val
                found = True
                break

        # --- train loss ---
        for pat in self._TRAIN_LOSS_RE:
            m = pat.search(line)
            if m:
                val = _safe_float(m.group(1))
                if val is not None and math.isnan(val):
                    snap.has_nan = True
                else:
                    snap.train_loss = val
                found = True
                break
        else:
            # generic loss fallback (only if no explicit train loss pattern)
            if snap.train_loss is None and snap.val_loss is None:
                m = self._GENERIC_LOSS_RE.search(line)
                if m:
                    val = _safe_float(m.group(1))
                    if val is not None and math.isnan(val):
                        snap.has_nan = True
                    else:
                        snap.train_loss = val
                    found = True

        # --- grad norm ---
        for pat in self._GRAD_NORM_RE:
            m = pat.search(line)
            if m:
                snap.grad_norm = _safe_float(m.group(1))
                found = True
                break

        # --- LR ---
        for pat in self._LR_RE:
            m = pat.search(line)
            if m:
                snap.lr = _safe_float(m.group(1))
                found = True
                break

        # --- AUROC ---
        for pat in self._AUROC_RE:
            m = pat.search(line)
            if m:
                snap.auroc = _safe_float(m.group(1))
                found = True
                break

        return snap if found else None

    def parse_lines(self, lines: List[str]) -> List[MetricsSnapshot]:
        """Parse multiple log lines, merging into per-epoch snapshots."""
        snapshots: List[MetricsSnapshot] = []
        current = MetricsSnapshot()

        for line in lines:
            snap = self.parse_line(line)
            if snap is None:
                continue

            # new epoch starts → flush current
            if (
                snap.epoch is not None
                and current.epoch is not None
                and snap.epoch != current.epoch
            ):
                if _snapshot_has_data(current):
                    snapshots.append(current)
                current = MetricsSnapshot()

            _merge(current, snap)

        if _snapshot_has_data(current):
            snapshots.append(current)

        return snapshots


# ------------------------------------------------------------------ helpers

def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _snapshot_has_data(snap: MetricsSnapshot) -> bool:
    return any(
        [
            snap.epoch is not None,
            snap.train_loss is not None,
            snap.val_loss is not None,
            snap.has_nan,
            snap.has_cuda_oom,
            snap.grad_norm is not None,
        ]
    )


def _merge(base: MetricsSnapshot, new: MetricsSnapshot) -> None:
    """Merge new fields into base (new overwrites None, accumulates flags)."""
    if new.epoch is not None:
        base.epoch = new.epoch
    if new.train_loss is not None:
        base.train_loss = new.train_loss
    if new.val_loss is not None:
        base.val_loss = new.val_loss
    if new.grad_norm is not None:
        base.grad_norm = new.grad_norm
    if new.lr is not None:
        base.lr = new.lr
    if new.auroc is not None:
        base.auroc = new.auroc
    if new.has_nan:
        base.has_nan = True
    if new.has_cuda_oom:
        base.has_cuda_oom = True
    if new.has_error:
        base.has_error = True
    if new.raw_line:
        base.raw_line = new.raw_line
    base.extra.update(new.extra)
