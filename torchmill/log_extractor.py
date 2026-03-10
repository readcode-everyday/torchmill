"""Parse RFM / torchmill training logs into structured metrics and errors.

This module is stdlib-only (plus dataclasses) so it can run on any machine.
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExperimentMetrics:
    val_loss: float = 999.0          # final val loss (999.0 if crash / missing)
    train_loss: float = 999.0        # final train loss
    auroc: float = 0.0               # 0.0 if not available
    best_val_loss: float = 999.0     # best across all epochs
    best_epoch: int = 0
    epochs_completed: int = 0
    epochs_total: int = 0
    peak_memory_gb: float = 0.0
    training_seconds: float = 0.0
    training_curve: List[Tuple[int, float, float]] = field(default_factory=list)
    # (epoch, train_loss, val_loss)


@dataclass
class ExperimentError:
    error_type: str = ""             # OOM, IndexError, NaN, Timeout, etc.
    error_file: str = ""             # file path where error occurred
    error_line: int = 0              # line number
    error_message: str = ""          # one-line summary
    context_lines: List[str] = field(default_factory=list)  # last 5 log lines before error
    full_traceback: str = ""         # full traceback text


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "Epoch 3: Train Loss 0.7012, Val Loss 0.6931"
_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)[:\s]+.*?[Tt]rain\s*[Ll]oss[:\s]+([\d.]+).*?[Vv]al\s*[Ll]oss[:\s]+([\d.]+)",
    re.IGNORECASE,
)

# Also handle "epoch=1 train_loss=0.7012 val_loss=0.6931" style
_EPOCH_KV_RE = re.compile(
    r"epoch[=:\s]+(\d+).*?train_loss[=:\s]+([\d.]+).*?val_loss[=:\s]+([\d.]+)",
    re.IGNORECASE,
)

# "AUROC: 0.7812" or "auroc=0.7812"
_AUROC_RE = re.compile(r"AUROC[=:\s]+([\d.]+)", re.IGNORECASE)

# "gradient norm: 1.23" or "grad_norm=1.23"
_GRADNORM_RE = re.compile(r"(?:gradient\s+norm|grad_norm)[=:\s]+([\d.eE+\-]+)", re.IGNORECASE)

# "lr=5e-4" or "lr: 5e-4"
_LR_RE = re.compile(r"\blr[=:\s]+([\d.eE+\-]+)", re.IGNORECASE)

# "Epoch 3 timing: total=12.34s"
_TIMING_RE = re.compile(r"[Ee]poch\s+\d+\s+timing:.*?total[=:\s]+([\d.]+)s", re.IGNORECASE)

# "total training time: 123.4s" or "Training completed in 123.4s"
_TOTAL_TIME_RE = re.compile(
    r"(?:total\s+training\s+time|[Tt]raining\s+(?:completed|finished|done)[^:]*)[:\s]+([\d.]+)\s*s",
    re.IGNORECASE,
)

# "Max memory: 8.2 GB" or "peak_memory_gb=8.2" or nvidia-smi style "8200MiB"
_MEMORY_GB_RE = re.compile(
    r"(?:max|peak)\s*memory[^:]*[=:\s]+([\d.]+)\s*[Gg][Bb]", re.IGNORECASE
)
# nvidia-smi "8400MiB / 24576MiB" — capture used (before /) not total
_MEMORY_MIB_USED_RE = re.compile(r"(\d+)\s*MiB\s*/\s*\d+\s*MiB", re.IGNORECASE)
_MEMORY_MIB_RE = re.compile(r"([\d]+)\s*MiB", re.IGNORECASE)

# Epoch count from lines like "epochs: 15" or "total_epochs=15"
_EPOCH_TOTAL_RE = re.compile(r"(?:total_epochs|epochs_total|epochs)[=:\s]+(\d+)", re.IGNORECASE)

# NaN detection
_NAN_RE = re.compile(r"\bnan\b", re.IGNORECASE)
_NAN_LOSS_RE = re.compile(r"(?:loss|grad)[^=\n]*=\s*nan", re.IGNORECASE)

# OOM
_OOM_RE = re.compile(r"out\s+of\s+memory|cuda\s+out\s+of\s+memory|CUDA_ERROR_OUT_OF_MEMORY", re.IGNORECASE)

# Python traceback
_TRACEBACK_START_RE = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_EXCEPTION_LINE_RE = re.compile(r"^(\w+(?:\.\w+)*Error|\w+Exception|KeyboardInterrupt|SystemExit|RuntimeError|ValueError|IndexError|TypeError|AttributeError|MemoryError|AssertionError):\s*(.*)", re.MULTILINE)

# File/line in traceback: '  File "foo.py", line 42, in bar'
_FILE_LINE_RE = re.compile(r'File "([^"]+)",\s+line\s+(\d+)')


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_metrics_and_errors(
    log_text: str,
    returncode: Optional[int] = None,
    timed_out: bool = False,
) -> Tuple[ExperimentMetrics, List[ExperimentError]]:
    """Parse raw log text and return (metrics, errors).

    Args:
        log_text: Combined stdout + stderr from the training process.
        returncode: Process exit code (None if unknown).
        timed_out: Whether the process was killed due to timeout.
    """
    lines = log_text.splitlines()
    errors: List[ExperimentError] = []
    metrics = ExperimentMetrics()

    # --- Training curve ---
    curve: List[Tuple[int, float, float]] = []
    for line in lines:
        m = _EPOCH_RE.search(line) or _EPOCH_KV_RE.search(line)
        if m:
            epoch, tr, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
            curve.append((epoch, tr, vl))

    if curve:
        metrics.training_curve = curve
        metrics.epochs_completed = max(ep for ep, _, _ in curve)
        metrics.train_loss = curve[-1][1]
        metrics.val_loss = curve[-1][2]
        best_ep, best_vl = min(((ep, vl) for ep, _, vl in curve), key=lambda x: x[1])
        metrics.best_val_loss = best_vl
        metrics.best_epoch = best_ep

    # --- AUROC (last occurrence) ---
    for line in reversed(lines):
        m = _AUROC_RE.search(line)
        if m:
            metrics.auroc = float(m.group(1))
            break

    # --- Timing ---
    epoch_times: List[float] = []
    for line in lines:
        m = _TIMING_RE.search(line)
        if m:
            epoch_times.append(float(m.group(1)))
    if epoch_times:
        metrics.training_seconds = sum(epoch_times)

    for line in lines:
        m = _TOTAL_TIME_RE.search(line)
        if m:
            metrics.training_seconds = float(m.group(1))
            break

    # --- Epochs total ---
    for line in lines:
        m = _EPOCH_TOTAL_RE.search(line)
        if m:
            metrics.epochs_total = int(m.group(1))
            break
    if metrics.epochs_total == 0 and metrics.epochs_completed > 0:
        metrics.epochs_total = metrics.epochs_completed

    # --- Peak memory ---
    for line in lines:
        m = _MEMORY_GB_RE.search(line)
        if m:
            metrics.peak_memory_gb = float(m.group(1))
            break
    if metrics.peak_memory_gb == 0.0:
        # Try nvidia-smi "used / total" pattern first (more accurate)
        used_mib = [int(m.group(1)) for m in _MEMORY_MIB_USED_RE.finditer(log_text)]
        if used_mib:
            metrics.peak_memory_gb = max(used_mib) / 1024.0
        else:
            # Fall back to any MiB value
            mib_values = [int(m.group(1)) for m in _MEMORY_MIB_RE.finditer(log_text)]
            if mib_values:
                metrics.peak_memory_gb = max(mib_values) / 1024.0

    # --- Errors ---
    if timed_out:
        ctx = _context_lines(lines, len(lines) - 1)
        errors.append(ExperimentError(
            error_type="Timeout",
            error_message="Training exceeded time budget and was killed",
            context_lines=ctx,
            full_traceback="[Timeout — process killed]",
        ))

    # OOM
    for i, line in enumerate(lines):
        if _OOM_RE.search(line):
            ctx = _context_lines(lines, i)
            errors.append(ExperimentError(
                error_type="OOM",
                error_message="CUDA out of memory",
                context_lines=ctx,
                full_traceback="\n".join(lines[max(0, i - 10): i + 5]),
            ))
            break  # one OOM is enough

    # NaN loss
    for i, line in enumerate(lines):
        if _NAN_LOSS_RE.search(line) or ("nan" in line.lower() and "loss" in line.lower()):
            ctx = _context_lines(lines, i)
            errors.append(ExperimentError(
                error_type="NaN",
                error_message="NaN detected in loss or gradients",
                context_lines=ctx,
                full_traceback="\n".join(lines[max(0, i - 5): i + 3]),
            ))
            break

    # Python tracebacks
    tb_errors = _extract_tracebacks(log_text, lines)
    errors.extend(tb_errors)

    # Non-zero exit with no other error
    if returncode not in (None, 0) and not errors:
        ctx = _context_lines(lines, len(lines) - 1)
        errors.append(ExperimentError(
            error_type="NonZeroExit",
            error_message=f"Process exited with code {returncode}",
            context_lines=ctx,
            full_traceback="\n".join(lines[-20:]),
        ))

    return metrics, errors


def _context_lines(lines: List[str], idx: int, n: int = 5) -> List[str]:
    start = max(0, idx - n)
    return lines[start: idx + 1]


def _extract_tracebacks(log_text: str, lines: List[str]) -> List[ExperimentError]:
    """Find all Python tracebacks in log_text and extract structured info."""
    errors: List[ExperimentError] = []
    seen_types: set = set()

    # Split on traceback markers
    parts = _TRACEBACK_START_RE.split(log_text)

    for part in parts[1:]:  # skip text before first traceback
        full_tb = "Traceback (most recent call last):\n" + part.split("\n\n")[0]

        # Find the exception type + message (last line of traceback)
        exc_match = None
        for line in reversed(full_tb.splitlines()):
            exc_match = _EXCEPTION_LINE_RE.match(line)
            if exc_match:
                break

        if not exc_match:
            continue

        exc_type = exc_match.group(1)
        exc_msg = exc_match.group(2).strip()

        # Deduplicate same error type
        if exc_type in seen_types:
            continue
        seen_types.add(exc_type)

        # Find last file/line reference in the traceback
        file_matches = list(_FILE_LINE_RE.finditer(full_tb))
        err_file, err_line = "", 0
        if file_matches:
            last = file_matches[-1]
            err_file = last.group(1)
            err_line = int(last.group(2))

        # Context = last 5 lines of the full log before this traceback
        tb_start_idx = _find_line_index(lines, "Traceback (most recent call last):")
        ctx = _context_lines(lines, tb_start_idx) if tb_start_idx >= 0 else []

        errors.append(ExperimentError(
            error_type=exc_type,
            error_file=err_file,
            error_line=err_line,
            error_message=exc_msg[:200],
            context_lines=ctx,
            full_traceback=full_tb[:2000],
        ))

    return errors


def _find_line_index(lines: List[str], needle: str) -> int:
    for i, line in enumerate(lines):
        if needle in line:
            return i
    return -1


# ---------------------------------------------------------------------------
# Config summary helper (used by autosearch)
# ---------------------------------------------------------------------------


def config_summary(cfg: dict, keys: Optional[List[str]] = None) -> str:
    """Produce a compact one-line summary of a training config dict.

    Example: "lr=5e-4 hd=128 nl=2 nh=4 bs=64 sched=cosine"
    """
    abbrev = {
        "lr": "lr",
        "hidden_dim": "hd",
        "num_layers": "nl",
        "num_heads": "nh",
        "dropout": "do",
        "batch_size": "bs",
        "scheduler": "sched",
        "warmup_steps": "ws",
        "weight_decay": "wd",
        "epochs": "ep",
    }
    if keys is None:
        keys = list(abbrev.keys())

    parts = []
    for k in keys:
        if k in cfg:
            short = abbrev.get(k, k)
            v = cfg[k]
            if isinstance(v, float) and v < 0.01:
                v = f"{v:.0e}"
            parts.append(f"{short}={v}")
    return " ".join(parts)
