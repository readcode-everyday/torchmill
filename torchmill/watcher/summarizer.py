"""Layer 3: Compress metrics history into a compact LLM-ready report.

Called ONLY when a brain_trigger fires — produces ~500-800 token summaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .log_parser import MetricsSnapshot


class Summarizer:
    def __init__(
        self,
        max_history_epochs: int = 20,
        include_experiment_history: bool = True,
        max_tokens: int = 800,
    ):
        self.max_history_epochs = max_history_epochs
        self.include_experiment_history = include_experiment_history
        self.max_tokens = max_tokens

    def summarize(
        self,
        snapshots: List[MetricsSnapshot],
        round_num: int = 1,
        current_config: Optional[Dict[str, Any]] = None,
        experiment_history: Optional[List[Dict[str, Any]]] = None,
        trigger_reason: str = "",
    ) -> str:
        """Generate compact text summary for the LLM brain."""
        lines: List[str] = []

        lines.append(f"=== Training Summary — Round {round_num} ===")
        if trigger_reason:
            lines.append(f"Trigger: {trigger_reason}")

        if snapshots:
            recent = snapshots[-self.max_history_epochs :]
            epochs_with_num = [s.epoch for s in recent if s.epoch is not None]
            current_epoch = max(epochs_with_num) if epochs_with_num else len(recent)
            total_epochs = len(snapshots)
            lines.append(f"Epochs seen: {total_epochs}  |  Current: {current_epoch}")

            # Best metrics across all history
            all_val = [s.val_loss for s in snapshots if s.val_loss is not None]
            all_train = [s.train_loss for s in snapshots if s.train_loss is not None]
            all_auroc = [s.auroc for s in snapshots if s.auroc is not None]

            if all_val:
                lines.append(f"Best val_loss   : {min(all_val):.6f}")
            if all_train:
                lines.append(f"Best train_loss : {min(all_train):.6f}")
            if all_auroc:
                lines.append(f"Best AUROC      : {max(all_auroc):.4f}")

            # Epoch table
            lines.append(f"\nLast {len(recent)} epochs:")
            lines.append("  epoch | train_loss  | val_loss   | grad_norm  | lr")
            for s in recent:
                row = (
                    f"  {_fmti(s.epoch):>5} | "
                    f"{_fmtf(s.train_loss):>11} | "
                    f"{_fmtf(s.val_loss):>10} | "
                    f"{_fmtf(s.grad_norm):>10} | "
                    f"{_fmtf(s.lr, 4)}"
                )
                lines.append(row)

            # Trend
            if len(all_val) >= 3:
                window = [s.val_loss for s in recent[-5:] if s.val_loss is not None]
                if len(window) >= 2:
                    direction = "rising" if window[-1] > window[0] else "falling"
                    delta = window[-1] - window[0]
                    lines.append(
                        f"\nVal loss trend: {direction} "
                        f"({delta:+.6f} over last {len(window)} epochs)"
                    )

        # Current hyperparams
        if current_config:
            lines.append("\nCurrent hyperparameters:")
            for k, v in current_config.items():
                lines.append(f"  {k}: {v}")

        # Experiment history
        if self.include_experiment_history and experiment_history:
            lines.append(
                f"\nExperiment history ({len(experiment_history)} previous rounds):"
            )
            for entry in experiment_history[-3:]:
                rn = entry.get("round", "?")
                outcome = entry.get("outcome", "?")
                analysis = str(entry.get("brain_analysis", ""))[:120]
                lines.append(f"  Round {rn}: {outcome}")
                if analysis:
                    lines.append(f"    Brain: {analysis}")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #


def _fmti(val: Optional[int]) -> str:
    return str(val) if val is not None else "?"


def _fmtf(val: Optional[float], precision: int = 6) -> str:
    if val is None:
        return "N/A"
    if val != 0 and (abs(val) < 1e-3 or abs(val) >= 1e5):
        return f"{val:.3e}"
    return f"{val:.{precision}f}"
