"""Experiment log — tracks all autopilot rounds in a persistent JSON file."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLog:
    """Append-only log of autopilot rounds stored in ``<dir>/experiment_log.json``."""

    def __init__(self, log_dir: str = "./torchmill_runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "experiment_log.json"
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------ #

    def add_round(
        self,
        round_num: int,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        brain_analysis: str = "",
        outcome: str = "",
    ) -> None:
        """Add or update a round entry and flush to disk."""
        entry: Dict[str, Any] = {
            "round": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config or {},
            "metrics": metrics or {},
            "brain_analysis": brain_analysis,
            "outcome": outcome,
        }
        # Replace existing entry for this round if present
        for i, r in enumerate(self._data["rounds"]):
            if r.get("round") == round_num:
                self._data["rounds"][i] = entry
                self._save()
                return
        self._data["rounds"].append(entry)
        self._save()

    def get_rounds(self) -> List[Dict[str, Any]]:
        return list(self._data.get("rounds", []))

    def get_round(self, round_num: int) -> Optional[Dict[str, Any]]:
        for r in self._data["rounds"]:
            if r.get("round") == round_num:
                return r
        return None

    def summary(self) -> str:
        rounds = self.get_rounds()
        if not rounds:
            return "No rounds recorded."
        lines = [f"Experiment history — {len(rounds)} round(s):"]
        for r in rounds:
            lines.append(
                f"  Round {r['round']:>2}: {r.get('outcome', '?'):<30} "
                f"@ {r.get('timestamp', '?')}"
            )
            analysis = str(r.get("brain_analysis", ""))
            if analysis:
                lines.append(f"           Brain: {analysis[:100]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #

    def _load(self) -> Dict[str, Any]:
        if self.log_path.exists():
            try:
                with open(self.log_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"rounds": []}

    def _save(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
