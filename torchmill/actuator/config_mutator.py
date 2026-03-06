"""Config mutator — reads a JSON or YAML training config, applies changes, writes back.

Supports flat and dotted-key notation for nested values:
  changes = {"lr": 3e-4, "optimizer.weight_decay": 0.05}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ConfigMutator:
    """Read-modify-write for JSON or YAML config files."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._fmt = self._detect_format()

    # ------------------------------------------------------------------ #

    def read(self) -> Dict[str, Any]:
        with open(self.config_path) as f:
            if self._fmt == "yaml":
                import yaml
                return yaml.safe_load(f) or {}
            return json.load(f)

    def write(self, data: Dict[str, Any]) -> None:
        with open(self.config_path, "w") as f:
            if self._fmt == "yaml":
                import yaml
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2)

    def apply_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Read config, apply changes, write back. Returns updated config."""
        if not changes:
            return self.read()
        data = self.read()
        _apply_flat(data, changes)
        self.write(data)
        return data

    # ------------------------------------------------------------------ #

    def _detect_format(self) -> str:
        suffix = self.config_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return "yaml"
        if suffix == ".json":
            return "json"
        raise ValueError(
            f"Unsupported config format '{suffix}'. Use .json or .yaml/.yml"
        )


def _apply_flat(data: Dict[str, Any], changes: Dict[str, Any]) -> None:
    """Apply flat key=value changes, supporting dotted nested keys."""
    for key, value in changes.items():
        if "." in key:
            head, tail = key.split(".", 1)
            if head not in data or not isinstance(data[head], dict):
                data[head] = {}
            _apply_flat(data[head], {tail: value})
        else:
            data[key] = value
