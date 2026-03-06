"""Pytest configuration — skip torch-dependent tests when torch is absent."""

import pathlib
import pytest


def pytest_ignore_collect(collection_path: pathlib.Path, config):
    """Skip test files that require torch when torch is not installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        if collection_path.name in {"test_registry.py", "test_trainer.py"}:
            return True
    return None
