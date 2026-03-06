"""Tests for ExperimentLog — persistent round tracking."""

import json
import os

import pytest

from torchmill.history.experiment_log import ExperimentLog


@pytest.fixture
def log(tmp_path):
    return ExperimentLog(log_dir=str(tmp_path / "torchmill_runs"))


# =========================================================================
# Basic operations
# =========================================================================

def test_creates_log_dir(tmp_path):
    log_dir = str(tmp_path / "runs" / "nested")
    ExperimentLog(log_dir=log_dir)
    assert os.path.isdir(log_dir)


def test_empty_log_has_no_rounds(log):
    assert log.get_rounds() == []


def test_add_round_and_retrieve(log):
    log.add_round(
        round_num=1,
        config={"lr": 1e-3},
        metrics={"best_val_loss": 0.42},
        brain_analysis="Training looks healthy",
        outcome="completed",
    )
    rounds = log.get_rounds()
    assert len(rounds) == 1
    r = rounds[0]
    assert r["round"] == 1
    assert r["config"]["lr"] == pytest.approx(1e-3)
    assert r["metrics"]["best_val_loss"] == pytest.approx(0.42)
    assert r["brain_analysis"] == "Training looks healthy"
    assert r["outcome"] == "completed"


def test_add_multiple_rounds(log):
    for i in range(1, 4):
        log.add_round(round_num=i, outcome=f"round_{i}")
    assert len(log.get_rounds()) == 3


def test_get_round_by_number(log):
    log.add_round(round_num=2, outcome="killed_overfitting")
    r = log.get_round(2)
    assert r is not None
    assert r["outcome"] == "killed_overfitting"


def test_get_missing_round_returns_none(log):
    assert log.get_round(99) is None


# =========================================================================
# Update (idempotent round numbers)
# =========================================================================

def test_update_existing_round(log):
    log.add_round(round_num=1, outcome="continue")
    log.add_round(round_num=1, outcome="completed")
    rounds = log.get_rounds()
    assert len(rounds) == 1
    assert rounds[0]["outcome"] == "completed"


# =========================================================================
# Persistence — survives reload
# =========================================================================

def test_persists_to_disk(tmp_path):
    log_dir = str(tmp_path / "runs")
    log = ExperimentLog(log_dir=log_dir)
    log.add_round(round_num=1, outcome="kill_and_retry", brain_analysis="Overfitting")

    # Reload from disk
    log2 = ExperimentLog(log_dir=log_dir)
    rounds = log2.get_rounds()
    assert len(rounds) == 1
    assert rounds[0]["outcome"] == "kill_and_retry"
    assert rounds[0]["brain_analysis"] == "Overfitting"


def test_log_file_is_valid_json(tmp_path):
    log_dir = str(tmp_path / "runs")
    log = ExperimentLog(log_dir=log_dir)
    log.add_round(round_num=1, config={"lr": 3e-4})
    with open(log.log_path) as f:
        data = json.load(f)
    assert "rounds" in data
    assert len(data["rounds"]) == 1


# =========================================================================
# Summary
# =========================================================================

def test_summary_empty(log):
    s = log.summary()
    assert "No rounds" in s


def test_summary_contains_round_info(log):
    log.add_round(round_num=1, outcome="completed")
    log.add_round(round_num=2, outcome="kill_and_retry")
    s = log.summary()
    assert "Round  1" in s or "Round 1" in s
    assert "completed" in s
    assert "kill_and_retry" in s


def test_summary_includes_brain_analysis(log):
    log.add_round(round_num=1, brain_analysis="Val loss diverging rapidly", outcome="kill")
    s = log.summary()
    assert "Val loss diverging" in s


# =========================================================================
# Corrupted log file gracefully handled
# =========================================================================

def test_corrupted_log_file_handled(tmp_path):
    log_dir = tmp_path / "runs"
    log_dir.mkdir()
    log_file = log_dir / "experiment_log.json"
    log_file.write_text("{ invalid json }")

    # Should not raise
    log = ExperimentLog(log_dir=str(log_dir))
    assert log.get_rounds() == []


# =========================================================================
# Timestamps
# =========================================================================

def test_round_has_timestamp(log):
    log.add_round(round_num=1, outcome="test")
    r = log.get_round(1)
    assert "timestamp" in r
    assert r["timestamp"] is not None
    assert len(r["timestamp"]) > 0
