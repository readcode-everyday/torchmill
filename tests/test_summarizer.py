"""Tests for Layer 3: Summarizer — compact LLM-ready reports."""

import pytest
from torchmill.watcher.log_parser import MetricsSnapshot
from torchmill.watcher.summarizer import Summarizer


def make_snap(epoch, train_loss=None, val_loss=None, grad_norm=None, lr=None):
    return MetricsSnapshot(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        grad_norm=grad_norm,
        lr=lr,
    )


@pytest.fixture
def summarizer():
    return Summarizer(max_history_epochs=10, include_experiment_history=True)


def make_history(n=5, start_loss=0.9, delta=-0.05):
    return [
        make_snap(i + 1, train_loss=start_loss + i * delta, val_loss=start_loss + i * delta + 0.05)
        for i in range(n)
    ]


# =========================================================================
# Basic structure
# =========================================================================

def test_summary_contains_round_number(summarizer):
    snaps = make_history(5)
    summary = summarizer.summarize(snaps, round_num=3)
    assert "Round 3" in summary


def test_summary_contains_trigger(summarizer):
    snaps = make_history(5)
    summary = summarizer.summarize(snaps, trigger_reason="Val loss rising for 5 epochs")
    assert "Val loss rising" in summary


def test_summary_contains_epoch_count(summarizer):
    snaps = make_history(8)
    summary = summarizer.summarize(snaps)
    assert "8" in summary


def test_summary_contains_best_metrics(summarizer):
    snaps = make_history(5)
    summary = summarizer.summarize(snaps)
    assert "Best val_loss" in summary
    assert "Best train_loss" in summary


def test_summary_contains_epoch_table(summarizer):
    snaps = make_history(5)
    summary = summarizer.summarize(snaps)
    assert "epoch" in summary.lower()
    assert "train_loss" in summary.lower()


# =========================================================================
# Hyperparams in summary
# =========================================================================

def test_summary_includes_current_config(summarizer):
    snaps = make_history(3)
    cfg = {"lr": 3e-4, "dropout": 0.1, "batch_size": 32}
    summary = summarizer.summarize(snaps, current_config=cfg)
    assert "lr" in summary
    assert "dropout" in summary


# =========================================================================
# Experiment history
# =========================================================================

def test_summary_includes_experiment_history(summarizer):
    snaps = make_history(5)
    history = [
        {"round": 1, "outcome": "killed_overfitting", "brain_analysis": "Val loss rose for 8 epochs"},
        {"round": 2, "outcome": "completed", "brain_analysis": "Training converged"},
    ]
    summary = summarizer.summarize(snaps, experiment_history=history)
    assert "killed_overfitting" in summary or "Round 1" in summary


def test_summary_excludes_history_when_disabled():
    summ = Summarizer(include_experiment_history=False)
    snaps = make_history(5)
    history = [{"round": 1, "outcome": "killed"}]
    summary = summ.summarize(snaps, experiment_history=history)
    assert "killed" not in summary


# =========================================================================
# max_history_epochs truncation
# =========================================================================

def test_summary_truncates_to_max_epochs():
    summ = Summarizer(max_history_epochs=3)
    snaps = make_history(20)
    summary = summ.summarize(snaps)
    # The table should only show 3 rows
    lines = summary.split("\n")
    table_lines = [l for l in lines if "|" in l and l.strip()[0].isspace()]
    # At most 3 data rows (plus header)
    assert len(table_lines) <= 4  # 1 header + 3 data


# =========================================================================
# Edge cases
# =========================================================================

def test_empty_snapshots(summarizer):
    summary = summarizer.summarize([])
    assert "Round" in summary  # still produces output


def test_no_val_loss_snapshots(summarizer):
    snaps = [make_snap(i, train_loss=0.5 - i * 0.01) for i in range(1, 6)]
    summary = summarizer.summarize(snaps)
    assert "train_loss" in summary.lower()
    assert "Best val_loss" not in summary


def test_val_loss_trend_rising(summarizer):
    snaps = [make_snap(i, val_loss=0.5 + i * 0.05) for i in range(1, 8)]
    summary = summarizer.summarize(snaps)
    assert "rising" in summary.lower()


def test_val_loss_trend_falling(summarizer):
    snaps = [make_snap(i, val_loss=0.9 - i * 0.05) for i in range(1, 8)]
    summary = summarizer.summarize(snaps)
    assert "falling" in summary.lower()
