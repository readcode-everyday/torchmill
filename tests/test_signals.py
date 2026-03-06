"""Tests for Layer 2: SignalDetector — local rule-based signal detection."""

import pytest
from torchmill.watcher.log_parser import MetricsSnapshot
from torchmill.watcher.signals import (
    Signal,
    SignalDetector,
    SignalSeverity,
    SignalType,
)


def make_snap(
    epoch=None,
    train_loss=None,
    val_loss=None,
    grad_norm=None,
    has_nan=False,
    has_cuda_oom=False,
) -> MetricsSnapshot:
    return MetricsSnapshot(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        grad_norm=grad_norm,
        has_nan=has_nan,
        has_cuda_oom=has_cuda_oom,
    )


@pytest.fixture
def detector():
    return SignalDetector(
        instant_kill_config={
            "nan_loss": True,
            "grad_norm_above": 1000,
            "loss_above": 100,
        },
        brain_trigger_config={
            "val_loss_rising_for": 3,
            "loss_plateau_for": 5,
            "every_n_epochs": 10,
        },
    )


# =========================================================================
# Instant-kill: NaN
# =========================================================================

def test_nan_triggers_kill(detector):
    signals = detector.process(make_snap(epoch=1, has_nan=True))
    kill_sigs = [s for s in signals if s.is_kill_trigger]
    assert len(kill_sigs) == 1
    assert kill_sigs[0].type == SignalType.NAN_LOSS
    assert kill_sigs[0].severity == SignalSeverity.CRITICAL


def test_nan_returns_immediately(detector):
    """NaN should short-circuit: no other signals computed."""
    signals = detector.process(
        make_snap(epoch=1, has_nan=True, grad_norm=9999, train_loss=999)
    )
    assert all(s.type == SignalType.NAN_LOSS for s in signals)


# =========================================================================
# Instant-kill: CUDA OOM
# =========================================================================

def test_cuda_oom_triggers_kill(detector):
    signals = detector.process(make_snap(epoch=2, has_cuda_oom=True))
    kill_sigs = [s for s in signals if s.is_kill_trigger]
    assert len(kill_sigs) >= 1
    assert any(s.type == SignalType.CUDA_OOM for s in signals)


# =========================================================================
# Instant-kill: gradient explosion
# =========================================================================

def test_grad_explosion_triggers_kill(detector):
    signals = detector.process(make_snap(epoch=5, grad_norm=1500.0))
    kill_sigs = [s for s in signals if s.is_kill_trigger]
    assert any(s.type == SignalType.GRAD_EXPLOSION for s in kill_sigs)


def test_grad_norm_below_threshold_no_kill(detector):
    signals = detector.process(make_snap(epoch=5, grad_norm=5.0))
    kill_sigs = [s for s in signals if s.is_kill_trigger]
    assert len(kill_sigs) == 0


# =========================================================================
# Instant-kill: loss above threshold
# =========================================================================

def test_loss_above_threshold_triggers_kill(detector):
    signals = detector.process(make_snap(epoch=3, train_loss=200.0))
    kill_sigs = [s for s in signals if s.is_kill_trigger]
    assert any(s.type == SignalType.LOSS_ABOVE_THRESHOLD for s in kill_sigs)


def test_loss_below_threshold_no_kill(detector):
    signals = detector.process(make_snap(epoch=3, train_loss=0.5))
    kill_sigs = [s for s in signals if s.is_kill_trigger]
    assert len(kill_sigs) == 0


# =========================================================================
# Brain trigger: overfitting (val loss rising)
# =========================================================================

def test_overfitting_detected(detector):
    """Feed 3 epochs of strictly increasing val loss → overfitting signal."""
    for i, vl in enumerate([0.5, 0.6, 0.7], start=1):
        signals = detector.process(make_snap(epoch=i, val_loss=vl))

    brain_sigs = [s for s in signals if s.is_brain_trigger]
    types = [s.type for s in brain_sigs]
    assert SignalType.OVERFITTING in types


def test_improving_val_loss_no_overfitting(detector):
    """Falling val loss should never trigger overfitting."""
    all_signals = []
    for i, vl in enumerate([0.9, 0.7, 0.5, 0.4, 0.35], start=1):
        sigs = detector.process(make_snap(epoch=i, val_loss=vl, train_loss=vl * 0.9))
        all_signals.extend(sigs)

    overfitting = [s for s in all_signals if s.type == SignalType.OVERFITTING]
    assert len(overfitting) == 0


# =========================================================================
# Brain trigger: plateau
# =========================================================================

def test_plateau_detected():
    det = SignalDetector(
        brain_trigger_config={"val_loss_rising_for": 10, "loss_plateau_for": 3}
    )
    for i, vl in enumerate([0.5, 0.5, 0.5, 0.5], start=1):
        signals = det.process(make_snap(epoch=i, val_loss=vl))

    plateau_sigs = [s for s in signals if s.type == SignalType.LOSS_PLATEAU]
    assert len(plateau_sigs) >= 1
    assert all(s.is_brain_trigger for s in plateau_sigs)


# =========================================================================
# Brain trigger: periodic check
# =========================================================================

def test_periodic_check_fires_at_multiple(detector):
    every_n = 10
    triggered_epochs = []
    for epoch in range(1, 31):
        sigs = detector.process(make_snap(epoch=epoch, val_loss=0.5 - epoch * 0.001))
        if any(s.type == SignalType.PERIODIC_CHECK for s in sigs):
            triggered_epochs.append(epoch)

    assert 10 in triggered_epochs
    assert 20 in triggered_epochs
    assert 30 in triggered_epochs


def test_periodic_check_not_duplicated(detector):
    """Same epoch should not fire periodic check twice."""
    sigs1 = detector.process(make_snap(epoch=10, val_loss=0.4))
    sigs2 = detector.process(make_snap(epoch=10, val_loss=0.4))
    periodic = [s for s in sigs1 + sigs2 if s.type == SignalType.PERIODIC_CHECK]
    assert len(periodic) <= 1


# =========================================================================
# training_completed_signal
# =========================================================================

def test_training_completed_signal(detector):
    sig = detector.training_completed_signal(epoch=100)
    assert sig.type == SignalType.TRAINING_COMPLETED
    assert sig.is_brain_trigger is True
    assert sig.is_kill_trigger is False
    assert sig.epoch == 100


# =========================================================================
# No false positives on healthy training
# =========================================================================

def test_healthy_training_no_kill():
    det = SignalDetector(
        instant_kill_config={"nan_loss": True, "grad_norm_above": 1000, "loss_above": 100},
        brain_trigger_config={"val_loss_rising_for": 5, "loss_plateau_for": 20, "every_n_epochs": 50},
    )
    # Simulate 20 epochs of improving training
    kill_signals = []
    for i in range(1, 21):
        loss = 1.0 - i * 0.04
        sigs = det.process(make_snap(epoch=i, train_loss=loss, val_loss=loss + 0.05))
        kill_signals.extend(s for s in sigs if s.is_kill_trigger)

    assert len(kill_signals) == 0
