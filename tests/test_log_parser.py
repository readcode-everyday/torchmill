"""Tests for Layer 1: LogParser — regex extraction from raw log lines."""

import pytest
from torchmill.watcher.log_parser import LogParser, MetricsSnapshot


@pytest.fixture
def parser():
    return LogParser()


# =========================================================================
# Epoch extraction
# =========================================================================

@pytest.mark.parametrize("line,expected_epoch", [
    ("Epoch 3/100 train_loss=0.45 val_loss=0.51", 3),
    ("Epoch 3: Train Loss 0.4521, Val Loss 0.5123", 3),
    ("Epoch: 42", 42),
    ("epoch=7 loss=0.1", 7),
    ("Epoch 100", 100),
    ("[Epoch 12] training...", 12),
])
def test_epoch_extraction(parser, line, expected_epoch):
    snap = parser.parse_line(line)
    assert snap is not None
    assert snap.epoch == expected_epoch


# =========================================================================
# Loss extraction
# =========================================================================

@pytest.mark.parametrize("line,expected_train,expected_val", [
    ("Epoch 3: Train Loss 0.4521, Val Loss 0.5123", 0.4521, 0.5123),
    ("Epoch 3/100 train_loss=0.45 val_loss=0.51", 0.45, 0.51),
    ("train_loss: 0.3  val_loss: 0.4", 0.3, 0.4),
    ("Training Loss: 1.23, Validation Loss: 2.34", 1.23, 2.34),
    ("val_loss=0.99", None, 0.99),
])
def test_loss_extraction(parser, line, expected_train, expected_val):
    snap = parser.parse_line(line)
    assert snap is not None
    if expected_train is not None:
        assert snap.train_loss == pytest.approx(expected_train, rel=1e-3)
    if expected_val is not None:
        assert snap.val_loss == pytest.approx(expected_val, rel=1e-3)


def test_generic_loss_fallback(parser):
    snap = parser.parse_line("loss: 0.4521")
    assert snap is not None
    assert snap.train_loss == pytest.approx(0.4521, rel=1e-3)
    assert snap.val_loss is None


def test_scientific_notation_loss(parser):
    snap = parser.parse_line("train_loss=1.23e-4 val_loss=5.67e-3")
    assert snap is not None
    assert snap.train_loss == pytest.approx(1.23e-4, rel=1e-3)
    assert snap.val_loss == pytest.approx(5.67e-3, rel=1e-3)


# =========================================================================
# Gradient norm
# =========================================================================

@pytest.mark.parametrize("line,expected_grad", [
    ("gradient norm: 12.4", 12.4),
    ("grad_norm=12.4", 12.4),
    ("Grad Norm: 0.982", 0.982),
    ("gradient_norm=1500.0", 1500.0),
])
def test_grad_norm_extraction(parser, line, expected_grad):
    snap = parser.parse_line(line)
    assert snap is not None
    assert snap.grad_norm == pytest.approx(expected_grad, rel=1e-3)


# =========================================================================
# Learning rate
# =========================================================================

@pytest.mark.parametrize("line,expected_lr", [
    ("lr=3e-4", 3e-4),
    ("learning_rate: 0.0003", 3e-4),
    ("lr: 0.001 loss: 0.5", 0.001),
    ("LR=1e-5", 1e-5),
])
def test_lr_extraction(parser, line, expected_lr):
    snap = parser.parse_line(line)
    assert snap is not None
    assert snap.lr == pytest.approx(expected_lr, rel=1e-3)


# =========================================================================
# AUROC / AUC
# =========================================================================

@pytest.mark.parametrize("line,expected_auroc", [
    ("AUROC: 0.72", 0.72),
    ("auc=0.72", 0.72),
    ("Epoch 5: AUROC=0.85, loss=0.3", 0.85),
])
def test_auroc_extraction(parser, line, expected_auroc):
    snap = parser.parse_line(line)
    assert snap is not None
    assert snap.auroc == pytest.approx(expected_auroc, rel=1e-3)


# =========================================================================
# NaN / error detection
# =========================================================================

def test_nan_text_detection(parser):
    snap = parser.parse_line("NaN detected in loss")
    assert snap is not None
    assert snap.has_nan is True


def test_nan_loss_text(parser):
    snap = parser.parse_line("nan loss encountered")
    assert snap is not None
    assert snap.has_nan is True


def test_cuda_oom(parser):
    snap = parser.parse_line("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert snap is not None
    assert snap.has_cuda_oom is True


def test_traceback_detection(parser):
    snap = parser.parse_line("Traceback (most recent call last):")
    assert snap is not None
    assert snap.has_error is True


def test_error_detection(parser):
    snap = parser.parse_line("RuntimeError: CUDA error: device-side assert triggered")
    assert snap is not None
    assert snap.has_error is True


# =========================================================================
# Lines that should produce nothing
# =========================================================================

@pytest.mark.parametrize("line", [
    "Loading dataset...",
    "Initializing model weights",
    "Using device: cuda",
    "   ",
    "",
    "Starting training loop",
])
def test_empty_lines_return_none(parser, line):
    snap = parser.parse_line(line)
    assert snap is None


# =========================================================================
# parse_lines — multi-line merging
# =========================================================================

def test_parse_lines_single_epoch():
    parser = LogParser()
    lines = [
        "Epoch 5/100",
        "train_loss=0.45",
        "val_loss=0.50",
        "grad_norm=0.9",
    ]
    snaps = parser.parse_lines(lines)
    assert len(snaps) >= 1
    snap = snaps[-1]
    assert snap.epoch == 5
    assert snap.train_loss == pytest.approx(0.45, rel=1e-3)
    assert snap.val_loss == pytest.approx(0.50, rel=1e-3)
    assert snap.grad_norm == pytest.approx(0.9, rel=1e-3)


def test_parse_lines_multiple_epochs():
    parser = LogParser()
    lines = [
        "Epoch 1: train_loss=0.8 val_loss=0.9",
        "Epoch 2: train_loss=0.7 val_loss=0.8",
        "Epoch 3: train_loss=0.6 val_loss=0.7",
    ]
    snaps = parser.parse_lines(lines)
    assert len(snaps) == 3
    assert snaps[0].epoch == 1
    assert snaps[1].epoch == 2
    assert snaps[2].epoch == 3


def test_parse_lines_real_format():
    """Test a realistic multi-line epoch log block."""
    parser = LogParser()
    lines = [
        "Epoch 10/100 [train] loss=0.3241, grad_norm=1.23, lr=3e-4",
        "Epoch 10/100 [val]   val_loss=0.3819, AUROC=0.8712",
    ]
    snaps = parser.parse_lines(lines)
    assert len(snaps) >= 1
    # Both lines belong to epoch 10
    snap = snaps[0]
    assert snap.epoch == 10
    assert snap.train_loss == pytest.approx(0.3241, rel=1e-3)
    assert snap.val_loss == pytest.approx(0.3819, rel=1e-3)
    assert snap.auroc == pytest.approx(0.8712, rel=1e-3)
