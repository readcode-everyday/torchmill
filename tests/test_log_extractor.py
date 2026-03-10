"""Tests for torchmill.log_extractor."""

import pytest
from torchmill.log_extractor import (
    ExperimentMetrics,
    ExperimentError,
    extract_metrics_and_errors,
    config_summary,
)


# ---------------------------------------------------------------------------
# Sample log fixtures
# ---------------------------------------------------------------------------

NORMAL_TRAINING_LOG = """
Starting training with lr=0.0005 hidden_dim=128 num_layers=3
Loading dataset: trial / study-outcome
Epoch 1: Train Loss 0.7812, Val Loss 0.7501
Epoch 1 timing: total=12.34s
Epoch 2: Train Loss 0.7201, Val Loss 0.7002
Epoch 2 timing: total=11.89s
Epoch 3: Train Loss 0.6910, Val Loss 0.6712
Epoch 3 timing: total=12.01s
AUROC: 0.6823
Total training time: 36.24s
Peak memory: 8.2 GB
Training complete.
"""

PARTIAL_TRAINING_LOG = """
Epoch 1: Train Loss 0.8100, Val Loss 0.7900
Epoch 2: Train Loss 0.7600, Val Loss 0.7400
Epoch 3: Train Loss 0.7200, Val Loss 0.7100
"""

NAN_LOSS_LOG = """
Epoch 1: Train Loss 0.7812, Val Loss 0.7501
Epoch 2: Train Loss nan, Val Loss nan
loss=nan detected, stopping training
NaN detected in gradients.
"""

OOM_LOG = """
Epoch 1: Train Loss 0.7500, Val Loss 0.7200
Epoch 2: Train Loss 0.6900, Val Loss 0.6700
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
"""

INDEX_ERROR_LOG = """
Epoch 1: Train Loss 0.7812, Val Loss 0.7501
Epoch 2: Train Loss 0.7201, Val Loss 0.7002
Training step 300...
Traceback (most recent call last):
  File "/home/wangk/kumorfm/models/kumorfm.py", line 342, in forward
    x = self.layers[idx](x)
IndexError: list index out of range
"""

RUNTIME_ERROR_LOG = """
Epoch 1: Train Loss 0.9000, Val Loss 0.8800
Traceback (most recent call last):
  File "train.py", line 50, in train
    loss.backward()
RuntimeError: Expected all tensors to be on the same device
"""

MULTIPLE_ERRORS_LOG = """
Epoch 1: Train Loss 0.8000, Val Loss 0.7800
RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB
Traceback (most recent call last):
  File "train.py", line 42, in step
    outputs = model(inputs)
RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB
"""

AUROC_LOG = """
Epoch 1: Train Loss 0.8100, Val Loss 0.7900
Epoch 2: Train Loss 0.7200, Val Loss 0.6800
AUROC: 0.5000
Epoch 3: Train Loss 0.6800, Val Loss 0.6500
AUROC: 0.7234
"""

KV_STYLE_LOG = """
epoch=1 train_loss=0.8100 val_loss=0.7900
epoch=2 train_loss=0.7200 val_loss=0.6800
"""

TIMING_LOG = """
Epoch 1: Train Loss 0.8100, Val Loss 0.7900
Epoch 1 timing: total=10.50s
Epoch 2: Train Loss 0.7200, Val Loss 0.6800
Epoch 2 timing: total=9.80s
"""

MEMORY_MIB_LOG = """
Epoch 1: Train Loss 0.8100, Val Loss 0.7900
|===========================================================================|
| NVIDIA-SMI ...                                                            |
|    8400MiB / 24576MiB                                                    |
Epoch 2: Train Loss 0.7200, Val Loss 0.6800
"""


# ---------------------------------------------------------------------------
# Normal training tests
# ---------------------------------------------------------------------------

class TestNormalTraining:
    def test_extracts_training_curve(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG)
        assert len(metrics.training_curve) == 3
        assert metrics.training_curve[0] == (1, pytest.approx(0.7812, abs=1e-3), pytest.approx(0.7501, abs=1e-3))
        assert metrics.training_curve[2] == (3, pytest.approx(0.6910, abs=1e-3), pytest.approx(0.6712, abs=1e-3))

    def test_final_losses(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG)
        assert metrics.val_loss == pytest.approx(0.6712, abs=1e-3)
        assert metrics.train_loss == pytest.approx(0.6910, abs=1e-3)

    def test_best_val_loss(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG)
        assert metrics.best_val_loss == pytest.approx(0.6712, abs=1e-3)
        assert metrics.best_epoch == 3

    def test_auroc(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG)
        assert metrics.auroc == pytest.approx(0.6823, abs=1e-3)

    def test_epochs_completed(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG)
        assert metrics.epochs_completed == 3

    def test_training_seconds(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG)
        assert metrics.training_seconds == pytest.approx(36.24, abs=0.1)

    def test_peak_memory_gb(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG)
        assert metrics.peak_memory_gb == pytest.approx(8.2, abs=0.05)

    def test_no_errors(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG, returncode=0)
        assert errors == []


# ---------------------------------------------------------------------------
# Error detection tests
# ---------------------------------------------------------------------------

class TestErrorDetection:
    def test_nan_loss_detected(self):
        metrics, errors = extract_metrics_and_errors(NAN_LOSS_LOG)
        nan_errors = [e for e in errors if e.error_type == "NaN"]
        assert len(nan_errors) >= 1
        assert "nan" in nan_errors[0].error_message.lower() or "NaN" in nan_errors[0].error_message

    def test_nan_detected_via_errors(self):
        # The NaN error is captured in the errors list.
        # val_loss still reflects whatever numeric epoch the extractor parsed
        # (epoch 1 = 0.7501); the NaN epoch itself doesn't produce a float.
        _, errors = extract_metrics_and_errors(NAN_LOSS_LOG)
        nan_errors = [e for e in errors if e.error_type == "NaN"]
        assert len(nan_errors) >= 1

    def test_oom_detected(self):
        metrics, errors = extract_metrics_and_errors(OOM_LOG, returncode=1)
        oom_errors = [e for e in errors if e.error_type == "OOM"]
        assert len(oom_errors) >= 1
        assert "out of memory" in oom_errors[0].error_message.lower()

    def test_index_error_detected(self):
        metrics, errors = extract_metrics_and_errors(INDEX_ERROR_LOG, returncode=1)
        idx_errors = [e for e in errors if e.error_type == "IndexError"]
        assert len(idx_errors) >= 1

    def test_index_error_location(self):
        metrics, errors = extract_metrics_and_errors(INDEX_ERROR_LOG, returncode=1)
        idx_error = next(e for e in errors if e.error_type == "IndexError")
        assert "kumorfm.py" in idx_error.error_file
        assert idx_error.error_line == 342

    def test_index_error_message(self):
        metrics, errors = extract_metrics_and_errors(INDEX_ERROR_LOG, returncode=1)
        idx_error = next(e for e in errors if e.error_type == "IndexError")
        assert "list index out of range" in idx_error.error_message

    def test_runtime_error_detected(self):
        metrics, errors = extract_metrics_and_errors(RUNTIME_ERROR_LOG, returncode=1)
        rt_errors = [e for e in errors if e.error_type == "RuntimeError"]
        assert len(rt_errors) >= 1

    def test_timeout_detected(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG, timed_out=True)
        timeout_errors = [e for e in errors if e.error_type == "Timeout"]
        assert len(timeout_errors) == 1
        assert "timeout" in timeout_errors[0].error_message.lower() or "killed" in timeout_errors[0].error_message.lower()

    def test_non_zero_exit_no_traceback(self):
        # Non-zero exit with no recognized error → NonZeroExit error
        simple_log = "Training started\nLoss: 0.5\n"
        metrics, errors = extract_metrics_and_errors(simple_log, returncode=137)
        non_zero = [e for e in errors if e.error_type == "NonZeroExit"]
        assert len(non_zero) == 1
        assert "137" in non_zero[0].error_message

    def test_context_lines_present(self):
        metrics, errors = extract_metrics_and_errors(INDEX_ERROR_LOG, returncode=1)
        idx_error = next(e for e in errors if e.error_type == "IndexError")
        assert isinstance(idx_error.context_lines, list)

    def test_full_traceback_present(self):
        metrics, errors = extract_metrics_and_errors(INDEX_ERROR_LOG, returncode=1)
        idx_error = next(e for e in errors if e.error_type == "IndexError")
        assert "Traceback" in idx_error.full_traceback


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_log(self):
        metrics, errors = extract_metrics_and_errors("", returncode=0)
        assert metrics.val_loss == 999.0
        assert metrics.epochs_completed == 0
        assert errors == []

    def test_no_epochs_crash(self):
        log = "Starting...\nError: something went wrong\n"
        metrics, errors = extract_metrics_and_errors(log, returncode=1)
        assert metrics.val_loss == 999.0

    def test_auroc_takes_last_value(self):
        metrics, errors = extract_metrics_and_errors(AUROC_LOG)
        assert metrics.auroc == pytest.approx(0.7234, abs=1e-3)

    def test_kv_style_epoch_parsing(self):
        metrics, errors = extract_metrics_and_errors(KV_STYLE_LOG)
        assert len(metrics.training_curve) == 2
        assert metrics.val_loss == pytest.approx(0.6800, abs=1e-3)

    def test_timing_from_epoch_lines(self):
        metrics, errors = extract_metrics_and_errors(TIMING_LOG)
        assert metrics.training_seconds == pytest.approx(10.50 + 9.80, abs=0.1)

    def test_memory_from_mib(self):
        metrics, errors = extract_metrics_and_errors(MEMORY_MIB_LOG)
        assert metrics.peak_memory_gb == pytest.approx(8400 / 1024, abs=0.1)

    def test_partial_run_metrics(self):
        metrics, errors = extract_metrics_and_errors(PARTIAL_TRAINING_LOG)
        assert metrics.epochs_completed == 3
        assert metrics.best_val_loss < metrics.val_loss or metrics.best_val_loss == metrics.val_loss

    def test_success_returncode_no_error(self):
        metrics, errors = extract_metrics_and_errors(NORMAL_TRAINING_LOG, returncode=0)
        assert not any(e.error_type == "NonZeroExit" for e in errors)


# ---------------------------------------------------------------------------
# config_summary tests
# ---------------------------------------------------------------------------

class TestConfigSummary:
    def test_basic_summary(self):
        cfg = {"lr": 0.001, "hidden_dim": 128, "num_layers": 3}
        s = config_summary(cfg)
        assert "lr=0.001" in s or "lr=1e-03" in s or "lr=1e-3" in s
        assert "hd=128" in s
        assert "nl=3" in s

    def test_small_float_scientific(self):
        cfg = {"lr": 0.0001}
        s = config_summary(cfg)
        # Small floats should be shown in scientific notation
        assert "lr=1e-04" in s or "lr=1e-4" in s or "lr=0.0001" not in s

    def test_missing_keys_skipped(self):
        cfg = {"lr": 0.001}
        s = config_summary(cfg, keys=["lr", "hidden_dim"])
        assert "lr=" in s
        assert "hd=" not in s

    def test_empty_config(self):
        s = config_summary({})
        assert s == ""

    def test_batch_size_abbreviated(self):
        cfg = {"batch_size": 64}
        s = config_summary(cfg)
        assert "bs=64" in s
