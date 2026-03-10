"""Tests for torchmill.autosearch."""

import json
import os
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from torchmill.autosearch import (
    AutosearchConfig,
    AutoGenerateConfig,
    DaemonLock,
    ExperimentResult,
    pick_from_queue,
    move_to_done,
    discard_inprogress,
    generate_random_config,
    load_config_dict,
    log_result,
    log_errors,
    write_heartbeat,
    short_hash,
    RESULTS_HEADER,
    ERRORS_HEADER,
)
from torchmill.log_extractor import ExperimentMetrics, ExperimentError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def minimal_autosearch_yaml(tmp_path):
    config = {
        "command_template": "echo training {config_path}",
        "workdir": str(tmp_path),
        "gpus": [0, 1],
        "time_budget": 300,
        "queue_dir": str(tmp_path / "queue"),
        "done_dir": str(tmp_path / "done"),
        "results_file": str(tmp_path / "results.tsv"),
        "errors_file": str(tmp_path / "errors.tsv"),
        "heartbeat_file": str(tmp_path / "heartbeat.txt"),
        "lock_file": str(tmp_path / "autosearch.lock"),
        "git_enabled": False,
        "git_pull_before_run": False,
        "git_commit_results": False,
        "git_keep_on_improve": True,
        "auto_generate": {
            "enabled": False,
            "search_space": {},
            "base_config": {},
        },
    }
    yaml_path = tmp_path / "autosearch.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)
    return yaml_path


@pytest.fixture
def cfg(tmp_path):
    ag = AutoGenerateConfig(
        enabled=True,
        search_space={
            "lr": [0.001, 0.0005],
            "hidden_dim": [64, 128],
            "batch_size": [32, 64],
        },
        base_config={
            "dataset": "trial",
            "epochs": 5,
        },
    )
    return AutosearchConfig(
        command_template="echo {config_path}",
        workdir=str(tmp_path),
        gpus=[0, 1],
        time_budget=60,
        queue_dir=str(tmp_path / "queue"),
        done_dir=str(tmp_path / "done"),
        results_file=str(tmp_path / "results.tsv"),
        errors_file=str(tmp_path / "errors.tsv"),
        heartbeat_file=str(tmp_path / "heartbeat.txt"),
        lock_file=str(tmp_path / "autosearch.lock"),
        git_enabled=False,
        git_pull_before_run=False,
        git_commit_results=False,
        git_keep_on_improve=True,
        auto_generate=ag,
    )


@pytest.fixture
def sample_result(tmp_path, cfg):
    metrics = ExperimentMetrics(
        val_loss=0.6931,
        train_loss=0.7012,
        auroc=0.50,
        best_val_loss=0.6800,
        best_epoch=2,
        epochs_completed=15,
        epochs_total=15,
        peak_memory_gb=8.2,
        training_seconds=120.0,
        training_curve=[(1, 0.7012, 0.6931), (2, 0.6800, 0.6800)],
    )
    return ExperimentResult(
        experiment_id="abc12345",
        gpu_id=0,
        config_path=str(tmp_path / "queue" / "test.json"),
        config_dict={"lr": 0.001, "hidden_dim": 128, "batch_size": 64},
        metrics=metrics,
        errors=[],
        returncode=0,
        timed_out=False,
        status="keep",
        git_commit_hash="a1b2c3d",
        timestamp="2026-03-10T12:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_from_yaml_basic(self, minimal_autosearch_yaml):
        cfg = AutosearchConfig.from_yaml(str(minimal_autosearch_yaml))
        assert cfg.command_template == "echo training {config_path}"
        assert cfg.gpus == [0, 1]
        assert cfg.time_budget == 300
        assert cfg.git_enabled is False

    def test_from_yaml_auto_generate(self, tmp_path):
        config = {
            "command_template": "python train.py --config {config_path}",
            "workdir": str(tmp_path),
            "auto_generate": {
                "enabled": True,
                "search_space": {
                    "lr": [0.001, 0.0005, 0.0001],
                    "hidden_dim": [64, 128, 256],
                },
                "base_config": {"dataset": "trial", "epochs": 5},
            },
        }
        yaml_path = tmp_path / "cfg.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
        loaded = AutosearchConfig.from_yaml(str(yaml_path))
        assert loaded.auto_generate.enabled is True
        assert loaded.auto_generate.search_space["lr"] == [0.001, 0.0005, 0.0001]
        assert loaded.auto_generate.base_config["dataset"] == "trial"

    def test_from_yaml_default_gpus(self, tmp_path):
        config = {"command_template": "echo {config_path}"}
        yaml_path = tmp_path / "cfg.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
        loaded = AutosearchConfig.from_yaml(str(yaml_path))
        assert loaded.gpus == [0]

    def test_from_yaml_single_gpu_int(self, tmp_path):
        config = {"command_template": "echo {config_path}", "gpus": 1}
        yaml_path = tmp_path / "cfg.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
        loaded = AutosearchConfig.from_yaml(str(yaml_path))
        assert loaded.gpus == [1]


# ---------------------------------------------------------------------------
# Queue management tests
# ---------------------------------------------------------------------------

class TestQueueManagement:
    def test_pick_from_empty_queue(self, tmp_path):
        q = tmp_path / "queue"
        q.mkdir()
        result = pick_from_queue(str(q))
        assert result is None

    def test_pick_from_nonexistent_queue(self, tmp_path):
        result = pick_from_queue(str(tmp_path / "nonexistent"))
        assert result is None

    def test_pick_json_config(self, tmp_path):
        q = tmp_path / "queue"
        q.mkdir()
        config_file = q / "exp001.json"
        config_file.write_text(json.dumps({"lr": 0.001}))

        result = pick_from_queue(str(q))
        assert result is not None
        assert result.endswith(".json.inprogress")
        assert not config_file.exists()  # original gone
        assert Path(result).exists()     # .inprogress exists

    def test_pick_yaml_config(self, tmp_path):
        q = tmp_path / "queue"
        q.mkdir()
        config_file = q / "exp001.yaml"
        config_file.write_text("lr: 0.001\n")

        result = pick_from_queue(str(q))
        assert result is not None
        assert result.endswith(".yaml.inprogress")

    def test_pick_oldest_first(self, tmp_path):
        q = tmp_path / "queue"
        q.mkdir()
        # Create files with alphabetical ordering
        (q / "aaa.json").write_text('{"lr": 0.001}')
        (q / "bbb.json").write_text('{"lr": 0.002}')
        (q / "ccc.json").write_text('{"lr": 0.003}')

        result = pick_from_queue(str(q))
        assert "aaa" in result

    def test_pick_ignores_inprogress(self, tmp_path):
        q = tmp_path / "queue"
        q.mkdir()
        # Only an in-progress file
        (q / "exp.json.inprogress").write_text('{"lr": 0.001}')
        result = pick_from_queue(str(q))
        assert result is None

    def test_move_to_done(self, tmp_path):
        q = tmp_path / "queue"
        done = tmp_path / "done"
        q.mkdir()

        # Create an in-progress file
        ip_file = q / "exp001.json.inprogress"
        ip_file.write_text('{"lr": 0.001}')

        move_to_done(str(ip_file), str(done))

        assert not ip_file.exists()
        assert (done / "exp001.json").exists()

    def test_move_to_done_creates_done_dir(self, tmp_path):
        q = tmp_path / "queue"
        done = tmp_path / "done_new"
        q.mkdir()
        ip_file = q / "exp.json.inprogress"
        ip_file.write_text("{}")
        move_to_done(str(ip_file), str(done))
        assert done.exists()

    def test_discard_inprogress_restores_file(self, tmp_path):
        q = tmp_path / "queue"
        q.mkdir()
        ip_file = q / "exp001.json.inprogress"
        ip_file.write_text('{"lr": 0.001}')

        discard_inprogress(str(ip_file))

        assert not ip_file.exists()
        assert (q / "exp001.json").exists()

    def test_concurrent_queue_pick(self, tmp_path):
        """Two threads should not pick the same config."""
        q = tmp_path / "queue"
        q.mkdir()
        for i in range(5):
            (q / f"exp{i:03d}.json").write_text(f'{{"lr": {i}}}')

        picked = []
        lock = threading.Lock()

        def pick():
            result = pick_from_queue(str(q))
            if result:
                with lock:
                    picked.append(result)

        threads = [threading.Thread(target=pick) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No duplicates
        assert len(picked) == len(set(picked))
        # At most 5 configs picked (only 5 exist)
        assert len(picked) <= 5


# ---------------------------------------------------------------------------
# Config generation tests
# ---------------------------------------------------------------------------

class TestConfigGeneration:
    def test_generates_json_file(self, cfg, tmp_path):
        Path(cfg.queue_dir).mkdir(parents=True, exist_ok=True)
        result = generate_random_config(cfg)
        assert result is not None
        assert Path(result).exists()
        assert result.endswith(".inprogress")

    def test_generated_config_has_base_fields(self, cfg, tmp_path):
        Path(cfg.queue_dir).mkdir(parents=True, exist_ok=True)
        result = generate_random_config(cfg)
        with open(result) as f:
            data = json.load(f)
        assert data["dataset"] == "trial"
        assert data["epochs"] == 5

    def test_generated_config_has_sampled_fields(self, cfg, tmp_path):
        Path(cfg.queue_dir).mkdir(parents=True, exist_ok=True)
        result = generate_random_config(cfg)
        with open(result) as f:
            data = json.load(f)
        assert data["lr"] in [0.001, 0.0005]
        assert data["hidden_dim"] in [64, 128]
        assert data["batch_size"] in [32, 64]

    def test_disabled_returns_none(self, cfg, tmp_path):
        cfg.auto_generate.enabled = False
        result = generate_random_config(cfg)
        assert result is None

    def test_empty_search_space_returns_none(self, cfg, tmp_path):
        cfg.auto_generate.search_space = {}
        result = generate_random_config(cfg)
        assert result is None

    def test_short_hash_unique(self):
        hashes = {short_hash() for _ in range(100)}
        assert len(hashes) > 90  # highly likely all unique


# ---------------------------------------------------------------------------
# Config dict loading tests
# ---------------------------------------------------------------------------

class TestLoadConfigDict:
    def test_load_json(self, tmp_path):
        f = tmp_path / "exp.json.inprogress"
        f.write_text('{"lr": 0.001, "epochs": 5}')
        data = load_config_dict(str(f))
        assert data["lr"] == 0.001
        assert data["epochs"] == 5

    def test_load_yaml(self, tmp_path):
        f = tmp_path / "exp.yaml.inprogress"
        f.write_text("lr: 0.001\nepochs: 5\n")
        data = load_config_dict(str(f))
        assert data["lr"] == pytest.approx(0.001)
        assert data["epochs"] == 5


# ---------------------------------------------------------------------------
# Heartbeat tests
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_writes_heartbeat(self, tmp_path):
        hb_file = str(tmp_path / "heartbeat.txt")
        write_heartbeat(hb_file, gpu_id=0)
        content = Path(hb_file).read_text()
        assert "gpu=0" in content
        assert "timestamp=" in content
        assert "pid=" in content

    def test_heartbeat_creates_parent_dirs(self, tmp_path):
        hb_file = str(tmp_path / "subdir" / "heartbeat.txt")
        write_heartbeat(hb_file, gpu_id=1)
        assert Path(hb_file).exists()

    def test_heartbeat_status(self, tmp_path):
        hb_file = str(tmp_path / "heartbeat.txt")
        write_heartbeat(hb_file, gpu_id=0, status="idle")
        content = Path(hb_file).read_text()
        assert "status=idle" in content


# ---------------------------------------------------------------------------
# TSV logging tests
# ---------------------------------------------------------------------------

class TestResultLogging:
    def test_log_result_creates_file(self, tmp_path, sample_result, cfg):
        log_result(sample_result, cfg.results_file)
        assert Path(cfg.results_file).exists()

    def test_log_result_has_header(self, tmp_path, sample_result, cfg):
        log_result(sample_result, cfg.results_file)
        content = Path(cfg.results_file).read_text()
        assert "commit\t" in content
        assert "val_loss\t" in content
        assert "status\t" in content

    def test_log_result_row_values(self, tmp_path, sample_result, cfg):
        log_result(sample_result, cfg.results_file)
        lines = Path(cfg.results_file).read_text().splitlines()
        assert len(lines) == 2  # header + 1 row
        row = lines[1]
        assert "a1b2c3d" in row
        assert "keep" in row
        assert "0.6931" in row

    def test_log_result_appends(self, tmp_path, sample_result, cfg):
        log_result(sample_result, cfg.results_file)
        log_result(sample_result, cfg.results_file)
        lines = Path(cfg.results_file).read_text().splitlines()
        assert len(lines) == 3  # header + 2 rows

    def test_log_result_header_written_once(self, tmp_path, sample_result, cfg):
        for _ in range(3):
            log_result(sample_result, cfg.results_file)
        content = Path(cfg.results_file).read_text()
        assert content.count("commit\t") == 1  # header appears once

    def test_log_errors_creates_file(self, tmp_path, sample_result, cfg):
        sample_result.errors = [ExperimentError(
            error_type="IndexError",
            error_file="models/model.py",
            error_line=42,
            error_message="list index out of range",
            context_lines=["line1", "line2"],
            full_traceback="Traceback...",
        )]
        log_errors(sample_result, cfg.errors_file)
        assert Path(cfg.errors_file).exists()

    def test_log_errors_row_values(self, tmp_path, sample_result, cfg):
        sample_result.errors = [ExperimentError(
            error_type="IndexError",
            error_file="models/model.py",
            error_line=42,
            error_message="list index out of range",
            context_lines=["a", "b", "c"],
            full_traceback="Traceback...",
        )]
        log_errors(sample_result, cfg.errors_file)
        content = Path(cfg.errors_file).read_text()
        assert "IndexError" in content
        assert "models/model.py:42" in content
        assert "list index out of range" in content

    def test_log_errors_no_errors_noop(self, tmp_path, sample_result, cfg):
        sample_result.errors = []
        log_errors(sample_result, cfg.errors_file)
        assert not Path(cfg.errors_file).exists()

    def test_concurrent_logging(self, tmp_path, sample_result, cfg):
        """Multiple threads logging simultaneously should not corrupt TSV."""
        def do_log():
            for _ in range(5):
                log_result(sample_result, cfg.results_file)

        threads = [threading.Thread(target=do_log) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = Path(cfg.results_file).read_text().splitlines()
        # 1 header + 20 rows
        assert len(lines) == 21
        # All rows should have the same number of tabs as header
        header_tabs = lines[0].count("\t")
        for row in lines[1:]:
            assert row.count("\t") == header_tabs


# ---------------------------------------------------------------------------
# DaemonLock tests
# ---------------------------------------------------------------------------

class TestDaemonLock:
    def test_acquire_and_release(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        lock = DaemonLock(lock_path)
        assert lock.acquire() is True
        lock.release()
        assert not Path(lock_path).exists()

    def test_double_acquire_fails(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        lock1 = DaemonLock(lock_path)
        lock2 = DaemonLock(lock_path)
        assert lock1.acquire() is True
        assert lock2.acquire() is False
        lock1.release()

    def test_acquire_after_release(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        lock = DaemonLock(lock_path)
        lock.acquire()
        lock.release()
        lock2 = DaemonLock(lock_path)
        assert lock2.acquire() is True
        lock2.release()
