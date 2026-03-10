"""autosearch.py — autonomous ML experiment runner for torchmill.

Runs as a bulletproof daemon on a GPU machine (no API keys required).
Manages N GPU workers in parallel, picks configs from a queue directory,
trains with timeout, captures metrics + errors, and logs results.

Usage:
    torchmill-autosearch --config autosearch.yaml
    python -m torchmill.autosearch --config autosearch.yaml
    python -m torchmill.autosearch --config autosearch.yaml --dry-run
    python -m torchmill.autosearch --config autosearch.yaml --status
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from torchmill.log_extractor import (
    ExperimentMetrics,
    ExperimentError,
    extract_metrics_and_errors,
    config_summary,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("autosearch")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AutoGenerateConfig:
    enabled: bool = False
    search_space: Dict[str, List[Any]] = field(default_factory=dict)
    base_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutosearchConfig:
    # Training
    command_template: str = "python train.py --config {config_path}"
    workdir: str = "."

    # GPU assignment
    gpus: List[int] = field(default_factory=lambda: [0])

    # Time budget
    time_budget: int = 1800  # seconds

    # Paths
    queue_dir: str = "./configs/queue"
    done_dir: str = "./configs/done"
    results_file: str = "./results.tsv"
    errors_file: str = "./errors.tsv"
    heartbeat_file: str = "./heartbeat.txt"
    lock_file: str = "./autosearch.lock"

    # Git
    git_enabled: bool = True
    git_pull_before_run: bool = True
    git_commit_results: bool = False
    git_keep_on_improve: bool = True

    # Auto-generate
    auto_generate: AutoGenerateConfig = field(default_factory=AutoGenerateConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "AutosearchConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        auto_gen_raw = raw.pop("auto_generate", {})
        auto_gen = AutoGenerateConfig(
            enabled=auto_gen_raw.get("enabled", False),
            search_space=auto_gen_raw.get("search_space", {}),
            base_config=auto_gen_raw.get("base_config", {}),
        )

        gpus = raw.get("gpus", [0])
        if isinstance(gpus, int):
            gpus = [gpus]

        return cls(
            command_template=raw.get("command_template", cls.command_template),
            workdir=raw.get("workdir", cls.workdir),
            gpus=gpus,
            time_budget=int(raw.get("time_budget", cls.time_budget)),
            queue_dir=raw.get("queue_dir", cls.queue_dir),
            done_dir=raw.get("done_dir", cls.done_dir),
            results_file=raw.get("results_file", cls.results_file),
            errors_file=raw.get("errors_file", cls.errors_file),
            heartbeat_file=raw.get("heartbeat_file", cls.heartbeat_file),
            lock_file=raw.get("lock_file", cls.lock_file),
            git_enabled=raw.get("git_enabled", cls.git_enabled),
            git_pull_before_run=raw.get("git_pull_before_run", cls.git_pull_before_run),
            git_commit_results=raw.get("git_commit_results", cls.git_commit_results),
            git_keep_on_improve=raw.get("git_keep_on_improve", cls.git_keep_on_improve),
            auto_generate=auto_gen,
        )


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    experiment_id: str
    gpu_id: int
    config_path: str
    config_dict: Dict[str, Any]
    metrics: ExperimentMetrics
    errors: List[ExperimentError]
    returncode: int
    timed_out: bool
    status: str           # "keep" | "discard" | "crash" | "timeout"
    git_commit_hash: str  # short hash before run
    timestamp: str


# ---------------------------------------------------------------------------
# File locking (prevent duplicate daemon instances)
# ---------------------------------------------------------------------------


class DaemonLock:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._fh = None

    def acquire(self) -> bool:
        """Return True if lock acquired, False if another instance is running."""
        try:
            self._fh = open(self.lock_path, "w")
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fh.write(str(os.getpid()))
            self._fh.flush()
            return True
        except (IOError, OSError):
            return False

    def release(self):
        if self._fh:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
                self._fh.close()
                Path(self.lock_path).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(args: List[str], cwd: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Run a git command; return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "git command timed out"
    except Exception as e:
        return -1, "", str(e)


def git_pull(workdir: str) -> bool:
    rc, out, err = _git(["pull", "--rebase", "--autostash"], cwd=workdir, timeout=60)
    if rc == 0:
        logger.info(f"git pull: {out or 'up to date'}")
        return True
    else:
        logger.warning(f"git pull failed (rc={rc}): {err}")
        return False


def git_current_hash(workdir: str) -> str:
    rc, out, _ = _git(["rev-parse", "--short", "HEAD"], cwd=workdir)
    return out if rc == 0 else "unknown"


def git_commit_experiment(workdir: str, experiment_id: str) -> str:
    """Stage all changes and commit with experiment label. Return new hash."""
    _git(["add", "-A"], cwd=workdir)
    msg = f"experiment: {experiment_id}"
    rc, out, err = _git(["commit", "-m", msg], cwd=workdir)
    if rc != 0 and "nothing to commit" not in err:
        logger.warning(f"git commit failed: {err}")
    return git_current_hash(workdir)


def git_reset_hard(workdir: str, ref: str = "HEAD~1") -> bool:
    """Discard last commit (revert to ref). Returns True on success."""
    rc, _, err = _git(["reset", "--hard", ref], cwd=workdir)
    if rc != 0:
        logger.warning(f"git reset failed: {err}")
    return rc == 0


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------


_QUEUE_LOCK = threading.Lock()


def pick_from_queue(queue_dir: str) -> Optional[str]:
    """Atomically claim the oldest JSON config file from queue_dir.

    Returns the path to the claimed file, or None if queue is empty.
    """
    queue_path = Path(queue_dir)
    if not queue_path.exists():
        return None

    with _QUEUE_LOCK:
        candidates = sorted(queue_path.glob("*.json")) + sorted(queue_path.glob("*.yaml"))
        for candidate in candidates:
            # Use .inprogress suffix as atomic claim marker
            claimed = candidate.with_suffix(candidate.suffix + ".inprogress")
            try:
                candidate.rename(claimed)
                return str(claimed)
            except (FileNotFoundError, OSError):
                continue  # another worker grabbed it first
    return None


def move_to_done(config_path: str, done_dir: str) -> None:
    """Move completed config to done_dir, stripping .inprogress suffix."""
    src = Path(config_path)
    done_path = Path(done_dir)
    done_path.mkdir(parents=True, exist_ok=True)

    # Strip .inprogress
    name = src.name
    if name.endswith(".inprogress"):
        name = name[: -len(".inprogress")]

    dest = done_path / name
    # Avoid overwrite by adding timestamp
    if dest.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = dest.stem
        suffix = dest.suffix
        dest = done_path / f"{stem}_{ts}{suffix}"

    try:
        shutil.move(str(src), str(dest))
    except Exception as e:
        logger.warning(f"Failed to move config to done: {e}")


def discard_inprogress(config_path: str) -> None:
    """Clean up a claimed config that we couldn't process."""
    src = Path(config_path)
    if src.exists() and src.suffix == ".inprogress":
        name = src.name[: -len(".inprogress")]
        queue_dir = src.parent
        src.rename(queue_dir / name)


# ---------------------------------------------------------------------------
# Random config generation
# ---------------------------------------------------------------------------


def short_hash(length: int = 8) -> str:
    """Generate a short random hex string."""
    return hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:length]


def generate_random_config(cfg: AutosearchConfig) -> Optional[str]:
    """Sample random hyperparams, merge with base_config, write to temp file.

    Returns path to the temp JSON file, or None if generation is disabled.
    """
    if not cfg.auto_generate.enabled:
        return None

    space = cfg.auto_generate.search_space
    if not space:
        return None

    sampled = {k: random.choice(v) for k, v in space.items() if v}
    config_dict = {**cfg.auto_generate.base_config, **sampled}

    queue_path = Path(cfg.queue_dir)
    queue_path.mkdir(parents=True, exist_ok=True)

    exp_id = short_hash()
    config_path = queue_path / f"autogen_{exp_id}.json.inprogress"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Generated random config: {config_summary(config_dict)}")
    return str(config_path)


def load_config_dict(config_path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file into a dict."""
    path = config_path.replace(".inprogress", "")
    # Check actual file path (might still have .inprogress)
    actual = config_path
    with open(actual) as f:
        if actual.endswith((".yaml", ".yaml.inprogress")):
            return yaml.safe_load(f) or {}
        else:
            return json.load(f)


# ---------------------------------------------------------------------------
# Training subprocess
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    returncode: int
    timed_out: bool
    log_text: str   # combined stdout + stderr
    duration_seconds: float


def run_training(
    command_template: str,
    config_path: str,
    workdir: str,
    gpu_id: int,
    timeout: int,
    num_gpus: int = 1,
) -> RunResult:
    """Run training as a subprocess and capture all output.
    
    If num_gpus > 1, wraps the python command with torchrun for DDP.
    """
    # Strip .inprogress for the actual config path passed to training
    # IMPORTANT: write clean copy to a TEMP location, NOT back into queue dir
    # (writing back to queue would let another worker pick it up — duplicate runs)
    actual_config = config_path
    if actual_config.endswith(".inprogress"):
        clean_name = Path(config_path).stem  # e.g. "p1_trial_lr0.0001.json"
        if clean_name.endswith(".json"):
            clean_name = clean_name  # already has .json
        else:
            clean_name = clean_name + ".json"
        clean_path = os.path.join(tempfile.gettempdir(), f"autosearch_{clean_name}")
        shutil.copy2(actual_config, clean_path)
        actual_config = clean_path

    cmd = command_template.replace("{config_path}", actual_config)

    # DDP: replace 'python ' with 'torchrun --nproc_per_node=N ' in the command
    if num_gpus > 1:
        cmd = cmd.replace("python relbdata/", f"torchrun --nproc_per_node={num_gpus} relbdata/")
        cmd = cmd.replace("python training/", f"torchrun --nproc_per_node={num_gpus} training/")
        cmd = cmd.replace("python scripts/", f"torchrun --nproc_per_node={num_gpus} scripts/")

    env = os.environ.copy()
    # Randomize MASTER_PORT to prevent collisions between concurrent experiments
    env["MASTER_PORT"] = str(random.randint(29500, 39999))
    if num_gpus == 1:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # For DDP, don't set CUDA_VISIBLE_DEVICES — let torchrun manage all GPUs

    logger.info(f"[GPU {gpu_id}] Running: {cmd}")

    timed_out = False
    start = time.monotonic()

    # Capture stdout + stderr combined into a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, prefix="autosearch_"
    ) as logfile:
        log_path = logfile.name

    try:
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=workdir,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        try:
            proc.wait(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = -9
            timed_out = True
            logger.warning(f"[GPU {gpu_id}] Training timed out after {timeout}s")
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed to launch training: {e}")
        returncode = -1

    duration = time.monotonic() - start

    try:
        with open(log_path) as f:
            log_text = f.read()
    except Exception:
        log_text = ""
    finally:
        try:
            os.unlink(log_path)
        except Exception:
            pass

    # Also clean up the non-inprogress copy
    if actual_config != config_path and Path(actual_config).exists():
        try:
            os.unlink(actual_config)
        except Exception:
            pass

    return RunResult(
        returncode=returncode,
        timed_out=timed_out,
        log_text=log_text,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# TSV logging
# ---------------------------------------------------------------------------

_TSV_LOCK = threading.Lock()

RESULTS_HEADER = "commit\tgpu\tval_loss\ttrain_loss\tauroc\tmemory_gb\tepochs\tstatus\tconfig_summary\terror_type\n"
ERRORS_HEADER = "commit\ttimestamp\tgpu\terror_type\terror_location\terror_message\tcontext\n"


def _ensure_header(path: str, header: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size == 0:
        with open(path, "w") as f:
            f.write(header)


def log_result(result: ExperimentResult, results_file: str) -> None:
    _ensure_header(results_file, RESULTS_HEADER)
    m = result.metrics
    e = result.errors
    epochs_str = f"{m.epochs_completed}/{m.epochs_total}" if m.epochs_total else str(m.epochs_completed)
    error_type = e[0].error_type if e else ""
    cfg_summary = config_summary(result.config_dict)

    row = (
        f"{result.git_commit_hash}\t"
        f"{result.gpu_id}\t"
        f"{m.val_loss:.4f}\t"
        f"{m.train_loss:.4f}\t"
        f"{m.auroc:.4f}\t"
        f"{m.peak_memory_gb:.1f}\t"
        f"{epochs_str}\t"
        f"{result.status}\t"
        f"{cfg_summary}\t"
        f"{error_type}\n"
    )
    with _TSV_LOCK:
        with open(results_file, "a") as f:
            f.write(row)


def log_errors(result: ExperimentResult, errors_file: str) -> None:
    if not result.errors:
        return
    _ensure_header(errors_file, ERRORS_HEADER)
    with _TSV_LOCK:
        with open(errors_file, "a") as f:
            for err in result.errors:
                location = f"{err.error_file}:{err.error_line}" if err.error_file else ""
                context = " | ".join(err.context_lines[-5:])
                row = (
                    f"{result.git_commit_hash}\t"
                    f"{result.timestamp}\t"
                    f"{result.gpu_id}\t"
                    f"{err.error_type}\t"
                    f"{location}\t"
                    f"{err.error_message[:200]}\t"
                    f"{context[:500]}\n"
                )
                f.write(row)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def write_heartbeat(heartbeat_file: str, gpu_id: int, status: str = "running") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    content = f"timestamp={ts}\ngpu={gpu_id}\nstatus={status}\npid={os.getpid()}\n"
    try:
        Path(heartbeat_file).parent.mkdir(parents=True, exist_ok=True)
        with open(heartbeat_file, "w") as f:
            f.write(content)
    except Exception as e:
        logger.debug(f"Failed to write heartbeat: {e}")


# ---------------------------------------------------------------------------
# Per-GPU worker loop
# ---------------------------------------------------------------------------


def worker_loop(gpu_id: int, cfg: AutosearchConfig, dry_run: bool = False, num_gpus: int = 1) -> None:
    """Main loop for a single GPU worker. Runs forever until killed.
    
    When num_gpus > 1, uses DDP via torchrun (all GPUs for each experiment).
    """
    mode = f"DDP({num_gpus})" if num_gpus > 1 else f"GPU {gpu_id}"
    log = logging.getLogger(f"autosearch.{mode}")
    best_val_loss = float("inf")
    pre_run_hash: Optional[str] = None

    log.info(f"Worker started (GPU {gpu_id})")

    while True:
        try:
            write_heartbeat(cfg.heartbeat_file, gpu_id)

            # Git pull
            if cfg.git_enabled and cfg.git_pull_before_run:
                git_pull(cfg.workdir)

            # Pick next config
            config_path = pick_from_queue(cfg.queue_dir)
            if config_path is None:
                if cfg.auto_generate.enabled:
                    config_path = generate_random_config(cfg)
                if config_path is None:
                    log.info("Queue empty, sleeping 60s...")
                    time.sleep(60)
                    continue

            # Load config dict for summary/logging
            try:
                config_dict = load_config_dict(config_path)
            except Exception as e:
                log.warning(f"Failed to load config {config_path}: {e}")
                discard_inprogress(config_path)
                continue

            experiment_id = short_hash()
            timestamp = datetime.now(timezone.utc).isoformat()

            log.info(f"Starting experiment {experiment_id}: {config_summary(config_dict)}")

            # Snapshot current commit
            if cfg.git_enabled:
                pre_run_hash = git_current_hash(cfg.workdir)
                # Commit any pending changes before run
                git_commit_experiment(cfg.workdir, experiment_id)
                commit_hash = git_current_hash(cfg.workdir)
            else:
                commit_hash = short_hash()

            if dry_run:
                log.info(f"[DRY RUN] Would run: {cfg.command_template.replace('{config_path}', config_path)}")
                move_to_done(config_path, cfg.done_dir)
                continue

            # Run training
            run_result = run_training(
                command_template=cfg.command_template,
                config_path=config_path,
                workdir=cfg.workdir,
                gpu_id=gpu_id,
                timeout=cfg.time_budget,
                num_gpus=num_gpus,
            )

            # Parse metrics + errors
            metrics, errors = extract_metrics_and_errors(
                log_text=run_result.log_text,
                returncode=run_result.returncode,
                timed_out=run_result.timed_out,
            )

            if metrics.training_seconds == 0.0:
                metrics.training_seconds = run_result.duration_seconds

            # Determine status
            if run_result.timed_out:
                status = "timeout"
            elif errors or run_result.returncode not in (0, None):
                status = "crash"
            elif metrics.val_loss < best_val_loss and cfg.git_keep_on_improve:
                status = "keep"
                best_val_loss = metrics.val_loss
                log.info(f"New best val_loss={metrics.val_loss:.4f} — keeping")
            else:
                status = "discard"

            # Git keep/discard
            if cfg.git_enabled and pre_run_hash is not None:
                if status in ("crash", "discard", "timeout"):
                    log.info(f"Status={status} — discarding git commit")
                    git_reset_hard(cfg.workdir, pre_run_hash)
                else:
                    log.info(f"Status=keep — branch advances")

            result = ExperimentResult(
                experiment_id=experiment_id,
                gpu_id=gpu_id,
                config_path=config_path,
                config_dict=config_dict,
                metrics=metrics,
                errors=errors,
                returncode=run_result.returncode,
                timed_out=run_result.timed_out,
                status=status,
                git_commit_hash=commit_hash,
                timestamp=timestamp,
            )

            log_result(result, cfg.results_file)
            if errors:
                log_errors(result, cfg.errors_file)

            log.info(
                f"Experiment {experiment_id} done — "
                f"val_loss={metrics.val_loss:.4f} status={status} "
                f"epochs={metrics.epochs_completed}/{metrics.epochs_total}"
            )

            move_to_done(config_path, cfg.done_dir)

        except KeyboardInterrupt:
            log.info("Worker interrupted by user")
            raise
        except Exception as e:
            log.error(f"Unexpected error in worker loop: {e}", exc_info=True)
            time.sleep(60)  # back off before retry


# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------


def show_status(cfg: AutosearchConfig) -> None:
    """Print current autosearch status to stdout."""
    print("\n=== autosearch status ===\n")

    # Heartbeat
    hb_path = Path(cfg.heartbeat_file)
    if hb_path.exists():
        print("Heartbeat:")
        print(hb_path.read_text())
    else:
        print("Heartbeat: not found (daemon not running?)\n")

    # Queue
    queue_path = Path(cfg.queue_dir)
    if queue_path.exists():
        pending = list(queue_path.glob("*.json")) + list(queue_path.glob("*.yaml"))
        inprog = list(queue_path.glob("*.inprogress"))
        print(f"Queue: {len(pending)} pending, {len(inprog)} in-progress")
    else:
        print("Queue: directory not found")

    # Done
    done_path = Path(cfg.done_dir)
    if done_path.exists():
        done = list(done_path.glob("*"))
        print(f"Done:  {len(done)} completed experiments")

    # Results
    results_path = Path(cfg.results_file)
    if results_path.exists():
        lines = results_path.read_text().splitlines()
        n_experiments = len(lines) - 1  # subtract header
        print(f"\nResults: {n_experiments} rows in {cfg.results_file}")
        if len(lines) > 1:
            print("Last 5 results:")
            for line in lines[-5:]:
                print(f"  {line}")

    # Lock
    lock_path = Path(cfg.lock_file)
    if lock_path.exists():
        print(f"\nLock file: {lock_path} (daemon may be running)")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="autosearch — autonomous ML experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  torchmill-autosearch --config autosearch.yaml
  torchmill-autosearch --config autosearch.yaml --dry-run
  torchmill-autosearch --config autosearch.yaml --status
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="autosearch.yaml",
        help="Path to autosearch config YAML (default: autosearch.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running training",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current daemon status and exit",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Run only this GPU index (overrides config.gpus, useful for debugging)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    cfg = AutosearchConfig.from_yaml(args.config)

    # Resolve relative paths from config file location
    cfg_dir = Path(args.config).parent.resolve()
    for attr in ("workdir", "queue_dir", "done_dir", "results_file", "errors_file",
                 "heartbeat_file", "lock_file"):
        val = getattr(cfg, attr)
        if not Path(val).is_absolute():
            setattr(cfg, attr, str(cfg_dir / val))

    # Ensure directories exist
    Path(cfg.queue_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.done_dir).mkdir(parents=True, exist_ok=True)

    if args.status:
        show_status(cfg)
        return

    # Determine GPUs to use
    gpus = [args.gpu] if args.gpu is not None else cfg.gpus

    # Acquire lock to prevent duplicate instances
    lock = DaemonLock(cfg.lock_file)
    if not lock.acquire():
        logger.error(
            f"Another autosearch instance appears to be running (lock: {cfg.lock_file}). "
            "Remove the lock file if this is stale."
        )
        sys.exit(1)

    logger.info(f"autosearch starting — GPUs: {gpus}, workdir: {cfg.workdir}")
    logger.info(f"Queue: {cfg.queue_dir} | Results: {cfg.results_file}")

    if args.dry_run:
        logger.info("DRY RUN mode — no training will be executed")

    try:
        if len(gpus) == 1:
            # Single GPU: run in main thread
            worker_loop(gpus[0], cfg, dry_run=args.dry_run, num_gpus=1)
        else:
            # Multiple GPUs: DDP mode — single sequential loop using all GPUs
            # torchrun manages multi-GPU, we just run one experiment at a time
            logger.info(f"DDP mode: {len(gpus)} GPUs, sequential experiments with torchrun")
            worker_loop(gpus[0], cfg, dry_run=args.dry_run, num_gpus=len(gpus))
    finally:
        lock.release()


if __name__ == "__main__":
    main()
