# torchmill

Autonomous ML experiment runner with LLM-powered analysis.

## What is torchmill?

torchmill is a two-machine system for running ML experiments 24/7 without babysitting:

1. **autosearch** — runs on the GPU machine (5090-1). No API keys needed. Manages a queue of experiment configs, trains in parallel across GPUs, captures metrics + errors, and integrates with git.

2. **brain** — runs on a separate machine with LLM access (ulap). Analyzes results.tsv and errors.tsv, suggests promising hyperparameter combinations, and pushes code fixes via git.

The machines communicate only through **git and files** — no sockets, no shared database, no cloud service.

## Architecture

```
┌──────────────────────────────────┐     git push/pull     ┌───────────────────────────────────┐
│         GPU Machine (5090-1)     │ ◄──────────────────── │        Brain Machine (ulap)        │
│                                  │                        │                                   │
│  torchmill-autosearch            │                        │  (Volt / LLM brain — separate)    │
│  ├── worker GPU 0                │ ──────────────────────►│                                   │
│  │   ├── git pull                │     results + errors   │  1. ssh: read results.tsv         │
│  │   ├── pick config from queue  │                        │  2. ssh: read errors.tsv          │
│  │   ├── run training (timeout)  │                        │  3. analyze with LLM              │
│  │   ├── parse logs → metrics    │                        │  4. write configs/queue/*.json    │
│  │   ├── write results.tsv       │                        │  5. git push                      │
│  │   └── git keep/discard        │                        │  6. (optionally push code fixes)  │
│  └── worker GPU 1 (same loop)   │                        │                                   │
│                                  │                        │                                   │
│  configs/queue/*.json  ◄─────────┼────────────────────────┘                                   │
│  results.tsv, errors.tsv ────────┼────────────────────────►                                   │
│  heartbeat.txt ──────────────────┼────────────────────────►                                   │
└──────────────────────────────────┘                        └───────────────────────────────────┘
```

## Quick Start

### GPU Machine (5090-1)

```bash
# Install torchmill (no API keys needed)
pip install -e .

# Copy and edit the example config
cp configs/autosearch_example.yaml autosearch.yaml
# Edit: set workdir, command_template, gpus, search_space

# Create the queue directory
mkdir -p configs/queue configs/done

# Drop an experiment config in the queue
echo '{"lr": 0.001, "hidden_dim": 128, "epochs": 15}' > configs/queue/exp001.json

# Start autosearch in a tmux session
tmux new -s autosearch
torchmill-autosearch --config autosearch.yaml

# Or run directly with Python
python -m torchmill.autosearch --config autosearch.yaml
```

### Brain Machine (ulap)

```bash
# Install with LLM deps
pip install torchmill openai

# Check results (via SSH or after git pull)
ssh 5090-1 cat /path/to/results.tsv

# The brain reads results and pushes new configs via git
# (See interface contract below)
```

## Prerequisites

### GPU Machine
- Python 3.10+
- PyTorch 2.0+
- `pyyaml` (autosearch only needs stdlib + yaml)
- Your training environment (conda, virtualenv, etc.)
- git

### Brain Machine
- Python 3.10+
- `openai>=1.0` or Anthropic API client
- SSH access to GPU machine (read-only is fine)
- git (with push access to shared repo)

## Configuration Reference

```yaml
# autosearch.yaml — complete reference

# Training command. {config_path} is replaced with the experiment config file.
# Use conda/venv activation if needed.
command_template: >-
  bash -c 'source ~/anaconda3/etc/profile.d/conda.sh &&
  conda activate myenv &&
  python train.py --config {config_path}'

# Working directory for git operations and relative paths
workdir: /home/user/my-ml-project

# GPU worker assignment (one thread per GPU)
gpus: [0, 1]

# Kill training after this many seconds
time_budget: 1800  # 30 minutes

# File paths (relative to this config file or absolute)
queue_dir: ./configs/queue
done_dir:  ./configs/done
results_file: ./results.tsv
errors_file:  ./errors.tsv
heartbeat_file: ./heartbeat.txt
lock_file: ./autosearch.lock

# Git integration
git_enabled: true
git_pull_before_run: true    # pull latest code before each experiment
git_commit_results: false    # don't auto-commit results.tsv
git_keep_on_improve: true    # advance branch only if val_loss improved

# Auto-generate: sample random configs when queue is empty
auto_generate:
  enabled: true
  search_space:
    lr: [0.0001, 0.0003, 0.0005, 0.001, 0.003]
    hidden_dim: [64, 128, 256]
    num_layers: [2, 3, 4]
    num_heads: [4, 8]
    dropout: [0.0, 0.05, 0.1, 0.2]
    batch_size: [32, 64, 128, 256]
    scheduler: [cosine, linear]
    warmup_steps: [0, 100, 500]
    weight_decay: [0.0, 0.01, 0.05]
  base_config:  # merged with sampled values; these stay fixed
    dataset: trial
    task: study-outcome
    epochs: 15
    device: cuda
    amp: true
    seed: 42
```

## How It Works

### Per-GPU Worker Loop

```
while True:
    write_heartbeat()
    git pull --rebase (picks up code fixes from brain)
    config = pick_from_queue() or generate_random()
    if no config: sleep(60); continue
    snapshot_commit = git_current_hash()
    run_training(config, timeout=time_budget)
    metrics, errors = parse_logs(stdout + stderr)
    write results.tsv row
    if errors: write errors.tsv row
    if val_loss improved: keep (branch advances)
    else: git reset --hard snapshot_commit (discard)
    move config to done/
```

### Git Keep/Discard Strategy

Each experiment runs on top of the current HEAD. If val_loss improves, the commit stands and the branch advances. If the result is worse (or crashes), we `git reset --hard` to the pre-experiment commit — reverting any model weight changes from that run. This means:

- The git branch always tracks the best-seen checkpoint
- Code fixes from the brain are automatically pulled before each run
- `results.tsv` and `errors.tsv` are never reverted — they accumulate all history

### Timeout Handling

If training exceeds `time_budget` seconds, the subprocess is killed (SIGKILL), an error row is written to errors.tsv with `error_type=Timeout`, and the worker moves immediately to the next experiment.

### Crash Recovery

The outer loop wraps everything in `try/except Exception`. Any unexpected crash (disk full, git conflict, import error) is logged and the worker sleeps 60s before retrying. The daemon never exits.

## CLI Reference

```bash
# Start daemon (foreground — use tmux or screen)
torchmill-autosearch --config autosearch.yaml

# Dry run: show commands without executing training
torchmill-autosearch --config autosearch.yaml --dry-run

# Show current status
torchmill-autosearch --config autosearch.yaml --status

# Run only one GPU (useful for debugging)
torchmill-autosearch --config autosearch.yaml --gpu 0

# Verbose logging
torchmill-autosearch --config autosearch.yaml -v

# Python module style
python -m torchmill.autosearch --config autosearch.yaml
```

## Output Files

### results.tsv

Tab-separated, one row per experiment. Appended after each run.

```
commit  gpu  val_loss  train_loss  auroc  memory_gb  epochs  status  config_summary  error_type
a1b2c3  0    0.6931    0.7012      0.50   8.2        15/15   keep    lr=5e-4 hd=128
b2c3d4  1    0.6512    0.6201      0.58   8.4        15/15   keep    lr=3e-4 hd=256
c3d4e5  0    999.0     999.0       0.0    0.0        3/15    crash   lr=1e-3 hd=512  IndexError
```

Fields:
- `commit` — short git hash at time of run
- `gpu` — CUDA device index
- `val_loss` — final validation loss (999.0 if crash/missing)
- `train_loss` — final training loss
- `auroc` — AUROC score (0.0 if not available)
- `memory_gb` — peak GPU memory used
- `epochs` — completed/total
- `status` — `keep`, `discard`, `crash`, or `timeout`
- `config_summary` — compact hyperparameter string (`lr=5e-4 hd=128 nl=3`)
- `error_type` — empty if success, else `IndexError`, `OOM`, `NaN`, `Timeout`, etc.

### errors.tsv

Tab-separated, one row per error (experiments with errors may have multiple rows).

```
commit    timestamp              gpu  error_type  error_location          error_message            context
b2c3d4    2026-03-10T14:30:00Z  0    IndexError  models/kumorfm.py:342  list index out of range  line1 | line2
```

### heartbeat.txt

Updated every loop iteration. The brain reads this to know the daemon is alive.

```
timestamp=2026-03-10T14:30:00+00:00
gpu=0
status=running
pid=12345
```

## Interface Contract (for the Brain)

The brain machine reads and writes via git. Here's the exact contract:

### Brain reads:
- `results.tsv` — experiment outcomes (git pull or SSH)
- `errors.tsv` — crash details including file, line, traceback context
- `heartbeat.txt` — liveness check
- `git log --oneline -20` — recent experiment commits
- `git diff HEAD~5 HEAD` — recent code changes

### Brain writes:
- `configs/queue/*.json` — new experiment configs to run
- Source code fixes (pushed via `git push`)

### Config format the brain should write:

```json
{
  "lr": 0.0003,
  "hidden_dim": 256,
  "num_layers": 3,
  "num_heads": 8,
  "dropout": 0.05,
  "batch_size": 64,
  "scheduler": "cosine",
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "dataset": "trial",
  "task": "study-outcome",
  "epochs": 15,
  "seed": 42
}
```

Files are picked in alphabetical order. Use numeric prefixes to set priority:
```
configs/queue/000_top_priority.json
configs/queue/001_normal.json
```

## Comparison with Alternatives

| Feature | torchmill autosearch | Ray Tune | Karpathy autoresearch |
|---------|---------------------|----------|----------------------|
| No API keys on GPU machine | ✓ | ✓ | ✗ |
| Multi-GPU parallel workers | ✓ | ✓ | ✗ |
| LLM brain (separate machine) | ✓ | ✗ | ✓ |
| Git-integrated keep/discard | ✓ | ✗ | ✗ |
| Error capture + structured log | ✓ | partial | ✗ |
| Bulletproof daemon | ✓ | ✗ | ✗ |
| Dependency footprint | minimal | heavy | medium |
| Timeout handling | ✓ | ✓ | ✗ |
| Queue from external source | ✓ | ✗ | ✗ |

## torchmill Training Framework (existing components)

autosearch is fully standalone, but torchmill also ships a general training framework:

- `torchmill/trainer.py` — model-agnostic training loop with callbacks
- `torchmill/config.py` — Pydantic config schema
- `torchmill/callbacks/` — checkpoint, early stopping, logging callbacks
- `torchmill/data/` — CSV and RelBench dataset adapters
- `torchmill/models/registry.py` — model registry for dynamic instantiation

### Quick training example:

```python
from torchmill.config import TorchmillConfig
from torchmill.trainer import Trainer

config = TorchmillConfig.from_yaml("config.yaml")
trainer = Trainer(config)
metrics = trainer.fit(model, train_dataset, val_dataset)
```

See `configs/example.yaml` for a full training config reference.

## Project Structure

```
torchmill/
├── torchmill/
│   ├── __init__.py           # version
│   ├── autosearch.py         # autonomous experiment runner (new)
│   ├── log_extractor.py      # training log parser (new)
│   ├── cli.py                # training CLI
│   ├── config.py             # Pydantic config schema
│   ├── trainer.py            # core training loop
│   ├── data/
│   │   ├── base.py
│   │   ├── csv_dataset.py
│   │   └── relbench.py
│   ├── models/
│   │   └── registry.py
│   ├── callbacks/
│   │   ├── checkpoint.py
│   │   ├── early_stopping.py
│   │   └── logging.py
│   └── utils/
│       └── device.py
├── configs/
│   ├── example.yaml
│   ├── autosearch_example.yaml   # autosearch config template (new)
│   └── queue/
│       └── README.md             # queue directory docs (new)
├── tests/
│   ├── test_config.py
│   ├── test_registry.py
│   ├── test_trainer.py
│   ├── test_log_extractor.py     # (new)
│   └── test_autosearch.py        # (new)
├── setup.py
└── requirements.txt
```

## Development

```bash
# Install in editable mode
pip install -e .

# Run all tests
pytest tests/ -v

# Run only autosearch tests
pytest tests/test_log_extractor.py tests/test_autosearch.py -v
```

## License

MIT
