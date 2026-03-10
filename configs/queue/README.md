# autosearch Queue Directory

This directory holds **pending experiment configs** that `autosearch.py` will pick up and run.

## How it works

1. Drop a `.json` or `.yaml` file here with training hyperparameters.
2. `autosearch.py` atomically claims the file (renames it to `*.inprogress`).
3. Training runs; file is moved to `../done/` when complete.

## File format

Each file is a flat key-value config that gets passed to your training script via `{config_path}`.

### Example: `configs/queue/run001.json`

```json
{
  "lr": 0.0005,
  "hidden_dim": 256,
  "num_layers": 3,
  "num_heads": 8,
  "dropout": 0.1,
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

### Example: `configs/queue/run002.yaml`

```yaml
lr: 0.001
hidden_dim: 128
num_layers: 2
num_heads: 4
dropout: 0.0
batch_size: 32
scheduler: linear
warmup_steps: 0
weight_decay: 0.0
dataset: trial
task: study-outcome
epochs: 15
seed: 123
```

## Priority

Files are picked **in alphabetical order** (oldest/lowest-numbered first).
Name files with a numeric prefix to control priority:

```
configs/queue/
  000_high_priority.json
  001_normal.json
  002_low_priority.yaml
```

## Interface contract (for the LLM brain on ulap)

The **brain machine** (Volt/ulap) reads `results.tsv` and `errors.tsv` via SSH,
analyzes patterns, and **writes new config files here** to push promising
hyperparameter combinations for the GPU machine to run.

The brain should:
1. `git pull` to get latest results
2. Read `results.tsv` and `errors.tsv`
3. Write new configs to this directory
4. `git push` — autosearch will pick them up on next `git pull`

## Files in this directory

- `*.json` / `*.yaml` — pending experiment configs
- `*.json.inprogress` / `*.yaml.inprogress` — currently running (do not modify)
- This `README.md` — documentation (ignored by autosearch)
