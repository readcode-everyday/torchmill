# RFM Autoresearch Program

Autonomous research agent for KumoRFM — a relational foundation model trained on temporal heterogeneous graphs.

## Environment

```bash
# Machine: GPU host (2x NVIDIA RTX GPUs, 32GB each)
# Conda env: pyg28
# Activation (MUST prefix every command):
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh && conda activate pyg28

# Project root:
cd /home/wangk/kumorfm-zhaoweizun
```

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files** (see "What you CAN modify" below).
4. **Verify data exists**: Check that `/home/wangk/data/relbench/` contains dataset directories. Available datasets:
   - `rel-trial` (small, fast — good for quick experiments)
   - `rel-f1` (small)
   - `rel-hm` (105K)
   - `rel-stack` (590K)
   - `rel-financial` (900K)
   - `rel-amazon` (1.8M)
   - `rel-supply-chain` (3.8M)
   - `rel-event` (11.2M)
   - `rel-avito` (5.9M)
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## What you CAN modify

### Experiment config (primary — preferred for hyperparameter tuning)

```
configs/autoresearch.json    ← YOUR experiment config (copy from an existing config)
```

This JSON config controls:
- `hidden_dim`, `num_layers`, `num_heads`, `dropout`
- `lr`, `scheduler`, `warmup_steps`, `weight_decay`
- `batch_size`, `max_samples_per_epoch`, `epochs`
- `dataset`, `task`
- `amp`, `device`, `seed`
- Any CLI arg accepted by `relbdata/train_on_relbench.py`

### Model code (for architecture changes)

```
models/kumorfm.py            ← Main model (KumoRFM)
models/relgt/                ← RelGT encoder
models/icl/                  ← In-context learning module
models/encoders/             ← Feature encoders
models/shared/               ← Shared components
config/model_config.py       ← Config dataclasses (KumoRFMConfig, etc.)
```

### Sampling (for data pipeline changes)

```
sampling/backward_sampler.py ← Subgraph extraction
sampling/context_sampler.py  ← ICL context sampling
sampling/context_label_table.py
```

### Training loop (for optimizer/scheduler changes)

```
training/trainer.py          ← Training loop
utils/training_utils.py      ← Optimizers, schedulers, utilities
```

## What you CANNOT modify

```
relbdata/train_on_relbench.py   ← Entry point (READ-ONLY — this is the "prepare.py")
relbdata/adapter.py             ← Dataset loading (READ-ONLY)
data/                           ← Data structures (READ-ONLY)
utils/path_constants.py         ← Path config (READ-ONLY)
utils/registry.py               ← Registries (READ-ONLY)
```

The human maintains these files. Do not touch them.

## Training command

```bash
# ALWAYS use this exact invocation:
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh && conda activate pyg28 && \
  cd /home/wangk/kumorfm-zhaoweizun && \
  python relbdata/train_on_relbench.py --config configs/autoresearch.json \
  > run.log 2>&1
```

**Do NOT use nohup or &.** Run in foreground so you can monitor and kill.

## Time budget

Each experiment has a **fixed time budget of 30 minutes** wall clock.

To enforce this, set in `configs/autoresearch.json`:
- `max_samples_per_epoch`: 5000–20000 (controls epoch size)
- `epochs`: 5–15 (controls total training duration)
- `dataset`: Start with `rel-trial` or `rel-f1` for fast iteration

If a run exceeds 30 minutes, kill it (`Ctrl+C` or `kill`) and treat as failure.

For **quick sanity checks** (architecture changes), use:
```json
{
  "dataset": "trial",
  "task": "study-outcome",
  "max_samples_per_epoch": 2000,
  "epochs": 3,
  "batch_size": 64
}
```
This runs in ~2–5 minutes.

## Metric

The primary metric is **val_loss** — lower is better. Extract from log:

```bash
grep "Epoch.*Val Loss" run.log | tail -1
```

Format: `Epoch N: Train Loss X.XXXX, Val Loss Y.YYYY`

Secondary metrics (when available):
```bash
grep "AUROC" run.log | tail -1
```

## Output format

After training completes, extract results:

```bash
# Primary metric
grep "Epoch.*Val Loss" run.log | tail -1

# Memory usage
grep "peak_vram\|CUDA memory\|torch.cuda.max_memory" run.log | tail -1
# Or check manually:
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Check for crashes
tail -20 run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header:

```
commit	val_loss	auroc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved — use 999.0 for crashes
3. auroc if available, else 0.0
4. peak GPU memory in GB (round to .1f) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_loss	auroc	memory_gb	status	description
a1b2c3d	0.6931	0.5000	8.2	keep	baseline: trial dataset, default config
b2c3d4e	0.6512	0.5830	8.4	keep	increase hidden_dim 128→256
c3d4e5f	0.7200	0.4800	8.2	discard	reduce num_heads 8→2
d4e5f6g	999.0	0.0	0.0	crash	double batch_size (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state and `results.tsv` to understand where you are.
2. Decide what to try next. Options:
   - **Config changes** (fast, safe): edit `configs/autoresearch.json` — lr, batch_size, hidden_dim, dropout, scheduler, etc.
   - **Architecture changes** (slower, riskier): edit model files — add layers, change attention, modify encoders, etc.
   - **Sampling changes**: edit sampling strategies — context window, num_hops, neighbor sampling.
3. Make the change. `git add -A && git commit -m "experiment: <description>"`
4. Run training:
   ```bash
   source /home/ubuntu/anaconda3/etc/profile.d/conda.sh && conda activate pyg28 && \
     cd /home/wangk/kumorfm-zhaoweizun && \
     python relbdata/train_on_relbench.py --config configs/autoresearch.json \
     > run.log 2>&1
   ```
5. Extract results: `grep "Epoch.*Val Loss" run.log | tail -1`
6. If grep is empty → crash. Run `tail -50 run.log` to diagnose.
7. Record results in `results.tsv` (do NOT commit results.tsv — keep it untracked).
8. If val_loss improved → **keep**: advance the branch.
9. If val_loss equal or worse → **discard**: `git reset --hard HEAD~1`

## Strategy guide

Start with config-only experiments (fast iteration), then move to architecture:

### Phase 1: Hyperparameter sweep (config changes only)
- Learning rate: try 1e-4, 5e-4, 1e-3, 5e-3
- Batch size: 32, 64, 128, 256
- Hidden dim: 64, 128, 256, 512
- Dropout: 0.0, 0.05, 0.1, 0.2
- Scheduler: cosine vs linear vs constant
- Warmup steps: 0, 100, 500, 1000

### Phase 2: Architecture experiments (model code changes)
- Number of RelGT layers: 1, 2, 4, 8
- Number of attention heads: 2, 4, 8, 16
- ICL module depth
- Encoder modifications
- Activation functions
- Normalization strategies

### Phase 3: Data/sampling experiments
- Different datasets (start small → scale up)
- Multi-task vs single-task
- Context window size
- Number of hops in subgraph sampling
- Sampling strategy: temporal, structural, mixed

### Phase 4: Scale up
Once you have a winning config on small data, scale:
- Increase `max_samples_per_epoch`
- Switch to larger datasets
- Extend `epochs`
- Verify gains hold at scale

## Important notes

- **Conda activation is REQUIRED** for every command. The Python in PATH is not the right one.
- **Two GPUs available**: `cuda:0` and `cuda:1`. Default uses `cuda:0`. For DDP, the training script handles multi-GPU automatically.
- **Existing outputs**: Previous experiment outputs are in `outputs/` and `relbench_outputs/`. Do not delete them.
- **The human may update RFM code independently**. Before starting a new experiment, always `git status` to check for upstream changes. If the human has pushed changes, `git pull` or rebase before continuing.
- **NEVER STOP**: Once the loop begins, do NOT pause to ask if you should continue. The human may be away. Run indefinitely until manually stopped.
- **Simplicity criterion**: All else equal, simpler is better. A tiny val_loss improvement that adds ugly complexity is not worth it. Removing code and getting equal results is a win.
