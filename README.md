# torchmill

A standalone, model-agnostic PyTorch training framework. Provide a model, a dataset config, and hyperparameters — torchmill handles the rest.

## Features

- **Config-driven** — single YAML file controls everything
- **Model-agnostic** — register any `nn.Module` by import path or name
- **Production-ready** — AMP mixed precision, gradient clipping, DDP multi-GPU, checkpointing
- **Extensible callbacks** — early stopping, checkpointing, TensorBoard + console logging
- **Multiple data sources** — CSV/Parquet and RelBench graph datasets
- **CLI first** — `torchmill train --config config.yaml`

## Installation

```bash
pip install -e .
# with RelBench support:
pip install -e ".[relbench]"
```

## Quick start

### 1. Define your model

```python
# mymodels.py
import torch.nn as nn
from torchmill.models import register_model

@register_model("my_mlp")
class MLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
```

### 2. Write a config

```yaml
# config.yaml
model:
  class: mymodels.MLP
  args:
    input_dim: 10
    hidden_dim: 128
    output_dim: 1

data:
  type: csv
  train_path: ./data/train.csv
  val_path: ./data/val.csv
  target_column: label
  batch_size: 64
  num_workers: 4

training:
  epochs: 100
  seed: 42
  device: auto          # auto-detects CUDA / MPS / CPU
  task: regression      # or: classification, auto
  amp: true
  gradient_clip: 1.0
  optimizer:
    type: adamw
    lr: 0.001
    weight_decay: 0.01
  scheduler:
    type: cosine
    warmup_steps: 200

callbacks:
  checkpoint:
    save_dir: ./checkpoints
    save_top_k: 3
    monitor: val_loss
    mode: min
  early_stopping:
    enabled: true
    patience: 10
    monitor: val_loss
  logging:
    tensorboard: true
    log_dir: ./runs
    log_every_n_steps: 10
```

### 3. Train

```bash
torchmill train --config config.yaml
```

Validate a config without training:

```bash
torchmill validate --config config.yaml
```

## Python API

```python
from torchmill.config import TorchmillConfig
from torchmill.trainer import Trainer
from torchmill.models import get_model
from torchmill.data import CSVDataset

config = TorchmillConfig.from_yaml("config.yaml")
model  = get_model(config.model.class_path, config.model.args)

train_ds = CSVDataset("data/train.csv", target_column="label")
val_ds   = CSVDataset("data/val.csv",   target_column="label")

trainer = Trainer(config)
metrics = trainer.fit(model, train_ds, val_ds)

test_ds      = CSVDataset("data/test.csv", target_column="label")
test_metrics = trainer.evaluate(model, test_ds)
```

## Config reference

### `model`
| Key | Type | Description |
|-----|------|-------------|
| `class` | str | Dotted import path **or** registry name of an `nn.Module` |
| `args` | dict | Keyword arguments forwarded to the model constructor |

### `data`
| Key | Default | Description |
|-----|---------|-------------|
| `type` | `csv` | `csv` \| `relbench` \| `custom` |
| `train_path` | — | CSV/Parquet path (required for `type: csv`) |
| `val_path` | — | Validation split path |
| `test_path` | — | Test split path |
| `target_column` | — | Column name for labels |
| `batch_size` | `32` | DataLoader batch size |
| `num_workers` | `0` | DataLoader worker processes |
| `dataset_name` | — | RelBench dataset name (required for `type: relbench`) |
| `task_name` | — | RelBench task name |

### `training`
| Key | Default | Description |
|-----|---------|-------------|
| `epochs` | `100` | Total training epochs |
| `seed` | `42` | Random seed |
| `device` | `auto` | `auto` / `cpu` / `cuda` / `mps` / `cuda:N` |
| `task` | `auto` | `classification` \| `regression` \| `auto` |
| `amp` | `false` | Mixed precision (CUDA only) |
| `gradient_clip` | `null` | Max gradient norm |
| `optimizer.type` | `adamw` | `adam` \| `adamw` \| `sgd` |
| `optimizer.lr` | `0.001` | Learning rate |
| `scheduler.type` | `none` | `cosine` \| `linear` \| `step` \| `exponential` \| `none` |
| `scheduler.warmup_steps` | `0` | Linear warmup steps |
| `ddp` | `false` | Enable DistributedDataParallel |

### `callbacks.checkpoint`
| Key | Default | Description |
|-----|---------|-------------|
| `save_dir` | `./checkpoints` | Directory for checkpoint files |
| `save_top_k` | `3` | Keep top-K checkpoints (0 = keep all) |
| `monitor` | `val_loss` | Metric to monitor |
| `mode` | `min` | `min` \| `max` |
| `save_every_n_epochs` | `null` | Also save every N epochs |

### `callbacks.early_stopping`
| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable early stopping |
| `patience` | `10` | Epochs with no improvement before stopping |
| `monitor` | `val_loss` | Metric to monitor |
| `mode` | `min` | `min` \| `max` |
| `min_delta` | `0.0` | Minimum improvement threshold |

### `callbacks.logging`
| Key | Default | Description |
|-----|---------|-------------|
| `tensorboard` | `true` | Write TensorBoard summaries |
| `log_dir` | `./runs` | TensorBoard log directory |
| `log_every_n_steps` | `10` | Step-level log frequency |
| `console` | `true` | Print metrics to stdout |

## Custom datasets

```python
from torch.utils.data import Dataset
from torchmill.data import register_dataset

@register_dataset("my_dataset")
class MyDataset(Dataset):
    def __init__(self, path, **kwargs):
        ...
    def __len__(self): ...
    def __getitem__(self, i): ...
```

Then in your config:

```yaml
data:
  type: custom
```

And instantiate manually in your training script, passing the dataset to `trainer.fit()`.

## Multi-GPU (DDP)

```yaml
training:
  ddp: true
  ddp_backend: nccl
```

Launch with:

```bash
torchrun --nproc_per_node=4 -m torchmill.cli train --config config.yaml
```

## Loading a checkpoint

```python
from torchmill.callbacks.checkpoint import CheckpointCallback
import torch.nn as nn

model = nn.Linear(10, 1)
state = CheckpointCallback.load_checkpoint("checkpoints/best.pt", model)
print(f"Loaded from epoch {state['epoch']}, metrics: {state['metrics']}")
```

## Project structure

```
torchmill/
├── torchmill/
│   ├── __init__.py           # version
│   ├── cli.py                # CLI entry point
│   ├── config.py             # Pydantic config schema
│   ├── trainer.py            # Core training loop
│   ├── data/
│   │   ├── base.py           # Dataset registry
│   │   ├── csv_dataset.py    # CSV/Parquet loader
│   │   └── relbench.py       # RelBench adapter
│   ├── models/
│   │   └── registry.py       # Model registry
│   ├── callbacks/
│   │   ├── checkpoint.py
│   │   ├── early_stopping.py
│   │   └── logging.py
│   └── utils/
│       └── device.py         # Device auto-detection
├── configs/
│   └── example.yaml
├── tests/
│   ├── test_trainer.py
│   ├── test_config.py
│   └── test_registry.py
├── setup.py
└── requirements.txt
```

## License

MIT
