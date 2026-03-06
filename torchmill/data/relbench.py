"""RelBench dataset adapter for temporal heterogeneous graph datasets."""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from .base import register_dataset


@register_dataset("relbench")
class RelBenchDataset(Dataset):
    """Adapter for RelBench (https://relbench.stanford.edu/) datasets.

    RelBench provides temporal heterogeneous graph datasets for relational ML.
    This adapter wraps the RelBench API so it fits the torchmill data pipeline.

    Requires the ``relbench`` package::

        pip install relbench

    Example config::

        data:
          type: relbench
          dataset_name: trial
          task_name: user-churn
          batch_size: 512
    """

    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        split: str = "train",
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import relbench  # noqa: F401
            from relbench import get_dataset as rb_get_dataset
        except ImportError as e:
            raise ImportError(
                "The 'relbench' package is required for RelBench datasets. "
                "Install it with: pip install relbench"
            ) from e

        self.dataset_name = dataset_name
        self.task_name = task_name
        self.split = split

        dataset = rb_get_dataset(name=dataset_name, download=True)
        task = dataset.get_task(task_name, download=True)

        table = task.get_table(split)
        self._data = table

        # Try to extract features and labels from the RelBench table
        import pandas as pd
        import numpy as np

        df = table.df
        # Labels
        if hasattr(task, "target_col") and task.target_col in df.columns:
            self.targets = torch.tensor(
                df[task.target_col].values.astype(np.float32)
            )
            df = df.drop(columns=[task.target_col])
        else:
            self.targets = None

        numeric = df.select_dtypes(include=[np.number])
        self.features = torch.tensor(numeric.values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.targets is not None:
            return x, self.targets[idx]
        return x
