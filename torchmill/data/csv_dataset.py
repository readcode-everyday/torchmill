"""CSV / Parquet dataset loader for generic tabular data."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .base import register_dataset


@register_dataset("csv")
class CSVDataset(Dataset):
    """Load a CSV or Parquet file as a PyTorch dataset.

    Numeric feature columns are used as inputs; ``target_column`` is the label.
    """

    def __init__(
        self,
        path: Union[str, Path],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        path = Path(path)
        if path.suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if target_column and target_column in df.columns:
            self.targets = torch.tensor(df[target_column].values, dtype=torch.float32)
            df = df.drop(columns=[target_column])
        else:
            self.targets = None

        if feature_columns:
            df = df[feature_columns]

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])
        self.features = torch.tensor(df.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.features[idx]
        if self.targets is not None:
            return x, self.targets[idx]
        return x

    @property
    def num_features(self) -> int:
        return self.features.shape[1]
