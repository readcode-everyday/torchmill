"""Tests for model and dataset registries."""

import pytest
import torch
import torch.nn as nn

from torchmill.models.registry import (
    ModelRegistry,
    register_model,
    get_model,
    _REGISTRY,
)
from torchmill.data.base import DatasetRegistry, register_dataset, get_dataset


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def setup_method(self):
        # Snapshot registry to restore after each test
        self._original = dict(_REGISTRY)

    def teardown_method(self):
        _REGISTRY.clear()
        _REGISTRY.update(self._original)

    def test_register_and_get(self):
        class TinyNet(nn.Module):
            def forward(self, x):
                return x

        ModelRegistry.register("tiny", TinyNet)
        cls = ModelRegistry.get("tiny")
        assert cls is TinyNet

    def test_decorator_registration(self):
        @register_model("decorated_net")
        class DecoratedNet(nn.Module):
            def forward(self, x):
                return x

        assert "decorated_net" in ModelRegistry.list()

    def test_decorator_default_name(self):
        @register_model()
        class AutoNamedNet(nn.Module):
            def forward(self, x):
                return x

        assert "AutoNamedNet" in ModelRegistry.list()

    def test_get_model_from_registry(self):
        @register_model("simple_linear")
        class SimpleLinear(nn.Module):
            def __init__(self, dim=4):
                super().__init__()
                self.fc = nn.Linear(dim, dim)

            def forward(self, x):
                return self.fc(x)

        model = get_model("simple_linear", {"dim": 8})
        assert isinstance(model, SimpleLinear)

    def test_get_model_by_dotted_path(self):
        model = get_model("torch.nn.Linear", {"in_features": 4, "out_features": 2})
        assert isinstance(model, nn.Linear)
        assert model.in_features == 4

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            ModelRegistry.get("nonexistent_model_xyz")

    def test_invalid_dotted_path_raises(self):
        with pytest.raises(ValueError):
            get_model("NoDotsAtAll", {})

    def test_list_returns_registered(self):
        @register_model("listed_model")
        class ListedModel(nn.Module):
            def forward(self, x):
                return x

        assert "listed_model" in ModelRegistry.list()


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


class TestDatasetRegistry:
    def test_register_and_get_class(self):
        from torch.utils.data import Dataset

        @register_dataset("my_ds")
        class MyDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, i):
                return torch.zeros(4)

        cls = DatasetRegistry.get("my_ds")
        assert cls is MyDataset

    def test_get_dataset_instantiates(self):
        from torch.utils.data import Dataset

        @register_dataset("instant_ds")
        class InstantDS(Dataset):
            def __init__(self, size=5):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, i):
                return torch.zeros(2)

        ds = get_dataset("instant_ds", size=7)
        assert len(ds) == 7

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            DatasetRegistry.get("totally_unknown_ds_xyz")
