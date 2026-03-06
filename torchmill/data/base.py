"""Base dataset interface and registry."""

from typing import Any, Dict, Optional, Type

from torch.utils.data import Dataset


_DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


class BaseDatasetConfig:
    """Marker base class for dataset-specific config objects."""


class DatasetRegistry:
    """Global registry mapping type names to Dataset classes."""

    @staticmethod
    def register(name: str, cls: Type[Dataset]) -> None:
        _DATASET_REGISTRY[name] = cls

    @staticmethod
    def get(name: str) -> Type[Dataset]:
        if name not in _DATASET_REGISTRY:
            raise KeyError(
                f"Dataset '{name}' not in registry. "
                f"Available: {list(_DATASET_REGISTRY.keys())}"
            )
        return _DATASET_REGISTRY[name]

    @staticmethod
    def list() -> list:
        return list(_DATASET_REGISTRY.keys())


def register_dataset(name: Optional[str] = None):
    """Decorator to register a Dataset class."""
    def decorator(cls: Type[Dataset]) -> Type[Dataset]:
        key = name if name is not None else cls.__name__
        DatasetRegistry.register(key, cls)
        return cls

    return decorator


def get_dataset(name: str, **kwargs: Any) -> Dataset:
    cls = DatasetRegistry.get(name)
    return cls(**kwargs)
