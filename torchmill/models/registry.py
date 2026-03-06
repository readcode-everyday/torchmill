"""Model registry — register any nn.Module by name for config-driven loading."""

import importlib
from typing import Any, Dict, Optional, Type

import torch.nn as nn

_REGISTRY: Dict[str, Type[nn.Module]] = {}


class ModelRegistry:
    """Global registry mapping string names to nn.Module classes."""

    @staticmethod
    def register(name: str, cls: Type[nn.Module]) -> None:
        _REGISTRY[name] = cls

    @staticmethod
    def get(name: str) -> Type[nn.Module]:
        if name not in _REGISTRY:
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available: {list(_REGISTRY.keys())}"
            )
        return _REGISTRY[name]

    @staticmethod
    def list() -> list:
        return list(_REGISTRY.keys())


def register_model(name: Optional[str] = None):
    """Decorator to register an nn.Module class.

    Usage::

        @register_model("my_model")
        class MyModel(nn.Module):
            ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name if name is not None else cls.__name__
        ModelRegistry.register(key, cls)
        return cls

    return decorator


def get_model(class_path: str, args: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Instantiate a model from a dotted class path or registry name.

    Args:
        class_path: Either a registry name (e.g. "my_model") or an importable
                    dotted path (e.g. "mypackage.models.MyModel").
        args: Keyword arguments forwarded to the model constructor.
    """
    args = args or {}

    # Try registry first
    if class_path in _REGISTRY:
        return _REGISTRY[class_path](**args)

    # Fall back to dynamic import
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"'{class_path}' is not in the registry and is not a valid "
            "dotted import path (expected 'module.ClassName')."
        )
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**args)
