"""GPU/CPU/MPS auto-detection and reproducibility utilities."""

import random
import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    """Return the best available device.

    Args:
        device: "auto" (default), "cpu", "cuda", "mps", or a specific CUDA
                device string like "cuda:1".
    """
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
