from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    X: np.ndarray
    Y: np.ndarray


def load_split_hdf5(path: str | Path) -> dict[str, SplitData]:
    """Load X_train/Y_train, X_val/Y_val and X_test/Y_test from an HDF5 file.

    Expected arrays:
        X_*: (N, H, W, C)
        Y_*: (N, H, W, 1)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    splits = {}
    with h5py.File(path, "r") as f:
        for name in ("train", "val", "test"):
            x_key = f"X_{name}"
            y_key = f"Y_{name}"
            if x_key not in f or y_key not in f:
                raise KeyError(
                    f"Expected '{x_key}' and '{y_key}' in {path}. "
                    f"Available keys: {list(f.keys())}"
                )
            x = np.asarray(f[x_key][:], dtype=np.float32)
            y = np.asarray(f[y_key][:], dtype=np.float32)

            if x.ndim != 4 or y.ndim != 4:
                raise ValueError(
                    f"{name}: expected X and Y to be 4-D arrays; "
                    f"got X={x.shape}, Y={y.shape}"
                )
            if y.shape[-1] != 1:
                raise ValueError(
                    f"{name}: expected a single-channel binary mask; got {y.shape}"
                )
            if x.shape[0] != y.shape[0] or x.shape[1:3] != y.shape[1:3]:
                raise ValueError(
                    f"{name}: X/Y dimensions do not match: X={x.shape}, Y={y.shape}"
                )

            splits[name] = SplitData(x, y)

    return splits


def to_torch(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert NHWC NumPy arrays to NCHW float tensors."""
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("Expected X and Y to be 4-D arrays.")
    x_t = torch.from_numpy(np.ascontiguousarray(x)).float().permute(0, 3, 1, 2)
    y_t = torch.from_numpy(np.ascontiguousarray(y)).float().permute(0, 3, 1, 2)
    return x_t, y_t


class GlacierDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.ascontiguousarray(x.astype(np.float32))
        self.y = np.ascontiguousarray(y.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.x[idx]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.y[idx]).permute(2, 0, 1).float()
        return image, mask
