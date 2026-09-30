#!/usr/bin/env python
"""Create a tiny synthetic HDF5 dataset for testing the repository."""

from pathlib import Path
import h5py
import numpy as np


def make_split(n, h=64, w=64, c=17, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.random((n, h, w, c), dtype=np.float32)
    y = np.zeros((n, h, w, 1), dtype=np.float32)

    yy, xx = np.ogrid[:h, :w]
    for i in range(n):
        cx = rng.integers(w // 4, 3 * w // 4)
        cy = rng.integers(h // 4, 3 * h // 4)
        r = rng.integers(min(h, w) // 8, min(h, w) // 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        y[i, ..., 0] = mask.astype(np.float32)

    return x, y


def main():
    out = Path("data/synthetic_hual_demo.h5")
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as f:
        for name, n, seed in [("train", 8, 1), ("val", 2, 2), ("test", 2, 3)]:
            x, y = make_split(n=n, seed=seed)
            f.create_dataset(f"X_{name}", data=x)
            f.create_dataset(f"Y_{name}", data=y)

    print(f"Created {out}")


if __name__ == "__main__":
    main()
