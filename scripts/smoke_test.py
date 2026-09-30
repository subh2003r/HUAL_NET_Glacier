#!/usr/bin/env python
"""Small local test that does not need the glacier dataset or trained weights."""

import numpy as np
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import torch

from src.hual import compute_entropy_map, select_top_k_patches
from src.model import make_multimodal_unet


def main():
    n, h, w, c = 4, 64, 64, 17
    x = torch.randn(n, c, h, w)

    model = make_multimodal_unet(c, device="cpu")
    with torch.no_grad():
        probs = torch.sigmoid(model(x))

    entropy = compute_entropy_map(probs)
    selected, scores = select_top_k_patches(entropy, k=2)

    assert probs.shape == (n, 1, h, w)
    assert entropy.shape == probs.shape
    assert len(selected) == 2
    assert len(scores) == n

    print("Smoke test passed.")
    print("Input:", tuple(x.shape))
    print("Probability map:", tuple(probs.shape))
    print("Selected patches:", selected.tolist())


if __name__ == "__main__":
    main()
