import torch

from src.hual import compute_entropy_map, select_top_k_patches


def test_entropy_is_low_for_confident_predictions():
    probs = torch.tensor([[[[0.99, 0.01]]]])
    entropy = compute_entropy_map(probs)
    assert float(entropy.max()) < 0.1


def test_top_k_returns_requested_number():
    probs = torch.full((5, 1, 4, 4), 0.5)
    entropy = compute_entropy_map(probs)
    indices, scores = select_top_k_patches(entropy, k=3)
    assert len(indices) == 3
    assert len(scores) == 5
