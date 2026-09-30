from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .losses import DiceBCELoss


@torch.no_grad()
def predict_probability_maps(model, x_nchw, batch_size: int = 8, device=None):
    """Run one deterministic inference pass and return sigmoid probability maps."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.eval()

    outputs = []
    for start in range(0, len(x_nchw), batch_size):
        xb = x_nchw[start:start + batch_size].to(device)
        outputs.append(torch.sigmoid(model(xb)).cpu())
    return torch.cat(outputs, dim=0)


def compute_entropy_map(probs, eps: float = 1e-6):
    """Binary predictive entropy from one deterministic probability map."""
    return -(
        probs * torch.log(probs + eps)
        + (1 - probs) * torch.log(1 - probs + eps)
    )


def compute_patch_uncertainty_scores(entropy_maps):
    """Use mean pixel entropy as the patch uncertainty score."""
    return entropy_maps.mean(dim=(1, 2, 3)).cpu().numpy()


def select_top_k_patches(entropy_maps, k: int = 400):
    scores = compute_patch_uncertainty_scores(entropy_maps)
    k = min(k, len(scores))
    indices = np.argsort(-scores)[:k]
    return indices, scores


def build_hual_dataset(x_base, y_base, x_selected, y_selected):
    """Append selected uncertain patches to the base training set."""
    return (
        np.concatenate([x_base, x_selected], axis=0),
        np.concatenate([y_base, y_selected], axis=0),
    )


def retrain_on_hual_samples(
    model,
    x_hual,
    y_hual,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device=None,
):
    """Fine-tune the supplied model on the HUAL-augmented training set."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    x = torch.from_numpy(x_hual).float().permute(0, 3, 1, 2)
    y = torch.from_numpy(y_hual).float().permute(0, 3, 1, 2)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = DiceBCELoss()

    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0

        for xb, yb in tqdm(
            loader,
            desc=f"HUAL epoch {epoch + 1}/{epochs}",
            leave=False,
        ):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        history.append(running / max(len(loader), 1))

    return model, history
