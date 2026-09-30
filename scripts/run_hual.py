#!/usr/bin/env python
"""Run the HUAL selection and retraining stage on a prepared HDF5 split file."""

import argparse
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

import numpy as np
import torch

from src.data import load_split_hdf5, to_torch
from src.hual import (a
    build_hual_dataset,
    compute_entropy_map,
    predict_probability_maps,
    retrain_on_hual_samples,
    select_top_k_patches,
)
from src.model import load_checkpoint, make_multimodal_unet
from src.metrics import evaluate_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="HDF5 file with X/Y train/val/test arrays.")
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint.")
    parser.add_argument("--output-dir", default="results/hual_run")
    parser.add_argument("--top-k", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_split_hdf5(args.data)

    x_train = splits["train"].X
    y_train = splits["train"].Y
    x_test = splits["test"].X
    y_test = splits["test"].Y

    x_train_t, _ = to_torch(x_train, y_train)
    model = make_multimodal_unet(x_train.shape[-1], device=device)
    model, checkpoint = load_checkpoint(model, args.checkpoint, device=device)

    # The final HUAL implementation uses one deterministic prediction.
    probs = predict_probability_maps(
        model, x_train_t, batch_size=args.batch_size, device=device
    )
    entropy = compute_entropy_map(probs)

    selected, scores = select_top_k_patches(entropy, k=args.top_k)
    np.save(out / "selected_patch_indices.npy", selected)
    np.save(out / "patch_uncertainty_scores.npy", scores)

    x_selected = x_train[selected]
    y_selected = y_train[selected]
    x_hual, y_hual = build_hual_dataset(
        x_train, y_train, x_selected, y_selected
    )

    np.save(out / "hual_train_indices.npy", selected)
    np.save(out / "hual_train_X.npy", x_hual)
    np.save(out / "hual_train_Y.npy", y_hual)

    model, history = retrain_on_hual_samples(
        model,
        x_hual,
        y_hual,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    # Evaluate sample-by-sample batches on the test split.
    model.eval()
    test_x, test_y = to_torch(x_test, y_test)
    dices, ious, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for start in range(0, len(test_x), args.batch_size):
            xb = test_x[start:start + args.batch_size].to(device)
            yb = test_y[start:start + args.batch_size].to(device)
            logits = model(xb)
            m = evaluate_batch(logits, yb, threshold=args.threshold)
            dices.append(m["dice"])
            ious.append(m["iou"])
            precisions.append(m["precision"])
            recalls.append(m["recall"])

    metrics = {
        "dice_mean": float(np.mean(dices)),
        "iou_mean": float(np.mean(ious)),
        "precision_mean": float(np.mean(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "selected_patches": int(len(selected)),
        "train_patches_before_hual": int(len(x_train)),
        "train_patches_after_hual": int(len(x_hual)),
        "hual_fraction_of_base_train": float(len(selected) / len(x_train)),
        "hual_loss_history": [float(v) for v in history],
    }

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {"model_state_dict": model.state_dict()},
        out / "hual_model.pth",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
