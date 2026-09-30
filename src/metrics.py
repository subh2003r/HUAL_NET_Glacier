import torch


def binarize(logits, threshold: float = 0.5):
    return (torch.sigmoid(logits) >= threshold).float()


def dice_score(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    preds = binarize(logits, threshold)
    dims = (2, 3)
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)
    return ((2 * intersection + eps) / (denominator + eps)).mean().item()


def iou_score(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    preds = binarize(logits, threshold)
    dims = (2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def precision_recall(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    preds = binarize(logits, threshold)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision.item(), recall.item()


def evaluate_batch(logits, targets, threshold: float = 0.5):
    precision, recall = precision_recall(logits, targets, threshold)
    return {
        "dice": dice_score(logits, targets, threshold),
        "iou": iou_score(logits, targets, threshold),
        "precision": precision,
        "recall": recall,
    }
