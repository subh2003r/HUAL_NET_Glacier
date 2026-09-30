import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.bce_weight * self.bce(logits, targets)
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.6, focal_weight: float = 0.4):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, logits, targets):
        return (
            self.dw * self.dice(logits, targets)
            + self.fw * self.focal(logits, targets)
        )


class DiceFocalBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits, targets):
        return (
            0.5 * self.dice(logits, targets)
            + 0.3 * self.focal(logits, targets)
            + 0.2 * self.boundary(logits, targets)
        )


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        )
        self.register_buffer("sx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sy", sobel_y.view(1, 1, 3, 3))

    def edges(self, x):
        gx = F.conv2d(x, self.sx, padding=1)
        gy = F.conv2d(x, self.sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return F.l1_loss(self.edges(probs), self.edges(targets))
