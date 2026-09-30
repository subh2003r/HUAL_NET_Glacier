#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

hdf5_path = f"/scratch/{os.getenv('USER')}/train.hdf5"


# In[2]:


import torch


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


print(torch.cuda.is_available())
print(torch.__version__)


# In[5]:


print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())


# In[6]:


import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_nl_means, estimate_sigma

# -----------------------------------------------------------
# REGION SPLIT
# -----------------------------------------------------------

def split_regions(region_keys, seed=42):
    train_r, temp_r = train_test_split(region_keys, test_size=0.2, random_state=seed)
    val_r, test_r = train_test_split(temp_r, test_size=0.1, random_state=seed)
    return train_r, val_r, test_r


# -----------------------------------------------------------
# PATCH EXTRACTION FROM GIVEN REGIONS (RAW STACKING)
# -----------------------------------------------------------

def extract_from_regions(f, region_list, patch_size, min_mask_pixels):

    modalities = {
        "co_pol_sar": 2,
        "cross_pol_sar": 2,
        "dem": 2,
        "in_sar": 2,
        "optical": 6
    }

    TOTAL_CHANNELS = sum(modalities.values())
    X, Y = [], []

    for region_key in region_list:
        region = f[region_key]

        raw_mask = np.array(region["outlines"], dtype=np.float32)
        if raw_mask.ndim == 3:
            raw_mask = np.argmax(raw_mask, axis=-1)

        mask = (raw_mask > 0).astype(np.float32)
        H, W = mask.shape

        sample = np.zeros((H, W, TOTAL_CHANNELS), dtype=np.float32)

        offset = 0
        for name, ch in modalities.items():

            if name not in region:
                offset += ch
                continue

            data = np.array(region[name], dtype=np.float32)

            # Convert (C,H,W) → (H,W,C)
            if data.ndim == 3 and data.shape[0] == ch:
                data = np.transpose(data, (1, 2, 0))

            sample[..., offset:offset+ch] = data
            offset += ch

        # -------- BALANCED PATCHES --------
        for i in range(0, H - patch_size + 1, patch_size // 2):
            for j in range(0, W - patch_size + 1, patch_size // 2):

                img = sample[i:i+patch_size, j:j+patch_size]
                m = mask[i:i+patch_size, j:j+patch_size]

                g = np.sum(m)

                if g < min_mask_pixels:
                    if np.random.rand() > 0.2:
                        continue

                X.append(img)
                Y.append(m[..., None])

    return np.stack(X), np.stack(Y)


# -----------------------------------------------------------
# MASTER FUNCTION
# -----------------------------------------------------------

def create_datasets_from_hdf5(hdf5_path, patch_size=256, min_mask_pixels=10):

    with h5py.File(hdf5_path, 'r') as f:
        regions = list(f.keys())
        train_r, val_r, test_r = split_regions(regions)

        print(f"Regions → Train:{len(train_r)} Val:{len(val_r)} Test:{len(test_r)}")

        X_train, Y_train = extract_from_regions(f, train_r, patch_size, min_mask_pixels)
        X_val, Y_val     = extract_from_regions(f, val_r, patch_size, min_mask_pixels)
        X_test, Y_test   = extract_from_regions(f, test_r, patch_size, min_mask_pixels)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# In[8]:


X_train, Y_train, X_val, Y_val, X_test, Y_test = create_datasets_from_hdf5(hdf5_path)


# In[9]:

# Create scratch directory
scratch_dir = f"/scratch/{os.environ['USER']}/glacier_changeLoss_project"
os.makedirs(scratch_dir, exist_ok=True)


np.save(os.path.join(scratch_dir, "X_Ltrain.npy"), X_train)
np.save(os.path.join(scratch_dir, "Y_Ltrain.npy"), Y_train)
np.save(os.path.join(scratch_dir, "X_Lval.npy"), X_val)
np.save(os.path.join(scratch_dir, "Y_Lval.npy"), Y_val)
np.save(os.path.join(scratch_dir, "X_Ltest.npy"), X_test)
np.save(os.path.join(scratch_dir, "Y_Ltest.npy"), Y_test)



# In[10]:


# X_train = np.load("X_Ltrain.npy")
# Y_train = np.load("Y_Ltrain.npy")

# X_val = np.load("X_Lval.npy")
# Y_val = np.load("Y_Lval.npy")

# X_test = np.load("X_Ltest.npy")
# Y_test = np.load("Y_Ltest.npy")


# In[11]:


import torch
from torch.utils.data import Dataset

class GlacierDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx]).permute(2, 0, 1).float()  # (C,H,W)
        y = torch.tensor(self.Y[idx]).permute(2, 0, 1).float()  # (1,H,W)
        return x, y


# In[12]:


from torch.utils.data import DataLoader

train_dataset = GlacierDataset(X_train, Y_train)
val_dataset   = GlacierDataset(X_val, Y_val)
test_dataset  = GlacierDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)


# In[13]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.mkldnn.enabled = (DEVICE.type == "cpu")
torch.backends.cudnn.benchmark = True


# In[14]:


from tqdm import tqdm


# In[15]:


def make_strong_unet(in_channels: int):
    """
    Multimodal-ready U-Net with SCSE attention.
    Safe for 16-channel glacier inputs.
    """

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,          # MUST be None for >3 channels
        in_channels=in_channels,       # e.g. 14 or 16
        classes=1,
        activation=None,
        decoder_attention_type="scse",
        encoder_depth=5,
        decoder_channels=[256,128,64,32,16]
    )

    # Important weight initialization since no ImageNet weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    return model.to(DEVICE)


# In[16]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
#                    LOSS FUNCTIONS
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        dims = (2, 3)  # per-image Dice

        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()


# ------------------------------------------------------------
# FOCAL LOSS (Stable Version for Batch Size 2)
# ------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal = self.alpha * (1 - pt) ** self.gamma * bce

        return focal.mean()


# ------------------------------------------------------------
# DICE + FOCAL COMBINED LOSS
# ------------------------------------------------------------

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.6, focal_weight: float = 0.4):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)

        return self.dw * dice_loss + self.fw * focal_loss


# ------------------------------------------------------------
# GET LOSS FUNCTION
# ------------------------------------------------------------

def get_losses(device):
    """
    Returns Combined Dice + Focal loss moved to device.
    """
    loss_fn = CombinedLoss().to(device)
    return loss_fn




# ============================================================
#                        METRICS
# ============================================================

def _get_preds_from_logits(
    logits: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Converts logits → probabilities → binary predictions
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).float().contiguous()


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    threshold: float = 0.5
) -> float:
    """
    Dice score computed properly from logits.
    """
    preds = _get_preds_from_logits(logits, threshold)

    dims = (2, 3)

    inter = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * inter + eps) / (denom + eps)

    return dice.mean().item()


def iou_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    threshold: float = 0.5
) -> float:
    """
    IoU score computed from logits.
    """
    preds = _get_preds_from_logits(logits, threshold)

    dims = (2, 3)

    inter = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - inter

    iou = (inter + eps) / (union + eps)

    return iou.mean().item()


def precision_recall_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    threshold: float = 0.5
) -> tuple[float, float]:
    """
    Precision and Recall for segmentation.
    """
    preds = _get_preds_from_logits(logits, threshold)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return precision.item(), recall.item()


# In[17]:


import torch

def validate_model_on_thresholds(model, val_loader, device):

    model.eval()

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    best_dice = -1.0
    best_threshold = 0.5
    best_stats = {}

    with torch.no_grad():

        for th in thresholds:

            dices = []
            ious = []
            precs = []
            recs = []

            for images, masks in val_loader:

                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)

                dices.append(dice_score_from_logits(logits, masks, threshold=th))
                ious.append(iou_score_from_logits(logits, masks, threshold=th))
                p, r = precision_recall_from_logits(logits, masks, threshold=th)

                precs.append(p)
                recs.append(r)

            if len(dices) == 0:
                continue

            mean_dice = sum(dices) / len(dices)

            if mean_dice > best_dice:
                best_dice = mean_dice
                best_threshold = th
                best_stats = {
                    "dice": mean_dice,
                    "iou": sum(ious) / len(ious),
                    "precision": sum(precs) / len(precs),
                    "recall": sum(recs) / len(recs),
                }

    model.train()

    return best_threshold, best_stats


# In[18]:


from typing import Dict, Any
import torch
import torch.optim as optim


def safe_torch_save(obj, save_path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Temporary file path
    temp_path = save_path + ".tmp"

    # Move tensors to CPU (safer for distributed/HPC systems)
    if "model_state_dict" in obj:
        obj["model_state_dict"] = {
            k: v.cpu() for k, v in obj["model_state_dict"].items()
        }

    # Save to temp file first
    torch.save(obj, temp_path)

    # Atomically replace
    os.replace(temp_path, save_path)


def train_unet_model(
    train_loader,
    val_loader,
    input_channels: int,
    num_epochs: int = 150,          # ↑ increase since focal needs more epochs
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    validate_every: int = 3,
    best_model_save_path: str = "unet_lossNew.pth",
    patience: int = 8,             # slightly increased
    min_delta: float = 1e-4,
    device: str = None
) -> Dict[str, Any]:

    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on {device}")

    model = make_strong_unet(input_channels).to(device)

    # NEW LOSS
    criterion = get_losses(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_dice = -1.0
    best_threshold = 0.5
    no_improve = 0

    train_losses = []
    val_dices = []

    print("\n" + "="*100)
    print("Epoch | Train Loss | LR | Val Dice | IoU | Prec | Rec | Th | Status")
    print("="*100)

    for epoch in range(1, num_epochs + 1):

        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        train_loss_avg = epoch_loss / max(len(train_loader), 1)
        train_losses.append(train_loss_avg)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss_avg:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")

        # ---------------- VALIDATION ----------------
        if epoch % validate_every == 0 or epoch == num_epochs:

            threshold, val_stats = validate_model_on_thresholds(model, val_loader, device)

            val_dice = val_stats["dice"]
            val_iou = val_stats["iou"]
            val_prec = val_stats["precision"]
            val_rec = val_stats["recall"]

            val_dices.append(val_dice)

            scheduler.step(val_dice)

            status = "BEST" if val_dice > best_val_dice + min_delta else "✔"

            print(f"{epoch:02d} | {train_loss_avg:.4f} | "
                  f"{optimizer.param_groups[0]['lr']:.1e} | "
                  f"{val_dice:.3f} | {val_iou:.3f} | "
                  f"{val_prec:.3f} | {val_rec:.3f} | "
                  f"{threshold:.2f} | {status}")

            if val_dice > best_val_dice + min_delta:

                best_val_dice = val_dice
                best_threshold = threshold

                safe_torch_save({
                    'model_state_dict': model.state_dict(),
                    'best_threshold': best_threshold,
                    'val_dice': best_val_dice
                }, best_model_save_path)

                no_improve = 0
            else:
                no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\nTRAINING COMPLETE")
    print(f"Best Dice: {best_val_dice:.4f} @ threshold {best_threshold}")

    return {
        'best_model_path': best_model_save_path,
        'best_dice': best_val_dice,
        'best_threshold': best_threshold,
        'train_losses': train_losses,
        'val_dices': val_dices,
    }


# In[1]:


input_channels = train_loader.dataset[0][0].shape[0]

results = train_unet_model(
    train_loader=train_loader,
    val_loader=val_loader,
    input_channels=input_channels,
    num_epochs=150,                 # important for focal
    learning_rate=1e-4,
    weight_decay=1e-4,
    validate_every=3,
    best_model_save_path=f"/scratch/{os.environ['USER']}/glacier_models/chageLoss_best_model.pth",
    patience=8,
    device="cuda"  # or leave None for auto-detect
)


# In[ ]:




