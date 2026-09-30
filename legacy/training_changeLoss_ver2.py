#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import math
import time

import numpy as np
import h5py



from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes


from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

import segmentation_models_pytorch as smp

import timm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.mkldnn.enabled = (DEVICE.type == "cpu")


# In[2]:


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# In[3]:


def split_regions(region_keys):

    train_r, temp_r = train_test_split(region_keys, test_size=0.1, random_state=42)
    val_r, test_r = train_test_split(temp_r, test_size=0.5, random_state=42)

    return train_r, val_r, test_r


# In[4]:


def extract_from_regions(f, region_list, patch_size=384):

    modalities = {
        "co_pol_sar": 2,
        "cross_pol_sar": 2,
        "dem": 2,
        "in_sar": 2,
        "optical": 6
    }

    TOTAL_CHANNELS = sum(modalities.values())

    X = []
    Y = []

    stride = patch_size // 2

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

            if data.ndim == 3 and data.shape[0] == ch:
                data = np.transpose(data, (1,2,0))

            sample[...,offset:offset+ch] = data
            offset += ch


        boundary = binary_dilation(mask.astype(bool)) ^ mask.astype(bool)


        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):

                img_patch = sample[i:i+patch_size, j:j+patch_size]
                mask_patch = mask[i:i+patch_size, j:j+patch_size]
                boundary_patch = boundary[i:i+patch_size, j:j+patch_size]

                glacier_pixels = np.sum(mask_patch)

                ratio = glacier_pixels / (patch_size*patch_size)

                keep = False

                if ratio > 0.05:
                    keep = True
                elif ratio > 0.01:
                    keep = np.random.rand() < 0.8
                else:
                    keep = np.random.rand() < 0.4
        
                # IMPORTANT: force keep boundary patches
                if np.sum(boundary_patch) > 20:
                    keep = True
        
                if keep:
                    X.append(img_patch)
                    Y.append(mask_patch[..., None])

    return np.stack(X), np.stack(Y)


# In[5]:


def create_datasets_from_hdf5(hdf5_path):

    with h5py.File(hdf5_path,'r') as f:

        regions = list(f.keys())

        train_r, val_r, test_r = split_regions(regions)

        X_train, Y_train = extract_from_regions(f, train_r)

    return X_train, Y_train


# In[6]:


def augment_sample(image, mask):

    if np.random.rand() < 0.5:
        image = np.flip(image,axis=1)
        mask = np.flip(mask,axis=1)

    if np.random.rand() < 0.5:
        image = np.flip(image,axis=0)
        mask = np.flip(mask,axis=0)

    k = np.random.randint(0,4)

    image = np.rot90(image,k)
    mask = np.rot90(mask,k)

    return image.copy(), mask.copy()


# In[7]:


class GlacierDataset(Dataset):

    def __init__(self,X,Y,augment=False):

        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):

        x = self.X[idx]
        y = self.Y[idx]

        if self.augment:
            x,y = augment_sample(x,y)

        x = torch.tensor(x).permute(2,0,1).float()
        y = torch.tensor(y).permute(2,0,1).float()

        return x,y


# In[8]:


def make_transformer_model(in_channels):

    model = smp.Unet(
        encoder_name="mit_b3",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None,
        decoder_attention_type="scse"
    )

    return model.to(DEVICE)


# In[9]:


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,logits,targets):

        probs = torch.sigmoid(logits)

        inter = (probs*targets).sum((2,3))
        denom = probs.sum((2,3)) + targets.sum((2,3))

        dice = (2*inter+1e-6)/(denom+1e-6)

        return 1-dice.mean()


# In[10]:


class FocalLoss(nn.Module):

    def __init__(self,alpha=0.25,gamma=2):
        super().__init__()
        self.alpha=alpha
        self.gamma=gamma

    def forward(self,logits,targets):

        bce = F.binary_cross_entropy_with_logits(logits,targets,reduction="none")

        probs = torch.sigmoid(logits)

        pt = torch.where(targets==1,probs,1-probs)

        focal = self.alpha*(1-pt)**self.gamma*bce

        return focal.mean()


# In[11]:


class BoundaryLoss(nn.Module):

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32)

        self.register_buffer("sx",sobel_x.view(1,1,3,3))
        self.register_buffer("sy",sobel_y.view(1,1,3,3))


    def edges(self,x):

        gx = F.conv2d(x,self.sx,padding=1)
        gy = F.conv2d(x,self.sy,padding=1)

        return torch.sqrt(gx**2 + gy**2 + 1e-6)


    def forward(self,logits,targets):

        probs = torch.sigmoid(logits)

        pe = self.edges(probs)
        te = self.edges(targets)

        return F.l1_loss(pe,te)


# In[12]:


class CombinedLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.boundary = BoundaryLoss()

    def forward(self,logits,targets):

        return (
            0.5*self.dice(logits,targets) +
            0.3*self.focal(logits,targets) +
            0.2*self.boundary(logits,targets)
        )


# In[13]:


def safe_torch_save(obj, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    temp_path = save_path + ".tmp"

    if "model_state_dict" in obj:
        obj["model_state_dict"] = {
            k: v.cpu() for k, v in obj["model_state_dict"].items()
        }

    torch.save(obj, temp_path)

    os.replace(temp_path, save_path)


# In[20]:


def train_model(train_loader, input_channels, epochs=2, lr=1e-4):
    train_losses = []
    learning_rates = []
    gradient_norms = []
    epoch_times = []
    batch_losses = []
    
    model = make_transformer_model(input_channels)

    loss_fn = CombinedLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    print("\nStarting Training...\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()

        running_loss = 0.0

        # progress_bar = tqdm(train_loader)

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):

                logits = model(images)
                logits = torch.clamp(logits, -20, 20)

                loss = loss_fn(logits, masks)

                # DEBUG Check
                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN loss detected")
                    print("Batch index:", batch_idx)
                    print("Image stats:", images.min().item(), images.max().item())
                    print("Mask stats:", masks.min().item(), masks.max().item())
                    break
                    
            # NaN detection
            if torch.isnan(loss) or torch.isinf(loss):
            
                print("\nNaN detected!")
                print("Batch:", batch_idx)
                print("Images min/max:", images.min().item(), images.max().item())
                print("Masks min/max:", masks.min().item(), masks.max().item())
                print("Logits min/max:", logits.min().item(), logits.max().item())
            
                break

            scaler.scale(loss).backward()

            # Gradient clipping (important for stability)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_norms.append(grad_norm.item())

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_losses.append(loss.item())

            # progress_bar.set_description(
            #     f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}"
            # )

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)           

        print(f"\nEpoch {epoch+1} Average Loss: {epoch_loss:.4f}\n")

    print("\nTraining Complete.\n")

    # Final model
    safe_torch_save(
                    {
                        "model_state_dict": model.state_dict()                    },
                    f"/scratch/{os.environ['USER']}/glacier_models/updated_changeloss_ver2.pth"
    )

    return model, {
        "train_losses": train_losses,
        "learning_rates": learning_rates,
        "gradient_norms": gradient_norms,
        "epoch_times": epoch_times,
        "batch_losses": batch_losses
    }


# In[21]:


hdf5_path = f"/scratch/{os.getenv('USER')}/gsgm_train_DEMO.hdf5"

X_train, Y_train = create_datasets_from_hdf5(hdf5_path)
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

train_dataset = GlacierDataset(X_train, Y_train, augment=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

model, training_metrics  = train_model(
    train_loader,
    input_channels=14,
    epochs=185
)

print("train_loss:", training_metrics["train_losses"])
print("learning_rates:", training_metrics["learning_rates"])
print("gradient_norms:", training_metrics["gradient_norms"])
print("epoch_times:", training_metrics["epoch_times"])
print("batch_losses:", training_metrics["batch_losses"])


# In[ ]:




