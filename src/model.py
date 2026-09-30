from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def make_multimodal_unet(in_channels: int, device: str | torch.device | None = None):
    """Create the ResNet34 U-Net with SCSE decoder attention used in the supplied work."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    )

    # Match the explicit initialization used in the supplied training script.
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    return model.to(device)


def load_checkpoint(model, checkpoint_path, device=None):
    if device is None:
        device = next(model.parameters()).device
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError(
            "Checkpoint must contain 'model_state_dict' or 'state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )
    model.load_state_dict(state_dict)
    return model, checkpoint
