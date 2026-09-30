# Final method configuration

This file records the method configuration to be used for the repository release.

## Input

The model uses 14 multimodal channels:

- 2 co-polarized SAR
- 2 cross-polarized SAR
- 2 DEM
- 2 InSAR
- 6 optical

Patch size:

```text
128 × 128
```

## Segmentation model

```text
U-Net
 └── ResNet34 encoder
 └── SCSE decoder attention
 └── 1-channel binary output
```

The multimodal model is trained without ImageNet encoder weights because the input
contains 14 channels rather than the standard three-channel RGB input.

## Uncertainty

The HUAL uncertainty score is deterministic predictive entropy.

```text
Input patch
    ↓
trained model
    ↓
sigmoid probability
    ↓
pixel-wise binary entropy
    ↓
mean entropy over patch
    ↓
uncertainty score
```

There are no repeated stochastic forward passes and no MC Dropout step.

## Patch selection

The patches are sorted from highest to lowest uncertainty.

The final configuration selects:

```text
K = 400
```

high-uncertainty patches.

## Retraining

The selected patches and their corresponding reference masks are added to the
training set and the segmentation model is fine-tuned.

This is the implementation that should be used consistently in the code,
figures, README and manuscript.
