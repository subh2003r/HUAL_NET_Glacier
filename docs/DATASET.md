# Dataset format

The final HUAL-Net experiment uses a prepared HDF5 file containing six arrays:

```text
X_train
Y_train
X_val
Y_val
X_test
Y_test
```

The input arrays are stored in NHWC order:

```text
X: (N, 128, 128, 14)
Y: (N, 128, 128, 1)
```

The repository converts the inputs to NCHW format before passing them to PyTorch.

`Y` is a binary glacier mask with one channel.

## Multimodal input channels

The final input contains 14 channels:

| Modality | Channels |
|---|---:|
| Co-polarized SAR | 2 |
| Cross-polarized SAR | 2 |
| DEM | 2 |
| InSAR | 2 |
| Optical | 6 |
| **Total** | **14** |

The exact ordering of the channels should remain the same between preprocessing,
training and inference.

## Patch preparation

The final model operates on 128 × 128 patches.

The original preprocessing workflow extracts glacier-containing patches and removes
patches with negligible glacier pixels. The corresponding ground-truth mask is kept
with every input patch.

## Data sharing

The real research dataset is intentionally not included in this repository.

If the dataset cannot be redistributed, keep it outside the Git repository and provide
instructions in the manuscript for obtaining it.

A small synthetic dataset generator is provided under:

```text
examples/create_synthetic_dataset.py
```

This synthetic data is only for testing the code and is not used for the scientific
results.
