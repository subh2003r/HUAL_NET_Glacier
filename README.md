# HUAL-Net: Multimodal Glacier Segmentation with Uncertainty-Aware Retraining

This repository contains the code used for the HUAL-Net glacier segmentation work.

The main idea is simple: first train a multimodal segmentation model, then use its
prediction uncertainty to find the patches where the model is most confused. Those
patches are added back to the training set and the model is fine-tuned on the
combined data.

The work uses a U-Net with a ResNet34 encoder and SCSE decoder attention.
The multimodal input is stored as a channel stack, and the HUAL stage ranks patches
using mean pixel-wise predictive entropy.

## What is in this repository?

- `src/` - cleaned, reusable implementation of the HUAL stage and supporting model,
  loss, metric and data-loading code.
- `scripts/run_hual.py` - command-line entry point for the HUAL stage.
- `examples/` - creates a small synthetic HDF5 dataset for testing.
- `scripts/smoke_test.py` - quick test that runs without the real glacier dataset.
- `notebooks/` - the original HUAL notebook supplied with the research work.
- `legacy/` - the original Python scripts supplied during development. They are kept
  for provenance and are not the recommended entry point.
- `docs/` - dataset, reproducibility and user documentation.

## Experimental setup

The repository follows the final experimental setup used for the work:

- **14 input channels**
- **Deterministic predictive entropy** for uncertainty estimation
- **U-Net with ResNet34 encoder**
- **SCSE decoder attention**
- **128 × 128 input patches**
- **400 high-uncertainty patches** selected in the HUAL stage

The 14-channel multimodal input consists of:

- 2 co-polarized SAR channels
- 2 cross-polarized SAR channels
- 2 DEM channels
- 2 InSAR channels
- 6 optical channels

The uncertainty stage uses one deterministic forward pass. The sigmoid probability
map is converted into a binary predictive entropy map, and the mean entropy of each
patch is used as its uncertainty score. The highest-uncertainty patches are selected
for the HUAL retraining stage.

## Installation

Create a clean Python environment and install the packages in `requirements.txt`.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can depend on the CUDA version of the machine. If the
generic PyPI installation is not appropriate for your GPU system, install the
matching PyTorch build first and then install the remaining requirements.

## Quick test

The following does not require the real glacier dataset.

```bash
python scripts/smoke_test.py
```

You can also create a small synthetic HDF5 file:

```bash
python examples/create_synthetic_dataset.py
```

The synthetic data is only for checking the repository and is not used for the
scientific results in the paper.

## Dataset

The HUAL notebook uses a prepared HDF5 file containing:

```text
X_train, Y_train
X_val,   Y_val
X_test,  Y_test
```

with the current supplied notebook showing shapes:

```text
X_train: (547, 128, 128, 17)
Y_train: (547, 128, 128, 1)

X_val:   (117, 128, 128, 17)
Y_val:   (117, 128, 128, 1)

X_test:  (118, 128, 128, 17)
Y_test:  (118, 128, 128, 1)
```

See `docs/DATASET.md` before uploading any real data. Large or restricted datasets
should not be committed to GitHub.

## HUAL workflow

The current implementation follows these steps:

1. Load the trained base segmentation model.
2. Run the model on the training patches.
3. Convert the output logits to probabilities.
4. Compute a pixel-wise binary entropy map.
5. Average entropy over each patch to obtain a patch uncertainty score.
6. Sort patches by uncertainty.
7. Select the top `K` patches (400 in the supplied notebook).
8. Append those patches and their corresponding ground-truth masks to the base
   training set. In the experiment this represents the extra annotation stage
   using the existing ground truth as the annotation simulation.
9. Fine-tune the model on the combined training set.
10. Evaluate the fine-tuned model on the held-out test split.

## Run HUAL

After preparing the real HDF5 split file and base model checkpoint:

```bash
python scripts/run_hual.py \
    --data /path/to/gsgm_splits_128.h5 \
    --checkpoint /path/to/unet_best_glacier.pth \
    --top-k 400 \
    --epochs 10 \
    --output-dir results/hual_run
```

The output directory contains selected patch indices, uncertainty scores,
the HUAL training arrays, the fine-tuned checkpoint and a JSON metrics file.

## Reproducibility

Please read `docs/REPRODUCIBILITY.md` before submission. It records the exact
workflow represented by the supplied files and also lists the items that still
need to be fixed or documented.

## License

The code is released under the MIT License. See `LICENSE`.

## Citation

A `CITATION.cff` file is included. Replace the placeholder DOI information after
the paper and repository have their final bibliographic details.
