# User guide

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Check the repository

```bash
python scripts/smoke_test.py
```

## 3. Create a small test dataset

```bash
python examples/create_synthetic_dataset.py
```

The generated file is:

`data/synthetic_hual_demo.h5`

It has the same general HDF5 split layout as the research input, but the data is
synthetic and must not be used for scientific results.

## 4. Prepare the real dataset

Place the prepared HDF5 split file outside the repository or under an ignored
local path. The file should contain:

```text
X_train, Y_train
X_val, Y_val
X_test, Y_test
```

See `DATASET.md`. The final experiment uses 14 input channels.

## 5. Prepare the base checkpoint

The HUAL stage starts from an already trained base segmentation model. Put the
checkpoint path into the command line; do not commit large model files unless the
repository and journal policy allow it.

## 6. Run HUAL

```bash
python scripts/run_hual.py         --data /path/to/gsgm_splits_128.h5         --checkpoint /path/to/unet_best_glacier.pth         --top-k 400         --epochs 10         --output-dir results/hual_run
```

## 7. Inspect the outputs

The run writes:

- `selected_patch_indices.npy`
- `patch_uncertainty_scores.npy`
- `hual_train_indices.npy`
- `hual_train_X.npy`
- `hual_train_Y.npy`
- `hual_model.pth`
- `metrics.json`

## 8. Do not commit generated data by accident

The `.gitignore` file excludes HDF5 data, NumPy arrays, checkpoints and generated
results. If a small example file is intentionally included, force-add it only
after checking that it contains no restricted data.
