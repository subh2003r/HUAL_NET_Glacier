# Reproducibility notes

This repository is organized around the final HUAL-Net experimental setup.

## Base segmentation model

The model uses:

- U-Net architecture
- ResNet34 encoder
- SCSE decoder attention
- 14-channel multimodal input
- one output channel for binary glacier segmentation
- no activation in the model output; sigmoid is applied during loss/evaluation
- no ImageNet encoder weights for the multimodal model
- explicit Kaiming initialization for convolution layers

## Input data

The model receives 128 × 128 patches with 14 channels:

```text
2 × co-polarized SAR
2 × cross-polarized SAR
2 × DEM
2 × InSAR
6 × optical
-------------------------
14 channels
```

## HUAL uncertainty calculation

The HUAL stage uses deterministic predictive entropy.

The procedure is:

1. Load the trained base segmentation model.
2. Run one deterministic forward pass on the training patches.
3. Apply sigmoid to obtain glacier probabilities.
4. Calculate binary pixel-wise entropy:

   `H(p) = -[p log(p) + (1-p) log(1-p)]`

5. Average the entropy values over each patch.
6. Rank patches according to their mean entropy.
7. Select the highest-uncertainty patches.
8. Add the selected patches and their corresponding ground-truth masks to the
   training data.
9. Fine-tune the segmentation model.
10. Evaluate the resulting model on the held-out test set.

The final experiment uses **400 selected high-uncertainty patches**.

No MC Dropout sampling is used in the final implementation.

## Annotation simulation

The selected patches use their existing ground-truth masks during the experiment.
Therefore, the HUAL annotation stage is a simulation using the available reference
annotations. The experiment should not be described as newly collected manual
annotations unless a separate annotation study was actually performed.

## Training

The supplied training workflow uses Adam/AdamW-based optimization depending on the
training stage, with the final configuration recorded with the corresponding
experiment.

For a final archival release, save the exact training configuration and environment
used to produce the reported results.

## Final experiment record

For every reported experiment, keep:

- repository commit hash
- Python version
- PyTorch version
- CUDA version
- GPU model
- random seed
- 14-channel input definition
- patch size
- train/validation/test split
- base model checkpoint
- number of selected uncertain patches
- entropy definition
- HUAL fine-tuning epochs
- learning rate
- weight decay
- batch size
- evaluation threshold
- Dice
- IoU
- precision
- recall

The final numerical results in the manuscript should be traceable to one repository
commit and one experiment configuration.
