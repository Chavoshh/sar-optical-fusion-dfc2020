# SAR-Optical Fusion for Land Cover Segmentation (DFC2020)

A comparative study of fusion strategies for combining Sentinel-1 SAR and Sentinel-2 optical imagery for land cover semantic segmentation, evaluated on the [2020 IEEE GRSS Data Fusion Contest](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) benchmark.

**Status:** Phase 1 complete — data inventory, exploration, and dataset statistics. Phase 2 (data pipeline) in progress.

## Project goals

## Status & current results

### Phase 3 (in progress): Baseline 1 — S2-only U-Net

A ResNet-18 U-Net trained on Sentinel-2 only (12 bands, B10 dropped), class-weighted cross-entropy, 30 epochs, batch size 16, AdamW with cosine learning-rate schedule, mixed-precision training.

| Metric | Value |
| --- | --- |
| Best val mCA (mean class accuracy) | **0.7765** |
| Best val pixel accuracy | 0.842 |
| Best val mean IoU | 0.646 |
| Training time | 13.2 min (GTX 1050 Ti, 4 GB VRAM) |
| Parameters | 14.36 M |

Per-class recall on the 197-patch validation set:

| Class | Recall |
| --- | --- |
| Forest | 0.944 |
| Shrubland | 0.410 |
| Grassland | 0.624 |
| Wetlands | 0.806 |
| Croplands | 0.748 |
| Urban | 0.874 |
| Barren | 0.813 |
| Water | 0.994 |

Shrubland is the hardest class, consistent with all published DFC2020 work. It is spectrally a continuum with Grassland (sparse vegetation) and the 10 m Sentinel-2 resolution often produces mixed pixels. Barren (the rarest class at 2.9% of training pixels) is recovered at 81% recall, which is the class-weighted loss compensating for the 12× class imbalance.

[W&B run](https://wandb.ai/chavosh-personal/sar-optical-fusion-dfc2020/runs/f1rh7skn)

Next: S1-only baseline, then early fusion, then late fusion. All four models will be evaluated on the held-out 5128-patch test set in Phase 7.

## Dataset

> Exploration notebook: [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) ([view on nbviewer](https://nbviewer.org/github/Chavoshh/sar-optical-fusion-dfc2020/blob/main/notebooks/01_data_exploration.ipynb) if GitHub's renderer fails)

This project uses the **2020 IEEE GRSS Data Fusion Contest** dataset — paired Sentinel-1 SAR and Sentinel-2 optical imagery with land cover labels, distributed as 256 × 256 patches.

| Split | Patches | Modalities per patch |
| --- | --- | --- |
| Validation (used for training and validation) | 986 | S1 (2 bands), S2 (13 bands), DFC label (1 band) |
| Test (held out for final evaluation) | 5,128 | S1 (2 bands), S2 (13 bands), DFC label (1 band) |

All S1/S2/label triples are pixel-aligned; pairing is by patch ID.

### Phase 1 findings

Exploratory analysis across all 986 validation patches (see `notebooks/01_data_exploration.ipynb`) produced the dataset statistics used throughout the project (`src/sar_optical_fusion/data/dataset_stats.json`). Key findings:

- **Effective 8-class problem.** Although DFC2020 nominally defines 10 land cover classes, the validation set contains zero pixels of class 3 (Savanna) and class 8 (Snow / Ice). The 8 classes actually present are: Forest, Shrubland, Grassland, Wetlands, Croplands, Urban, Barren, and Water. This matches the official challenge evaluation scheme.
- ### Class index mapping

To produce contiguous indices for `CrossEntropyLoss`, the 8 present raw DFC class IDs are remapped:

| Train index | Raw DFC ID | Class | Pixel share |
| --- | --- | --- | --- |
| 0 | 1 | Forest | 9.1% |
| 1 | 2 | Shrubland | 5.2% |
| 2 | 4 | Grassland | 11.8% |
| 3 | 5 | Wetlands | 17.4% |
| 4 | 6 | Croplands | 13.0% |
| 5 | 7 | Urban | 5.4% |
| 6 | 9 | Barren | 2.9% |
| 7 | 10 | Water | 35.0% |

Mapping follows the convention in Schmitt et al. (2020), making results directly comparable to published DFC2020 baselines. The mapping is defined as `RAW_TO_TRAIN_ID` in `src/sar_optical_fusion/data/dataset.py`.
- **Significant class imbalance.** Water dominates at 35% of labeled pixels; Barren is the rarest present class at 2.9%, giving a 12× imbalance ratio. Class-weighted cross-entropy is used to compensate.
- **S2 band B10 (cirrus) carries no surface information.** Dataset-wide mean ≈ 11, std ≈ 5 — effectively constant. It is excluded from model input, reducing S2 from 13 to 12 channels.
- **Per-channel outlier clipping.** Both modalities contain rare extreme values (S1 backscatter outside ±30 dB; S2 reflectance > 1.0 after scaling). Inputs are clipped to the dataset-wide p1–p99 range before standardization.
- **No "no data" pixels.** Class 0 does not appear, so no `ignore_index` is required in the loss function.

### S1 normalization constants (dB)

| Channel | Mean | Std | p1 | p99 |
| --- | --- | --- | --- | --- |
| VV | −13.95 | 4.33 | −23.18 | −4.16 |
| VH | −21.54 | 6.00 | −34.39 | −11.79 |

### S2 normalization constants (reflectance × 10000, uint16)

Per-band means range from 638 (B9) to 2370 (B8A); standard deviations from 170 (B1) to 1490 (B8A). Full per-band statistics are in `dataset_stats.json`.

### Train/validation split

The 986 patches in the DFC2020 validation set are partitioned into 80% train (789 patches) and 20% validation (197 patches). The 5,128 patches in the test set are held out entirely until final evaluation.

The split is generated by `build_train_val_split` in `src/sar_optical_fusion/data/splits.py` using `seed=53`. This seed was selected after evaluating all seeds in 0–99 on per-class pixel distribution preservation; seed 53 produces a maximum class-percentage deviation of 0.96% between train and val (vs. 9.28% for the more conventional seed=42). The chosen seed is pinned as `DEFAULT_SEED` in code and the resulting partition is committed as `src/sar_optical_fusion/data/splits.json` to guarantee identical training data across the four model variants.

| Class | Train % | Val % | Diff |
| --- | --- | --- | --- |
| Forest | 8.95% | 9.90% | +0.96% |
| Shrubland | 5.33% | 4.63% | −0.70% |
| Grassland | 11.73% | 12.32% | +0.59% |
| Wetlands | 17.57% | 16.94% | −0.64% |
| Croplands | 13.11% | 12.59% | −0.52% |
| Urban | 5.45% | 5.40% | −0.05% |
| Barren | 2.90% | 3.00% | +0.10% |
| Water | 34.96% | 35.22% | +0.26% |

## Tech stack

- PyTorch 2.5 (CUDA 12.1) + segmentation-models-pytorch
- rasterio for geospatial I/O
- albumentations for augmentation
- Hydra for configuration
- Weights & Biases for experiment tracking
- uv for environment management

## Reproducibility

Environment is fully pinned via `uv.lock`. To reproduce:

```Bash
git clone https://github.com/Chavoshh/sar-optical-fusion-dfc2020.git
cd sar-optical-fusion-dfc2020
uv sync
```

Data (~19 GB) must be obtained separately from the [DFC2020 page](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) and placed in `data/` — see `data/README.md` for the expected layout.

## License

MIT — see [LICENSE](LICENSE).