# SAR-Optical Fusion for Land Cover Segmentation (DFC2020)

A comparative study of fusion strategies for combining Sentinel-1 SAR and Sentinel-2 optical imagery for land cover semantic segmentation, evaluated on the [2020 IEEE GRSS Data Fusion Contest](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) benchmark.

**Status:** Phase 1 complete — data inventory, exploration, and dataset statistics. Phase 2 (data pipeline) in progress.

## Project goals

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