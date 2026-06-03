# SAR-Optical Fusion for Land Cover Segmentation (DFC2020)

A comparative study of fusion strategies for combining Sentinel-1 SAR and Sentinel-2 optical imagery for land cover semantic segmentation, evaluated on the [2020 IEEE GRSS Data Fusion Contest](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) benchmark.

**Status:** Phase 3 in progress. Two single-modality baselines (S1, S2) trained and evaluated; fusion experiments next.

---

## Project goals

Train and compare four U-Net-based segmentation models on the DFC2020 benchmark, using identical training infrastructure so any performance differences reflect the modality choice and not implementation details:

1. **S1-only** baseline: Sentinel-1 SAR (VV + VH).
2. **S2-only** baseline: Sentinel-2 optical (12 bands; B10 dropped).
3. **Early fusion**: channel-wise concatenation of S1 and S2 at the input.
4. **Late fusion**: dual-encoder architecture with feature-level fusion.

The central research question: **does SAR-optical fusion help, and if so, where does the gain come from?** The single-modality baselines establish what each sensor contributes alone, so the fusion models can be evaluated against the *better* of the two — not against a strawman.

A stretch goal in Phase 8 is to add predictive uncertainty quantification (MC-Dropout or ensembles) to the best fusion model, since reliable confidence estimation is a known weak spot in operational EO products.

## Approach

The project is built to run on consumer hardware (single GTX 1050 Ti, 4 GB VRAM) with full reproducibility. Key design decisions:

- **Identical training pipeline across all four models.** Same loss, optimizer, schedule, augmentation, and train/val split. Only the model architecture and input channel count change.
- **Honest evaluation.** Class weights computed on the training partition only (not on the full validation set). Best-model selection by mean class accuracy, not pixel accuracy, to avoid rewarding models that ignore minority classes.
- **Empirically selected split seed.** Seeds 0–99 were scored on per-class pixel-distribution preservation; seed 53 (max class deviation 0.96 %) was selected, well below the default seed 42 at 9.28 %.
- **Mixed-precision training (AMP)** to fit the U-Net on 4 GB VRAM at batch size 16.
- **No domain-specific tricks.** Standard ResNet-18 encoder with ImageNet pre-init, AdamW, cosine LR schedule, D4 augmentation (flips + 90° rotations). Results are intended to be a clean reference, not a leaderboard chase.

## Results

### Single-modality baselines (Phase 3)

Both baselines use a ResNet-18 U-Net, class-weighted cross-entropy, 30 epochs, batch size 16, AdamW with cosine LR, mixed-precision. Only input modality and channel count differ.

| Metric | S1-only (2 ch) | S2-only (12 ch) |
| --- | --- | --- |
| Best val mCA | 0.714 | **0.777** |
| Best val pixel accuracy | 0.787 | 0.842 |
| Best val mean IoU | 0.560 | 0.646 |
| Best epoch | 29 / 30 | 30 / 30 |
| Training time | 11.9 min | 13.2 min |
| Parameters | 14.33 M | 14.36 M |

W&B runs: [S1-only](https://wandb.ai/chavosh-personal/sar-optical-fusion-dfc2020/runs/mqszt3tn) · [S2-only](https://wandb.ai/chavosh-personal/sar-optical-fusion-dfc2020/runs/f1rh7skn)

Per-class recall on the 197-patch validation set:

| Class | S1 | S2 | Δ (S2 − S1) |
| --- | --- | --- | --- |
| Forest | 0.924 | 0.944 | +0.020 |
| Shrubland | **0.451** | 0.410 | **−0.041** |
| Grassland | 0.630 | 0.624 | −0.005 |
| Wetlands | 0.643 | 0.806 | +0.163 |
| Croplands | 0.619 | 0.748 | +0.129 |
| Urban | 0.804 | 0.874 | +0.071 |
| Barren | 0.639 | 0.813 | +0.175 |
| Water | 0.987 | 0.994 | +0.007 |

**Key finding.** Sentinel-1 outperforms Sentinel-2 on Shrubland (the only class where SAR wins) by 4.1 percentage points. C-band cross-polarized backscatter encodes structural information about woody vegetation (branch density, surface roughness) that the Sentinel-2 spectral signature does not capture, especially at 10 m resolution where shrub patches often produce mixed pixels.

This complementarity motivates the fusion experiments in Phases 5 and 6: can a fused model preserve S1's Shrubland advantage while retaining S2's strength on the spectrally separable classes?

### Coming next

- **Phase 5:** Early fusion (S1+S2 concatenated, 14 input channels).
- **Phase 6:** Late fusion (dual encoder).
- **Phase 7:** All four models evaluated on the held-out 5,128-patch test set.

## Dataset

See the [data exploration notebook](notebooks/01_data_exploration.ipynb) ([nbviewer link](https://nbviewer.org/github/Chavoshh/sar-optical-fusion-dfc2020/blob/main/notebooks/01_data_exploration.ipynb)) for the full analysis. Summary below.

The 2020 IEEE GRSS Data Fusion Contest dataset provides paired Sentinel-1 SAR and Sentinel-2 optical imagery with land cover labels, distributed as 256 × 256 patches.

| Split | Patches | Modalities per patch |
| --- | --- | --- |
| Validation (used for training and validation) | 986 | S1 (2 bands), S2 (13 bands), DFC label (1 band) |
| Test (held out for final evaluation) | 5,128 | S1 (2 bands), S2 (13 bands), DFC label (1 band) |

S1, S2, and label patches with matching IDs are pixel-aligned.

### Phase 1 findings

Exploratory analysis across all 986 validation patches produced the dataset statistics in `src/sar_optical_fusion/data/dataset_stats.json`, used throughout the pipeline. Headline findings:

- **Effective 8-class problem.** Classes 3 (Savanna) and 8 (Snow / Ice) have zero pixels in our data. The 8 classes present are Forest, Shrubland, Grassland, Wetlands, Croplands, Urban, Barren, and Water — matching the official DFC2020 evaluation scheme.
- **12× class imbalance.** Water alone is 35 % of pixels; Barren is the rarest at 2.9 %. Addressed via class-weighted cross-entropy with inverse-square-root frequency weights.
- **B10 (cirrus) carries no surface signal** (dataset-wide mean ≈ 11, std ≈ 5). Dropped from model input, reducing Sentinel-2 from 13 to 12 channels.
- **Per-channel outlier clipping.** S1 and S2 inputs are clipped to dataset-wide [p1, p99] before z-score normalization, preventing extreme values from dominating gradients.
- **No "no data" pixels.** Class 0 (no data) is absent throughout; no `ignore_index` needed.

### Class index mapping

`CrossEntropyLoss` requires contiguous indices. The 8 raw DFC class IDs are remapped:

| Train index | Raw DFC ID | Class | Pixel share |
| --- | --- | --- | --- |
| 0 | 1 | Forest | 9.1 % |
| 1 | 2 | Shrubland | 5.2 % |
| 2 | 4 | Grassland | 11.8 % |
| 3 | 5 | Wetlands | 17.4 % |
| 4 | 6 | Croplands | 13.0 % |
| 5 | 7 | Urban | 5.4 % |
| 6 | 9 | Barren | 2.9 % |
| 7 | 10 | Water | 35.0 % |

Mapping follows Schmitt et al. (2020), making results comparable to published baselines. Defined as `RAW_TO_TRAIN_ID` in `src/sar_optical_fusion/data/dataset.py`.

### Normalization constants

**S1 (dB):**

| Channel | Mean | Std | p1 | p99 |
| --- | --- | --- | --- | --- |
| VV | −13.95 | 4.33 | −23.18 | −4.16 |
| VH | −21.54 | 6.00 | −34.39 | −11.79 |

**S2 (reflectance × 10000, uint16):** per-band means range 638 (B9) to 2370 (B8A); per-band standard deviations 170 (B1) to 1490 (B8A). Full per-band statistics in `dataset_stats.json`.

### Train/validation split

The 986 validation-set patches are partitioned 80 % train (789) / 20 % validation (197). The 5,128 test patches are held out for Phase 7.

The split is generated by `build_train_val_split` in `src/sar_optical_fusion/data/splits.py` using `seed=53` (pinned as `DEFAULT_SEED`). The seed was selected by evaluating seeds 0–99 on per-class pixel-distribution preservation; seed 53 produces a 0.96 % max class deviation between train and val versus 9.28 % for seed 42. The resulting partition is committed as `src/sar_optical_fusion/data/splits.json` to guarantee identical training data across all four model variants.

| Class | Train % | Val % | Diff |
| --- | --- | --- | --- |
| Forest | 8.95 | 9.90 | +0.96 |
| Shrubland | 5.33 | 4.63 | −0.70 |
| Grassland | 11.73 | 12.32 | +0.59 |
| Wetlands | 17.57 | 16.94 | −0.64 |
| Croplands | 13.11 | 12.59 | −0.52 |
| Urban | 5.45 | 5.40 | −0.05 |
| Barren | 2.90 | 3.00 | +0.10 |
| Water | 34.96 | 35.22 | +0.26 |

## Tech stack

- PyTorch 2.5 (CUDA 12.1) + segmentation-models-pytorch
- rasterio for geospatial I/O
- albumentations for augmentation
- Hydra for configuration
- Weights & Biases for experiment tracking
- uv for environment management

## Reproducibility

Environment is fully pinned via `uv.lock`. To reproduce:

```bash
git clone https://github.com/Chavoshh/sar-optical-fusion-dfc2020.git
cd sar-optical-fusion-dfc2020
uv sync
```

Data (~19 GB) must be obtained separately from the [DFC2020 page](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) and placed in `data/` - see [`data/README.md`](data/README.md) for the expected layout.

Run any of the three trained experiments with:

```bash
uv run python scripts/train.py experiment=s2_only
uv run python scripts/train.py experiment=s1_only
uv run python scripts/train.py experiment=early_fusion
```

## License

MIT - see [LICENSE](LICENSE).