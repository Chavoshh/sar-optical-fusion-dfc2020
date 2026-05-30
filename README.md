# SAR-Optical Fusion for Land Cover Segmentation (DFC2020)

A comparative study of fusion strategies for combining Sentinel-1 SAR and Sentinel-2 optical imagery for land cover semantic segmentation, evaluated on the [2020 IEEE GRSS Data Fusion Contest](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) benchmark.

**Status:** In development.

## Project goals

Train and compare four U-Net-based segmentation models on the DFC2020 benchmark:

1. **S1-only** baseline — Sentinel-1 (VV+VH) input.
2. **S2-only** baseline — Sentinel-2 multispectral input.
3. **Early fusion** — channel-wise concatenation of S1 and S2 at the input.
4. **Late fusion** — dual-encoder architecture with feature-level fusion.

The aim is an honest, reproducible ablation on consumer hardware (single GTX 1050 Ti, 4 GB VRAM), with predictive uncertainty quantification as a stretch goal.

## Tech stack

- PyTorch 2.5 (CUDA 12.1) + segmentation-models-pytorch
- rasterio for geospatial I/O
- albumentations for augmentation
- Hydra for configuration
- Weights & Biases for experiment tracking
- uv for environment management

## Reproducibility

Environment is fully pinned via `uv.lock`. To reproduce:

\\Bash
git clone https://github.com/Chavoshh/sar-optical-fusion-dfc2020.git
cd sar-optical-fusion-dfc2020
uv sync
\\\

Data (~19 GB) must be obtained separately from the [DFC2020 page](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) and placed in `data/` — see `data/README.md` for the expected layout.

## License

MIT — see [LICENSE](LICENSE).