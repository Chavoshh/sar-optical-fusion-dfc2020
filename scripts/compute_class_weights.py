"""Compute per-class loss weights from the training partition only.

The weights are saved as JSON next to dataset_stats.json. The training loop
loads them and passes to nn.CrossEntropyLoss(weight=...).

Weighting scheme: inverse-square-root frequency, normalized to mean 1.
This is the median-frequency variant of the canonical inverse-frequency
weight (1/n), which is more stable than pure inverse-frequency (avoids
giving extreme weights to very rare classes).

    w_c = 1 / sqrt(freq_c)
    w_c <- w_c / mean(w)   so weights have mean 1, comparable to unweighted

Reference: Eigen & Fergus, "Predicting depth, surface normals and semantic
labels with a common multi-scale convolutional architecture" (2015).
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np
import rasterio
from tqdm import tqdm

from sar_optical_fusion.data.dataset import (
    N_CLASSES,
    TRAIN_ID_TO_NAME,
    remap_labels,
)
from sar_optical_fusion.data.splits import load_split


def main() -> None:
    data_root = Path("data/raw/ROIs0000_validation")
    dfc_dir = data_root / "dfc_validation"
    split = load_split("src/sar_optical_fusion/data/splits.json")
    train_ids = split["train"]

    print(f"Computing class frequencies over {len(train_ids)} training patches...")
    counts = np.zeros(N_CLASSES, dtype=np.int64)

    for pid in tqdm(train_ids):
        path = dfc_dir / f"ROIs0000_validation_dfc_0_{pid}.tif"
        with rasterio.open(path) as src:
            raw = src.read(1)
        labels = remap_labels(raw)
        vals, cnts = np.unique(labels, return_counts=True)
        for v, c in zip(vals, cnts):
            counts[v] += c

    total = int(counts.sum())
    freq = counts / total

    # Inverse-sqrt weighting, normalized to mean 1
    weights = 1.0 / np.sqrt(freq)
    weights = weights / weights.mean()

    print(f"\n{'Class':<12s} {'Pixels':>12s} {'Freq':>8s} {'Weight':>8s}")
    print("-" * 44)
    for c in range(N_CLASSES):
        print(f"{TRAIN_ID_TO_NAME[c]:<12s} {counts[c]:>12,d} "
              f"{freq[c]:>8.4f} {weights[c]:>8.4f}")

    print(f"\nWeight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"Weight mean:  {weights.mean():.3f}  (should be exactly 1.0)")

    # Save
    out = {
        "method": "inverse_sqrt_frequency_normalized",
        "source_split": "train",
        "source_n_patches": len(train_ids),
        "total_pixels": total,
        "per_class": [
            {
                "train_id": c,
                "name": TRAIN_ID_TO_NAME[c],
                "pixel_count": int(counts[c]),
                "frequency": float(freq[c]),
                "weight": float(weights[c]),
            }
            for c in range(N_CLASSES)
        ],
        "weights": [float(w) for w in weights],  # ready to pass to torch.tensor()
    }
    out_path = Path("src/sar_optical_fusion/data/class_weights.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()