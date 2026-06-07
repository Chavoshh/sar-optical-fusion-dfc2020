"""Select three representative DFC2020 test patches for qualitative figures.

Selection criteria:
  1. Shrubland-rich patch:  the class where fusion clearly wins.
                            Picked: largest Shrubland fraction.
  2. Wetlands-heavy patch:  the universal failure case.
                            Picked: largest Wetlands fraction.
  3. Urban+Water patch:     the saturated/easy case (all models tie).
                            Picked: high combined Urban+Water fraction with
                            both classes meaningfully present.

The script writes the chosen patch IDs to outputs/selected_test_patches.json
so the rendering step is reproducible without re-scanning the test set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from sar_optical_fusion.data.dataset import (
    N_CLASSES,
    TRAIN_ID_TO_NAME,
    remap_labels,
)


TEST_ROOT = Path("data/raw/ROIs0000_test")
DFC_DIR = TEST_ROOT / "dfc_0"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def class_fractions(patch_path: Path) -> np.ndarray:
    """Return an array of shape (N_CLASSES,) with the pixel fraction per class."""
    with rasterio.open(patch_path) as src:
        raw = src.read(1)
    labels = remap_labels(raw)
    counts = np.bincount(labels.ravel(), minlength=N_CLASSES).astype(np.float64)
    return counts / counts.sum()


def main() -> None:
    label_files = sorted(
        DFC_DIR.glob("*.tif"),
        key=lambda p: int(p.stem.split("_")[-1][1:]),
    )
    print(f"Scanning {len(label_files)} test labels...")

    fractions = np.zeros((len(label_files), N_CLASSES), dtype=np.float64)
    patch_ids: list[str] = []
    for i, p in enumerate(tqdm(label_files)):
        fractions[i] = class_fractions(p)
        patch_ids.append(p.stem.split("_")[-1])

    # Class index lookup (TRAIN_ID_TO_NAME maps train_id -> name)
    name_to_id = {name: tid for tid, name in TRAIN_ID_TO_NAME.items()}

    # Selection 1: Shrubland-rich
    shrubland_idx = name_to_id["Shrubland"]
    best_shrub = int(np.argmax(fractions[:, shrubland_idx]))

    # Selection 2: Wetlands-heavy
    wetlands_idx = name_to_id["Wetlands"]
    best_wetlands = int(np.argmax(fractions[:, wetlands_idx]))

    # Selection 3: Urban + Water mix.
    # Score = min(urban_frac, water_frac) -- rewards patches with BOTH classes
    # meaningfully present. argmax = the most "city by water" patch.
    urban_idx = name_to_id["Urban"]
    water_idx = name_to_id["Water"]
    urban_water_score = np.minimum(fractions[:, urban_idx], fractions[:, water_idx])
    best_urban_water = int(np.argmax(urban_water_score))

    selected = {
        "shrubland_rich": {
            "patch_id": patch_ids[best_shrub],
            "rationale": "Largest Shrubland fraction; class where fusion clearly wins.",
            "class_fractions": {
                TRAIN_ID_TO_NAME[c]: float(fractions[best_shrub, c])
                for c in range(N_CLASSES)
            },
        },
        "wetlands_heavy": {
            "patch_id": patch_ids[best_wetlands],
            "rationale": "Largest Wetlands fraction; universal failure mode on test.",
            "class_fractions": {
                TRAIN_ID_TO_NAME[c]: float(fractions[best_wetlands, c])
                for c in range(N_CLASSES)
            },
        },
        "urban_water_mix": {
            "patch_id": patch_ids[best_urban_water],
            "rationale": "Both Urban and Water meaningfully present; saturated easy case.",
            "class_fractions": {
                TRAIN_ID_TO_NAME[c]: float(fractions[best_urban_water, c])
                for c in range(N_CLASSES)
            },
        },
    }

    print("\nSelected patches:")
    for tag, info in selected.items():
        pid = info["patch_id"]
        top = sorted(info["class_fractions"].items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"{n} {f:.0%}" for n, f in top if f > 0.01)
        print(f"  {tag:<18s}  {pid:>6s}   top classes: {top_str}")

    out_path = OUT_DIR / "selected_test_patches.json"
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved selection to: {out_path}")


if __name__ == "__main__":
    main()