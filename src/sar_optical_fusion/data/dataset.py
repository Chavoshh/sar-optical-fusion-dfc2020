"""DFC2020 paired-patch dataset for SAR-optical fusion.

This module provides a PyTorch Dataset that loads co-registered Sentinel-1,
Sentinel-2, and DFC land-cover label patches. The same class supports all four
experimental settings (S1-only, S2-only, early fusion, late fusion) via a
single `modality` argument.

Design decisions (from Phase 1 exploration of all 986 validation patches):
    * S2 band B10 (cirrus) is dropped — near-zero dynamic range, no surface signal.
    * Class 3 (Savanna) and class 8 (Snow/Ice) are absent in the validation set,
      so we remap the 8 present raw class IDs to contiguous indices 0-7.
    * Class 0 (No data) never appears, so no ignore_index is required.
    * Per-channel inputs are clipped to dataset-wide [p1, p99] then z-score
      normalized.

Normalization constants and class statistics are loaded from
`dataset_stats.json` in the same directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Class scheme (DFC2020)
# --------------------------------------------------------------------------
# Raw class IDs in the DFC GeoTIFFs use the simplified IGBP scheme:
#
#   0  No data    *absent in DFC2020 validation/test*
#   1  Forest
#   2  Shrubland
#   3  Savanna    *absent in DFC2020 validation/test*
#   4  Grassland
#   5  Wetlands
#   6  Croplands
#   7  Urban / Built-up
#   8  Snow / Ice  *absent in DFC2020 validation/test*
#   9  Barren
#   10 Water
#
# The model sees contiguous indices 0-7 (CrossEntropyLoss requires this).
# Mapping convention follows Schmitt et al. 2020 (DFC2020 benchmark paper).

RAW_TO_TRAIN_ID: dict[int, int] = {
    1:  0,   # Forest
    2:  1,   # Shrubland
    4:  2,   # Grassland
    5:  3,   # Wetlands
    6:  4,   # Croplands
    7:  5,   # Urban
    9:  6,   # Barren
    10: 7,   # Water
}

TRAIN_ID_TO_NAME: dict[int, str] = {
    0: "Forest",
    1: "Shrubland",
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban",
    6: "Barren",
    7: "Water",
}

N_CLASSES = len(TRAIN_ID_TO_NAME)
assert N_CLASSES == 8

# Sentinel-2 band indices to keep (drop B10, index 10).
# Resulting order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
S2_BAND_INDICES_KEEP: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
N_S2_BANDS = len(S2_BAND_INDICES_KEEP)
assert N_S2_BANDS == 12

N_S1_CHANNELS = 2  # VV, VH

# --------------------------------------------------------------------------
# Load dataset statistics computed in Phase 1
# --------------------------------------------------------------------------

_STATS_PATH = Path(__file__).parent / "dataset_stats.json"
with open(_STATS_PATH, "r") as _f:
    DATASET_STATS = json.load(_f)