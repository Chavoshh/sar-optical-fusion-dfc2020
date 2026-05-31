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

    # --------------------------------------------------------------------------
# Label remapping
# --------------------------------------------------------------------------

# Build a lookup table once at import time.
# Index = raw class ID, value = train-index (or 255 for unmapped IDs).
# We use 255 as a sentinel for "should never appear"; if it ever shows up
# in a label tensor, downstream code will fail loudly rather than silently.
_LABEL_LUT = np.full(256, fill_value=255, dtype=np.uint8)
for _raw, _train in RAW_TO_TRAIN_ID.items():
    _LABEL_LUT[_raw] = _train


def remap_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Convert raw DFC class IDs to contiguous train indices [0, N_CLASSES).

    Parameters
    ----------
    raw_labels : np.ndarray
        Integer array of any shape with values in the raw DFC class scheme.

    Returns
    -------
    np.ndarray
        Same shape as input, dtype uint8, with values in [0, N_CLASSES).
        Any raw value not in RAW_TO_TRAIN_ID maps to 255 (sentinel).

    Raises
    ------
    ValueError
        If any pixel in the output is the sentinel value 255, indicating an
        unexpected raw class ID was present.
    """
    # Cast to uint8 indexing range. Raw labels are int32 in the GeoTIFFs but
    # values are 0-10 so this is lossless.
    if raw_labels.min() < 0 or raw_labels.max() > 255:
        raise ValueError(
            f"Raw labels outside uint8 range: [{raw_labels.min()}, {raw_labels.max()}]"
        )
    remapped = _LABEL_LUT[raw_labels.astype(np.uint8)]
    if (remapped == 255).any():
        bad_raw_values = np.unique(raw_labels[remapped == 255])
        raise ValueError(
            f"Unmapped raw class IDs found in label patch: {bad_raw_values.tolist()}. "
            f"Expected only {sorted(RAW_TO_TRAIN_ID.keys())}."
        )
    return remapped

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

# Pre-compute normalization arrays at import time. Doing this once rather
# than re-reading the stats dict for every patch shaves dataloader overhead.
def _build_norm_arrays(modality: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-channel clip/normalize arrays for one modality.

    Returns four 1D float32 arrays of length C (number of channels):
    (clip_lo, clip_hi, mean, std). All in the *original* value domain
    (S1: dB, S2: uint16 reflectance * 10000).
    """
    if modality == "s1":
        keys = ["vv", "vh"]
        stats = DATASET_STATS["s1"]
    elif modality == "s2":
        # We use only the bands we keep (B10 excluded). Order matches
        # S2_BAND_INDICES_KEEP, which is the order channels come out of
        # _load_s2_patch below.
        all_keys = ["b1", "b2", "b3", "b4", "b5", "b6", "b7",
                    "b8", "b8a", "b9", "b10", "b11", "b12"]
        keys = [all_keys[i] for i in S2_BAND_INDICES_KEEP]
        stats = DATASET_STATS["s2"]
    else:
        raise ValueError(f"Unknown modality: {modality!r}")

    clip_lo = np.array([stats[k]["p1"] for k in keys], dtype=np.float32)
    clip_hi = np.array([stats[k]["p99"] for k in keys], dtype=np.float32)
    mean = np.array([stats[k]["mean"] for k in keys], dtype=np.float32)
    std = np.array([stats[k]["std"] for k in keys], dtype=np.float32)
    return clip_lo, clip_hi, mean, std


_S1_CLIP_LO, _S1_CLIP_HI, _S1_MEAN, _S1_STD = _build_norm_arrays("s1")
_S2_CLIP_LO, _S2_CLIP_HI, _S2_MEAN, _S2_STD = _build_norm_arrays("s2")


def normalize_s1(s1: np.ndarray) -> np.ndarray:
    """Clip to [p1, p99] then z-score normalize per channel.

    Parameters
    ----------
    s1 : np.ndarray of shape (2, H, W), float32 or float64
        Raw S1 patch in dB.

    Returns
    -------
    np.ndarray of shape (2, H, W), float32
        Normalized S1 patch. Mean ~0, std ~1 across the dataset.
    """
    if s1.shape[0] != N_S1_CHANNELS:
        raise ValueError(f"Expected {N_S1_CHANNELS} S1 channels, got {s1.shape[0]}")
    # Broadcast the per-channel constants over (H, W).
    # Shape: (2,) -> (2, 1, 1) for broadcasting.
    lo = _S1_CLIP_LO[:, None, None]
    hi = _S1_CLIP_HI[:, None, None]
    mean = _S1_MEAN[:, None, None]
    std = _S1_STD[:, None, None]
    out = np.clip(s1.astype(np.float32), lo, hi)
    out = (out - mean) / std
    return out


def normalize_s2(s2: np.ndarray) -> np.ndarray:
    """Clip to [p1, p99] then z-score normalize per band.

    Parameters
    ----------
    s2 : np.ndarray of shape (12, H, W), uint16 or float
        S2 patch with B10 *already dropped*. Channel order matches
        S2_BAND_INDICES_KEEP.

    Returns
    -------
    np.ndarray of shape (12, H, W), float32
        Normalized S2 patch.
    """
    if s2.shape[0] != N_S2_BANDS:
        raise ValueError(f"Expected {N_S2_BANDS} S2 bands, got {s2.shape[0]} "
                         f"(did you forget to drop B10?)")
    lo = _S2_CLIP_LO[:, None, None]
    hi = _S2_CLIP_HI[:, None, None]
    mean = _S2_MEAN[:, None, None]
    std = _S2_STD[:, None, None]
    out = np.clip(s2.astype(np.float32), lo, hi)
    out = (out - mean) / std
    return out

# --------------------------------------------------------------------------
# File loading
# --------------------------------------------------------------------------

import rasterio  # noqa: E402  -- placed here to keep "constants" section above

# DFC2020 patches always come from scene 0 in this release.
_SCENE_ID = "0"


def _load_s1_patch(path: Path) -> np.ndarray:
    """Read a single S1 patch GeoTIFF, return (2, H, W) float32."""
    with rasterio.open(path) as src:
        arr = src.read()  # (2, H, W) float64 from source
    return arr.astype(np.float32)


def _load_s2_patch(path: Path) -> np.ndarray:
    """Read a single S2 patch GeoTIFF, drop B10, return (12, H, W) uint16."""
    with rasterio.open(path) as src:
        arr = src.read()  # (13, H, W) uint16
    # Drop B10 — keep only the bands listed in S2_BAND_INDICES_KEEP.
    return arr[S2_BAND_INDICES_KEEP]


def _load_label_patch(path: Path) -> np.ndarray:
    """Read a single DFC label patch GeoTIFF, return (H, W) int32."""
    with rasterio.open(path) as src:
        return src.read(1)  # (H, W) int32 from source

# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

import albumentations as A  # noqa: E402


def build_train_augmentation() -> A.Compose:
    """Geometric-only augmentation safe for both SAR and optical modalities.

    We use only flips and 90-degree rotations. These four operations together
    form the D4 dihedral group - the natural symmetry group of square aerial
    imagery. They are *lossless*: every pixel from the original appears in
    the output, just at a permuted location. No interpolation, no padding,
    no fractional rotations.

    Why no other augmentations?
      * Color jitter / brightness changes: would distort S1's physical dB
        backscatter and S2's reflectance values, breaking the dataset
        normalization stats.
      * Arbitrary-angle rotation: requires interpolation, which produces
        non-physical S1 values and creates label-boundary artifacts.
      * Random crops: would change patch dimensions, complicating the
        downstream U-Net which expects fixed (H, W).
      * Gaussian noise, blur: rarely helpful for satellite imagery and
        can interact badly with SAR speckle.

    Returns
    -------
    albumentations.Compose
        Transform configured for multi-modal application via
        `additional_targets`. Apply via:

            t = transform(image=s1, image2=s2, mask=label)
            s1_aug, s2_aug, label_aug = t["image"], t["image2"], t["mask"]
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"image2": "image"},
    )


def build_eval_augmentation() -> A.Compose | None:
    """No augmentation for validation/test. Returns None for clarity."""
    return None

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402


class DFC2020Dataset(Dataset):
    """Paired (S1, S2, label) patches from the DFC2020 benchmark.

    The same class supports all four experimental settings via the
    `modality` argument:

      * "s1"     - returns (s1, label),       s1 is (2, H, W)
      * "s2"     - returns (s2, label),       s2 is (12, H, W)
      * "fusion" - returns (s1, s2, label),   late-fusion models consume both

    For *early-fusion* models, use modality="fusion" and concatenate the two
    tensors along dim 0 in the model's forward pass (or in a collate_fn).

    Parameters
    ----------
    data_root : str | Path
        Directory containing the three modality subfolders (e.g.
        `data/raw/ROIs0000_validation`).
    patch_ids : list[str]
        Patch identifiers like "p0", "p1", ... determining which patches
        this dataset instance serves.
    split_name : str
        Name of the split as it appears in filenames: "validation" or "test".
        Used to construct file paths (e.g. `s1_validation`, `s1_0`).
    modality : str
        One of "s1", "s2", "fusion". Controls what `__getitem__` returns.
    """

    VALID_MODALITIES = ("s1", "s2", "fusion")

    def __init__(
        self,
        data_root: str | Path,
        patch_ids: list[str],
        split_name: str,
        modality: str = "fusion",
        transform: A.Compose | None = None,
    ) -> None:
        if modality not in self.VALID_MODALITIES:
            raise ValueError(
                f"modality must be one of {self.VALID_MODALITIES}, got {modality!r}"
            )
        if split_name not in ("validation", "0"):
            # "validation" is the validation set, "0" is the test set
            # (file name suffix differs: s1_validation vs s1_0).
            raise ValueError(
                f"split_name must be 'validation' or '0', got {split_name!r}"
            )

        self.data_root = Path(data_root)
        self.patch_ids = list(patch_ids)
        self.split_name = split_name
        self.modality = modality
        self.transform = transform

        # Resolve the three modality subdirectories once.
        # Folder naming: s1_validation, s2_validation, dfc_validation
        # vs           : s1_0,          s2_0,          dfc_0
        self._s1_dir = self.data_root / f"s1_{split_name}"
        self._s2_dir = self.data_root / f"s2_{split_name}"
        self._dfc_dir = self.data_root / f"dfc_{split_name}"

        for d in (self._s1_dir, self._s2_dir, self._dfc_dir):
            if not d.exists():
                raise FileNotFoundError(f"Modality directory missing: {d}")

        # Filename templates
        self._s1_name = f"ROIs0000_{'validation' if split_name == 'validation' else 'test'}_s1_{_SCENE_ID}_{{pid}}.tif"
        self._s2_name = f"ROIs0000_{'validation' if split_name == 'validation' else 'test'}_s2_{_SCENE_ID}_{{pid}}.tif"
        self._dfc_name = f"ROIs0000_{'validation' if split_name == 'validation' else 'test'}_dfc_{_SCENE_ID}_{{pid}}.tif"

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pid = self.patch_ids[idx]

        # Load label
        label_path = self._dfc_dir / self._dfc_name.format(pid=pid)
        raw_label = _load_label_patch(label_path)
        label = remap_labels(raw_label)  # uint8, (H, W)

        # Load S1 if needed
        s1: np.ndarray | None = None
        if self.modality in ("s1", "fusion"):
            s1_path = self._s1_dir / self._s1_name.format(pid=pid)
            s1 = _load_s1_patch(s1_path)         # (2, H, W) float32
            s1 = normalize_s1(s1)                # normalized

        # Load S2 if needed
        s2: np.ndarray | None = None
        if self.modality in ("s2", "fusion"):
            s2_path = self._s2_dir / self._s2_name.format(pid=pid)
            s2 = _load_s2_patch(s2_path)         # (12, H, W) uint16
            s2 = normalize_s2(s2)                # (12, H, W) float32

        # Apply augmentation jointly across modalities and label.
        # Albumentations expects (H, W, C) for "image"-like targets, so we
        # transpose, apply, then transpose back. The label stays (H, W).
        if self.transform is not None:
            kwargs: dict[str, np.ndarray] = {"mask": label}
            if s1 is not None:
                kwargs["image"] = np.transpose(s1, (1, 2, 0))    # (H, W, 2)
            if s2 is not None:
                # Albumentations uses "image" and "image2" for two inputs.
                # We use "image" for the primary modality.
                if "image" in kwargs:
                    kwargs["image2"] = np.transpose(s2, (1, 2, 0))   # (H, W, 12)
                else:
                    kwargs["image"] = np.transpose(s2, (1, 2, 0))

            transformed = self.transform(**kwargs)
            label = transformed["mask"]
            if s1 is not None and s2 is not None:
                s1 = np.transpose(transformed["image"], (2, 0, 1))
                s2 = np.transpose(transformed["image2"], (2, 0, 1))
            elif s1 is not None:
                s1 = np.transpose(transformed["image"], (2, 0, 1))
            elif s2 is not None:
                s2 = np.transpose(transformed["image"], (2, 0, 1))

        # Convert to tensors
        out: dict[str, torch.Tensor] = {
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "patch_id": pid,
        }
        if s1 is not None:
            out["s1"] = torch.from_numpy(np.ascontiguousarray(s1))
        if s2 is not None:
            out["s2"] = torch.from_numpy(np.ascontiguousarray(s2))
        return out