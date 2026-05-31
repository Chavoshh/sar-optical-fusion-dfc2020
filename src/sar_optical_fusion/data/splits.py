"""Reproducible train/validation patch ID splits for DFC2020.

The DFC2020 release does not ship pre-defined train/val splits within the
"validation" set. We construct our own deterministic split here, so every
model trains and is evaluated on exactly the same patch partition.

The split is seeded, so re-running this code on any machine produces the
same partition. The result is also serializable to JSON for inclusion in
W&B run configs and the project README.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# The official train/val split seed for this project.
# Selected after evaluating seeds 0-99 on per-class pixel distribution;
# seed=53 gives max class deviation 0.96% (compare seed=42 at 9.28%).
# Reproduce: `build_train_val_split(data_root, val_fraction=0.2, seed=DEFAULT_SEED)`.
DEFAULT_SEED: int = 53

"""def build_train_val_split(
    data_root: str | Path,
    val_fraction: float = 0.2,
   seed: int = 42,
    split_name: str = "validation",
) -> dict[str, list[str]]:"""
def build_train_val_split(
    data_root: str | Path,
    val_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,           # was: seed: int = 42
    split_name: str = "validation",
) -> dict[str, list[str]]:
    """Build a reproducible train/val split from the patches on disk.

    Patches are discovered by listing the S1 directory; we trust that the
    S2 and DFC directories have matching IDs (verified in Phase 1).

    Parameters
    ----------
    data_root : str | Path
        Path to e.g. data/raw/ROIs0000_validation.
    val_fraction : float
        Fraction of patches reserved for validation (default 0.2 = 20%).
    seed : int
        Seed for the Python `random` module's shuffle.
    split_name : str
        DFC split suffix used in folder names. Default "validation".

    Returns
    -------
    dict
        {"train": [patch_id, ...], "val": [patch_id, ...]}
    """
    data_root = Path(data_root)
    s1_dir = data_root / f"s1_{split_name}"
    if not s1_dir.exists():
        raise FileNotFoundError(s1_dir)

    # Extract patch IDs from filenames: ROIs0000_validation_s1_0_p0.tif -> p0
    patch_ids = sorted(
        f.stem.split("_")[-1] for f in s1_dir.glob("*.tif")
    )
    if not patch_ids:
        raise RuntimeError(f"No patches found in {s1_dir}")

    # Sort numerically by the integer part of the patch ID (p0, p1, ..., p10
    # would otherwise sort lexicographically as p0, p1, p10, p100, p11, ...).
    patch_ids.sort(key=lambda pid: int(pid[1:]))

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled = list(patch_ids)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:n_val], key=lambda pid: int(pid[1:]))
    train_ids = sorted(shuffled[n_val:], key=lambda pid: int(pid[1:]))

    return {"train": train_ids, "val": val_ids}


def save_split(split: dict[str, list[str]], path: str | Path) -> None:
    """Write a split dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Load a previously saved split."""
    with open(path, "r") as f:
        return json.load(f)