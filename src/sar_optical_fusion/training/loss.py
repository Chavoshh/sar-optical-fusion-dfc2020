"""Loss functions for DFC2020 segmentation.

The default is class-weighted cross-entropy with weights computed from the
training partition (see scripts/compute_class_weights.py).

We expose a build_loss() factory so the training entry point can pull a
loss out of a config string without conditional imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

LossName = Literal["ce", "weighted_ce"]


def load_class_weights(path: str | Path | None = None) -> torch.Tensor:
    """Load class weights computed by scripts/compute_class_weights.py.

    Parameters
    ----------
    path : str | Path | None
        Path to class_weights.json. If None, uses the default packaged path.

    Returns
    -------
    torch.Tensor of shape (N_CLASSES,), float32
        Per-class weights, ready to pass to nn.CrossEntropyLoss(weight=...).
    """
    if path is None:
        path = Path(__file__).parent.parent / "data" / "class_weights.json"
    with open(path) as f:
        data = json.load(f)
    return torch.tensor(data["weights"], dtype=torch.float32)


def build_loss(
    name: LossName = "weighted_ce",
    class_weights_path: str | Path | None = None,
) -> nn.Module:
    """Build a loss module by name.

    Parameters
    ----------
    name : str
        "ce" for unweighted cross-entropy.
        "weighted_ce" for class-weighted cross-entropy (default).
    class_weights_path : str | Path | None
        Path to class_weights.json. Only used when name == "weighted_ce".

    Returns
    -------
    nn.Module
        A loss module taking (logits, target) where logits is
        (N, n_classes, H, W) and target is (N, H, W) int64.
    """
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        weights = load_class_weights(class_weights_path)
        return nn.CrossEntropyLoss(weight=weights)
    raise ValueError(f"Unknown loss name: {name!r}")