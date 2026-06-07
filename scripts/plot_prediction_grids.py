"""Render qualitative prediction grids for the selected test patches.

For each patch chosen by scripts/select_test_patches.py, this script:
  1. Loads the patch's S1, S2, and ground-truth label.
  2. Runs all four trained models on the patch (one batch of size 1).
  3. Produces a single PNG with a 2x4 grid:
     row 1: S1 false-color | S2 false-color | Ground truth | Legend
     row 2: S1-only pred  | S2-only pred   | Early fusion | Late fusion

Output: outputs/figures/prediction_grid_<tag>.png  (one PNG per patch)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from sar_optical_fusion.data.dataset import (
    DFC2020Dataset,
    N_CLASSES,
    TRAIN_ID_TO_NAME,
)
from sar_optical_fusion.models.unet import build_unet


CHECKPOINTS = Path("checkpoints")
TEST_ROOT = Path("data/raw/ROIs0000_test")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Discrete color scheme matching the DFC2020 convention loosely.
# We define one color per train index 0-7.
CLASS_COLORS = [
    "#1B7837",  # 0 Forest      - dark green
    "#C2A33E",  # 1 Shrubland   - olive
    "#A6CE39",  # 2 Grassland   - light green
    "#27AAE1",  # 3 Wetlands    - cyan
    "#D55E00",  # 4 Croplands   - vermillion
    "#888888",  # 5 Urban       - gray
    "#F9DEC9",  # 6 Barren      - pale tan
    "#0033A0",  # 7 Water       - deep blue
]
CMAP = ListedColormap(CLASS_COLORS)
NORM = BoundaryNorm(np.arange(-0.5, N_CLASSES + 0.5, 1), CMAP.N)


# --- Model loading ----------------------------------------------------------

def reconstruct_model_for_inference(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt["cfg"]["extra_config"]["model"]
    architecture = model_cfg["architecture"]

    if architecture == "unet":
        model = build_unet(
            in_channels=model_cfg["in_channels"],
            n_classes=N_CLASSES,
            encoder_name=model_cfg["encoder_name"],
            encoder_weights=None,
        )
    elif architecture == "dual_encoder_unet":
        from sar_optical_fusion.models.dual_encoder_unet import build_dual_encoder_unet
        model = build_dual_encoder_unet(
            encoder_name=model_cfg["encoder_name"],
            encoder_weights=None,
            in_channels_a=model_cfg["in_channels_a"],
            in_channels_b=model_cfg["in_channels_b"],
            n_classes=N_CLASSES,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, model_cfg


@torch.no_grad()
def predict_single_patch(
    model: torch.nn.Module,
    model_cfg: dict[str, Any],
    sample: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Run one model on one patch; return predicted class indices as (H, W)."""
    keys = model_cfg["model_input_keys"]
    fusion_type = model_cfg.get("fusion_type", "early")
    is_dual = fusion_type == "late" and len(keys) == 2

    # Add batch dim and move to device
    def prep(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).to(device)

    if is_dual:
        logits = model(prep(sample[keys[0]]), prep(sample[keys[1]]))
    elif len(keys) == 1:
        logits = model(prep(sample[keys[0]]))
    else:
        x = torch.cat([prep(sample[k]) for k in keys], dim=1)
        logits = model(x)

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


# --- Visualization helpers --------------------------------------------------

def s1_false_color(s1_normalized: torch.Tensor) -> np.ndarray:
    """Build a 3-channel false color image from normalized S1 (VV, VH).

    The s1 tensor has been clip-and-z-scored; we just rescale to [0,1] per
    channel for display.
    """
    arr = s1_normalized.cpu().numpy()  # (2, H, W)
    vv, vh = arr[0], arr[1]

    def to_01(x):
        lo, hi = np.percentile(x, [2, 98])
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    r = to_01(vv)
    g = to_01(vh)
    b = to_01(vv - vh)
    return np.dstack([r, g, b])


def s2_false_color(s2_normalized: torch.Tensor) -> np.ndarray:
    """Build a false-color RGB from S2 (NIR, Red, Green).

    s2_normalized has 12 channels after B10 was dropped; the channel index
    map after dropping B10 is:
        original B1..B9    -> idx 0..8
        original B11, B12  -> idx 9, 10  (wait: 12 channels total -- check)

    Actually after dropping B10:
        index: 0  1  2  3  4   5   6   7  8   9    10   11
        band:  B1 B2 B3 B4 B5  B6  B7  B8 B8A B9   B11  B12
    So Red=B4 -> idx 3, Green=B3 -> idx 2, NIR=B8 -> idx 7
    """
    arr = s2_normalized.cpu().numpy()  # (12, H, W)
    nir = arr[7]
    red = arr[3]
    green = arr[2]

    def to_01(x):
        lo, hi = np.percentile(x, [2, 98])
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    return np.dstack([to_01(nir), to_01(red), to_01(green)])


def legend_handles() -> list[Patch]:
    """Color-coded legend entries for the 8 classes."""
    return [
        Patch(facecolor=CLASS_COLORS[c], edgecolor="black", linewidth=0.3,
              label=TRAIN_ID_TO_NAME[c])
        for c in range(N_CLASSES)
    ]


# --- Main pipeline ----------------------------------------------------------

def render_one_patch(
    patch_id: str,
    rationale: str,
    sample: dict[str, torch.Tensor],
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Compose the 2x4 grid for one patch and save it."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    fig.suptitle(
        f"Patch {patch_id}  -  {rationale}",
        fontsize=11, y=0.99, x=0.05, ha="left",
    )

    s1_img = s1_false_color(sample["s1"])
    s2_img = s2_false_color(sample["s2"])
    gt = sample["label"].cpu().numpy()

    # Top row: inputs, GT, legend
    axes[0, 0].imshow(s1_img)
    axes[0, 0].set_title("S1 false-color (VV, VH, VV-VH)", fontsize=10)

    axes[0, 1].imshow(s2_img)
    axes[0, 1].set_title("S2 false-color (NIR, R, G)", fontsize=10)

    axes[0, 2].imshow(gt, cmap=CMAP, norm=NORM, interpolation="nearest")
    axes[0, 2].set_title("Ground truth", fontsize=10)

    axes[0, 3].axis("off")
    axes[0, 3].legend(
        handles=legend_handles(),
        loc="center", frameon=False, fontsize=9, ncol=1,
    )

    # Bottom row: predictions
    model_order = [
        ("s1_only", "S1-only"),
        ("s2_only", "S2-only"),
        ("early_fusion", "Early fusion"),
        ("late_fusion", "Late fusion"),
    ]
    for j, (name, label) in enumerate(model_order):
        ax = axes[1, j]
        ax.imshow(predictions[name], cmap=CMAP, norm=NORM, interpolation="nearest")
        ax.set_title(label, fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    with open("outputs/selected_test_patches.json") as f:
        selected = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all four models once
    print("Loading checkpoints...")
    models: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
    for name in ["s1_only", "s2_only", "early_fusion", "late_fusion"]:
        ckpt_path = CHECKPOINTS / name / "best.pt"
        model, model_cfg = reconstruct_model_for_inference(ckpt_path, device)
        models[name] = (model, model_cfg)
        print(f"  loaded {name}")

    # For each selected patch, fetch the data + run all models + render
    for tag, info in selected.items():
        patch_id = info["patch_id"]
        rationale = info["rationale"]
        print(f"\n--- {tag}: {patch_id} ---")

        # Need both modalities. Use the "fusion" mode of the dataset to get
        # s1, s2, label all at once.
        ds = DFC2020Dataset(
            data_root=TEST_ROOT,
            patch_ids=[patch_id],
            split_name="0",
            modality="fusion",
            transform=None,
        )
        sample = ds[0]

        # Run each model
        predictions: dict[str, np.ndarray] = {}
        for name, (model, model_cfg) in models.items():
            predictions[name] = predict_single_patch(model, model_cfg, sample, device)

        # Render
        out_path = OUT_DIR / f"prediction_grid_{tag}.png"
        render_one_patch(patch_id, rationale, sample, predictions, out_path)


if __name__ == "__main__":
    main()