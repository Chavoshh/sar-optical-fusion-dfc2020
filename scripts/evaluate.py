"""Evaluate a trained checkpoint on the DFC2020 test set.

Use:
    uv run python scripts/evaluate.py --checkpoint checkpoints/s2_only/best.pt
    uv run python scripts/evaluate.py --checkpoint checkpoints/late_fusion/best.pt

The script:
  1. Loads the checkpoint and the embedded training config.
  2. Reconstructs the model (single- or dual-encoder U-Net based on architecture).
  3. Builds a DataLoader over the 5128 test patches.
  4. Runs inference, accumulates a confusion matrix.
  5. Saves per-class metrics, PA, mCA, mIoU, and the confusion matrix as JSON
     and NumPy files alongside the checkpoint.

No retraining. No hyperparameter changes. This is the held-out-test-set
single-shot evaluation Phase 7 exists to produce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sar_optical_fusion.data.dataset import (
    DFC2020Dataset,
    N_CLASSES,
    TRAIN_ID_TO_NAME,
)
from sar_optical_fusion.evaluation.metrics import ConfusionMatrixAccumulator
from sar_optical_fusion.models.unet import build_unet


def reconstruct_model(ckpt: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the model from a saved checkpoint dict.

    Returns
    -------
    model : torch.nn.Module
        The model with weights restored.
    model_cfg : dict
        The model config block, useful for the caller to know which modality
        keys to feed.
    """
    train_cfg = ckpt["cfg"]
    extra = train_cfg.get("extra_config", {})
    model_cfg = extra["model"]
    architecture = model_cfg["architecture"]

    if architecture == "unet":
        model = build_unet(
            in_channels=model_cfg["in_channels"],
            n_classes=N_CLASSES,
            encoder_name=model_cfg["encoder_name"],
            encoder_weights=None,  # weights come from the checkpoint
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
    return model, model_cfg


@torch.no_grad()
def evaluate_on_test(
    checkpoint_path: Path,
    test_data_root: Path,
    test_split_name: str = "0",
    batch_size: int = 16,
    num_workers: int = 2,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run a single-shot test-set evaluation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and reconstruct model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, model_cfg = reconstruct_model(ckpt)
    model = model.to(device).eval()

    modality = model_cfg["modality"]
    model_input_keys = model_cfg["model_input_keys"]
    fusion_type = model_cfg.get("fusion_type", "early")
    is_dual = (fusion_type == "late") and (len(model_input_keys) == 2)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Architecture: {model_cfg['architecture']}")
    print(f"  Modality: {modality}    Fusion type: {fusion_type}")
    print(f"  Trained val mCA: {ckpt.get('val_mca', float('nan')):.4f} "
          f"(epoch {ckpt.get('epoch', '?')})")

    # Build test loader.
    patch_ids = sorted(
        [p.stem.split("_")[-1] for p in (test_data_root / f"s1_{test_split_name}").glob("*.tif")],
        key=lambda pid: int(pid[1:]),
    )
    print(f"  Test patches: {len(patch_ids)}")

    test_ds = DFC2020Dataset(
        data_root=test_data_root,
        patch_ids=patch_ids,
        split_name=test_split_name,
        modality=modality,
        transform=None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    # Inference + confusion matrix
    cm = ConfusionMatrixAccumulator(N_CLASSES)
    for batch in tqdm(test_loader, desc="Evaluating"):
        y = batch["label"].to(device, non_blocking=True)
        if is_dual:
            s1 = batch[model_input_keys[0]].to(device, non_blocking=True)
            s2 = batch[model_input_keys[1]].to(device, non_blocking=True)
            logits = model(s1, s2)
        else:
            if len(model_input_keys) == 1:
                x = batch[model_input_keys[0]].to(device, non_blocking=True)
            else:
                x = torch.cat(
                    [batch[k].to(device, non_blocking=True) for k in model_input_keys],
                    dim=1,
                )
            logits = model(x)
        pred = logits.argmax(dim=1)
        cm.update(y, pred)

    metrics = cm.compute()
    out = {
        "checkpoint": str(checkpoint_path),
        "n_test_patches": len(patch_ids),
        "test_pixel_accuracy": metrics.pixel_accuracy,
        "test_mean_class_accuracy": metrics.mean_class_accuracy,
        "test_mean_iou": metrics.mean_iou,
        "test_per_class_accuracy": {
            TRAIN_ID_TO_NAME[c]: float(metrics.per_class_accuracy[c])
            for c in range(N_CLASSES)
        },
        "test_per_class_iou": {
            TRAIN_ID_TO_NAME[c]: float(metrics.per_class_iou[c])
            for c in range(N_CLASSES)
        },
        "train_val_mca": float(ckpt.get("val_mca", float("nan"))),
        "train_best_epoch": int(ckpt.get("epoch", -1)),
        "architecture": model_cfg["architecture"],
        "modality": modality,
        "fusion_type": fusion_type,
    }

    # Save alongside the checkpoint
    out_dir = checkpoint_path.parent
    metrics_path = out_dir / "test_metrics.json"
    cm_path = out_dir / "test_confusion_matrix.npy"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    np.save(cm_path, metrics.confusion_matrix)

    print(f"\nTest set results:")
    print(f"  Pixel accuracy:        {metrics.pixel_accuracy:.4f}")
    print(f"  Mean class accuracy:   {metrics.mean_class_accuracy:.4f}")
    print(f"  Mean IoU:              {metrics.mean_iou:.4f}")
    print(f"\nPer-class recall:")
    for c in range(N_CLASSES):
        print(f"  {TRAIN_ID_TO_NAME[c]:<12s} {metrics.per_class_accuracy[c]:.4f}")
    print(f"\nSaved metrics to:          {metrics_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best.pt from a training run.",
    )
    parser.add_argument(
        "--test-data-root",
        type=str,
        default="data/raw/ROIs0000_test",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    evaluate_on_test(
        checkpoint_path=Path(args.checkpoint),
        test_data_root=Path(args.test_data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()