"""End-to-end smoke test of the training pipeline.

Trains an S2-only U-Net on 5 patches for 2 epochs. The goal is to verify
that data loading, augmentation, model forward/backward, loss, metrics,
checkpoint saving, and history logging all work together.

If this script runs to completion without error, the full training run
in scripts/train.py is unlikely to fail for plumbing reasons. (It may
still produce a bad model -- that's a research problem, not a plumbing
problem.)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sar_optical_fusion.data.dataset import (
    DFC2020Dataset,
    N_CLASSES,
    build_train_augmentation,
)
from sar_optical_fusion.data.splits import load_split
from sar_optical_fusion.models.unet import build_unet
from sar_optical_fusion.training.loop import TrainConfig, fit
from sar_optical_fusion.training.loss import build_loss


def main() -> None:
    # Small subset for a fast smoke test
    split = load_split("src/sar_optical_fusion/data/splits.json")
    train_ids = split["train"][:5]
    val_ids = split["val"][:5]
    print(f"Smoke test: {len(train_ids)} train patches, {len(val_ids)} val patches.")

    data_root = Path("data/raw/ROIs0000_validation")

    train_ds = DFC2020Dataset(
        data_root=data_root,
        patch_ids=train_ids,
        split_name="validation",
        modality="s2",
        transform=build_train_augmentation(),
    )
    val_ds = DFC2020Dataset(
        data_root=data_root,
        patch_ids=val_ids,
        split_name="validation",
        modality="s2",
        transform=None,
    )

    # On Windows, num_workers > 0 requires this script to be run as a
    # module (which it is). We use num_workers=0 in the smoke test for
    # simplicity; the real training run will use workers.
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True,
    )

    # 12-channel input for S2
    model = build_unet(in_channels=12, n_classes=N_CLASSES, encoder_name="resnet18")
    loss_fn = build_loss("weighted_ce")

    cfg = TrainConfig(
        experiment_name="smoke_test_s2",
        model_input_keys=["s2"],
        num_epochs=2,
        batch_size=2,
        num_workers=0,
        lr=1e-3,
        use_wandb=False,
        amp=True,
        checkpoint_dir="checkpoints",
    )

    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        cfg=cfg,
        n_classes=N_CLASSES,
    )

    print("\n=== Smoke test summary ===")
    print(f"Best val mCA:    {summary['best_val_mca']:.4f}")
    print(f"Best epoch:      {summary['best_epoch']}")
    print(f"Checkpoint:      {summary['checkpoint_path']}")
    print(f"Total time:      {summary['total_time_seconds']:.1f} s")
    print(f"Epochs in history: {len(summary['history'])}")

    # Sanity checks on what we expect to exist after running
    ckpt = Path(summary["checkpoint_path"])
    history = Path("checkpoints/smoke_test_s2/history.json")
    assert ckpt.exists(), f"Checkpoint missing: {ckpt}"
    assert history.exists(), f"History missing: {history}"
    print(f"\nArtifacts on disk:\n  {ckpt}\n  {history}")


if __name__ == "__main__":
    main()