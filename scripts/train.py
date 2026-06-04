"""Training entry point.

Use:
    uv run python scripts/train.py experiment=s2_only
    uv run python scripts/train.py experiment=s1_only training.num_epochs=10
    uv run python scripts/train.py experiment=early_fusion training.lr=5e-4

The script reads a Hydra config, builds the dataset/model/loss/loaders,
and hands them to fit() in sar_optical_fusion.training.loop. All
heavy lifting lives in the library; this script is plumbing.
"""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from sar_optical_fusion.data.dataset import (
    DFC2020Dataset,
    N_CLASSES,
    build_train_augmentation,
)
from sar_optical_fusion.data.splits import load_split
from sar_optical_fusion.models.unet import build_unet, count_parameters
from sar_optical_fusion.training.loop import TrainConfig, fit
from sar_optical_fusion.training.loss import build_loss


def set_seed(seed: int) -> None:
    """Seed all random sources we control. DataLoader worker seeding is
    handled by PyTorch via the base seed; we do not pursue full bit-level
    determinism (which would require disabling cuDNN benchmarking and
    cost significant throughput)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and validation loaders from a resolved Hydra config."""
    split = load_split(cfg.dataset.split_path)
    data_root = Path(cfg.dataset.data_root)

    aug = build_train_augmentation() if cfg.dataset.augmentation == "d4" else None

    train_ds = DFC2020Dataset(
        data_root=data_root,
        patch_ids=split["train"],
        split_name=cfg.dataset.split_name,
        modality=cfg.model.modality,
        transform=aug,
    )
    val_ds = DFC2020Dataset(
        data_root=data_root,
        patch_ids=split["val"],
        split_name=cfg.dataset.split_name,
        modality=cfg.model.modality,
        transform=None,
    )

    print(f"Train patches: {len(train_ds)}    Val patches: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.num_workers > 0,
        drop_last=False,
    )
    return train_loader, val_loader


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 70)

    set_seed(cfg.seed)

    # Data
    train_loader, val_loader = build_dataloaders(cfg)

   # Model — dispatch on architecture
    if cfg.model.architecture == "unet":
        model = build_unet(
            in_channels=cfg.model.in_channels,
            n_classes=N_CLASSES,
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
        )
    elif cfg.model.architecture == "dual_encoder_unet":
        from sar_optical_fusion.models.dual_encoder_unet import build_dual_encoder_unet
        model = build_dual_encoder_unet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels_a=cfg.model.in_channels_a,
            in_channels_b=cfg.model.in_channels_b,
            n_classes=N_CLASSES,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.model.architecture!r}")
    total, trainable = count_parameters(model)
    print(f"Model: {cfg.model.name}  ({total/1e6:.2f}M params, "
          f"{trainable/1e6:.2f}M trainable)")

    # Loss
    loss_fn = build_loss(cfg.training.loss)

    # Build the TrainConfig that the loop expects
    train_cfg = TrainConfig(
        experiment_name=cfg.experiment.name,
        model_input_keys=list(cfg.model.model_input_keys),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        num_epochs=cfg.training.num_epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        checkpoint_dir=cfg.training.checkpoint_dir,
        log_every_n_steps=cfg.training.log_every_n_steps,
        use_wandb=cfg.training.use_wandb,
        wandb_project=cfg.training.wandb_project,
        seed=cfg.seed,
        amp=cfg.training.amp,
        extra_config={
            "fusion_type": cfg.model.get("fusion_type", "early"),
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
            "training": OmegaConf.to_container(cfg.training, resolve=True),
            "experiment_description": cfg.experiment.description,
        },
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        cfg=train_cfg,
        n_classes=N_CLASSES,
        device=device,
    )

    print("\n" + "=" * 70)
    print(f"Training finished.")
    print(f"  Experiment:       {cfg.experiment.name}")
    print(f"  Best val mCA:     {summary['best_val_mca']:.4f}")
    print(f"  Best epoch:       {summary['best_epoch']}/{cfg.training.num_epochs}")
    print(f"  Checkpoint:       {summary['checkpoint_path']}")
    print(f"  Total time:       {summary['total_time_seconds']:.0f}s "
          f"({summary['total_time_seconds']/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()