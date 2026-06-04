"""Training loop for DFC2020 segmentation.

The loop is modality-agnostic: it pulls 's1', 's2', or both from the batch
dict based on the `model_input_keys` config. This is what lets the same
loop train all four model variants without modification.

Key design choices:
    * Mixed precision (autocast + GradScaler) to halve VRAM usage and run
      ~1.5x faster on the 1050 Ti. Critical for 4 GB cards.
    * Per-epoch validation by mean class accuracy (the official metric).
    * Best-model checkpointing keyed on val mCA, not train loss.
    * Optional W&B logging; runs locally if wandb is disabled.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from sar_optical_fusion.evaluation.metrics import (
    ConfusionMatrixAccumulator,
    SegmentationMetrics,
)


@dataclass
class TrainConfig:
    """Hyperparameters and runtime settings for a training run."""
    # Required
    experiment_name: str
    model_input_keys: list[str]   # e.g. ["s2"] or ["s1"] or ["s1", "s2"]
    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Schedule
    num_epochs: int = 30
    warmup_epochs: int = 0
    # Data
    batch_size: int = 8
    num_workers: int = 2
    # Logging / checkpointing
    checkpoint_dir: str = "checkpoints"
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "sar-optical-fusion-dfc2020"
    # Misc
    seed: int = 42
    amp: bool = True   # mixed-precision training
    extra_config: dict[str, Any] = field(default_factory=dict)


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch dict to device; leave non-tensors alone."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _stack_inputs(batch: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    """Build the model input by concatenating one or more modality tensors.

    For early fusion, pass keys=["s1", "s2"]; this concatenates along channel
    axis (dim=1). For single-modality models, pass keys=["s2"] or keys=["s1"].
    """
    if len(keys) == 1:
        return batch[keys[0]]
    return torch.cat([batch[k] for k in keys], dim=1)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    """One epoch of training. Returns mean loss for the epoch."""
    model.train()
    running_loss = 0.0
    running_n = 0
    is_dual = len(cfg.model_input_keys) == 2 and cfg.extra_config.get(
        "fusion_type", "early"
    ) == "late"
    pbar = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for step, batch in enumerate(pbar):
        batch = _move_to_device(batch, device)
        y = batch["label"]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                if is_dual:
                    logits = model(batch[cfg.model_input_keys[0]],
                                   batch[cfg.model_input_keys[1]])
                else:
                    x = _stack_inputs(batch, cfg.model_input_keys)
                    logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if is_dual:
                logits = model(batch[cfg.model_input_keys[0]],
                               batch[cfg.model_input_keys[1]])
            else:
                x = _stack_inputs(batch, cfg.model_input_keys)
                logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        # batch size for averaging - pick from any tensor in the batch
        n = y.size(0)
        running_loss += loss.item() * n
        running_n += n
        if step % cfg.log_every_n_steps == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": running_loss / max(running_n, 1)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    n_classes: int,
) -> tuple[SegmentationMetrics, float]:
    """One full evaluation pass. Returns (metrics, mean_loss)."""
    model.eval()
    cm = ConfusionMatrixAccumulator(n_classes)
    running_loss = 0.0
    running_n = 0
    is_dual = len(cfg.model_input_keys) == 2 and cfg.extra_config.get(
        "fusion_type", "early"
    ) == "late"
    for batch in tqdm(loader, desc="Validating", leave=False):
        batch = _move_to_device(batch, device)
        y = batch["label"]
        if is_dual:
            logits = model(batch[cfg.model_input_keys[0]],
                           batch[cfg.model_input_keys[1]])
        else:
            x = _stack_inputs(batch, cfg.model_input_keys)
            logits = model(x)
        loss = loss_fn(logits, y)
        n = y.size(0)
        running_loss += loss.item() * n
        running_n += n
        pred = logits.argmax(dim=1)
        cm.update(y, pred)
    return cm.compute(), running_loss / max(running_n, 1)


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    cfg: TrainConfig,
    n_classes: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train a model end-to-end and return a summary.

    Returns
    -------
    dict
        Keys:
            best_val_mca: float            mean class accuracy of best epoch
            best_epoch: int
            history: list[dict]            per-epoch metrics
            checkpoint_path: str
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = loss_fn.to(device) #: nn.Module objects don't automatically move their non-parameter buffers to the GPU just because the model does. The class weights aren't a parameter (they don't get learned), but they aren't a registered buffer either — they're just an attribute of the CrossEntropyLoss module set at construction time. So .to(device) won't pull them along even if you call it.

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    # scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type == "cuda") else None
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None

    # Checkpoint directory: checkpoints/<experiment_name>/
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"
    history_path = ckpt_dir / "history.json"

    # Optional W&B
    wandb_run = None
    if cfg.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.experiment_name,
            config={
                **{k: v for k, v in cfg.__dict__.items() if k != "extra_config"},
                **cfg.extra_config,
            },
        )

    best_val_mca = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []

    t_start = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        # Train
        train_stats = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, cfg, epoch, scaler,
        )
        # Validate
        val_metrics, val_loss = evaluate(
            model, val_loader, loss_fn, device, cfg, n_classes,
        )
        scheduler.step()

        # Log to console
        elapsed = time.time() - t_start
        msg = (
            f"Epoch {epoch:>3d}/{cfg.num_epochs} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_PA={val_metrics.pixel_accuracy:.4f} | "
            f"val_mCA={val_metrics.mean_class_accuracy:.4f} | "
            f"val_mIoU={val_metrics.mean_iou:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"elapsed={elapsed:.0f}s"
        )
        print(msg)

        # Log to W&B
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_stats["loss"],
                    "val/loss": val_loss,
                    "val/pixel_accuracy": val_metrics.pixel_accuracy,
                    "val/mean_class_accuracy": val_metrics.mean_class_accuracy,
                    "val/mean_iou": val_metrics.mean_iou,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

        # Record history
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_loss,
                "val_pa": val_metrics.pixel_accuracy,
                "val_mca": val_metrics.mean_class_accuracy,
                "val_miou": val_metrics.mean_iou,
                "per_class_acc": val_metrics.per_class_accuracy.tolist(),
                "per_class_iou": val_metrics.per_class_iou.tolist(),
                "lr": scheduler.get_last_lr()[0],
            }
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save best checkpoint
        if val_metrics.mean_class_accuracy > best_val_mca:
            best_val_mca = val_metrics.mean_class_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_mca": best_val_mca,
                    "cfg": cfg.__dict__,
                },
                best_ckpt,
            )

    summary = {
        "best_val_mca": best_val_mca,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint_path": str(best_ckpt),
        "total_time_seconds": time.time() - t_start,
    }
    if wandb_run is not None:
        wandb_run.summary["best_val_mca"] = best_val_mca
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()
    return summary