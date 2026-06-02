"""Segmentation metrics computed from a confusion matrix.

We accumulate a per-batch confusion matrix during validation and compute
metrics at the end. This is more numerically stable and memory-efficient
than collecting all predictions in a list, and it allows us to compute
many metrics from the same accumulator.

Metrics provided:
    * Overall pixel accuracy (PA): correctly classified pixels / total.
      Dominated by majority classes; can look "high" while rare classes fail.
    * Per-class recall (a.k.a. class accuracy): TP_c / (TP_c + FN_c).
      The diagonal of the row-normalized confusion matrix.
    * Mean class accuracy (mCA): mean of per-class recall.
      This is the OFFICIAL DFC2020 evaluation metric.
    * Per-class IoU (Jaccard): TP_c / (TP_c + FP_c + FN_c).
    * Mean IoU (mIoU): mean of per-class IoU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SegmentationMetrics:
    """Container for segmentation metric results."""
    pixel_accuracy: float
    mean_class_accuracy: float
    mean_iou: float
    per_class_accuracy: np.ndarray  # shape (n_classes,)
    per_class_iou: np.ndarray       # shape (n_classes,)
    confusion_matrix: np.ndarray    # shape (n_classes, n_classes); rows=true, cols=pred


class ConfusionMatrixAccumulator:
    """Builds a confusion matrix incrementally over many batches.

    Use:
        cm = ConfusionMatrixAccumulator(n_classes=8)
        for batch in val_loader:
            preds = model(batch["s2"]).argmax(dim=1)
            cm.update(batch["label"], preds)
        metrics = cm.compute()
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self._cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, target: torch.Tensor, pred: torch.Tensor) -> None:
        """Add a batch to the confusion matrix.

        Parameters
        ----------
        target : torch.Tensor (N, H, W) int
            Ground-truth class indices in [0, n_classes).
        pred : torch.Tensor (N, H, W) int
            Predicted class indices in [0, n_classes).
        """
        t = target.detach().cpu().numpy().ravel()
        p = pred.detach().cpu().numpy().ravel()
        # Single-pass confusion matrix via integer binning. ~3x faster than
        # a Python loop over classes, especially for big patches.
        idx = t * self.n_classes + p
        bincount = np.bincount(idx, minlength=self.n_classes ** 2)
        self._cm += bincount.reshape(self.n_classes, self.n_classes)

    def reset(self) -> None:
        self._cm.fill(0)

    def compute(self) -> SegmentationMetrics:
        cm = self._cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        total = cm.sum()

        # Per-class recall (= class accuracy). Avoid divide-by-zero.
        denom_recall = tp + fn
        per_class_acc = np.where(denom_recall > 0, tp / np.maximum(denom_recall, 1), 0.0)

        # Per-class IoU
        denom_iou = tp + fp + fn
        per_class_iou = np.where(denom_iou > 0, tp / np.maximum(denom_iou, 1), 0.0)

        # Mean over classes that actually appear in the ground truth
        present = denom_recall > 0
        mean_class_acc = per_class_acc[present].mean() if present.any() else 0.0
        mean_iou = per_class_iou[present].mean() if present.any() else 0.0

        pixel_acc = tp.sum() / total if total > 0 else 0.0

        return SegmentationMetrics(
            pixel_accuracy=float(pixel_acc),
            mean_class_accuracy=float(mean_class_acc),
            mean_iou=float(mean_iou),
            per_class_accuracy=per_class_acc,
            per_class_iou=per_class_iou,
            confusion_matrix=self._cm.copy(),
        )