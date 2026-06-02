"""U-Net wrapper around segmentation_models_pytorch.

The same factory function builds U-Nets for all four experimental settings
by varying `in_channels`:

    s1-only:        in_channels = 2   (VV, VH)
    s2-only:        in_channels = 12  (B1..B12 with B10 dropped)
    early fusion:   in_channels = 14  (2 + 12)
    late fusion:    uses build_dual_encoder_unet() instead (Phase 6)

The encoder is a small ResNet by default. The 1050 Ti has 4 GB VRAM; we
budget for batch 8 at 256x256 with mixed precision, which an encoder of
resnet18-resnet34 size accommodates comfortably.
"""

from __future__ import annotations

from typing import Literal

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from sar_optical_fusion.data.dataset import N_CLASSES

EncoderName = Literal["resnet18", "resnet34", "mobilenet_v2"]


def build_unet(
    in_channels: int,
    n_classes: int = N_CLASSES,
    encoder_name: EncoderName = "resnet18",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build a U-Net for semantic segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Use 2 for S1-only, 12 for S2-only,
        14 for early fusion.
    n_classes : int
        Number of output classes. Default is N_CLASSES (8 for DFC2020).
    encoder_name : str
        Backbone name as recognized by segmentation_models_pytorch.
        "resnet18" is the lightest reasonable choice; "resnet34" if VRAM
        allows; "mobilenet_v2" as an even smaller fallback.
    encoder_weights : str | None
        Pre-training to use for the encoder. "imagenet" loads ImageNet
        weights. For non-3-channel inputs, smp adapts the first conv layer
        by repeating/averaging the pretrained weights. None means random init.

    Returns
    -------
    nn.Module
        The U-Net model. Forward signature: (N, in_channels, H, W) -> (N, n_classes, H, W).
        Logits, NOT softmax probabilities. nn.CrossEntropyLoss takes logits.
    """
    if encoder_weights is not None and in_channels != 3:
        # smp does support pretrained weights with non-3 inputs (it averages
        # or replicates the first conv kernel), but emit a one-time note so
        # we know it's happening.
        pass  # handled silently by smp; we keep this branch for documentation

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=n_classes,
        activation=None,  # raw logits; loss applies softmax internally
    )
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total params, trainable params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable