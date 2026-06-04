"""Dual-encoder U-Net for late-fusion SAR-optical segmentation.

Architecture
------------

Two independent encoders process S1 and S2 separately. At each of the five
encoder scales, the two feature maps are concatenated along the channel
axis and reduced back to the encoder's native channel count by a learned
1x1 convolution. The fused multi-scale features are then consumed by a
standard U-Net decoder.

    S1 (N, 2, H, W)  ---> EncoderA ---> [f_a^0, f_a^1, f_a^2, f_a^3, f_a^4, f_a^5]
                                                                      |
                                                                     fuse_k:
                                                                     cat -> 1x1conv
                                                                      |
    S2 (N, 12, H, W) ---> EncoderB ---> [f_b^0, f_b^1, f_b^2, f_b^3, f_b^4, f_b^5]
                                                                      |
                                                                      v
                                       [g^0, g^1, g^2, g^3, g^4, g^5] -> UnetDecoder -> (N, C, H, W)

Notes
-----
* segmentation_models_pytorch (smp) encoders return a list of feature maps
  including the input "stem" at index 0 and progressively downsampled
  features. We fuse at every scale, including the stem.
* The decoder is smp.decoders.unet.decoder.UnetDecoder, the same one used
  by smp.Unet, so the only architectural difference vs single-encoder
  U-Net is the dual encoder + per-scale fusion module.
* Both encoders share the same architecture (e.g. resnet18) but have
  independent weights. Pretrained ImageNet weights are applied to both;
  smp adapts the first conv layer to the modality's input channels.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder

from sar_optical_fusion.data.dataset import N_CLASSES

EncoderName = Literal["resnet18", "resnet34", "mobilenet_v2"]


class ScaleFusion(nn.Module):
    """Concatenate two same-shape feature tensors, then 1x1 conv to halve channels.

    Input  (N, C, H, W) and (N, C, H, W)
    Output (N, C, H, W)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # No batch norm or activation here: the receiving decoder block
        # applies its own normalization. A bare 1x1 conv is just a learned
        # weighted combination, which is exactly what we want.
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([a, b], dim=1))


class DualEncoderUnet(nn.Module):
    """U-Net with two encoders, one per modality.

    See module docstring for architecture details.
    """

    def __init__(
        self,
        encoder_name: EncoderName = "resnet18",
        encoder_weights: str | None = "imagenet",
        in_channels_a: int = 2,
        in_channels_b: int = 12,
        n_classes: int = N_CLASSES,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        input_projection_channels: int = 8,
    ) -> None:
        super().__init__()

        # Two independent encoders.
        self.encoder_a = get_encoder(
            name=encoder_name,
            in_channels=in_channels_a,
            depth=5,
            weights=encoder_weights,
        )
        self.encoder_b = get_encoder(
            name=encoder_name,
            in_channels=in_channels_b,
            depth=5,
            weights=encoder_weights,
        )

        # smp encoders report channels at every scale, including index 0 = raw input.
        # The raw inputs have different channel counts (e.g. 2 vs 12), so we project
        # both to a common channel count before fusing at the input scale.
        a_channels = self.encoder_a.out_channels  # e.g. (2, 64, 64, 128, 256, 512)
        b_channels = self.encoder_b.out_channels  # e.g. (12, 64, 64, 128, 256, 512)

        # Encoder feature scales (everything after the raw input) MUST match.
        assert a_channels[1:] == b_channels[1:], (
            "Encoder feature scales must match between the two encoders; got "
            f"A={a_channels[1:]} vs B={b_channels[1:]}"
        )

        # Project raw inputs to a common channel count at the input scale.
        self.proj_a_in = nn.Conv2d(a_channels[0], input_projection_channels,
                                    kernel_size=1, bias=False)
        self.proj_b_in = nn.Conv2d(b_channels[0], input_projection_channels,
                                    kernel_size=1, bias=False)

        # The fused channel layout the decoder will see:
        # index 0 = input_projection_channels (after fusion)
        # index 1..5 = encoder feature channels (same as either encoder)
        fused_channels = (input_projection_channels,) + tuple(a_channels[1:])

        # Per-scale fusion modules, one per scale.
        self.fuse = nn.ModuleList(
            [ScaleFusion(c) for c in fused_channels]
        )

        # U-Net decoder operating on the fused multi-scale features.
        self.decoder = UnetDecoder(
            encoder_channels=fused_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_norm="batchnorm",
            add_center_block=False,
            attention_type=None,
        )

        # Final 1x1 conv to project decoder output to class logits.
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=n_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        feats_a = list(self.encoder_a(s1))
        feats_b = list(self.encoder_b(s2))

        # Replace the raw inputs at index 0 with projected versions
        # so the fusion module sees same-channel tensors.
        feats_a[0] = self.proj_a_in(feats_a[0])
        feats_b[0] = self.proj_b_in(feats_b[0])

        fused = [self.fuse[i](feats_a[i], feats_b[i]) for i in range(len(feats_a))]
        decoder_out = self.decoder(fused)
        return self.segmentation_head(decoder_out)


def build_dual_encoder_unet(
    encoder_name: EncoderName = "resnet18",
    encoder_weights: str | None = "imagenet",
    in_channels_a: int = 2,
    in_channels_b: int = 12,
    n_classes: int = N_CLASSES,
) -> DualEncoderUnet:
    """Factory matching the style of build_unet() in models/unet.py."""
    return DualEncoderUnet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels_a=in_channels_a,
        in_channels_b=in_channels_b,
        n_classes=n_classes,
    )