"""Fixed DINOv2-Small network for eight ordered 2D box corners."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


IMAGE_SIZE = (168, 224)  # H, W; divisible by DINOv2 patch size 14.
DECODER_DIM = 128


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class CornerHeatmapNet(nn.Module):
    """Frozen DINOv2-Small features plus a spatial fusion decoder."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
        )
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        feature_dim = self.backbone.embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(feature_dim, DECODER_DIM, 1) for _ in range(4)
        ])
        self.fuse = nn.Sequential(
            ConvNormAct(DECODER_DIM * 4, DECODER_DIM),
            ConvNormAct(DECODER_DIM, DECODER_DIM),
        )
        self.decode = nn.ModuleList([
            ConvNormAct(DECODER_DIM, DECODER_DIM),
            ConvNormAct(DECODER_DIM, DECODER_DIM // 2),
        ])
        self.head = nn.Conv2d(DECODER_DIM // 2, 8, 1)
        nn.init.constant_(self.head.bias, -4.0)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image):
        if tuple(image.shape[-2:]) != IMAGE_SIZE:
            raise ValueError(f"expected input size {IMAGE_SIZE}, got {image.shape[-2:]}")
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                image, n=4, reshape=True, norm=True,
            )
        projected = [layer(feature)
                     for layer, feature in zip(self.projections, features)]
        decoded = self.fuse(torch.cat(projected, dim=1))
        decoded = F.interpolate(
            decoded, scale_factor=2.0, mode="bilinear", align_corners=False,
        )
        decoded = self.decode[0](decoded)
        decoded = F.interpolate(
            decoded, scale_factor=2.0, mode="bilinear", align_corners=False,
        )
        decoded = self.decode[1](decoded)
        decoded = F.interpolate(
            decoded, size=IMAGE_SIZE, mode="bilinear", align_corners=False,
        )
        return torch.sigmoid(self.head(decoded))
