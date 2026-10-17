"""DINOv2-Small network for eight geometrically related 2D box corners."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


IMAGE_SIZE = (168, 224)  # H, W; divisible by DINOv2 patch size 14.
DECODER_DIM = 128
CORNER_COUNT = 8


def load_dinov2_backbone():
    project_cache = Path(__file__).resolve().parents[2] / ".torch-cache" / "hub"
    hub_dirs = (Path(torch.hub.get_dir()), project_cache)
    for hub_dir in hub_dirs:
        repository = hub_dir / "facebookresearch_dinov2_main"
        if repository.is_dir():
            return torch.hub.load(
                str(repository), "dinov2_vits14", source="local",
            )
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class CornerHeatmapNet(nn.Module):
    """Predict joint corner heatmaps and an intrinsic-free projective cuboid.

    The convolutional path preserves local detail. Eight learned queries exchange
    information and attend to DINO tokens before producing a shared heatmap
    correction. An auxiliary projective head maps the canonical 3D cuboid to all
    eight normalized 2D corners with one transform.
    """
    def __init__(self):
        super().__init__()
        self.backbone = load_dinov2_backbone()
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.backbone.blocks[-2:].parameters():
            parameter.requires_grad_(True)
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad_(True)

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

        # The legacy convolutional head predicts every heatmap independently.
        # Corner queries communicate through self-attention before attending to
        # image tokens, so evidence for one box corner can inform the others.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=DECODER_DIM,
            nhead=8,
            dim_feedforward=DECODER_DIM * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.corner_queries = nn.Parameter(
            torch.empty(1, CORNER_COUNT, DECODER_DIM),
        )
        self.image_position = nn.Parameter(
            torch.empty(1, 12 * 16, DECODER_DIM),
        )
        self.query_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.pixel_projection = nn.Conv2d(
            DECODER_DIM // 2, DECODER_DIM, kernel_size=1,
        )
        # Start small so the stable convolutional path dominates early training.
        # The checkpoint loader resets this to zero for legacy checkpoints.
        self.query_scale = nn.Parameter(torch.tensor(0.05))

        # Predict one normalized 3x4 projective transform (11 free parameters)
        # rather than eight unrelated points. It needs no explicit intrinsics,
        # does not assume parallel image edges, and generates all corners jointly.
        self.geometry_head = nn.Sequential(
            nn.Linear(DECODER_DIM, DECODER_DIM),
            nn.GELU(),
            nn.Linear(DECODER_DIM, 11),
        )
        corner_signs = torch.tensor([
            [x, y, z]
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
        ], dtype=torch.float32)
        self.register_buffer("corner_signs", corner_signs, persistent=False)

        nn.init.trunc_normal_(self.corner_queries, std=0.02)
        nn.init.trunc_normal_(self.image_position, std=0.02)
        nn.init.normal_(self.geometry_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.geometry_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        if mode == True:
            self.backbone.train()
        else :
            self.backbone.eval()
        return self

    def forward(self, image, return_aux=False):
        if tuple(image.shape[-2:]) != IMAGE_SIZE:
            raise ValueError(f"expected input size {IMAGE_SIZE}, got {image.shape[-2:]}")

        features = self.backbone.get_intermediate_layers(
                image, n=4, reshape=True, norm=True,
        )
        projected = [layer(feature)
                     for layer, feature in zip(self.projections, features)]
        decoded = self.fuse(torch.cat(projected, dim=1))
        fused_features = decoded
        decoded = F.interpolate(
            decoded, scale_factor=2.0, mode="bilinear", align_corners=False,
        )
        decoded = self.decode[0](decoded)
        decoded = F.interpolate(
            decoded, scale_factor=2.0, mode="bilinear", align_corners=False,
        )
        decoded = self.decode[1](decoded)
        low_resolution = decoded
        decoded = F.interpolate(
            decoded, size=IMAGE_SIZE, mode="bilinear", align_corners=False,
        )
        base_logits = self.head(decoded)

        memory = projected[-1].flatten(2).transpose(1, 2)
        memory = memory + self.image_position
        queries = self.corner_queries.expand(image.shape[0], -1, -1)
        queries = self.query_decoder(queries, memory)
        pixel_features = self.pixel_projection(low_resolution)
        query_logits = torch.einsum(
            "bcd,bdhw->bchw", queries, pixel_features,
        ) / (DECODER_DIM ** 0.5)
        query_logits = F.interpolate(
            query_logits, size=IMAGE_SIZE, mode="bilinear", align_corners=False,
        )
        heatmaps = torch.sigmoid(base_logits + self.query_scale * query_logits)

        geometry = self.geometry_head(fused_features.mean(dim=(-2, -1)))
        centre = torch.sigmoid(geometry[:, :2])
        numerator_basis = 0.75 * torch.tanh(
            geometry[:, 2:8].reshape(-1, 2, 3),
        )
        perspective = 0.5 * torch.tanh(geometry[:, 8:11])
        numerator = centre[:, :, None] + torch.einsum(
            "kc,bdc->bdk", self.corner_signs, numerator_basis,
        )
        denominator = 1.0 + torch.einsum(
            "kc,bc->bk", self.corner_signs, perspective,
        )
        coarse_corners = numerator.transpose(1, 2) / denominator[:, :, None]
        if return_aux:
            return {
                "heatmaps": heatmaps,
                "coarse_corners": coarse_corners,
            }
        return heatmaps
