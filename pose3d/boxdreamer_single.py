"""Single-image corner heatmap network (trimmed BoxDreamer adaptation).

Task goal: RGB image -> 8 corner heatmaps -> peak extraction (corners only,
no PnP/pose stage). Adapted from BoxDreamer Sec 3.2 with these trims:
- The reference-image branch is removed entirely (single fixed object).
- The formerly separate query/pos tables are merged into one learnable
  position embedding: they were only ever used via their sum.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class PoseHeatmapDataset(Dataset):
    def __init__(self, root, split="train", image_size=(180, 240)):
        self.root = Path(root)
        meta = json.loads((self.root / "corner_labels/annotations.json").read_text())
        splits = json.loads((self.root / "corner_labels/splits.json").read_text())
        wanted = set(splits[split])
        self.records = [r for r in meta["records"] if r["id"] in wanted]
        self.image_size = tuple(image_size)  # H, W

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        image = Image.open(self.root / r["image"]).convert("RGB")
        image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        # ImageNet normalization, compatible with DINO-style backbones.
        x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        h = np.load(self.root / r["heatmap"])["heatmaps"].astype(np.float32) / 255.0
        return {"image": x, "heatmap": torch.from_numpy(h), "id": r["id"]}


class BoxDreamerSingle(nn.Module):
    """RGB -> 8-channel corner heatmap.

    patchify (Conv2d) -> +learnable position embedding -> Transformer
    -> per-token patch prediction (Linear) -> fold -> sigmoid.
    """

    def __init__(self, image_size=(180, 240), patch_size=15, dim=256,
                 depth=6, heads=8, dropout=0.1, out_channels=8):
        super().__init__()
        h, w = image_size
        if h % patch_size or w % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid = (h // patch_size, w // patch_size)
        self.patch_size, self.dim, self.out_channels = patch_size, dim, out_channels
        self.image_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        num_tokens = self.grid[0] * self.grid[1]
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels * patch_size * patch_size)
        # Sparse targets: start near zero instead of sigmoid's 0.5 midpoint.
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, image):
        # (B,3,H,W) -> (B,N,dim) token sequence, N = (H/p)*(W/p)
        z = self.image_embed(image).flatten(2).transpose(1, 2)
        z = self.transformer(z + self.pos)
        # each token predicts its own out*p*p heatmap patch
        y = self.head(self.norm(z)).transpose(1, 2)
        y = F.fold(y, output_size=(self.grid[0] * self.patch_size, self.grid[1] * self.patch_size),
                   kernel_size=self.patch_size, stride=self.patch_size)
        return torch.sigmoid(y)


def heatmap_loss(pred, target, peak_weight=100.0):
    """Foreground-weighted smooth-L1 loss for sparse corner heatmaps.

    ~99.7% of target pixels are background, so a plain loss lets the network
    cheat by predicting zeros everywhere (validated: loss plateaus with peaks
    never localizing). Weighting each pixel by 1 + peak_weight * target makes
    peak regions dominate the gradient again; with this the 2-sample overfit
    smoke drives peak error from ~110px to <1px.
    """
    weights = 1.0 + peak_weight * target
    diff = torch.abs(pred - target)
    elementwise = torch.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)
    return (elementwise * weights).mean()
