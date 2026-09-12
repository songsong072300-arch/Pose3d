"""Single-image BoxDreamer-style corner heatmap model.

This adapts Section 3.2 of BoxDreamer to the current fixed-object task:
RGB image -> patch tokens -> Transformer -> 8 corner heatmaps.
The original reference-image branch is intentionally optional because the
current dataset contains one query image with two identical instances.
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
    def __init__(self, image_size=(180, 240), patch_size=15, dim=256,
                 depth=6, heads=8, dropout=0.1, out_channels=8):
        super().__init__()
        h, w = image_size
        if h % patch_size or w % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid = (h // patch_size, w // patch_size)
        self.patch_size, self.dim, self.out_channels = patch_size, dim, out_channels
        self.image_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.heatmap_embed = nn.Linear(out_channels * patch_size * patch_size, dim)
        self.query = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], dim))
        self.pos = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], dim))
        nn.init.trunc_normal_(self.query, std=.02)
        nn.init.trunc_normal_(self.pos, std=.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels * patch_size * patch_size)

    def forward(self, image, reference_heatmap=None):
        # reference_heatmap is accepted for API compatibility with Section 3.2.
        z = self.image_embed(image).flatten(2).transpose(1, 2)
        if reference_heatmap is not None:
            hp = F.unfold(reference_heatmap, self.patch_size, stride=self.patch_size).transpose(1, 2)
            z = z + self.heatmap_embed(hp)
        z = self.transformer(z + self.query + self.pos)
        y = self.head(self.norm(z)).transpose(1, 2)
        y = F.fold(y, output_size=(self.grid[0] * self.patch_size, self.grid[1] * self.patch_size),
                   kernel_size=self.patch_size, stride=self.patch_size)
        return torch.sigmoid(y)


def heatmap_loss(pred, target):
    """Smooth-L1 coarse loss from paper Section 3.3."""
    return F.smooth_l1_loss(pred, target)

