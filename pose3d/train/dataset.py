"""Single-instance ROI dataset for ordered corner supervision."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from ..common.image import image_to_tensor, letterbox_image
from ..common.model import IMAGE_SIZE


def expanded_corner_box(corners, image_size, pad=0.15, jitter=0.0):
    """Build a padded, randomly shifted crop containing all corners."""
    image_width, image_height = image_size
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    object_width = max(float(x2 - x1), 2.0)
    object_height = max(float(y2 - y1), 2.0)
    extra = random.uniform(-jitter, jitter) if jitter else 0.0
    effective_pad = max(0.08, pad + extra)
    shift_x = random.uniform(-jitter, jitter) * object_width if jitter else 0.0
    shift_y = random.uniform(-jitter, jitter) * object_height if jitter else 0.0
    xa = min(x1 - 1.0, x1 - effective_pad * object_width + shift_x)
    ya = min(y1 - 1.0, y1 - effective_pad * object_height + shift_y)
    xb = max(x2 + 1.0, x2 + effective_pad * object_width + shift_x)
    yb = max(y2 + 1.0, y2 + effective_pad * object_height + shift_y)
    return (
        max(0, math.floor(xa)),
        max(0, math.floor(ya)),
        min(image_width, math.ceil(xb)),
        min(image_height, math.ceil(yb)),
    )


class RoiCornerDataset(torch.utils.data.Dataset):
    """Derive one distortion-free corner sample per annotated instance."""

    def __init__(self, root, split="train", pad=0.15, crop_jitter=0.05):
        self.root = Path(root)
        metadata = json.loads(
            (self.root / "corner_labels/annotations.json").read_text()
        )
        splits = json.loads(
            (self.root / "corner_labels/splits.json").read_text()
        )
        wanted = set(splits[split])
        self.items = [
            (record, instance)
            for record in metadata["records"] if record["id"] in wanted
            for instance in record["instances"]
        ]
        self.pad = pad
        self.crop_jitter = crop_jitter if split == "train" else 0.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record, instance = self.items[index]
        image = Image.open(self.root / record["image"]).convert("RGB")
        corners = np.asarray(instance["corners_2d"], dtype=np.float32)
        crop_box = expanded_corner_box(
            corners, image.size, self.pad, self.crop_jitter,
        )
        crop = image.crop(crop_box)
        crop, scale_xy, pad_x, pad_y = letterbox_image(crop)
        roi_corners = ((corners - np.asarray(crop_box[:2], dtype=np.float32))
                       * scale_xy)
        roi_corners += np.asarray((pad_x, pad_y), dtype=np.float32)

        height, width = IMAGE_SIZE
        yy, xx = np.mgrid[0:height, 0:width]
        heatmaps = np.zeros((8, height, width), dtype=np.float32)
        projected_extent = np.ptp(roi_corners, axis=0)
        sigma = max(2.0, float(projected_extent.min()) / 18.0)
        for channel, (x, y) in enumerate(roi_corners):
            heatmaps[channel] = np.exp(
                -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2)
            )
        return {
            "image": image_to_tensor(crop),
            "heatmap": torch.from_numpy(heatmaps),
            "corners": torch.from_numpy(roi_corners),
            "id": f"{record['id']}_{instance['instance_id']}",
        }
