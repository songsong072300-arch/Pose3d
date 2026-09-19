"""Image transforms shared by training and inference."""
from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from .model import IMAGE_SIZE


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    # 进行image的归一化
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def letterbox_image(image, fill=(114, 114, 114)):
    """Resize without distortion and return image, x/y scale, and padding."""
    output_height, output_width = IMAGE_SIZE
    source_width, source_height = image.size
    scale = min(output_width / source_width, output_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    pad_x = (output_width - resized_width) // 2
    pad_y = (output_height - resized_height) // 2
    canvas = Image.new("RGB", (output_width, output_height), fill)
    canvas.paste(resized, (pad_x, pad_y))
    scale_xy = np.asarray(
        (resized_width / source_width, resized_height / source_height),
        dtype=np.float32,
    )
    return canvas, scale_xy, float(pad_x), float(pad_y)
