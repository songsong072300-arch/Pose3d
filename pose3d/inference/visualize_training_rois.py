"""Visualize the ROI crops used by RoiCornerDataset."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..common.image import IMAGENET_MEAN, IMAGENET_STD
from ..train.dataset import RoiCornerDataset


def tensor_to_image(tensor):
    array = tensor.detach().cpu().numpy()
    array = array * IMAGENET_STD.numpy() + IMAGENET_MEAN.numpy()
    array = np.clip(array.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="TripoSR/dataset/lm/new_train_pbr")
    parser.add_argument("--labels-dir", default="new_corner_labels_v1")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--out", default="outputs/training_roi_visualization")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = RoiCornerDataset(args.root, args.split, labels_dir=args.labels_dir)
    if not dataset:
        raise SystemExit("dataset split is empty")
    count = min(args.count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count).round().astype(int)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    panels = []

    for panel_id, index in enumerate(indices):
        batch = dataset[int(index)]
        record, instance = dataset.items[int(index)]
        image = Image.open(Path(args.root) / record["image"]).convert("RGB")
        corners = np.rint(np.asarray(instance["corners_2d"], dtype=np.float32)).astype(int)
        origin = batch["crop_origin"].numpy().astype(int)
        padding = batch["letterbox_padding"].numpy()
        scale = batch["scale_xy"].numpy()
        crop_width = max(1, int(round((224 - 2 * padding[0]) / scale[0])))
        crop_height = max(1, int(round((168 - 2 * padding[1]) / scale[1])))
        crop_box = (
            int(origin[0]), int(origin[1]),
            min(image.width, int(origin[0]) + crop_width),
            min(image.height, int(origin[1]) + crop_height),
        )

        original = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.rectangle(original, crop_box[:2], (crop_box[2] - 1, crop_box[3] - 1),
                      (0, 220, 255), 3)
        for point in corners:
            cv2.circle(original, tuple(point), 5, (40, 220, 40), -1, cv2.LINE_AA)
        original = cv2.resize(original, (448, 336), interpolation=cv2.INTER_AREA)
        crop = tensor_to_image(batch["image"])
        crop = cv2.resize(crop, (448, 336), interpolation=cv2.INTER_NEAREST)
        cv2.putText(original, "original + training ROI", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(crop, "network input 224x168", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
        panel = np.concatenate([original, crop], axis=1)
        cv2.putText(panel, f"sample {panel_id}  {record['id']}",
                    (10, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
        cv2.imwrite(str(output / f"roi_{panel_id:02d}.jpg"), panel)

    cv2.imwrite(str(output / "contact_sheet.jpg"), np.concatenate(panels, axis=0))
    print(output / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
