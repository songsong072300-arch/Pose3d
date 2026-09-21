#!/usr/bin/env python3
"""Overlay ordered BOP 3D-box corner ground truth on rendered RGB images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


EDGES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)
CORNER_COLORS = (
    (50, 80, 255), (40, 180, 255), (40, 230, 230), (50, 210, 80),
    (255, 190, 40), (255, 90, 50), (220, 70, 210), (255, 255, 255),
)
INSTANCE_COLORS = ((30, 240, 80), (255, 170, 30), (220, 70, 220))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Optional label IDs such as 000000_000000")
    parser.add_argument("--instance-crops", action="store_true",
                        help="Also save enlarged crops around each projected cuboid")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    annotations = json.loads(
        (dataset / "corner_labels" / "annotations.json").read_text())
    output_dir = args.output_dir or dataset / "corner_labels" / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.ids) if args.ids else None

    written = 0
    for record in annotations["records"]:
        if requested is not None and record["id"] not in requested:
            continue
        image = cv2.imread(str(dataset / record["image"]))
        if image is None:
            raise FileNotFoundError(dataset / record["image"])

        for instance_index, instance in enumerate(record["instances"]):
            points = np.rint(np.asarray(instance["corners_2d"])).astype(np.int32)
            valid = np.asarray(instance["corners_in_frame"], dtype=bool)
            instance_color = INSTANCE_COLORS[instance_index % len(INSTANCE_COLORS)]
            for start, end in EDGES:
                if valid[start] and valid[end]:
                    cv2.line(image, tuple(points[start]), tuple(points[end]),
                             instance_color, 2, cv2.LINE_AA)
            for corner_index, (point, is_valid) in enumerate(zip(points, valid)):
                if not is_valid:
                    continue
                location = tuple(point)
                cv2.circle(image, location, 7, (15, 15, 15), -1, cv2.LINE_AA)
                cv2.circle(image, location, 5, CORNER_COLORS[corner_index],
                           -1, cv2.LINE_AA)
                label = f"{instance_index}:{corner_index}"
                cv2.putText(image, label, (location[0] + 7, location[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10),
                            3, cv2.LINE_AA)
                cv2.putText(image, label, (location[0] + 7, location[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255),
                            1, cv2.LINE_AA)

            if args.instance_crops and valid.any():
                visible_points = points[valid]
                x1, y1 = visible_points.min(axis=0)
                x2, y2 = visible_points.max(axis=0)
                pad = max(24, int(max(x2 - x1, y2 - y1) * 0.45))
                height, width = image.shape[:2]
                xa, ya = max(0, x1 - pad), max(0, y1 - pad)
                xb, yb = min(width, x2 + pad), min(height, y2 + pad)
                crop = image[ya:yb, xa:xb]
                if crop.size:
                    crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_CUBIC)
                    crop_path = output_dir / f"{record['id']}_instance_{instance_index}.jpg"
                    cv2.imwrite(str(crop_path), crop)

        cv2.putText(image, "label = instance:corner", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(image, "label = instance:corner", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        output_path = output_dir / f"{record['id']}_corners.jpg"
        cv2.imwrite(str(output_path), image)
        print(output_path)
        written += 1

    print(f"Wrote {written} corner overlays")


if __name__ == "__main__":
    main()
