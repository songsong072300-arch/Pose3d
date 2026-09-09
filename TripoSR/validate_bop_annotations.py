#!/usr/bin/env python3
"""Overlay projected BOP 3D bounding boxes on rendered RGB images."""

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
COLORS = ((40, 220, 40), (255, 150, 20), (40, 80, 255), (220, 40, 220))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "demo-bin-picking",
    )
    parser.add_argument("--scene", default="000000")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frames", nargs="*", type=int, default=None,
                        help="只检查指定帧，例如 --frames 50 300")
    return parser.parse_args()


def bbox_corners(info):
    return np.array([
        [x, y, z]
        for x in (info["min_x"], info["max_x"])
        for y in (info["min_y"], info["max_y"])
        for z in (info["min_z"], info["max_z"])
    ], dtype=np.float64)


def main():
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    scene_dir = dataset / "train_pbr" / args.scene
    output_dir = args.output_dir or (scene_dir / "bbox_overlay")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = json.loads((dataset / "models" / "models_info.json").read_text())
    scene_gt = json.loads((scene_dir / "scene_gt.json").read_text())
    scene_camera = json.loads((scene_dir / "scene_camera.json").read_text())

    for frame_key, instances in scene_gt.items():
        if args.frames is not None and int(frame_key) not in args.frames:
            continue
        image_path = scene_dir / "rgb" / f"{int(frame_key):06d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        K = np.asarray(scene_camera[frame_key]["cam_K"], dtype=np.float64).reshape(3, 3)

        for index, instance in enumerate(instances):
            corners = bbox_corners(model_info[str(instance["obj_id"])])
            R = np.asarray(instance["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(instance["cam_t_m2c"], dtype=np.float64).reshape(1, 3)
            camera_points = corners @ R.T + t
            pixels_h = camera_points @ K.T
            pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
            pixels = np.rint(pixels).astype(np.int32)
            color = COLORS[index % len(COLORS)]
            for start, end in EDGES:
                cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), color, 2, cv2.LINE_AA)
            center = np.rint(np.mean(pixels, axis=0)).astype(np.int32)
            cv2.putText(image, f"instance {index + 1}", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        output_path = output_dir / f"{int(frame_key):06d}_bbox.jpg"
        cv2.imwrite(str(output_path), image)
        print(output_path)


if __name__ == "__main__":
    main()
