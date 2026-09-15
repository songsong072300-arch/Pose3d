#!/usr/bin/env python3
"""Generate 8-corner coordinates and heatmaps from a rendered BOP dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

# 设置命令行参数
def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "demo-bin-picking")
    parser.add_argument("--heatmap-width", type=int, default=240)
    parser.add_argument("--heatmap-height", type=int, default=180)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def bbox_corners(info):
    # Stable channel order: binary XYZ, with X changing slowest and Z fastest.
    return np.array([
        [x, y, z]
        for x in (info["min_x"], info["max_x"])
        for y in (info["min_y"], info["max_y"])
        for z in (info["min_z"], info["max_z"])
    ], dtype=np.float64)


def project(corners, instance, K):
    R = np.asarray(instance["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(instance["cam_t_m2c"], dtype=np.float64).reshape(1, 3)
    points_camera = corners @ R.T + t
    homogeneous = points_camera @ K.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, points_camera[:, 2]


def draw_gaussian(heatmap, center_x, center_y, sigma):
    radius = int(np.ceil(3.0 * sigma))
    x0 = max(0, int(np.floor(center_x)) - radius)
    x1 = min(heatmap.shape[1], int(np.floor(center_x)) + radius + 1)
    y0 = max(0, int(np.floor(center_y)) - radius)
    y1 = min(heatmap.shape[0], int(np.floor(center_y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma ** 2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)


def main():
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    models_info = json.loads((dataset / "models" / "models_info.json").read_text())
    # 渲染器若对模型做了整体缩放（render_config.json），角点真值需同步缩放，
    # 否则热图标签与渲染机身不重合。
    model_scale = 1.0
    render_config_path = dataset / "render_config.json"
    if render_config_path.exists():
        model_scale = float(json.loads(render_config_path.read_text()).get("model_scale", 1.0))
    output_root = dataset / "corner_labels"
    heatmap_dir = output_root / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    records = []
    channel_order = (bbox_corners(models_info["1"]) * model_scale).tolist()
    for scene_dir in sorted((dataset / "train_pbr").glob("[0-9][0-9][0-9][0-9][0-9][0-9]")):
        scene_gt = json.loads((scene_dir / "scene_gt.json").read_text())
        scene_camera = json.loads((scene_dir / "scene_camera.json").read_text())
        for frame_key in sorted(scene_gt, key=int):
            image_path = scene_dir / "rgb" / f"{int(frame_key):06d}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            height, width = image.shape[:2]
            K = np.asarray(scene_camera[frame_key]["cam_K"], dtype=np.float64).reshape(3, 3)
            heatmaps = np.zeros((8, args.heatmap_height, args.heatmap_width), dtype=np.float32)
            instances_out = []
            for instance_index, instance in enumerate(scene_gt[frame_key]):
                corners_3d = bbox_corners(models_info[str(instance["obj_id"])]) * model_scale
                pixels, depths = project(corners_3d, instance, K)
                in_frame = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
                            & (depths > 0))
                scaled = pixels * np.array([
                    args.heatmap_width / width, args.heatmap_height / height])
                for channel, ((x, y), valid) in enumerate(zip(scaled, in_frame)):
                    if valid:
                        draw_gaussian(heatmaps[channel], x, y, args.sigma)
                instances_out.append({
                    "instance_id": instance_index,
                    "obj_id": instance["obj_id"],
                    "corners_2d": pixels.tolist(),
                    "corners_in_frame": in_frame.tolist(),
                })

            label_id = f"{int(scene_dir.name):06d}_{int(frame_key):06d}"
            heatmap_path = heatmap_dir / f"{label_id}.npz"
            np.savez_compressed(heatmap_path, heatmaps=np.rint(heatmaps * 255).astype(np.uint8))
            records.append({
                "id": label_id,
                "image": str(image_path.relative_to(dataset)),
                "heatmap": str(heatmap_path.relative_to(dataset)),
                "image_size": [width, height],
                "heatmap_size": [args.heatmap_width, args.heatmap_height],
                "instances": instances_out,
            })

    rng = random.Random(args.seed)
    rng.shuffle(records)
    count = len(records)
    train_end = int(count * 0.8)
    val_end = train_end + int(count * 0.1)
    splits = {
        "train": [item["id"] for item in records[:train_end]],
        "val": [item["id"] for item in records[train_end:val_end]],
        "test": [item["id"] for item in records[val_end:]],
    }
    (output_root / "annotations.json").write_text(json.dumps({
        "corner_channel_order_mm": channel_order,
        "records": sorted(records, key=lambda item: item["id"]),
    }, indent=2))
    (output_root / "splits.json").write_text(json.dumps(splits, indent=2))
    print(f"Generated {count} labels: "
          f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
