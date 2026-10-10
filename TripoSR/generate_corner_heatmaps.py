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
    parser.add_argument("--image-split", default="train_pbr",
                        help="BOP scene 子目录，例如 new_train_pbr")
    parser.add_argument(
        "--split-mode", choices=("scene", "frame"), default="scene",
        help=("scene keeps complete BOP scenes in one split and prevents adjacent-frame "
              "leakage; frame retains the legacy random frame split."),
    )
    parser.add_argument(
        "--val-scene-count", type=int, default=1,
        help="Number of complete scenes assigned to validation in scene mode.",
    )
    parser.add_argument(
        "--test-scene-count", type=int, default=0,
        help="Number of complete scenes assigned to test in scene mode.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Directory for annotations.json, splits.json, and heatmaps; "
             "defaults to <dataset>/corner_labels.",
    )
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


def build_splits(records, split_mode, seed, val_scene_count, test_scene_count):
    """Split label IDs while keeping complete render scenes isolated by default."""
    rng = random.Random(seed)
    count = len(records)
    if split_mode == "frame":
        shuffled = records.copy()
        rng.shuffle(shuffled)
        train_end = int(count * 0.8)
        val_end = train_end + int(count * 0.1)
        return {
            "train": [item["id"] for item in shuffled[:train_end]],
            "val": [item["id"] for item in shuffled[train_end:val_end]],
            "test": [item["id"] for item in shuffled[val_end:]],
            "mode": "frame",
        }

    scenes = sorted({item["id"].split("_", 1)[0] for item in records})
    if val_scene_count < 1:
        raise ValueError("--val-scene-count must be at least 1 in scene mode")
    if test_scene_count < 0:
        raise ValueError("--test-scene-count cannot be negative")
    held_out = val_scene_count + test_scene_count
    if len(scenes) <= held_out:
        raise ValueError(
            f"scene split needs more than {held_out} scenes, found {len(scenes)}")

    rng.shuffle(scenes)
    test_scenes = set(scenes[:test_scene_count])
    val_start = test_scene_count
    val_scenes = set(scenes[val_start:val_start + val_scene_count])
    train_scenes = set(scenes[val_start + val_scene_count:])
    split_scene_sets = {
        "train": train_scenes,
        "val": val_scenes,
        "test": test_scenes,
    }
    splits = {name: [] for name in split_scene_sets}
    for item in records:
        scene = item["id"].split("_", 1)[0]
        for name, scene_set in split_scene_sets.items():
            if scene in scene_set:
                splits[name].append(item["id"])
                break
    return {
        **splits,
        "mode": "scene",
        "scenes": {
            name: sorted(scene_set)
            for name, scene_set in split_scene_sets.items()
        },
    }


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
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else dataset / "corner_labels"
    )
    heatmap_dir = output_root / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    records = []
    channel_order = (bbox_corners(models_info["1"]) * model_scale).tolist()
    for scene_dir in sorted((dataset / args.image_split).glob("[0-9][0-9][0-9][0-9][0-9][0-9]")):
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
            try:
                heatmap_record_path = heatmap_path.relative_to(dataset)
            except ValueError:
                heatmap_record_path = heatmap_path
            records.append({
                "id": label_id,
                "image": str(image_path.relative_to(dataset)),
                "heatmap": str(heatmap_record_path),
                "image_size": [width, height],
                "heatmap_size": [args.heatmap_width, args.heatmap_height],
                "instances": instances_out,
            })

    count = len(records)
    if not count:
        raise ValueError(f"No BOP frames found under {dataset / args.image_split}")
    splits = build_splits(
        records,
        split_mode=args.split_mode,
        seed=args.seed,
        val_scene_count=args.val_scene_count,
        test_scene_count=args.test_scene_count,
    )
    (output_root / "annotations.json").write_text(json.dumps({
        "corner_channel_order_mm": channel_order,
        "records": sorted(records, key=lambda item: item["id"]),
    }, indent=2))
    (output_root / "splits.json").write_text(json.dumps(splits, indent=2))
    print(f"Generated {count} labels: "
          f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
