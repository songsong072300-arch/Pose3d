"""Render BOP RGB/depth, generate corner heatmaps, and export YOLO data.

Run with the pose3d environment's Python; all paths are resolved independently
of the current working directory. Existing datasets are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def run(arguments):
    print(subprocess.list2cmdline([str(arg) for arg in arguments]), flush=True)
    subprocess.run([str(arg) for arg in arguments], cwd=PROJECT, check=True)


def verify(root, labels, yolo, expected):
    annotations = json.loads((labels / "annotations.json").read_text())
    splits = json.loads((labels / "splits.json").read_text())
    records = annotations["records"]
    if len(records) != expected:
        raise ValueError(f"Expected {expected} images, got {len(records)}")
    counts = {}
    for subset in ("train", "val"):
        if not splits[subset]:
            raise ValueError(f"Empty {subset} split")
        images = sorted((yolo / "images" / subset).glob("*.jpg"))
        if len(images) != len(splits[subset]):
            raise ValueError(f"YOLO export lost images in {subset}")
        boxes = 0
        for image in images:
            lines = (yolo / "labels" / subset / f"{image.stem}.txt").read_text().splitlines()
            for line in lines:
                fields = [float(value) for value in line.split()]
                if len(fields) != 5 or fields[0] != 0 or not all(
                        np.isfinite(value) and 0 <= value <= 1 for value in fields[1:]):
                    raise ValueError(f"Invalid YOLO box: {line}")
                if fields[3] <= 0 or fields[4] <= 0:
                    raise ValueError(f"Empty YOLO box: {line}")
            boxes += len(lines)
        counts[subset] = {"images": len(images), "boxes": boxes}
    for record in records:
        image = cv2.imread(str(root / record["image"]))
        if image is None or list(image.shape[1::-1]) != record["image_size"]:
            raise ValueError(f"Missing or invalid RGB: {record['image']}")
        with np.load(root / record["heatmap"]) as data:
            maps = data["heatmaps"]
            width, height = record["heatmap_size"]
            if maps.shape != (8, height, width) or maps.dtype != np.uint8 or not maps.any():
                raise ValueError(f"Invalid heatmaps: {record['heatmap']}")
        for instance in record["instances"]:
            points = np.asarray(instance["corners_2d"])
            if points.shape != (8, 2) or not np.isfinite(points).all():
                raise ValueError(f"Invalid corners: {record['id']}")
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Train/val overlap")
    report = {"verified_images": len(records), "splits": counts,
              "corner_instances": sum(len(record["instances"]) for record in records)}
    (labels / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT / "TripoSR/dataset/lm")
    parser.add_argument("--output-split", default="new_train_pbr")
    parser.add_argument("--labels-dir", default="new_corner_labels")
    parser.add_argument("--yolo-out", type=Path, default=PROJECT / "checkpoints/yolo_new_train_pbr")
    parser.add_argument("--frames-per-scene", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--render-device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    labels = root / args.labels_dir
    yolo = args.yolo_out.resolve()
    if args.verify_only:
        verify(root, labels, yolo, args.num_images)
        return
    chunk = args.frames_per_scene or min(1000, max(1, (args.num_images + 5) // 6))
    if args.num_images < 2 or chunk < 1 or chunk >= args.num_images:
        parser.error("Require at least 2 images and 1 <= frames-per-scene < num-images")
    for path in (root / args.output_split, labels, yolo):
        if path.exists() and any(path.iterdir()):
            parser.error(f"Output already contains data; choose a fresh output: {path}")
    blenderproc = Path(sys.executable).parent / "Scripts/blenderproc.exe"
    if not blenderproc.is_file():
        parser.error("Use the pose3d environment Python (Scripts/blenderproc.exe must exist)")
    run([blenderproc, "run", PROJECT / "TripoSR/render_dataset.py", args.num_images,
         "--dataset-dir", root, "--output-split", args.output_split,
         "--frames-per-scene", chunk, "--seed", args.seed,
         "--samples", args.samples, "--render-device", args.render_device])
    run([sys.executable, PROJECT / "TripoSR/generate_corner_heatmaps.py", root,
         "--image-split", args.output_split, "--output-root", labels,
         "--split-mode", "scene", "--val-scene-count", 1])
    run([sys.executable, "-m", "pose3d.train.export_yolo_labels", "--root", root,
         "--labels-dir", args.labels_dir, "--out", yolo])
    verify(root, labels, yolo, args.num_images)


if __name__ == "__main__":
    main()
