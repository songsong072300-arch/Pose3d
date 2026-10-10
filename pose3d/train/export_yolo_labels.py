"""Export BOP corner annotations as a leakage-resistant YOLO dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def instance_box(corners_2d, width, height):
    xs = [point[0] for point in corners_2d]
    ys = [point[1] for point in corners_2d]
    x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
    x2, y2 = min(float(width), max(xs)), min(float(height), max(ys))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def materialize_image(source, destination):
    """Link an image when possible, falling back for restricted Windows hosts."""
    try:
        destination.symlink_to(source.resolve())
        return "symlink"
    except OSError:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"


def unique_name(prefix, path):
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}{path.suffix.lower()}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="TripoSR/dataset/demo-bin-picking")
    parser.add_argument(
        "--labels-dir", default="corner_labels",
        help="Label directory relative to --root (default: corner_labels).",
    )
    parser.add_argument("--out", default="checkpoints/yolo_dataset")
    parser.add_argument(
        "--negative-dir", action="append", default=[], type=Path,
        help=("Directory of camera-free images. May be repeated. Images receive "
              "empty labels and are added to train only."),
    )
    parser.add_argument(
        "--max-negatives", type=int, default=0,
        help="Maximum negatives to add (0 uses every image).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def validate_splits(splits):
    names = ("train", "val", "test")
    split_ids = {name: set(splits.get(name, [])) for name in names}
    if not split_ids["train"] or not split_ids["val"]:
        raise ValueError("splits.json must contain non-empty train and val splits")
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise ValueError(
                    f"split leakage: {left}/{right} share {len(overlap)} IDs"
                )
    return split_ids


def collect_negative_images(directories):
    images = []
    for directory in directories:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"negative directory does not exist: {directory}")
        images.extend(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return sorted(set(images))


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to mix with an existing dataset: {output}; choose a new --out"
        )

    labels_root = root / args.labels_dir
    annotations = json.loads((labels_root / "annotations.json").read_text())
    splits = json.loads((labels_root / "splits.json").read_text())
    split_ids = validate_splits(splits)
    id_to_split = {
        item_id: subset
        for subset, item_ids in split_ids.items()
        for item_id in item_ids
    }
    subsets = ("train", "val", "test") if split_ids["test"] else ("train", "val")
    for subset in subsets:
        (output / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output / "labels" / subset).mkdir(parents=True, exist_ok=True)

    counts = {
        subset: {"positive_images": 0, "negative_images": 0, "boxes": 0}
        for subset in subsets
    }
    skipped = 0
    image_modes = {"symlink": 0, "hardlink": 0, "copy": 0}
    for record in annotations["records"]:
        subset = id_to_split.get(record["id"])
        if subset is None:
            continue
        image_path = root / record["image"]
        if not image_path.exists():
            skipped += 1
            continue
        width, height = record["image_size"]
        lines = []
        for instance in record["instances"]:
            box = instance_box(instance["corners_2d"], width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            lines.append(
                f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            )
        if not lines:
            continue
        stem = record["id"].replace("/", "_")
        destination = output / "images" / subset / f"synthetic_{stem}.jpg"
        image_modes[materialize_image(image_path, destination)] += 1
        (output / "labels" / subset / f"synthetic_{stem}.txt").write_text(
            "\n".join(lines) + "\n"
        )
        counts[subset]["positive_images"] += 1
        counts[subset]["boxes"] += len(lines)

    negative_images = collect_negative_images(args.negative_dir)
    rng = random.Random(args.seed)
    rng.shuffle(negative_images)
    if args.max_negatives > 0:
        negative_images = negative_images[:args.max_negatives]
    for image_path in negative_images:
        filename = unique_name("negative", image_path)
        destination = output / "images" / "train" / filename
        image_modes[materialize_image(image_path, destination)] += 1
        (output / "labels" / "train" / f"{Path(filename).stem}.txt").write_text("")
        counts["train"]["negative_images"] += 1

    yaml_lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in subsets:
        yaml_lines.append("test: images/test")
    yaml_lines.extend(("nc: 1", "names: ['action_camera']"))
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    manifest = {
        "source_root": str(root),
        "labels_root": str(labels_root),
        "seed": args.seed,
        "negative_directories": [str(path.resolve()) for path in args.negative_dir],
        "counts": counts,
        "skipped_missing_images": skipped,
        "image_storage": image_modes,
        "note": "Real final-test frames are intentionally excluded.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"dataset: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
