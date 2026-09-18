"""Export corner annotations as an Ultralytics YOLO detection dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def instance_box(corners_2d, width, height):
    xs = [point[0] for point in corners_2d]
    ys = [point[1] for point in corners_2d]
    x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
    x2, y2 = min(float(width), max(xs)), min(float(height), max(ys))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="TripoSR/dataset/demo-bin-picking")
    parser.add_argument("--out", default="checkpoints/yolo_dataset")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.out)
    annotations = json.loads(
        (root / "corner_labels/annotations.json").read_text()
    )
    splits = json.loads((root / "corner_labels/splits.json").read_text())
    val_ids = set(splits.get("val", [])) | set(splits.get("test", []))
    for subset in ("train", "val"):
        (output / f"images/{subset}").mkdir(parents=True, exist_ok=True)
        (output / f"labels/{subset}").mkdir(parents=True, exist_ok=True)

    image_count = box_count = skipped = 0
    for record in annotations["records"]:
        image_path = root / record["image"]
        if not image_path.exists():
            skipped += 1
            continue
        width, height = record["image_size"]
        subset = "val" if record["id"] in val_ids else "train"
        lines = []
        for instance in record["instances"]:
            box = instance_box(instance["corners_2d"], width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            lines.append(
                f"0 {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )
            box_count += 1
        if not lines:
            continue
        stem = record["id"].replace("/", "_")
        link = output / "images" / subset / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(image_path.resolve())
        (output / "labels" / subset / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n"
        )
        image_count += 1

    train_count = len(list((output / "images/train").glob("*.jpg")))
    val_count = len(list((output / "images/val").glob("*.jpg")))
    (output / "data.yaml").write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\nval: images/val\n"
        "nc: 1\nnames: ['action_camera']\n"
    )
    print(
        f"exported {image_count} images ({train_count} train / "
        f"{val_count} val), {box_count} boxes, {skipped} missing images skipped"
    )
    print(f"dataset: {output}/data.yaml")


if __name__ == "__main__":
    main()
