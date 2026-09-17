"""Export existing corner annotations as YOLO detection labels.

The renderer already stores per-instance corner ground truth
(corner_labels/annotations.json -> instances[].corners_2d), so the
axis-aligned bounding box of each instance's 8 corners is a free detection
label: no manual annotation needed.

Builds a ultralytics-ready dataset:

    <out>/images/{train,val}/...jpg   (symlinks to the BOP rgb frames)
    <out>/labels/{train,val}/....txt  (YOLO format: class cx cy w h, normalized)
    <out>/data.yaml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def instance_box(corners_2d, width, height):
    xs = [p[0] for p in corners_2d]
    ys = [p[1] for p in corners_2d]
    x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
    x2, y2 = min(float(width), max(xs)), min(float(height), max(ys))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="TripoSR/dataset/demo-bin-picking")
    ap.add_argument("--out", default="checkpoints/yolo_dataset")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    ann = json.loads((root / "corner_labels/annotations.json").read_text())
    splits = json.loads((root / "corner_labels/splits.json").read_text())
    val_ids = set(splits.get("val", [])) | set(splits.get("test", []))

    (out / "images/train").mkdir(parents=True, exist_ok=True)
    (out / "images/val").mkdir(parents=True, exist_ok=True)
    (out / "labels/train").mkdir(parents=True, exist_ok=True)
    (out / "labels/val").mkdir(parents=True, exist_ok=True)

    n_img, n_box, skipped = 0, 0, 0
    for r in ann["records"]:
        image_path = root / r["image"]
        if not image_path.exists():
            skipped += 1
            continue
        width, height = r["image_size"]
        subset = "val" if r["id"] in val_ids else "train"
        lines = []
        for ins in r["instances"]:
            box = instance_box(ins["corners_2d"], width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            w, h = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_box += 1
        if not lines:
            continue
        stem = r["id"].replace("/", "_")
        link = out / "images" / subset / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(image_path.resolve())
        (out / "labels" / subset / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_img += 1

    n_train = len(list((out / "images/train").glob("*.jpg")))
    n_val = len(list((out / "images/val").glob("*.jpg")))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: 1\nnames: ['action_camera']\n")
    print(f"exported {n_img} images ({n_train} train / {n_val} val), "
          f"{n_box} boxes, {skipped} missing images skipped")
    print(f"dataset: {out}/data.yaml")


if __name__ == "__main__":
    main()
