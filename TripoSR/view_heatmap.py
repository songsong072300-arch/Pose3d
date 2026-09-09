#!/usr/bin/env python3
"""Inspect one generated corner Heatmap and overlay it on the source RGB image."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label_id", help="例如 000000_000000")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset" / "demo-bin-picking")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    labels = dataset / "corner_labels"
    annotations = json.loads((labels / "annotations.json").read_text())
    record = next((r for r in annotations["records"] if r["id"] == args.label_id), None)
    if record is None:
        raise SystemExit(f"找不到标签: {args.label_id}")

    data = np.load(dataset / record["heatmap"])
    heatmaps = data["heatmaps"].astype(np.float32) / 255.0
    image = cv2.imread(str(dataset / record["image"]))
    if image is None:
        raise FileNotFoundError(record["image"])
    h, w = image.shape[:2]
    hh, hw = heatmaps.shape[1:]
    print(f"image={record['image']}  heatmap={record['heatmap']}")
    print(f"shape={heatmaps.shape}, dtype=uint8(on disk), range=[{heatmaps.min():.3f}, {heatmaps.max():.3f}]")

    # Print strongest local maxima candidates. A channel normally has two peaks.
    for ch, hm in enumerate(heatmaps):
        work = hm.copy()
        peaks = []
        for _ in range(2):
            y, x = np.unravel_index(np.argmax(work), work.shape)
            value = float(work[y, x])
            if value <= 0:
                break
            peaks.append((round(x * w / hw, 1), round(y * h / hh, 1), round(value, 3)))
            cv2.circle(work, (x, y), 8, 0, -1)
        print(f"ch{ch}: {peaks}")

    # 8-channel color montage.
    cell_w, cell_h = 320, 240
    montage = np.zeros((cell_h * 2, cell_w * 4, 3), np.uint8)
    for ch, hm in enumerate(heatmaps):
        color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        color = cv2.resize(color, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(color, f"channel {ch}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        montage[(ch // 4) * cell_h:(ch // 4 + 1) * cell_h,
                (ch % 4) * cell_w:(ch % 4 + 1) * cell_w] = color

    # Maximum response across channels overlaid on RGB.
    maximum = heatmaps.max(axis=0)
    overlay = cv2.applyColorMap((maximum * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    out = args.output or (labels / f"{args.label_id}_inspect.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([cv2.resize(montage, (w, int(h * 2 / 3))), blended]))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
