"""Detect up to two camera instances and export their ordered corners."""
import argparse
import json
import math
from pathlib import Path

import cv2

from .pipeline import VideoCornerPipeline, draw_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--corner-weights", required=True)
    parser.add_argument("--out", default="outputs/corners.jpg")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--corner-threshold", type=float, default=0.15)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    pipeline = VideoCornerPipeline(
        args.corner_weights,
        args.detector,
        device=args.device,
        corner_threshold=args.corner_threshold,
    )
    results = pipeline(image)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), draw_results(image.copy(), results))

    json_path = Path(args.json_out) if args.json_out else output_path.with_suffix(".json")
    payload = []
    for index, result in enumerate(results):
        corners = [
            [float(x), float(y)] if math.isfinite(x) and math.isfinite(y) else None
            for x, y in result.corners
        ]
        payload.append({
            "instance_id": index,
            "box_xyxy": result.box.tolist(),
            "detection_score": result.detection_score,
            "corners_xy": corners,
            "corner_scores": result.corner_scores.tolist(),
        })
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"image: {output_path}")
    print(f"corners: {json_path}")


if __name__ == "__main__":
    main()
