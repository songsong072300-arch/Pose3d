"""Compare YOLO detectors on an image directory without using it for training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_model(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use NAME=WEIGHTS syntax")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("model must use NAME=WEIGHTS syntax")
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, type=parse_model)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("install ultralytics in the pose3d environment") from error

    image_paths = sorted(
        path for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"no images found in {args.images}")
    args.out.mkdir(parents=True, exist_ok=True)
    device = "0" if torch.cuda.is_available() else "cpu"
    report = {
        "images": str(args.images.resolve()),
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "note": "Evaluation only; these images are not added to a training dataset.",
        "models": {},
    }

    for model_name, weights in args.model:
        weights = weights.expanduser().resolve()
        output_dir = args.out / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        results = YOLO(str(weights)).predict(
            source=[str(path) for path in image_paths],
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            verbose=False,
            stream=False,
        )
        frames = []
        for image_path, result in zip(image_paths, results):
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            predictions = [
                {"xyxy": [round(value, 2) for value in box],
                 "confidence": round(float(score), 6)}
                for box, score in zip(boxes, scores)
            ]
            predictions.sort(key=lambda item: item["confidence"], reverse=True)
            frames.append({"image": image_path.name, "detections": predictions})
            cv2.imwrite(str(output_dir / image_path.name), result.plot())

        detection_counts = [len(frame["detections"]) for frame in frames]
        report["models"][model_name] = {
            "weights": str(weights),
            "frame_count": len(frames),
            "frames_with_no_detections": sum(count == 0 for count in detection_counts),
            "frames_with_exactly_two": sum(count == 2 for count in detection_counts),
            "total_detections": sum(detection_counts),
            "frames": frames,
        }

    report_path = args.out / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for name, values in report["models"].items():
        print(
            f"{name}: no_detection={values['frames_with_no_detections']}/"
            f"{values['frame_count']}, exactly_two={values['frames_with_exactly_two']}/"
            f"{values['frame_count']}, total={values['total_detections']}"
        )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
