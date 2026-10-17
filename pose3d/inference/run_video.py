"""Run two-instance corner detection on a video."""
import argparse
from pathlib import Path

import cv2

from .pipeline import VideoCornerPipeline, draw_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--corner-weights", required=True)
    parser.add_argument("--out", default="outputs/corners.mp4")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--corner-threshold", type=float, default=0.15)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, choices=(1, 2), default=2)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None,
                        help="process at most this many seconds")
    args = parser.parse_args()

    pipeline = VideoCornerPipeline(
        args.corner_weights,
        args.detector,
        device=args.device,
        corner_threshold=args.corner_threshold,
        detector_conf=args.detector_conf,
        detector_iou=args.detector_iou,
        max_det=args.max_det,
    )
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start_seconds < 0:
        raise SystemExit("--start-seconds must be non-negative")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    start_frame = int(round(args.start_seconds * fps))
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = None
    if args.duration is not None:
        end_frame = start_frame + int(round(args.duration * fps))
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width, height),
    )
    try:
        while True:
            if end_frame is not None and capture.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                break
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(draw_results(frame, pipeline(frame)))
    finally:
        capture.release()
        writer.release()
    print(output_path)


if __name__ == "__main__":
    main()
