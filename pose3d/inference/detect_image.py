"""Run only the two-object YOLO detection stage on one image."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="outputs/detections.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("install ultralytics in the pose3d environment") from error
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available() else "cpu"
    )
    prediction = YOLO(args.weights).predict(
        image, conf=args.conf, verbose=False, device=device,
    )[0]
    boxes = prediction.boxes.xyxy.detach().cpu().numpy()
    scores = prediction.boxes.conf.detach().cpu().numpy()
    for instance_id, index in enumerate(np.argsort(scores)[::-1][:2]):
        x1, y1, x2, y2 = boxes[index].astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image, f"ACTION4 {scores[index]:.2f}",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        print(instance_id, boxes[index].tolist(), float(scores[index]))
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
