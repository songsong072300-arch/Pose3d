"""Two-instance RGB corner inference pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image
import torch

from ..common.image import image_to_tensor, letterbox_image
from ..common.model import CornerHeatmapNet


def box_crop(frame_size, box, pad=0.15):
    """Expand a detector xyxy box and clamp it to the image."""
    image_height, image_width = frame_size
    x1, y1, x2, y2 = map(float, box)
    pad_width = (x2 - x1) * pad
    pad_height = (y2 - y1) * pad
    return (
        max(0, math.floor(x1 - pad_width)),
        max(0, math.floor(y1 - pad_height)),
        min(image_width, math.ceil(x2 + pad_width)),
        min(image_height, math.ceil(y2 + pad_height)),
    )


def extract_peaks(heatmaps, threshold=0.15, return_scores=False):
    """Extract one sub-pixel peak and confidence from each heatmap channel."""
    points = []
    scores = []
    for channel in heatmaps:
        _, score, _, location = cv2.minMaxLoc(channel.astype(np.float32))
        x, y = location
        scores.append(float(score))
        if score < threshold:
            points.append([np.nan, np.nan])
            continue
        x0, y0 = max(0, x - 1), max(0, y - 1)
        patch = channel[y0:min(channel.shape[0], y + 2),
                        x0:min(channel.shape[1], x + 2)]
        yy, xx = np.mgrid[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
        weights = np.maximum(patch, 0)
        denominator = max(float(weights.sum()), 1e-6)
        points.append([
            float((xx * weights).sum() / denominator),
            float((yy * weights).sum() / denominator),
        ])
    points = np.asarray(points, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    return (points, scores) if return_scores else points


@dataclass
class CornerResult:
    box: np.ndarray
    corners: np.ndarray
    corner_scores: np.ndarray
    detection_score: float


def load_corner_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "decoder" not in checkpoint:
        raise ValueError("checkpoint must contain a 'decoder' state")
    model = CornerHeatmapNet().to(device)
    incompatible = model.load_state_dict(checkpoint["decoder"], strict=False)
    invalid_missing = [name for name in incompatible.missing_keys
                       if not name.startswith("backbone.")]
    if invalid_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"incompatible checkpoint: missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.eval()


class VideoCornerPipeline:
    """Detect up to two instances and return eight corners for each one."""

    def __init__(self, corner_weights, detector_weights, device="auto",
                 crop_pad=0.15, corner_threshold=0.15):
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available() else
                "mps" if getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        self.net = load_corner_model(corner_weights, device)
        from ultralytics import YOLO
        self.detector = YOLO(detector_weights)
        self.crop_pad = crop_pad
        self.corner_threshold = corner_threshold

    def _detect(self, frame):
        prediction = self.detector.predict(
            frame, verbose=False, device=self.device,
        )[0]
        boxes = prediction.boxes.xyxy.detach().cpu().numpy()
        scores = prediction.boxes.conf.detach().cpu().numpy()
        selected = np.argsort(scores)[::-1][:2]
        return [(boxes[index], float(scores[index])) for index in selected]

    def __call__(self, frame):
        results = []
        for box, detection_score in self._detect(frame):
            xa, ya, xb, yb = box_crop(frame.shape[:2], box, self.crop_pad)
            rgb = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2RGB)
            crop, scale_xy, pad_x, pad_y = letterbox_image(Image.fromarray(rgb))
            tensor = image_to_tensor(crop)[None].to(self.device)
            with torch.no_grad():
                heatmaps = self.net(tensor)[0].cpu().numpy()
            corners, corner_scores = extract_peaks(
                heatmaps, self.corner_threshold, return_scores=True,
            )
            corners[:, 0] = (corners[:, 0] - pad_x) / scale_xy[0] + xa
            corners[:, 1] = (corners[:, 1] - pad_y) / scale_xy[1] + ya
            results.append(CornerResult(
                box=np.asarray(box, dtype=np.float32),
                corners=corners,
                corner_scores=corner_scores,
                detection_score=detection_score,
            ))
        return sorted(results, key=lambda result: float(result.box[0]))


def draw_results(frame, results):
    colors = [
        (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 128, 0), (255, 0, 255), (128, 0, 255),
    ]
    for instance_id, result in enumerate(results):
        x1, y1, x2, y2 = result.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"camera {instance_id + 1}", (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        for corner_id, (x, y) in enumerate(result.corners):
            if not np.isfinite((x, y)).all():
                continue
            point = (round(float(x)), round(float(y)))
            cv2.circle(frame, point, 4, colors[corner_id], -1)
            cv2.putText(frame, str(corner_id), (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[corner_id], 1)
    return frame
