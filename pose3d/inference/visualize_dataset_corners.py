"""Visualize ROI corner predictions in the original synthetic image."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common.model import CornerHeatmapNet
from ..train.dataset import RoiCornerDataset
from .pipeline import extract_peaks


BOX_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]
GT_COLOR = (40, 210, 40)
PRED_COLOR = (220, 40, 220)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CornerHeatmapNet()
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    elif "decoder" in checkpoint:
        incompatible = model.load_state_dict(checkpoint["decoder"], strict=False)
        invalid_missing = [
            name for name in incompatible.missing_keys
            if not name.startswith("backbone.")
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise ValueError(
                f"incompatible checkpoint: missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    else:
        raise ValueError("checkpoint must contain model_state or decoder")
    return model.to(device).eval(), checkpoint


def original_predictions(heatmaps, batch):
    predictions = []
    scores = []
    for heatmap in heatmaps:
        points, confidence = extract_peaks(
            heatmap, threshold=0.0, return_scores=True,
        )
        predictions.append(points)
        scores.append(confidence)
    predictions = np.stack(predictions)
    scale = batch["scale_xy"].numpy()[:, None, :]
    origin = batch["crop_origin"].numpy()[:, None, :]
    padding = batch["letterbox_padding"].numpy()[:, None, :]
    return (predictions - padding) / scale + origin, np.stack(scores)


def draw_box(image, corners, color, thickness, labels=False):
    points = np.rint(corners).astype(np.int32)
    for start, end in BOX_EDGES:
        cv2.line(image, tuple(points[start]), tuple(points[end]), color,
                 thickness, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(image, tuple(point), 5, color, -1, cv2.LINE_AA)
        if labels:
            cv2.putText(
                image, str(index), (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )


def choose_quantile_records(record_errors, count):
    ranked = sorted(record_errors, key=record_errors.get)
    if count >= len(ranked):
        return ranked
    indices = np.rint(np.linspace(0, len(ranked) - 1, count)).astype(int)
    return [ranked[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="TripoSR/dataset/demo-bin-picking")
    parser.add_argument("--weights", default="checkpoints/unfreezing_net.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="outputs/corner_eval_unfreezing")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    dataset = RoiCornerDataset(root, args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=args.device == "cuda",
    )
    model, checkpoint = load_model(args.weights, args.device)

    predictions_by_id = {}
    targets_by_id = {}
    errors_by_id = {}
    scores_by_id = {}
    record_errors = defaultdict(list)
    records = {}
    for record, instance in dataset.items:
        sample_id = f"{record['id']}_{instance['instance_id']}"
        records[record["id"]] = record
        targets_by_id[sample_id] = np.asarray(
            instance["corners_2d"], dtype=np.float32,
        )

    with torch.inference_mode():
        for batch in loader:
            heatmaps = model(
                batch["image"].to(args.device, non_blocking=True),
            ).cpu().numpy()
            predicted, scores = original_predictions(heatmaps, batch)
            target = batch["original_corners"].numpy()
            errors = np.linalg.norm(predicted - target, axis=-1)
            for index, sample_id in enumerate(batch["id"]):
                record_id = sample_id.rsplit("_", 1)[0]
                predictions_by_id[sample_id] = predicted[index]
                scores_by_id[sample_id] = scores[index]
                errors_by_id[sample_id] = errors[index]
                record_errors[record_id].extend(errors[index].tolist())

    record_mean_errors = {
        record_id: float(np.mean(errors))
        for record_id, errors in record_errors.items()
    }
    selected = choose_quantile_records(record_mean_errors, args.count)
    rendered = []
    for rank, record_id in enumerate(selected, start=1):
        record = records[record_id]
        image = cv2.imread(str(root / record["image"]))
        if image is None:
            raise FileNotFoundError(root / record["image"])
        instance_text = []
        for instance in record["instances"]:
            sample_id = f"{record_id}_{instance['instance_id']}"
            target = targets_by_id[sample_id]
            predicted = predictions_by_id[sample_id]
            errors = errors_by_id[sample_id]
            draw_box(image, target, GT_COLOR, 2)
            draw_box(image, predicted, PRED_COLOR, 2, labels=True)
            instance_text.append(
                f"I{instance['instance_id']}: {errors.mean():.2f}px"
            )

        cv2.rectangle(image, (0, 0), (image.shape[1], 74), (20, 20, 20), -1)
        cv2.putText(
            image,
            f"{args.split} {record_id} | full-image mean: "
            f"{record_mean_errors[record_id]:.2f}px",
            (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "GT: green | prediction: magenta | " + "  ".join(instance_text),
            (14, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
            (230, 230, 230), 1, cv2.LINE_AA,
        )
        path = output / f"{rank:02d}_{record_id}_{record_mean_errors[record_id]:.2f}px.jpg"
        cv2.imwrite(str(path), image)
        rendered.append((path, image))

    tile_width, tile_height = 480, 360
    columns = min(3, len(rendered))
    rows = int(np.ceil(len(rendered) / columns))
    contact = np.full((rows * tile_height, columns * tile_width, 3), 245,
                      dtype=np.uint8)
    for index, (_, image) in enumerate(rendered):
        tile = cv2.resize(image, (tile_width, tile_height),
                          interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        contact[row * tile_height:(row + 1) * tile_height,
                column * tile_width:(column + 1) * tile_width] = tile
    contact_path = output / "contact_sheet.jpg"
    cv2.imwrite(str(contact_path), contact)

    epoch = checkpoint.get("epoch", "unknown")
    print(f"checkpoint epoch: {epoch}")
    print(f"evaluated {len(record_mean_errors)} complete {args.split} images")
    for path, _ in rendered:
        print(path)
    print(f"contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
