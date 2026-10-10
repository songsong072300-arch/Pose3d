"""Train the fixed DINOv2-Small eight-corner heatmap network."""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common.model import CornerHeatmapNet
from .dataset import RoiCornerDataset
from .losses import corner_heatmap_loss


def heatmap_peak_coordinates(pred):
    """Return Bx8x2 argmax coordinates in ROI pixels."""
    pred = pred.detach().cpu().numpy()
    coordinates = np.empty((pred.shape[0], pred.shape[1], 2), dtype=np.float32)
    for batch_index in range(pred.shape[0]):
        for channel in range(pred.shape[1]):
            y, x = np.unravel_index(
                pred[batch_index, channel].argmax(),
                pred[batch_index, channel].shape,
            )
            coordinates[batch_index, channel] = (x, y)
    return coordinates


def corner_errors(pred, batch):
    """Return mean corner errors in ROI and original-image pixels."""
    predicted_roi = heatmap_peak_coordinates(pred)
    roi_corners = batch["corners"].detach().cpu().numpy()
    roi_error = np.linalg.norm(predicted_roi - roi_corners, axis=-1).mean()

    scale_xy = batch["scale_xy"].detach().cpu().numpy()[:, None, :]
    crop_origin = batch["crop_origin"].detach().cpu().numpy()[:, None, :]
    padding = batch["letterbox_padding"].detach().cpu().numpy()[:, None, :]
    predicted_original = (
        (predicted_roi - padding) / scale_xy + crop_origin
    )
    original_corners = batch["original_corners"].detach().cpu().numpy()
    original_error = np.linalg.norm(
        predicted_original - original_corners, axis=-1,
    ).mean()
    return float(roi_error), float(original_error)


def select_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if (getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="TripoSR/dataset/demo-bin-picking")
    parser.add_argument("--labels-dir", default="corner_labels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="checkpoints/dino_corner_net.pt")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true",
                        help="enable CUDA mixed precision")
    parser.add_argument("--coordinate-weight", type=float, default=2.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    use_amp = args.amp and device == "cuda"
    print(f"device: {device}  amp: {use_amp}  seed: {args.seed}")

    train_set = RoiCornerDataset(args.root, "train", labels_dir=args.labels_dir)
    val_set = RoiCornerDataset(args.root, "val", labels_dir=args.labels_dir)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    net = CornerHeatmapNet().to(device)
    for parameter in net.backbone.blocks[-2:].parameters():
        parameter.requires_grad_(True)
    # 开放最后的 LayerNorm 层
    for parameter in net.backbone.norm.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in net.parameters()
                 if parameter.requires_grad]
    print(
        f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M  "
        f"trainable: {sum(p.numel() for p in trainable) / 1e6:.2f}M  "
        f"train: {len(train_set)}  val: {len(val_set)}"
    )
    skipped = train_set.skipped_invisible + val_set.skipped_invisible
    if skipped:
        print(f"skipped fully out-of-frame instances: {skipped}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_corner_error = float("inf")
    best_original_corner_error = float("inf")

    for epoch in range(args.epochs):
        net.train()
        train_total = train_coarse = train_fine = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["heatmap"].to(device)
            corners = batch["corners"].to(device)
            with torch.autocast("cuda", enabled=use_amp):
                prediction = net(images)
                loss, coarse, fine = corner_heatmap_loss(
                    prediction, targets, corners,
                    coordinate_weight=args.coordinate_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += loss.item()
            train_coarse += coarse.item()
            train_fine += fine.item()
        scheduler.step()

        net.eval()
        val_total = val_roi_error_sum = val_original_error_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                prediction = net(batch["image"].to(device))
                loss, _, _ = corner_heatmap_loss(
                    prediction,
                    batch["heatmap"].to(device),
                    batch["corners"].to(device),
                    coordinate_weight=args.coordinate_weight,
                )
                count = batch["image"].shape[0]
                val_total += loss.item()
                roi_error, original_error = corner_errors(prediction, batch)
                val_roi_error_sum += roi_error * count
                val_original_error_sum += original_error * count
                val_samples += count

        train_batches = max(1, len(train_loader))
        val_loss = val_total / max(1, len(val_loader))
        val_roi_error = val_roi_error_sum / max(1, val_samples)
        val_original_error = val_original_error_sum / max(1, val_samples)
        print(
            f"epoch {epoch + 1:03d} "
            f"train={train_total / train_batches:.5f} "
            f"coarse={train_coarse / train_batches:.5f} "
            f"fine={train_fine / train_batches:.5f} "
            f"val={val_loss:.5f} "
            f"corner_err_roi={val_roi_error:.2f}px "
            f"corner_err_original={val_original_error:.2f}px"
        )
        if val_roi_error < best_corner_error:
            best_corner_error = val_roi_error
            best_original_corner_error = val_original_error
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            decoder_state = {
                name: tensor for name, tensor in net.state_dict().items()
                if not name.startswith("backbone.")
            }
            torch.save({
                "model_state": net.state_dict(),
                "decoder": decoder_state,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_corner_err": val_roi_error,
                "val_corner_err_roi": val_roi_error,
                "val_corner_err_original": val_original_error,
            }, output_path)
    print(
        f"saved: {args.out}  "
        f"best ROI corner error: {best_corner_error:.2f}px  "
        f"original-image error: {best_original_corner_error:.2f}px"
    )


if __name__ == "__main__":
    main()
