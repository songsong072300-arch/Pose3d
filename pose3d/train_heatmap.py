"""Train the single-image corner heatmap network."""
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .boxdreamer_single import PoseHeatmapDataset, heatmap_loss, build_net
from .roi_pipeline import RoiPoseDataset


def corner_error(pred, corners):
    """Mean pixel distance between per-channel argmax peaks and corner GT.

    pred: (B, 8, H, W) predicted heatmaps; corners: (B, 8, 2) GT in the same
    grid. Directly measures the task objective (corner localization), which
    the heatmap loss only proxies.
    """
    pred = pred.detach().cpu().numpy()
    corners = corners.detach().cpu().numpy()
    errs = []
    for b in range(pred.shape[0]):
        for c in range(8):
            py, px = np.unravel_index(pred[b, c].argmax(), pred[b, c].shape)
            gx, gy = corners[b, c]
            errs.append(float(np.hypot(px - gx, py - gy)))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="TripoSR/dataset/demo-bin-picking")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="checkpoints/boxdreamer_single.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="CUDA 混合精度（游戏本提速）")
    ap.add_argument("--roi", action="store_true", help="train on per-instance ROI labels")
    ap.add_argument("--arch", default="scratch", choices=["scratch", "dinov2"],
                    help="scratch=从零小Transformer；dinov2=冻结DINOv2+解码头（论文方案）")
    ap.add_argument("--dinov2-variant", default="base", choices=["small", "base", "large"],
                    help="DINOv2 规格：small(22M)/base(86M,论文同款)/large(300M)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    use_amp = args.amp and device == "cuda"
    print(f"device: {device}  amp: {use_amp}  seed: {args.seed}")

    DatasetClass = RoiPoseDataset if args.roi else PoseHeatmapDataset
    ds = DatasetClass(args.root, "train")
    va = DatasetClass(args.root, "val")
    use_pin = device == "cuda"
    dl = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=use_pin)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=use_pin)
    if args.arch == "dinov2":
        net = build_net("dinov2", variant=args.dinov2_variant).to(device)
    else:
        net = build_net("scratch").to(device)
    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f"arch: {args.arch}({args.dinov2_variant if args.arch == 'dinov2' else '-'})  "
          f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M  "
          f"trainable: {sum(p.numel() for p in trainable) / 1e6:.2f}M  "
          f"train: {len(ds)}  val: {len(va)}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = float("inf")
    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for b in dl:
            with torch.autocast("cuda", enabled=use_amp):
                loss = heatmap_loss(net(b["image"].to(device)),
                                    b["heatmap"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()

        net.eval()
        val, cerr_sum, cerr_n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in vl:
                pred = net(b["image"].to(device))
                val += heatmap_loss(pred, b["heatmap"].to(device)).item()
                if "corners" in b:  # ROI 模式直接量化角点像素误差
                    cerr_sum += corner_error(pred, b["corners"]) * b["image"].shape[0]
                    cerr_n += b["image"].shape[0]
        val /= max(1, len(vl))
        cerr = cerr_sum / max(1, cerr_n) if cerr_n else None
        print(f"epoch {epoch + 1:03d} train={total / max(1, len(dl)):.5f} val={val:.5f}"
              + (f" corner_err={cerr:.2f}px" if cerr is not None else ""))
        if val < best:
            best = val
            import pathlib
            path = pathlib.Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "epoch": epoch + 1,
                        "val_loss": val, "val_corner_err": cerr,
                        "arch": args.arch,
                        "dinov2_variant": args.dinov2_variant}, path)
    print(f"saved: {args.out}  best val_loss: {best:.5f}")


if __name__ == "__main__":
    main()
