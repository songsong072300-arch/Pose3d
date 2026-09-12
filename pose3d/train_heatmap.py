"""Train the Section 3.2 single-image baseline in the conda environment."""
import argparse, torch
from torch.utils.data import DataLoader
from .boxdreamer_single import PoseHeatmapDataset, BoxDreamerSingle, heatmap_loss
from .roi_pipeline import RoiPoseDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="TripoSR/dataset/demo-bin-picking")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/boxdreamer_single.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--roi", action="store_true", help="train on per-instance ROI labels")
    args = ap.parse_args()
    if args.device == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    else:
        device = args.device
    print(f"device: {device}")
    DatasetClass = RoiPoseDataset if args.roi else PoseHeatmapDataset
    ds = DatasetClass(args.root, "train")
    va = DatasetClass(args.root, "val")
    use_pin = device == "cuda"
    dl = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=use_pin)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=use_pin)
    net = BoxDreamerSingle().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(args.epochs):
        net.train(); total = 0.0
        for b in dl:
            p = net(b["image"].to(device)); loss = heatmap_loss(p, b["heatmap"].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total += loss.item()
        net.eval(); val = 0.0
        with torch.no_grad():
            for b in vl: val += heatmap_loss(net(b["image"].to(device)), b["heatmap"].to(device)).item()
        val /= max(1, len(vl)); print(f"epoch {epoch+1:03d} train={total/max(1,len(dl)):.5f} val={val:.5f}")
        if val < best:
            best = val; path = __import__('pathlib').Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "epoch": epoch+1, "val_loss": val}, path)
    print(f"saved: {args.out}")
if __name__ == "__main__": main()
