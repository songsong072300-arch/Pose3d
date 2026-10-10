"""Parse pose3d training logs and plot loss/error curves.

Example:
    python -m pose3d.train.plot_training_curves \
        --log outputs/merged_action4_train.log \
        --out outputs/merged_action4_curves
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+"
    r"train=(?P<train>[0-9.eE+-]+)\s+"
    r"coarse=(?P<coarse>[0-9.eE+-]+)\s+"
    r"fine=(?P<fine>[0-9.eE+-]+)\s+"
    r"val=(?P<val>[0-9.eE+-]+)\s+"
    r"corner_err_roi=(?P<roi>[0-9.eE+-]+)px\s+"
    r"corner_err_original=(?P<original>[0-9.eE+-]+)px"
)

# This fallback keeps the tool usable with logs from before the two error
# metrics were added, while leaving unavailable columns empty in the CSV.
LEGACY_EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+"
    r"train=(?P<train>[0-9.eE+-]+)\s+"
    r"coarse=(?P<coarse>[0-9.eE+-]+)\s+"
    r"fine=(?P<fine>[0-9.eE+-]+)\s+"
    r"val=(?P<val>[0-9.eE+-]+)\s+"
    r"corner_err=(?P<roi>[0-9.eE+-]+)px"
)


def parse_log(path: Path):
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        match = EPOCH_RE.search(line) or LEGACY_EPOCH_RE.search(line)
        if match is None:
            continue
        values = match.groupdict()
        rows.append({
            "epoch": int(values["epoch"]),
            "train": float(values["train"]),
            "coarse": float(values["coarse"]),
            "fine": float(values["fine"]),
            "val": float(values["val"]),
            "corner_err_roi": float(values["roi"]),
            "corner_err_original": (
                float(values["original"])
                if values.get("original") is not None else None
            ),
            "line": line_number,
        })
    if not rows:
        raise ValueError(
            f"no epoch lines found in {path}; expected lines like "
            "epoch 001 train=... coarse=... fine=... val=..."
        )
    rows.sort(key=lambda row: row["epoch"])
    return rows


def write_csv(rows, path: Path):
    fields = [
        "epoch", "train", "coarse", "fine", "val",
        "corner_err_roi", "corner_err_original", "line",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, output: Path, title: str):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    axes[0, 0].plot(epochs, [row["train"] for row in rows], label="total train")
    axes[0, 0].plot(epochs, [row["val"] for row in rows], label="total val")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].set_ylabel("loss")

    axes[0, 1].plot(epochs, [row["coarse"] for row in rows], label="coarse")
    axes[0, 1].plot(epochs, [row["fine"] for row in rows], label="fine")
    axes[0, 1].set_title("Loss components")
    axes[0, 1].set_ylabel("loss")

    roi = [row["corner_err_roi"] for row in rows]
    axes[1, 0].plot(epochs, roi, color="tab:orange", label="ROI")
    best_roi = min(range(len(rows)), key=lambda index: rows[index]["corner_err_roi"])
    axes[1, 0].scatter(
        [epochs[best_roi]], [roi[best_roi]], color="red", zorder=3,
        label=f"best {roi[best_roi]:.2f}px",
    )
    axes[1, 0].set_title("ROI corner error")
    axes[1, 0].set_ylabel("pixels")

    original_rows = [row for row in rows if row["corner_err_original"] is not None]
    if original_rows:
        original_epochs = [row["epoch"] for row in original_rows]
        original = [row["corner_err_original"] for row in original_rows]
        axes[1, 1].plot(
            original_epochs, original, color="tab:green",
            label="original image",
        )
        best_original = min(range(len(original_rows)),
                            key=lambda index: original[index])
        axes[1, 1].scatter(
            [original_epochs[best_original]], [original[best_original]],
            color="red", zorder=3,
            label=f"best {original[best_original]:.2f}px",
        )
        axes[1, 1].set_title("Original-image corner error")
        axes[1, 1].set_ylabel("pixels")
    else:
        axes[1, 1].text(
            0.5, 0.5, "No corner_err_original in log",
            ha="center", va="center", transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Original-image corner error")

    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", default="outputs/training_curves", type=Path)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    rows = parse_log(args.log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_png = args.out.with_suffix(".png")
    output_csv = args.out.with_suffix(".csv")
    write_csv(rows, output_csv)
    plot_rows(rows, output_png, args.title or args.log.stem)

    best_roi = min(rows, key=lambda row: row["corner_err_roi"])
    original_rows = [row for row in rows
                     if row["corner_err_original"] is not None]
    print(f"parsed epochs: {len(rows)} ({rows[0]['epoch']}..{rows[-1]['epoch']})")
    print(f"best ROI error: epoch {best_roi['epoch']} "
          f"{best_roi['corner_err_roi']:.2f}px")
    if original_rows:
        best_original = min(original_rows,
                            key=lambda row: row["corner_err_original"])
        print(f"best original-image error: epoch {best_original['epoch']} "
              f"{best_original['corner_err_original']:.2f}px")
    print(f"plot: {output_png}")
    print(f"csv: {output_csv}")


if __name__ == "__main__":
    main()
