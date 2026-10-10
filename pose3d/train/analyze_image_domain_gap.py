"""Measure simple, actionable image-domain gaps between real and synthetic sets.

The report intentionally focuses on quantities that the renderer/compositor can
control: resolution, luminance, contrast, saturation, sharpness, edge density,
noise and encoded bytes per pixel. It is not a replacement for real validation,
but it prevents tuning exposure/blur/JPEG parameters by eye alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METRIC_NAMES = (
    "luminance_mean",
    "luminance_std",
    "saturation_mean",
    "laplacian_variance",
    "edge_density",
    "high_frequency_noise",
    "encoded_bytes_per_pixel",
)


def parse_domain(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("domain must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("domain name cannot be empty")
    return name, Path(raw_path).expanduser()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", action="append", type=parse_domain, required=True,
        help="Image domain in NAME=PATH form. Repeat for every domain.",
    )
    parser.add_argument(
        "--reference", required=True,
        help="Domain name treated as the real/reference distribution.",
    )
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def collect_images(root: Path):
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def image_metrics(path: Path):
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image: {path}")
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 80, 160)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    residual = gray.astype(np.float32) - smooth.astype(np.float32)
    return {
        "width": float(width),
        "height": float(height),
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "saturation_mean": float(hsv[..., 1].mean()),
        "laplacian_variance": float(laplacian.var()),
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "high_frequency_noise": float(residual.std()),
        "encoded_bytes_per_pixel": float(path.stat().st_size / (width * height)),
    }


def summarize(rows):
    summary = {"image_count": len(rows)}
    for name in ("width", "height", *METRIC_NAMES):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }
    return summary


def normalized_gap(domain, reference):
    metrics = {}
    values = []
    for name in METRIC_NAMES:
        domain_mean = domain[name]["mean"]
        reference_mean = reference[name]["mean"]
        scale = max(reference[name]["std"], abs(reference_mean) * 0.10, 1e-6)
        distance = abs(domain_mean - reference_mean) / scale
        metrics[name] = distance
        values.append(distance)
    return {"mean_normalized_gap": float(np.mean(values)), "metrics": metrics}


def main():
    args = parse_args()
    if args.max_images < 1:
        raise ValueError("--max-images must be positive")
    domains = dict(args.domain)
    if len(domains) != len(args.domain):
        raise ValueError("domain names must be unique")
    if args.reference not in domains:
        raise ValueError(f"unknown reference domain: {args.reference}")

    rng = random.Random(args.seed)
    summaries = {}
    failures = {}
    for name, root in domains.items():
        paths = collect_images(root)
        rng.shuffle(paths)
        paths = paths[: args.max_images]
        rows = []
        failed = []
        for path in paths:
            try:
                rows.append(image_metrics(path))
            except (OSError, ValueError) as error:
                failed.append({"path": str(path), "error": str(error)})
        if not rows:
            raise RuntimeError(f"no readable images in domain {name}: {root}")
        summaries[name] = summarize(rows)
        failures[name] = failed

    reference = summaries[args.reference]
    gaps = {
        name: normalized_gap(summary, reference)
        for name, summary in summaries.items()
        if name != args.reference
    }
    report = {
        "reference": args.reference,
        "max_images": args.max_images,
        "seed": args.seed,
        "domains": summaries,
        "gaps_to_reference": gaps,
        "decode_failures": failures,
        "interpretation": (
            "Lower mean_normalized_gap is better. Use the per-metric gaps to tune "
            "renderer exposure, blur, noise and JPEG settings; final acceptance must "
            "still use labeled real images."
        ),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
