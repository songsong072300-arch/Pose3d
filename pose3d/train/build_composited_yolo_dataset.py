"""Build a YOLO dataset with rendered cameras composited onto real backgrounds."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-yolo", type=Path, required=True)
    parser.add_argument("--skip-base-train", action="store_true",
                        help="Keep base validation but replace old training appearances.")
    parser.add_argument("--positive-dir", action="append", type=Path, default=[],
                        help="RGB preview directory with labels/STEM.txt; imported into train.")
    parser.add_argument(
        "--layers", action="append", type=Path, required=True,
        help="Directory containing RGBA camera layers. May be repeated.",
    )
    parser.add_argument("--background-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--negative-dir", action="append", type=Path, default=[],
        help="Verified camera-free images used only for empty-label negatives. "
             "Defaults to background-dir when omitted. May be repeated.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument(
        "--negative-count", type=int, default=256,
        help="Number of camera-free hard-negative composites added to train.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=2)
    parser.add_argument("--min-width-fraction", type=float, default=0.08)
    parser.add_argument("--max-width-fraction", type=float, default=0.34)
    parser.add_argument("--max-roll", type=float, default=35.0)
    parser.add_argument(
        "--style-reference-dir", action="append", type=Path, default=[],
        help=("Images used only as photometric references. Their pixels are never "
              "copied into the generated dataset. May be repeated."),
    )
    parser.add_argument(
        "--style-match-probability", type=float, default=0.75,
        help="Probability of matching each generated image to reference color statistics.",
    )
    parser.add_argument(
        "--bottom-placement-probability", type=float, default=0.55,
        help="Probability of sampling a wrist-like lower-frame placement.",
    )
    parser.add_argument(
        "--partial-visibility-probability", type=float, default=0.25,
        help="Probability of deliberately placing a target partly outside the frame.",
    )
    parser.add_argument(
        "--motion-blur-probability", type=float, default=0.30,
        help="Probability of applying whole-frame directional motion blur.",
    )
    parser.add_argument(
        "--sensor-noise", type=float, default=3.0,
        help="Maximum Gaussian sensor-noise sigma in 8-bit intensity units.",
    )
    parser.add_argument("--alpha-threshold", type=int, default=8)
    parser.add_argument(
        "--min-layer-edge-density", type=float, default=0.02,
        help=("Reject nearly featureless rendered views (for example a blank CG back "
              "face). Set to 0 to disable."),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def image_files(root):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def materialize(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve())
        return "symlink"
    except OSError:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"


def alpha_box(image, threshold):
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > threshold)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_layer(path, threshold):
    image = Image.open(path).convert("RGBA")
    box = alpha_box(image, threshold)
    if box is None:
        raise ValueError(f"layer has no visible alpha pixels: {path}")
    return image.crop(box)


def layer_edge_density(image, threshold):
    array = np.asarray(image)
    mask = array[..., 3] > threshold
    if not np.any(mask):
        return 0.0
    gray = cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 120) > 0
    return float(np.count_nonzero(edges & mask) / np.count_nonzero(mask))


def cover_crop(image, width, height, rng):
    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (math.ceil(image.width * scale), math.ceil(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = rng.randint(0, resized.width - width)
    top = rng.randint(0, resized.height - height)
    return resized.crop((left, top, left + width, top + height))


def transform_layer(layer, canvas_width, args, rng):
    target_width = round(
        canvas_width * math.exp(rng.uniform(
            math.log(args.min_width_fraction), math.log(args.max_width_fraction)
        ))
    )
    scale = target_width / layer.width
    target_height = max(2, round(layer.height * scale))
    result = layer.resize((target_width, target_height), Image.Resampling.LANCZOS)
    result = result.rotate(
        rng.uniform(-args.max_roll, args.max_roll),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )

    rgb = result.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.62, 1.30))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.76, 1.24))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.65, 1.18))
    if rng.random() < 0.45:
        rgb = rgb.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 1.15)))
    alpha = result.getchannel("A")
    # Contract then feather the matte. This removes the bright CG fringe that
    # otherwise becomes an extremely easy synthetic-only feature.
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(
        ImageFilter.GaussianBlur(max(0.35, result.width * 0.0025))
    )
    rgb.putalpha(alpha)
    return rgb


def intersection_over_area(left, top, width, height, canvas_width, canvas_height):
    visible_width = max(0, min(left + width, canvas_width) - max(left, 0))
    visible_height = max(0, min(top + height, canvas_height) - max(top, 0))
    return visible_width * visible_height / max(1, width * height)


def choose_position(layer, canvas_width, canvas_height, existing, args, rng):
    for _ in range(80):
        partial = rng.random() < args.partial_visibility_probability
        margin_x = round(layer.width * (0.45 if partial else 0.12))
        margin_y = round(layer.height * (0.45 if partial else 0.12))
        left = rng.randint(-margin_x, canvas_width - layer.width + margin_x)
        if rng.random() < args.bottom_placement_probability:
            center_y = rng.uniform(0.58, 0.94) * canvas_height
            top = round(center_y - layer.height / 2)
            top = max(-margin_y, min(canvas_height - layer.height + margin_y, top))
        else:
            top = rng.randint(-margin_y, canvas_height - layer.height + margin_y)
        if intersection_over_area(
            left, top, layer.width, layer.height, canvas_width, canvas_height
        ) < 0.65:
            continue
        candidate = (left, top, left + layer.width, top + layer.height)
        if all(box_iou(candidate, box) < 0.12 for box in existing):
            return left, top
    return max(0, (canvas_width - layer.width) // 2), max(
        0, (canvas_height - layer.height) // 2
    )


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


def paste_with_shadow(canvas, layer, left, top, rng):
    alpha = layer.getchannel("A")
    if rng.random() < 0.45:
        shadow_alpha = alpha.filter(
            ImageFilter.GaussianBlur(max(1.0, layer.width * rng.uniform(0.012, 0.035)))
        )
        opacity = rng.uniform(0.12, 0.30)
        shadow_alpha = shadow_alpha.point(lambda value: round(value * opacity))
        shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        shadow.putalpha(shadow_alpha)
        offset_x = round(layer.width * rng.uniform(-0.020, 0.035))
        offset_y = round(layer.width * rng.uniform(0.008, 0.040))
        canvas.alpha_composite(shadow, (left + offset_x, top + offset_y))
    canvas.alpha_composite(layer, (left, top))


def image_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def match_reference_statistics(image, reference, strength=0.70):
    """Match low-order Lab statistics without copying reference pixels."""
    source = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB).astype(np.float32)
    target = cv2.cvtColor(np.asarray(reference), cv2.COLOR_RGB2LAB).astype(np.float32)
    result = source.copy()
    for channel in range(3):
        src = source[..., channel]
        dst = target[..., channel]
        src_mean, src_std = float(src.mean()), max(float(src.std()), 1.0)
        dst_mean, dst_std = float(dst.mean()), max(float(dst.std()), 1.0)
        matched = (src - src_mean) * (dst_std / src_std) + dst_mean
        channel_strength = strength if channel == 0 else strength * 0.55
        result[..., channel] = src * (1.0 - channel_strength) + matched * channel_strength
    rgb = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb, mode="RGB")


def directional_blur(image, rng):
    size = rng.choice((3, 5, 7, 9))
    kernel = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[size // 2, :] = 1.0
    else:
        np.fill_diagonal(kernel, 1.0)
        if rng.random() < 0.5:
            kernel = np.fliplr(kernel)
    kernel /= kernel.sum()
    array = cv2.filter2D(np.asarray(image), -1, kernel)
    return Image.fromarray(array, mode="RGB")


def apply_sensor_pipeline(image, args, rng, style_references):
    if style_references and rng.random() < args.style_match_probability:
        reference = image_rgb(rng.choice(style_references))
        image = match_reference_statistics(image, reference, rng.uniform(0.48, 0.82))
    if rng.random() < args.motion_blur_probability:
        image = directional_blur(image, rng)
    elif rng.random() < 0.55:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.85)))
    if args.sensor_noise > 0:
        array = np.asarray(image).astype(np.float32)
        sigma = rng.uniform(0.0, args.sensor_noise)
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        array += noise_rng.normal(0.0, sigma, array.shape)
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    return image


def visible_box(layer, left, top, canvas_width, canvas_height, threshold):
    box = alpha_box(layer, threshold)
    if box is None:
        return None
    x1 = max(0, left + box[0])
    y1 = max(0, top + box[1])
    x2 = min(canvas_width, left + box[2])
    y2 = min(canvas_height, top + box[3])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def yolo_line(box, width, height):
    x1, y1, x2, y2 = box
    return (
        f"0 {(x1 + x2) / (2 * width):.6f} "
        f"{(y1 + y2) / (2 * height):.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def validate_args(args):
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.negative_count < 0:
        raise ValueError("--negative-count cannot be negative")
    if args.width < 32 or args.height < 32:
        raise ValueError("canvas dimensions must be at least 32 pixels")
    if not 1 <= args.min_objects <= args.max_objects:
        raise ValueError("require 1 <= min-objects <= max-objects")
    if not 0 < args.min_width_fraction <= args.max_width_fraction <= 1:
        raise ValueError("width fractions must satisfy 0 < min <= max <= 1")
    for name in (
        "style_match_probability", "bottom_placement_probability",
        "partial_visibility_probability", "motion_blur_probability",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.sensor_noise < 0:
        raise ValueError("--sensor-noise cannot be negative")
    if not 0.0 <= args.min_layer_edge_density <= 1.0:
        raise ValueError("--min-layer-edge-density must be in [0, 1]")


def main():
    args = parse_args()
    validate_args(args)
    base = args.base_yolo.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not (base / "data.yaml").is_file():
        raise FileNotFoundError(f"not a YOLO dataset: {base}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    layer_paths = sorted({
        path
        for directory in args.layers
        for path in image_files(directory)
    })
    background_paths = sorted({
        path
        for directory in args.background_dir
        for path in image_files(directory)
    })
    negative_paths = sorted({
        path
        for directory in (args.negative_dir or args.background_dir)
        for path in image_files(directory)
    })
    # BOP RGB scenes contain labeled cameras; depth maps are not RGB backgrounds.
    for path in set(background_paths + negative_paths):
        if "train_pbr" in path.parts or "depth" in path.parts:
            raise ValueError(f"unsafe background/negative source: {path}")
    if args.negative_count and not negative_paths:
        raise ValueError("camera-free negative images are required")
    style_reference_paths = sorted({
        path
        for directory in args.style_reference_dir
        for path in image_files(directory)
    })
    candidate_layers = [
        (path, crop_layer(path, args.alpha_threshold)) for path in layer_paths
    ]
    layers = [
        (path, layer) for path, layer in candidate_layers
        if layer_edge_density(layer, args.alpha_threshold) >= args.min_layer_edge_density
    ]
    if not layers or not background_paths:
        raise ValueError("at least one layer and one background are required")

    storage = {"symlink": 0, "hardlink": 0, "copy": 0}
    base_counts = {}
    for subset in ("train", "val", "test"):
        if subset == "train" and args.skip_base_train:
            continue
        source_images = base / "images" / subset
        source_labels = base / "labels" / subset
        if not source_images.is_dir():
            continue
        images = image_files(source_images)
        base_counts[subset] = len(images)
        for source in images:
            storage[materialize(source, output / "images" / subset / source.name)] += 1
            label = source_labels / f"{source.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"missing YOLO label: {label}")
            storage[materialize(label, output / "labels" / subset / label.name)] += 1

    train_images = output / "images" / "train"
    train_labels = output / "labels" / "train"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    imported = []
    for directory_index, directory in enumerate(args.positive_dir):
        directory = directory.expanduser().resolve()
        for source in image_files(directory):
            label = directory / "labels" / f"{source.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"missing rendered positive label: {label}")
            lines = label.read_text().splitlines()
            for line in lines:
                category, cx, cy, width, height = map(float, line.split())
                if (category != 0 or not all(math.isfinite(v) for v in (cx, cy, width, height))
                        or not 0 < width <= 1 or not 0 < height <= 1
                        or cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6
                        or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6):
                    raise ValueError(f"invalid rendered label: {label}")
            stem = f"wrist_{directory_index}_{source.stem}"
            storage[materialize(source, train_images / f"{stem}{source.suffix}")] += 1
            storage[materialize(label, train_labels / f"{stem}.txt")] += 1
            imported.append({"image": f"{stem}{source.suffix}", "source": str(source),
                             "label": str(label), "box_count": len(lines)})
    rng = random.Random(args.seed)
    generated = []
    for index in range(args.count):
        background_path = background_paths[index % len(background_paths)]
        background = cover_crop(Image.open(background_path), args.width, args.height, rng)
        canvas = background.convert("RGBA")
        count = rng.randint(args.min_objects, args.max_objects)
        boxes = []
        used_layers = []
        for _ in range(count):
            layer_path, original = rng.choice(layers)
            layer = transform_layer(original, args.width, args, rng)
            left, top = choose_position(
                layer, args.width, args.height, boxes, args, rng
            )
            box = visible_box(
                layer, left, top, args.width, args.height, args.alpha_threshold
            )
            if box is None:
                continue
            paste_with_shadow(canvas, layer, left, top, rng)
            boxes.append(box)
            used_layers.append(str(layer_path))
        if not boxes:
            raise RuntimeError(f"failed to place an object for composite {index}")

        image = apply_sensor_pipeline(
            canvas.convert("RGB"), args, rng, style_reference_paths
        )
        image_path = train_images / f"composited_{index:06d}.jpg"
        label_path = train_labels / f"composited_{index:06d}.txt"
        image.save(image_path, quality=rng.randint(72, 94), subsampling=2)
        label_path.write_text(
            "\n".join(yolo_line(box, args.width, args.height) for box in boxes) + "\n"
        )
        generated.append({
            "image": image_path.name,
            "background": str(background_path),
            "layers": used_layers,
            "boxes_xyxy": boxes,
        })

    generated_negatives = []
    for index in range(args.negative_count):
        background_path = negative_paths[index % len(negative_paths)]
        image = cover_crop(image_rgb(background_path), args.width, args.height, rng)
        image = apply_sensor_pipeline(image, args, rng, style_reference_paths)
        image_path = train_images / f"hard_negative_{index:06d}.jpg"
        label_path = train_labels / f"hard_negative_{index:06d}.txt"
        image.save(image_path, quality=rng.randint(68, 94), subsampling=2)
        label_path.write_text("")
        generated_negatives.append({
            "image": image_path.name,
            "background": str(background_path),
        })

    yaml_lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if (output / "images" / "test").is_dir():
        yaml_lines.append("test: images/test")
    yaml_lines.extend(("nc: 1", "names: ['action_camera']"))
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    manifest = {
        "base_yolo": str(base),
        "skip_base_train": args.skip_base_train,
        "positive_directories": [str(path.resolve()) for path in args.positive_dir],
        "rendered_train_images": len(imported),
        "rendered_empty_images": sum(item["box_count"] == 0 for item in imported),
        "imported": imported,
        "layer_directories": [str(path.resolve()) for path in args.layers],
        "background_directories": [str(path.resolve()) for path in args.background_dir],
        "negative_directories": [
            str(path.resolve()) for path in (args.negative_dir or args.background_dir)
        ],
        "negative_source_count": len(negative_paths),
        "style_reference_directories": [
            str(path.resolve()) for path in args.style_reference_dir
        ],
        "seed": args.seed,
        "canvas": [args.width, args.height],
        "base_image_counts": base_counts,
        "composited_train_images": len(generated),
        "composited_boxes": sum(len(item["boxes_xyxy"]) for item in generated),
        "hard_negative_images": len(generated_negatives),
        "layer_count": len(layers),
        "rejected_layer_count": len(candidate_layers) - len(layers),
        "background_count": len(background_paths),
        "image_storage": storage,
        "generated": generated,
        "generated_negatives": generated_negatives,
        "note": (
            "Generated positives and hard negatives are train-only; base validation "
            "and test are unchanged. Style references contribute statistics only."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in {"generated", "generated_negatives", "imported"}}, indent=2))
    print(f"dataset: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
