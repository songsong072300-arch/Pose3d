#!/usr/bin/env python3
"""Convert a TripoSR OBJ into a metric BOP PLY model.

The default axis mapping is specific to the camera used by this project:
X = depth (32.8 mm), Y = width (70.5 mm), Z = height (44.2 mm).
The inspected TripoSR source faces +X, so it is rotated 180 degrees around Z
by default to establish the project convention that the lens faces -X.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "output" / "camera_v1" / "0" / "mesh.obj"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset" / "demo-bin-picking" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale a TripoSR OBJ to millimetres and export a BOP model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--x-mm", type=float, default=32.8, help="X size: camera depth")
    parser.add_argument("--y-mm", type=float, default=70.5, help="Y size: camera width")
    parser.add_argument("--z-mm", type=float, default=44.2, help="Z size: camera height")
    parser.add_argument(
        "--origin",
        choices=("min", "center"),
        default="min",
        help="Place the coordinate origin at the bounding-box minimum or center",
    )
    parser.add_argument(
        "--source-lens-direction",
        choices=("+x", "-x"),
        default="+x",
        help="Lens direction in the source OBJ; output is always normalized to -X",
    )
    parser.add_argument(
        "--texture",
        type=Path,
        default=None,
        help="Texture image; defaults to texture.png beside the input OBJ",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up an existing PLY/models_info.json before overwriting",
    )
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"No usable triangular mesh found in {path}")
    return mesh


def model_diameter(vertices: np.ndarray, block_size: int = 2048) -> float:
    """Return the largest vertex distance using only convex-hull vertices."""
    hull_vertices = trimesh.Trimesh(vertices=vertices, process=False).convex_hull.vertices
    max_squared = 0.0
    for start in range(0, len(hull_vertices), block_size):
        block = hull_vertices[start : start + block_size]
        squared = np.sum((block[:, None, :] - hull_vertices[None, :, :]) ** 2, axis=2)
        max_squared = max(max_squared, float(np.max(squared)))
    return float(np.sqrt(max_squared))


def add_texture_comment(ply_path: Path, texture_name: str) -> None:
    """Add the TextureFile header used by PLY/BOP loaders."""
    data = ply_path.read_bytes()
    marker = b"end_header\n"
    offset = data.find(marker)
    if offset < 0:
        raise ValueError(f"Invalid PLY header in {ply_path}")
    header_lines = [
        line
        for line in data[:offset].splitlines(keepends=True)
        if not line.startswith(b"comment TextureFile ")
    ]
    texture_line = f"comment TextureFile {texture_name}\n".encode("ascii")
    insert_at = next(
        (index + 1 for index, line in enumerate(header_lines) if line.startswith(b"format ")),
        1,
    )
    header_lines.insert(insert_at, texture_line)
    ply_path.write_bytes(b"".join(header_lines) + data[offset:])


def backup_if_needed(path: Path, stamp: str, enabled: bool) -> None:
    if enabled and path.exists():
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, backup)
        print(f"Backup: {backup}")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    texture_path = (
        args.texture.expanduser().resolve()
        if args.texture is not None
        else input_path.with_name("texture.png")
    )

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    target_size = np.array([args.x_mm, args.y_mm, args.z_mm], dtype=np.float64)
    if np.any(target_size <= 0):
        raise ValueError("All target dimensions must be positive")

    mesh = load_mesh(input_path)
    source_size = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(source_size <= 0):
        raise ValueError(f"Invalid source extents: {source_size}")

    # Scale each axis independently to the measured physical dimensions.
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * (target_size / source_size)
    if args.source_lens_direction == "+x":
        # Proper 180-degree rotation around Z: +X -> -X without mirroring.
        mesh.vertices[:, 0] *= -1.0
        mesh.vertices[:, 1] *= -1.0
    bounds = mesh.bounds
    if args.origin == "min":
        mesh.vertices -= bounds[0]
    else:
        mesh.vertices -= (bounds[0] + bounds[1]) / 2.0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"obj_{args.obj_id:06d}"
    output_ply = output_dir / f"{stem}.ply"
    output_texture = output_dir / f"{stem}.png"
    info_path = output_dir / "models_info.json"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_enabled = not args.no_backup
    backup_if_needed(output_ply, stamp, backup_enabled)
    backup_if_needed(info_path, stamp, backup_enabled)
    if texture_path.is_file():
        backup_if_needed(output_texture, stamp, backup_enabled)

    # BlenderProc 4.x reads and rewrites textured PLY files as text, so the
    # source PLY must be ASCII when it contains a TextureFile comment.
    mesh.export(output_ply, file_type="ply", encoding="ascii")
    if texture_path.is_file():
        shutil.copy2(texture_path, output_texture)
        add_texture_comment(output_ply, output_texture.name)
    else:
        print(f"Warning: texture not found: {texture_path}")

    mins, maxs = mesh.bounds
    sizes = maxs - mins
    model_info = {
        "diameter": model_diameter(np.asarray(mesh.vertices)),
        "min_x": float(mins[0]),
        "min_y": float(mins[1]),
        "min_z": float(mins[2]),
        "max_x": float(maxs[0]),
        "max_y": float(maxs[1]),
        "max_z": float(maxs[2]),
        "size_x": float(sizes[0]),
        "size_y": float(sizes[1]),
        "size_z": float(sizes[2]),
    }
    existing = {}
    if info_path.is_file():
        with info_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    existing[str(args.obj_id)] = model_info
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(f"Input size:  {source_size.tolist()} (source units)")
    print(f"Output size: {sizes.tolist()} mm")
    print(f"PLY:         {output_ply}")
    print(f"Model info:  {info_path}")
    print(f"Source lens: {args.source_lens_direction}; output lens: -X")


if __name__ == "__main__":
    main()
