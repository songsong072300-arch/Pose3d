#!/usr/bin/env python3
"""Render six quick orthographic views of a GLB or textured PLY model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--texture", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    suffix = model_path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(model_path))
    else:
        raise ValueError("Only GLB/GLTF and PLY are supported")

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in {model_path}")
    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

    if suffix == ".ply" and args.texture:
        material = bpy.data.materials.new("diagnostic_texture")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        image = bpy.data.images.load(str(args.texture.expanduser().resolve()))
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.interpolation = "Linear"
        material.node_tree.links.new(texture.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
        for obj in meshes:
            obj.data.materials.clear()
            obj.data.materials.append(material)

    # Work in the model's native units and derive framing from its bounds.
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    mins = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maxs = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (mins + maxs) / 2.0
    size = max(maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = size * 1.35
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(center.x - size, center.y - size, center.z + size))
    bpy.context.object.data.energy = 700
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = size * 2.0
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("diagnostic_world")
    bpy.context.scene.world.color = (0.12, 0.12, 0.12)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"

    views = {
        "front_minus_x": Vector((-1, 0, 0)),
        "back_plus_x": Vector((1, 0, 0)),
        "left_minus_y": Vector((0, -1, 0)),
        "right_plus_y": Vector((0, 1, 0)),
        "bottom_minus_z": Vector((0, 0, -1)),
        "top_plus_z": Vector((0, 0, 1)),
    }
    for name, direction in views.items():
        camera.location = center + direction * size * 2.5
        look_at(camera, center)
        scene.render.filepath = str(output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name}")


if __name__ == "__main__":
    main()
