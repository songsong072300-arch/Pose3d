import blenderproc as bproc  # 必须放在最顶端第一行
import os
import json
import argparse
import random
import glob
import shutil
import tempfile

import numpy as np
import bpy
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_CAMERA_MODEL_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "相机3D模型", "相机3D模型",
))
REFINED_CAMERA_MODEL_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "refined_action_camera"))
if os.path.isfile(os.path.join(REFINED_CAMERA_MODEL_DIR, "refined_camera.blend")):
    DEFAULT_CAMERA_MODEL_DIR = REFINED_CAMERA_MODEL_DIR


def axis_rotation(axis, angle):
    """Create a 3x3 right-handed rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrist_camera_rotation(obj_location, camera_location, view_mode, rng, world_up=None):
    """Orient the model using the rear/side-heavy distribution in the real video."""
    to_camera = camera_location - obj_location
    to_camera /= np.linalg.norm(to_camera)
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    if view_mode in {"front", "back"}:
        local_x_world = -to_camera if view_mode == "front" else to_camera
        local_z_world = world_up - np.dot(world_up, local_x_world) * local_x_world
        local_z_world /= np.linalg.norm(local_z_world)
        local_y_world = np.cross(local_z_world, local_x_world)
    else:
        local_y_world = to_camera if view_mode == "side_plus_y" else -to_camera
        local_z_world = world_up - np.dot(world_up, local_y_world) * local_y_world
        local_z_world /= np.linalg.norm(local_z_world)
        local_x_world = np.cross(local_y_world, local_z_world)
    base = np.column_stack((local_x_world, local_y_world, local_z_world))

    if view_mode.startswith("side"):
        perturbation = (
            axis_rotation("x", rng.uniform(-0.14, 0.14))
            @ axis_rotation("y", rng.uniform(-0.18, 0.18))
            @ axis_rotation("z", rng.uniform(-0.28, 0.28))
        )
    else:
        perturbation = (
            axis_rotation("x", rng.uniform(-0.22, 0.22))
            @ axis_rotation("y", rng.uniform(-0.28, 0.28))
            @ axis_rotation("z", rng.uniform(-0.42, 0.42))
        )
    return base @ perturbation


def sample_composition(rng, lower_middle_probability):
    """Sample several ego-view compositions instead of one fixed shortcut."""
    lower_middle_probability = float(np.clip(lower_middle_probability, 0.0, 1.0))
    remaining = 1.0 - lower_middle_probability
    band = rng.choices(
        ["lower_middle", "center", "left_edge", "right_edge", "raised"],
        weights=[lower_middle_probability, remaining * 0.35, remaining * 0.20,
                 remaining * 0.20, remaining * 0.25],
        k=1,
    )[0]
    if band == "left_edge":
        group_x = rng.uniform(-0.22, -0.08)
    elif band == "right_edge":
        group_x = rng.uniform(0.08, 0.22)
    else:
        group_x = rng.uniform(-0.12, 0.12)

    ranges = {
        "lower_middle": ((0.00, 0.20), (0.015, 0.060)),
        "center": ((0.12, 0.34), (0.025, 0.090)),
        "left_edge": ((0.02, 0.28), (0.015, 0.080)),
        "right_edge": ((0.02, 0.28), (0.015, 0.080)),
        "raised": ((0.22, 0.48), (0.070, 0.180)),
    }
    y_range, z_range = ranges[band]
    return band, group_x, np.array([
        group_x + rng.uniform(-0.035, 0.035),
        rng.uniform(*y_range),
        rng.uniform(*z_range),
    ])


def set_local_pose(part, object_location, object_rotation, local_location,
                   local_rotation=None):
    part.set_location(object_location + object_rotation @ np.asarray(local_location))
    rotation = object_rotation
    if local_rotation is not None:
        rotation = object_rotation @ local_rotation
    part.set_rotation_mat(rotation)


def sample_ego_pose(rng, mode):
    """Jointly sample observer and wrists in metres above the work surface."""
    if mode == "mixed":
        mode = rng.choices(["raised", "work", "fold"], [0.25, 0.50, 0.25])[0]
    cam = np.array([rng.uniform(-0.04, 0.04), -0.38, rng.uniform(0.52, 0.65)])
    if mode == "raised":
        y, z, spread = -0.08, rng.uniform(0.36, 0.46), rng.uniform(0.12, 0.20)
    elif mode == "fold":
        cam[1], cam[2] = rng.uniform(-0.08, -0.03), rng.uniform(0.48, 0.55)
        y, z, spread = 0.09, rng.uniform(0.12, 0.20), rng.uniform(0.07, 0.13)
    else:
        cam[1], cam[2] = rng.uniform(-0.08, -0.03), rng.uniform(0.48, 0.55)
        y, z, spread = rng.uniform(0.03, 0.12), rng.uniform(0.08, 0.15), rng.uniform(0.14, 0.22)
    wrists = [np.array([side * spread + rng.uniform(-0.025, 0.025),
                       y + rng.uniform(-0.025, 0.025), z + rng.uniform(-0.025, 0.025)])
              for side in (-1.0, 1.0)]
    focus = np.mean(wrists, axis=0)
    if mode != "raised":
        # Look down at the work surface, rather than ahead at raised wrists.
        focus = np.array([focus[0], rng.uniform(0.08, 0.14), 0.035])
    return mode, cam, wrists, focus


def projected_object_bounds(obj, cam2world, intrinsics, width, height):
    """Audit unclipped geometry; Blender camera looks along local -Z."""
    vertices = np.array([list(v) for v in obj.blender_obj.bound_box])
    world = (np.array(obj.blender_obj.matrix_world) @
             np.column_stack((vertices, np.ones(8))).T).T
    camera = (np.linalg.inv(cam2world) @ world.T).T
    depth = -camera[:, 2]
    if np.any(depth <= 1e-5):
        return {"valid": False, "reason": "crosses_camera_plane"}
    pixels = np.column_stack((intrinsics[0, 0] * camera[:, 0] / depth + width / 2,
                              -intrinsics[1, 1] * camera[:, 1] / depth + height / 2))
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    clipped_lo, clipped_hi = np.maximum(lo, 0), np.minimum(hi, [width, height])
    return {"valid": True, "xyxy": [*lo.tolist(), *hi.tolist()],
            "width_fraction": float((hi[0] - lo[0]) / width),
            "visible_box_fraction": float(np.prod(np.maximum(clipped_hi - clipped_lo, 0)) /
                                          max(np.prod(hi - lo), 1e-9))}


def configure_cloth_grid(material):
    """Surface bands replace subpixel geometry and its shadow artifacts."""
    mat = material.blender_obj
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    coords = nodes.new("ShaderNodeTexCoord")
    bands = []
    for direction, scale in (("X", 8.6), ("Y", 6.7)):
        wave = nodes.new("ShaderNodeTexWave")
        wave.bands_direction = direction
        wave.wave_profile = "SAW"
        wave.inputs["Scale"].default_value = scale
        wave.inputs["Distortion"].default_value = 0.12
        links.new(coords.outputs["Generated"], wave.inputs["Vector"])
        threshold = nodes.new("ShaderNodeMath")
        threshold.operation = "LESS_THAN"
        threshold.inputs[1].default_value = 0.035
        links.new(wave.outputs["Fac"], threshold.inputs[0])
        bands.append(threshold)
    maximum = nodes.new("ShaderNodeMath")
    maximum.operation = "MAXIMUM"
    for i, band in enumerate(bands):
        links.new(band.outputs[0], maximum.inputs[i])
    colors = nodes.new("ShaderNodeMixRGB")
    colors.inputs[1].default_value = (0.65, 0.68, 0.70, 1)
    colors.inputs[2].default_value = (0.018, 0.024, 0.030, 1)
    links.new(maximum.outputs[0], colors.inputs[0])
    links.new(colors.outputs[0], shader.inputs["Base Color"])


def configure_screen_content(material, image_path=None):
    """Configure either the supplied Action 4 screen or a procedural fallback."""
    mat = material.blender_obj
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    if image_path:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"camera screen texture missing: {image_path}")
        coords = nodes.new("ShaderNodeTexCoord")
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(image_path, check_existing=True)
        texture.image.colorspace_settings.name = "sRGB"
        texture.extension = "CLIP"
        links.new(coords.outputs["UV"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        emission = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
        if emission is not None:
            links.new(texture.outputs["Color"], emission)
        strength = shader.inputs.get("Emission Strength")
        if strength is not None:
            strength.default_value = 0.12
        shader.inputs["Roughness"].default_value = 0.24
        return

    coords = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 2.0
    links.new(coords.outputs["Generated"], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.30
    ramp.color_ramp.elements[0].color = (0.02, 0.035, 0.05, 1)
    ramp.color_ramp.elements[1].position = 0.68
    ramp.color_ramp.elements[1].color = (0.55, 0.62, 0.65, 1)
    links.new(noise.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    emission = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission is not None:
        links.new(ramp.outputs["Color"], emission)
    strength = shader.inputs.get("Emission Strength")
    if strength is not None:
        strength.default_value = 0.35


def set_parts_hidden(parts, hidden):
    for part in parts:
        if part is not None:
            part.hide(hidden)


def create_camera_appearance_parts(proxy_scale, materials):
    """Create removable screen/label/tag details missing from the base mesh."""
    bezel_material, display_material, white_material, black_material, led_material = materials
    parts = {}
    parts["bezel"] = bproc.object.create_primitive(
        "CUBE", scale=(np.array([0.0010, 0.0310, 0.0188]) * proxy_scale).tolist())
    parts["bezel"].replace_materials(bezel_material)
    parts["display"] = bproc.object.create_primitive(
        "PLANE", scale=(np.array([0.0190, 0.0295, 1.0]) * proxy_scale).tolist())
    parts["display"].replace_materials(display_material)

    parts["top_label"] = bproc.object.create_primitive(
        "CUBE", scale=(np.array([0.0100, 0.0200, 0.00035]) * proxy_scale).tolist())
    parts["top_label"].replace_materials(white_material)

    parts["tag_base"] = bproc.object.create_primitive(
        "CUBE", scale=(np.array([0.00035, 0.0095, 0.0095]) * proxy_scale).tolist())
    parts["tag_base"].replace_materials(white_material)
    tag_pixels = []
    for y_offset, z_offset in ((-0.0048, -0.0048), (-0.0048, 0.0048),
                               (0.0048, -0.0048), (0.0020, 0.0020)):
        pixel = bproc.object.create_primitive(
            "CUBE", scale=(np.array([0.00045, 0.0023, 0.0023]) * proxy_scale).tolist())
        pixel.replace_materials(black_material)
        tag_pixels.append((pixel, y_offset, z_offset))
    parts["tag_pixels"] = tag_pixels

    parts["status_led"] = bproc.object.create_primitive(
        "SPHERE", scale=(np.array([0.0010, 0.0010, 0.0010]) * proxy_scale).tolist())
    parts["status_led"].replace_materials(led_material)
    return parts


def place_camera_appearance_parts(parts, location, rotation, proxy_scale,
                                  show_screen, show_label, show_tag, show_led,
                                  face_sign=1.0):
    if not parts:  # Refined asset already contains all physical appearance parts.
        return
    set_local_pose(parts["bezel"], location, rotation,
                   np.array([face_sign * 0.0170, 0.0, 0.0]) * proxy_scale)
    set_local_pose(parts["display"], location, rotation,
                   np.array([face_sign * 0.0182, 0.0, 0.0]) * proxy_scale,
                   axis_rotation("y", face_sign * np.pi / 2.0))
    set_local_pose(parts["top_label"], location, rotation,
                   np.array([0.0000, 0.0040, 0.02245]) * proxy_scale)
    set_local_pose(parts["tag_base"], location, rotation,
                   np.array([0.01675, -0.0160, 0.0015]) * proxy_scale)
    for pixel, y_offset, z_offset in parts["tag_pixels"]:
        set_local_pose(pixel, location, rotation,
                       np.array([0.01715, -0.0160 + y_offset, 0.0015 + z_offset])
                       * proxy_scale)
    set_local_pose(parts["status_led"], location, rotation,
                   np.array([0.0169, 0.0250, -0.0140]) * proxy_scale)

    parts["bezel"].hide(not show_screen)
    parts["display"].hide(not show_screen)
    parts["top_label"].hide(not show_label)
    parts["tag_base"].hide(not show_tag)
    for pixel, _, _ in parts["tag_pixels"]:
        pixel.hide(not show_tag)
    parts["status_led"].hide(not show_led)


def all_camera_parts(parts):
    if not parts:
        return []
    result = [parts["bezel"], parts["display"], parts["top_label"],
              parts["tag_base"], parts["status_led"]]
    result.extend(pixel for pixel, _, _ in parts["tag_pixels"])
    return result


def attach_model_pbr_maps(obj, model_dir):
    """Attach the BOP texture explicitly; PLY import does not bind TextureFile."""
    material = obj.get_materials()[0].blender_obj
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")

    color_path = os.path.join(model_dir, "obj_000001.png")
    if not os.path.isfile(color_path):
        raise FileNotFoundError(f"camera base-color texture missing: {color_path}")
    if not obj.blender_obj.data.uv_layers:
        raise RuntimeError("camera mesh has no UV layer; cannot map its texture")
    color_texture = nodes.new("ShaderNodeTexImage")
    color_texture.image = bpy.data.images.load(color_path, check_existing=True)
    color_texture.image.colorspace_settings.name = "sRGB"
    links.new(color_texture.outputs["Color"], principled.inputs["Base Color"])

    for map_name, input_name in (("roughness", "Roughness"), ("metallic", "Metallic")):
        path = os.path.join(model_dir, f"obj_000001_{map_name}.png")
        if not os.path.isfile(path):
            continue
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(path, check_existing=True)
        texture.image.colorspace_settings.name = "Non-Color"
        links.new(texture.outputs["Color"], principled.inputs[input_name])

    normal_path = os.path.join(model_dir, "obj_000001_normal.png")
    if os.path.isfile(normal_path):
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(normal_path, check_existing=True)
        texture.image.colorspace_settings.name = "Non-Color"
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.55
        links.new(texture.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])


def attach_asset_pbr_maps(obj, asset_dir):
    """Bind the texture set exported alongside the clothing/clutter OBJ."""
    materials = obj.get_materials()
    if not materials:
        material = bproc.material.create("asset_material")
        obj.replace_materials(material)
        materials = [material]
    material = materials[0].blender_obj
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    def image(name, colorspace):
        path = os.path.join(asset_dir, name)
        if not os.path.isfile(path):
            return None
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(path, check_existing=True)
        tex.image.colorspace_settings.name = colorspace
        return tex
    diffuse = image("texture_diffuse.png", "sRGB")
    if diffuse:
        links.new(diffuse.outputs["Color"], shader.inputs["Base Color"])
    roughness = image("texture_roughness.png", "Non-Color")
    if roughness:
        links.new(roughness.outputs["Color"], shader.inputs["Roughness"])
    metallic = image("texture_metallic.png", "Non-Color")
    if metallic:
        links.new(metallic.outputs["Color"], shader.inputs["Metallic"])
    normal = image("texture_normal.png", "Non-Color")
    if normal:
        node = nodes.new("ShaderNodeNormalMap")
        node.inputs["Strength"].default_value = 0.45
        links.new(normal.outputs["Color"], node.inputs["Color"])
        links.new(node.outputs["Normal"], shader.inputs["Normal"])
    shader.inputs["Roughness"].default_value = max(float(shader.inputs["Roughness"].default_value), 0.72)


def attach_camera_asset_pbr_maps(obj, asset_dir):
    """Bind the PBR texture set shipped with the replacement camera mesh."""
    materials = obj.get_materials()
    if not materials:
        material = bproc.material.create("replacement_camera_material")
        obj.replace_materials(material)
        materials = [material]

    for material_index, material in enumerate(materials):
        blender_material = material.blender_obj
        blender_material.use_nodes = True
        nodes, links = blender_material.node_tree.nodes, blender_material.node_tree.links
        shader = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")

        def texture(name, colorspace):
            path = os.path.join(asset_dir, name)
            if not os.path.isfile(path):
                return None
            node = nodes.new("ShaderNodeTexImage")
            node.image = bpy.data.images.load(path, check_existing=True)
            node.image.colorspace_settings.name = colorspace
            return node

        diffuse = texture("texture_diffuse.png", "sRGB")
        if diffuse:
            links.new(diffuse.outputs["Color"], shader.inputs["Base Color"])
        roughness = texture("texture_roughness.png", "Non-Color")
        if roughness:
            links.new(roughness.outputs["Color"], shader.inputs["Roughness"])
        metallic = texture("texture_metallic.png", "Non-Color")
        if metallic:
            links.new(metallic.outputs["Color"], shader.inputs["Metallic"])
        normal = texture("texture_normal.png", "Non-Color")
        if normal:
            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.inputs["Strength"].default_value = 0.35
            links.new(normal.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
        shader.inputs["Base Color"].default_value = [0.012, 0.014, 0.016, 1.0]
        shader.inputs["Roughness"].default_value = 0.52
        blender_material.name = f"replacement_camera_material_{material_index}"


def load_replacement_camera_objects(asset_dir, count=2, target_width=0.065):
    """Load, orient and metrically normalize the user-supplied camera mesh.

    The OBJ front is +Z with width along X and height along Y. The renderer uses
    +X as the front, +Y as width and +Z as height, hence the Z/X/Y permutation.
    """
    blend_path = os.path.join(asset_dir, "refined_camera.blend")
    if os.path.isfile(blend_path):
        with bpy.data.libraries.load(blend_path, link=False) as (source, loaded):
            loaded.objects = ["refined_action_camera"]
        mesh = loaded.objects[0]
        if mesh is None:
            raise ValueError(f"refined_action_camera mesh missing in {blend_path}")
        bpy.context.collection.objects.link(mesh)
        template = bproc.types.MeshObject(mesh)
        with open(os.path.join(asset_dir, "model_info.json"), encoding="utf-8") as f:
            bounds = json.load(f)["corner_bounds_mm"]
        # Read bounds from geometry, rather than trusting a stale sidecar.
        vertices = np.asarray([v.co[:] for v in mesh.data.vertices]) * 1000.0
        lo, hi = vertices.min(0), vertices.max(0)
        for i, axis in enumerate("xyz"):
            if not np.allclose([bounds[f"min_{axis}"], bounds[f"max_{axis}"]],
                               [lo[i], hi[i]], atol=0.001):
                raise ValueError("refined camera geometry and model_info.json disagree")
        objects = [template] + [template.duplicate(linked=True) for _ in range(count - 1)]
        for index, obj in enumerate(objects):
            obj.set_name(f"refined_action_camera_{index + 1}")
            obj.set_cp("category_id", 1)
        return objects, bounds
    obj_path = os.path.join(asset_dir, "base.obj")
    loaded = bproc.loader.load_obj(obj_path, forward_axis="Y", up_axis="Z")
    if len(loaded) != 1:
        raise RuntimeError(f"expected one camera mesh in {obj_path}, found {len(loaded)}")
    template = loaded[0]
    attach_camera_asset_pbr_maps(template, asset_dir)
    template.set_shading_mode("auto")

    vertices = np.asarray([
        list(vertex.co) for vertex in template.blender_obj.data.vertices
    ], dtype=np.float64)
    oriented = vertices[:, [2, 0, 1]]
    lo, hi = oriented.min(axis=0), oriented.max(axis=0)
    centre = (lo + hi) / 2.0
    scale = float(target_width) / max(float(hi[1] - lo[1]), 1e-9)
    normalized = (oriented - centre) * scale
    for vertex, point in zip(template.blender_obj.data.vertices, normalized):
        vertex.co = point
    template.blender_obj.data.update()
    template.set_scale([1.0, 1.0, 1.0])

    objects = [template]
    for _ in range(1, count):
        objects.append(template.duplicate(linked=True))
    for index, obj in enumerate(objects):
        obj.set_name(f"replacement_action_camera_{index + 1}")
        obj.set_cp("category_id", 1)

    normalized_lo = normalized.min(axis=0)
    normalized_hi = normalized.max(axis=0)
    bounds_mm = {
        "min_x": float(normalized_lo[0] * 1000.0),
        "min_y": float(normalized_lo[1] * 1000.0),
        "min_z": float(normalized_lo[2] * 1000.0),
        "size_x": float((normalized_hi[0] - normalized_lo[0]) * 1000.0),
        "size_y": float((normalized_hi[1] - normalized_lo[1]) * 1000.0),
        "size_z": float((normalized_hi[2] - normalized_lo[2]) * 1000.0),
    }
    for axis in "xyz":
        bounds_mm[f"max_{axis}"] = bounds_mm[f"min_{axis}"] + bounds_mm[f"size_{axis}"]
    return objects, bounds_mm


def load_asset_object(asset_dir, name, target_size, location, rotation=None, floor_z=0.0):
    """Load an OBJ, normalize its largest dimension, and place it on the table."""
    loaded = bproc.loader.load_obj(os.path.join(asset_dir, "base.obj"),
                                  forward_axis="Y", up_axis="Z")
    if not loaded:
        raise RuntimeError(f"asset OBJ contains no objects: {asset_dir}")
    if len(loaded) != 1:
        raise RuntimeError(f"Expected one mesh per asset: {asset_dir}")
    obj = loaded[0]
    obj.set_name(name)
    attach_asset_pbr_maps(obj, asset_dir)
    obj.set_shading_mode("auto")
    vertices = np.array([list(v.co) for v in obj.blender_obj.data.vertices])
    # These exports already use Blender's Z-up coordinates. In particular,
    # rotating clutter around X would tip the pile over below the tabletop.
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    for vertex, point in zip(obj.blender_obj.data.vertices, vertices - center):
        vertex.co = point
    obj.set_rotation_euler([0, 0, 0])
    obj.set_scale([1, 1, 1])
    dims = hi - lo
    scale = float(target_size) / max(float(np.max(dims)), 1e-6)
    obj.set_scale(np.asarray(obj.get_scale()) * scale)
    if rotation is not None:
        obj.set_rotation_euler(rotation)
    obj.set_location(location)
    bpy.context.view_layer.update()
    # Recompute the bottom after rotation/scale and rest it just above the table.
    world_corners = np.array([list(v) for v in obj.blender_obj.bound_box])
    world_corners = (np.array(obj.blender_obj.matrix_world) @
                     np.column_stack((world_corners, np.ones(8))).T).T[:, :3]
    loc = np.asarray(obj.get_location()).copy()
    loc[2] += floor_z + 0.004 - float(world_corners[:, 2].min())
    obj.set_location(loc)
    obj.set_cp("asset_name", name)
    return obj


def create_segment(start, end, radius, material):
    """Create a capsule-like cylinder between two world-space points."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    z_axis = direction / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    segment = bproc.object.create_primitive("CYLINDER", scale=[radius, radius, length / 2.0],
                                            location=(start + end) / 2.0)
    segment.set_rotation_mat(np.column_stack((x_axis, y_axis, z_axis)))
    segment.replace_materials(material)
    segment.set_shading_mode("smooth")
    return segment


def rotation_from_z_axis(z_axis):
    """Return an orthonormal frame whose local Z axis follows z_axis."""
    z_axis = np.asarray(z_axis, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))


def add_wrist_context(wrist, elbow, mount, axis, arm_radius, band_radius,
                      cam_center, camera_rotation, skin_material, strap_material,
                      cable_material, rng, with_camera=True, mount_probability=0.85,
                      device_materials=None, device_probability=1.0):
    """Build a smooth first-person forearm anchored to a wrist camera."""
    wrist = np.asarray(wrist, dtype=float)
    context = []
    arm = create_segment(wrist, elbow, arm_radius, skin_material)
    if arm is not None:
        context.append(arm)
    for point, radius in ((wrist, arm_radius * 0.96), (elbow, arm_radius * 1.02)):
        cap = bproc.object.create_primitive("SPHERE", scale=[radius] * 3, location=point)
        cap.replace_materials(skin_material)
        cap.set_shading_mode("smooth")
        context.append(cap)

    band_width = rng.uniform(0.030, 0.040)
    band = create_segment(mount - axis * band_width * 0.5,
                          mount + axis * band_width * 0.5,
                          band_radius, strap_material)
    if band is not None:
        context.append(band)

    # The real gray cuff/white wrist device is a distractor, not the camera.
    # Include it independently of target presence, including camera-free renders.
    if device_materials is not None and rng.random() < device_probability:
        cuff_material, device_material, trim_material = device_materials
        if band is not None:
            band.replace_materials(cuff_material)
        top_normal = np.array([0.0, 0.0, 1.0])
        top_normal -= np.dot(top_normal, axis) * axis
        top_normal /= np.linalg.norm(top_normal)
        cross_axis = np.cross(axis, top_normal)
        cuff_rotation = np.column_stack((cross_axis, axis, top_normal))
        cuff = bproc.object.create_primitive("CUBE", scale=[0.031, band_width * 0.55, 0.004])
        cuff.set_location(mount + top_normal * (band_radius + 0.002))
        cuff.set_rotation_mat(cuff_rotation)
        cuff.replace_materials(cuff_material)
        context.append(cuff)
        device = bproc.object.create_primitive(
            "SPHERE", scale=[0.015, 0.025, rng.uniform(0.008, 0.010)])
        device.set_location(mount + top_normal * (band_radius + 0.015)
                            + cross_axis * rng.uniform(-0.004, 0.004))
        device.set_rotation_mat(cuff_rotation)
        device.replace_materials(device_material)
        device.set_shading_mode("smooth")
        context.append(device)
        for sign in (-1, 1):
            center = mount + axis * sign * band_width * 0.5
            trim = create_segment(center - axis * 0.002, center + axis * 0.002,
                                  band_radius + 0.002, trim_material)
            if trim is not None:
                context.append(trim)
            # Continue the green edging across the raised gray cuff. A ring
            # alone sits below this panel and disappears from overhead views.
            edge = bproc.object.create_primitive(
                "CUBE", scale=[0.032, 0.002, 0.0015])
            edge.set_rotation_mat(cuff_rotation)
            edge.set_location(center + top_normal * (band_radius + 0.0065))
            edge.replace_materials(trim_material)
            context.append(edge)

    # Real footage contains a prominent black wrist plate and rotating camera
    # mount. Omitting it makes the rendered camera silhouette and attachment
    # geometry fundamentally different from the target domain.
    if with_camera and rng.random() < mount_probability:
        mount_plate = bproc.object.create_primitive(
            "CUBE", scale=[0.016, 0.027, 0.006])
        set_local_pose(mount_plate, cam_center, camera_rotation,
                       np.array([0.0, 0.0, -0.030]))
        mount_plate.replace_materials(strap_material)
        context.append(mount_plate)
        mount_ball = bproc.object.create_primitive(
            "SPHERE", scale=[0.014, 0.014, 0.014])
        set_local_pose(mount_ball, cam_center, camera_rotation,
                       np.array([0.0, 0.0, -0.022]))
        mount_ball.replace_materials(strap_material)
        mount_ball.set_shading_mode("smooth")
        context.append(mount_ball)

    hand_direction = wrist - mount
    hand_direction /= np.linalg.norm(hand_direction)
    palm_center = wrist + hand_direction * rng.uniform(0.030, 0.045)
    palm = bproc.object.create_primitive("SPHERE", scale=[0.034, 0.020, 0.055],
                                         location=palm_center)
    hand_rotation = rotation_from_z_axis(hand_direction)
    palm.set_rotation_mat(hand_rotation)
    palm.replace_materials(skin_material)
    palm.set_shading_mode("smooth")
    context.append(palm)

    finger_count = rng.randint(3, 4)
    for finger_index in range(finger_count):
        lateral = (finger_index - (finger_count - 1) / 2.0) * 0.011
        start = palm_center + hand_direction * 0.038 + hand_rotation[:, 0] * lateral
        end = start + hand_direction * rng.uniform(0.028, 0.050)
        finger = create_segment(start, end, rng.uniform(0.0055, 0.007), skin_material)
        if finger is not None:
            context.append(finger)


    if with_camera and rng.random() < 0.80:
        cable_start = cam_center + camera_rotation @ np.array([0.0, 0.030, -0.004])
        cable_mid = mount + np.array([rng.uniform(-0.012, 0.012), 0.015,
                                      -(arm_radius + 0.008)])
        cable_end = elbow + np.array([rng.uniform(-0.01, 0.01), 0.03, 0.0])
        for a, b in ((cable_start, cable_mid), (cable_mid, cable_end)):
            cable = create_segment(a, b, rng.uniform(0.0016, 0.0026), cable_material)
            if cable is not None:
                context.append(cable)
    return context


def create_target_scene(denim_material, metal_material, pants_material, rng, asset_root):
    """Create the stable white grid, scalable denim jacket, and pants context."""
    scene_objects = []
    garment_objects = []

    # Photogrammetric garments are the primary clothing context.  Keep all
    # variants loaded once and select one per frame to avoid accumulated scale.
    garment_specs = [
        ("牛仔模型", 0.68, 0.64, [0.0, 0.18, 0.0]),
        ("牛仔模型_2", 0.19, 0.55, [0.0, 0.18, 0.0]),
        ("牛仔外套_3", 0.13, 0.58, [0.0, 0.18, 0.0]),
    ]
    for asset_name, _, target_size, location in garment_specs:
        asset_dir = os.path.join(asset_root, asset_name)
        garment_objects.append(load_asset_object(
            asset_dir, f"garment_{asset_name}", target_size, location,
            rotation=[0.0, 0.0, rng.uniform(-0.18, 0.18)]))

    # === 衣服全局缩放系数 (1.0为默认大小，这里设置为 1.3 放大) ===
    g_scale = 1.3
    base_y = 0.34  # 衣服在桌面上的中心 Y 坐标基准点

    # The old primitive garment geometry is intentionally omitted; the loaded
    # meshes above preserve pockets, seams and the real silhouette.
    """  # legacy proxy garment retained in source for reference
    # 1. 缩放躯干
    for x_offset, scale in ((0.0, [0.14, 0.17, 0.018]),
                            (-0.07, [0.09, 0.15, 0.014]),
                            (0.07, [0.09, 0.15, 0.014])):
        torso = bproc.object.create_primitive(
            "SPHERE",
            scale=[scale[0] * g_scale, scale[1] * g_scale, scale[2]],
            location=[x_offset * g_scale, base_y, 0.015]
        )
        torso.replace_materials(denim_material)
        torso.set_shading_mode("smooth")
        garment_objects.append(torso)

    # 2. 缩放袖子和金属扣
    for side in (-1.0, 1.0):
        sleeve = create_segment(
            [side * 0.12 * g_scale, base_y + 0.04 * g_scale, 0.018],
            [side * 0.34 * g_scale, base_y + 0.08 * g_scale, 0.018],
            0.048 * g_scale, denim_material
        )
        if sleeve is not None:
            garment_objects.append(sleeve)

        for button_y_offset in (-0.10, -0.01, 0.08):
            button = bproc.object.create_primitive(
                "CYLINDER",
                scale=[0.007 * g_scale, 0.007 * g_scale, 0.003],
                location=[side * 0.025 * g_scale, base_y + button_y_offset * g_scale, 0.025]
            )
            button.replace_materials(metal_material)
            garment_objects.append(button)

    """
    # ================== 修正的环境干扰构建 ==================

   # === 1. 模拟操作者的躯干与大腿 (占据画面底部) ===
    pants_material = bproc.material.create("black_pants")
    pants_material.set_principled_shader_value("Base Color", [0.015, 0.015, 0.018, 1.0])
    pants_material.set_principled_shader_value("Roughness", 0.9)

    # 使用一个被极度压扁的宽大球体模拟操作者的腹部/躯干
    operator_lap = bproc.object.create_primitive(
        "SPHERE",
        # 【关键修复】：大幅压扁 Z 轴厚度 (从 0.35 降至 0.12)，X 轴加宽以覆盖整个底部
        scale=[0.55, 0.15, 0.12],
        # 【关键修复】：大幅降低 Z 轴高度 (降至 0.05)，使其沉降到桌子(0.1)边缘的下方
        # Y=-0.35 确保它位于桌子边缘，在相机的前下方
        location=[0.0, -0.35, 0.05]
    )
    # 稍微向后倾斜
    operator_lap.set_rotation_euler([0.15, 0, 0])
    operator_lap.replace_materials(pants_material)
    operator_lap.set_shading_mode("smooth")
    scene_objects.append(operator_lap)

    # === 2. 模拟白墙与暗色地板 ===
    wall_material = bproc.material.create("white_wall")
    wall_material.set_principled_shader_value("Base Color", [0.7, 0.7, 0.72, 1.0])
    wall_material.set_principled_shader_value("Roughness", 0.95)
    
    floor_material = bproc.material.create("wood_floor")
    floor_material.set_principled_shader_value("Base Color", [0.15, 0.1, 0.05, 1.0]) 
    floor_material.set_principled_shader_value("Roughness", 0.8)

    # 墙壁保持在前方
    wall = bproc.object.create_primitive("PLANE", scale=[2.0, 2.0, 1.0], location=[0, 0.8, 0])
    wall.set_rotation_euler([np.pi/2, 0, 0]) 
    wall.replace_materials(wall_material)
    
    # 地板保持在下方
    floor = bproc.object.create_primitive("PLANE", scale=[3.0, 3.0, 1.0], location=[0, 0, -0.8])
    floor.replace_materials(floor_material)
    
    scene_objects.extend([wall, floor])

    # === 3. 左侧真实杂物资产与右侧纸箱 ===
    box_material = bproc.material.create("cardboard_box")
    box_material.set_principled_shader_value("Base Color", [0.4, 0.3, 0.15, 1.0])
    box_material.set_principled_shader_value("Roughness", 0.85)

    clutter_dir = os.path.join(asset_root, "左侧杂物")
    left_clutter = load_asset_object(
        clutter_dir, "left_clutter_asset", 0.95, [-0.77, 0.10, -0.8],
        rotation=[0.0, 0.0, 0.0], floor_z=-0.8)
    scene_objects.append(left_clutter)
        
    # 右侧：象征性放 1 个浅色纸箱，制造非对称感
    right_box = bproc.object.create_primitive(
        "CUBE", 
        scale=[0.15, 0.2, 0.25], 
        location=[rng.uniform(0.7, 0.9), rng.uniform(0.2, 0.5), -0.4]
    )
    right_box.replace_materials(box_material)
    scene_objects.append(right_box)

    return scene_objects, garment_objects

def uint8_color(color):
    pixels = np.asarray(color)
    if np.issubdtype(pixels.dtype, np.floating) and pixels.max(initial=0.0) <= 1.0:
        pixels = pixels * 255.0
    return np.clip(pixels, 0, 255).astype(np.uint8)


def motion_blur(image, rng):
    # Pillow's generic Kernel filter is reliably supported for 3x3/5x5.
    size = rng.choice([3, 5])
    kernel = np.zeros((size, size), dtype=np.float64)
    mode = rng.choice(["horizontal", "vertical", "diag", "anti_diag"])
    if mode == "horizontal":
        kernel[size // 2, :] = 1.0
    elif mode == "vertical":
        kernel[:, size // 2] = 1.0
    elif mode == "diag":
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    kernel /= kernel.sum()
    return image.filter(ImageFilter.Kernel((size, size), kernel.ravel(), scale=1.0))


def process_color(color, rng, args):
    """Approximate wide-angle action-camera exposure and temporal artifacts."""
    image = Image.fromarray(uint8_color(color), mode="RGB")
    soft = args.lighting_profile == "frames_soft"
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.02) if soft else rng.uniform(0.78, 1.12))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 0.98) if soft else rng.uniform(0.88, 1.14))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.88, 1.0) if soft else rng.uniform(0.82, 1.18))

    if rng.random() < args.motion_blur_probability:
        image = motion_blur(image, rng)
    elif rng.random() < 0.55:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.20, 0.90)))

    pixels = np.asarray(image).astype(np.float32)
    height, width = pixels.shape[:2]
    if rng.random() < args.vignette_probability:
        yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
        radius = np.sqrt(xx * xx + yy * yy)
        strength = rng.uniform(0.08, 0.26)
        vignette = np.clip(1.0 - strength * radius ** rng.uniform(1.5, 2.6), 0.68, 1.0)
        pixels *= vignette[..., None]

    if args.sensor_noise_std > 0:
        noise_std = rng.uniform(0.0, args.sensor_noise_std)
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        pixels += noise_rng.normal(0.0, noise_std, pixels.shape)

    # Small channel offsets mimic demosaicing/chromatic aberration near a
    # wide-angle action-camera lens without introducing a fixed pattern.
    if rng.random() < 0.35:
        offset = rng.choice([-2, -1, 1, 2])
        pixels[..., 0] = np.roll(pixels[..., 0], offset, axis=1)
        pixels[..., 2] = np.roll(pixels[..., 2], -offset, axis=1)
    return np.clip(pixels, 0, 255).astype(np.uint8)


def jitter_reference_color(color, rng, scale_range=(0.86, 1.14), channel_range=(0.94, 1.06)):
    """Apply restrained RGB variation without changing the material identity."""
    reference = np.asarray(color, dtype=np.float64)[:3]
    scale = rng.uniform(*scale_range)
    channels = np.asarray([rng.uniform(*channel_range) for _ in range(3)])
    return np.concatenate((np.clip(reference * scale * channels, 0.0, 1.0), [1.0]))


def place_on_surface(location, rotation, clearance=0.003):
    """Set Z so the rotated metric bounding box rests above the table."""
    half_extents = np.array([0.0164, 0.03525, 0.0221])
    corners = np.array([
        [x, y, z]
        for x in (-half_extents[0], half_extents[0])
        for y in (-half_extents[1], half_extents[1])
        for z in (-half_extents[2], half_extents[2])
    ])
    rotated = corners @ rotation.T
    result = location.copy()
    result[2] = clearance - np.min(rotated[:, 2])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_images", type=int, help="渲染图片数量")
    parser.add_argument("cc0textures", type=str, nargs="?", default=None,
                        help="generic 场景所需的 cc0textures-512 材质库路径")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="BOP 数据集根目录；默认使用当前工作目录")
    parser.add_argument("--output-split", default="train_pbr",
                        help="渲染输出子目录，例如 new_train_pbr；不修改其他 split")
    parser.add_argument("--frames-per-scene", type=int, default=1000,
                        help="每个 BOP scene 的帧数；2 张流程测试可设为 1")
    parser.add_argument("--fov", type=float, default=82.0, help="水平视场角(度)")
    parser.add_argument("--fov-min", type=float, default=74.0,
                        help="逐帧随机水平视场角下限")
    parser.add_argument("--fov-max", type=float, default=100.0,
                        help="逐帧随机水平视场角上限")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--arms", action=argparse.BooleanOptionalAction, default=True,
                        help="生成与左右腕绑定的前臂、手掌、手指、腕带和线缆（默认开启）")
    parser.add_argument("--front-probability", type=float, default=0.25,
                        help="正面视角概率")
    parser.add_argument("--back-probability", type=float, default=0.35,
                        help="背面视角概率；侧面 = 1 - 正面 - 背面")
    parser.add_argument("--lower-middle-probability", type=float, default=0.55,
                        help="目标集中在画面下方的概率；其余样本覆盖中心、边缘和抬手构图")
    parser.add_argument("--occlusion-probability", type=float, default=0.35,
                        help="每个实例出现遮挡的总概率；遮挡内部再随机轻/中/重等级")
    parser.add_argument("--min-cameras", type=int, default=1,
                        help="普通正样本中最少可见相机数")
    parser.add_argument("--max-cameras", type=int, default=2,
                        help="普通正样本中最多可见相机数")
    parser.add_argument("--single-camera-probability", type=float, default=0.40,
                        help="允许 1-2 台相机时，只有一台可见的概率")
    parser.add_argument("--context-dropout-probability", type=float, default=0.30,
                        help="将简化手臂上下文替换为桌面摆放的概率，抑制人体代理捷径")
    parser.add_argument("--mount-probability", type=float, default=0.90,
                        help="相机附带腕部底座/转轴的概率")
    parser.add_argument("--wrist-side-probability", type=float, default=0.70,
                        help="相机位于腕部侧面而非顶部的概率")
    parser.add_argument("--wrist-device-probability", type=float, default=1.0,
                        help="生成灰色袖带/白色塑料件/绿色条边的概率，默认每只手腕固定生成")
    parser.add_argument("--pose-profile", choices=["mixed", "raised", "work", "fold"],
                        default="mixed", help="联动头部视角和双腕位置的第一视角动作模板")
    parser.add_argument("--min-target-width-fraction", type=float, default=0.075,
                        help="投影框宽度占图像比例的审计阈值；不满足时写入元数据")
    parser.add_argument("--render-device", choices=["auto", "cpu"], default="auto",
                        help="小批预览可选 cpu，避免特定 GPU 的内核初始化开销")
    parser.add_argument("--lighting-profile", choices=["frames_soft", "legacy"], default="frames_soft")
    parser.add_argument("--garment-variant", choices=["mixed", "main", "second", "third"], default="mixed",
                        help="mixed 按 68/19/13%% 选姿态；其他选项用于检查单个资产")
    parser.add_argument("--label-probability", type=float, default=0.70,
                        help="相机顶部出现白色标签的概率")
    parser.add_argument("--marker-probability", type=float, default=0.60,
                        help="背面出现高对比标记的概率")
    parser.add_argument("--samples", type=int, default=8, help="Cycles 每像素采样数")
    parser.add_argument("--scene-profile", choices=("target", "generic"), default="target",
                        help="target 匹配 frames 中的室内腕戴场景；generic 使用 CC0 地面")
    parser.add_argument("--garment-probability", type=float, default=0.85,
                        help="target 场景中出现蓝色衣物代理的概率；避免衣物与相机强绑定")
    parser.add_argument("--sensor-effects", action=argparse.BooleanOptionalAction, default=True,
                        help="对 target RGB 应用接近测试视频的曝光、饱和度、模糊和 JPEG 退化")
    parser.add_argument("--rear-screen-proxy", action=argparse.BooleanOptionalAction, default=True,
                        help="添加具有随机亮度的背屏代理（默认开启）")
    parser.add_argument(
        "--camera-model-dir",
        default=DEFAULT_CAMERA_MODEL_DIR if os.path.isdir(DEFAULT_CAMERA_MODEL_DIR) else None,
        help=("相机资产目录；优先使用 assets/refined_action_camera 的完整模型，支持旧 OBJ/PBR 目录，"
              "传入空字符串可回退到 BOP models/obj_000001.ply"),
    )
    parser.add_argument("--motion-blur-probability", type=float, default=0.28,
                        help="整帧方向性运动模糊概率")
    parser.add_argument("--vignette-probability", type=float, default=0.45,
                        help="模拟广角运动相机暗角的概率")
    parser.add_argument("--sensor-noise-std", type=float, default=3.5,
                        help="8-bit RGB 传感器高斯噪声标准差上限")
    parser.add_argument("--preview-dir", type=str, default=None,
                        help="仅保存 RGB 样图到此目录，不追加或改写 BOP 数据集")
    parser.add_argument("--camera-free-preview", action="store_true",
                        help="预览模式中隐藏相机，生成含手臂/衣物的困难负样本")
    parser.add_argument("--preview-yolo-labels", action="store_true",
                        help="为普通 RGB 预览输出基于可见相机分割的 YOLO 标签，不修改 BOP 数据")
    parser.add_argument("--object-layer-preview", action="store_true",
                        help="将每个相机分别输出为透明 PNG，供真实背景合成")
    parser.add_argument("--object-layer-scale", type=float, default=4.0,
                        help="透明相机层的模型放大倍率，用于保留合成缩放所需的像素细节")
    parser.add_argument("--bop-masks", action="store_true",
                        help="额外生成 BOP mask/info/COCO（较慢，macOS 使用单进程）")
    args = parser.parse_args()
    if args.num_images < 1 or args.frames_per_scene < 1:
        parser.error("num_images and --frames-per-scene must be positive")
    if not args.output_split or args.output_split in {".", ".."} or any(
            char in args.output_split for char in "/\\"):
        parser.error("--output-split must be a single directory name")

    if (args.camera_free_preview or args.object_layer_preview) and not args.preview_dir:
        parser.error("layer preview modes require --preview-dir")
    if args.camera_free_preview and args.object_layer_preview:
        parser.error("camera-free and object-layer preview modes are mutually exclusive")
    if args.preview_yolo_labels and (
            not args.preview_dir or args.camera_free_preview or args.object_layer_preview):
        parser.error("--preview-yolo-labels requires ordinary RGB --preview-dir")
    if args.preview_yolo_labels and os.path.isdir(args.preview_dir) and os.listdir(args.preview_dir):
        parser.error("labeled preview output must be a new or empty directory")
    if args.object_layer_scale <= 0:
        parser.error("--object-layer-scale must be positive")
    if not 1 <= args.min_cameras <= args.max_cameras <= 2:
        parser.error("require 1 <= --min-cameras <= --max-cameras <= 2")
    if args.front_probability < 0 or args.back_probability < 0 or \
            args.front_probability + args.back_probability > 1:
        parser.error("front/back probabilities must be non-negative and sum to <= 1")
    for name in ("single_camera_probability", "context_dropout_probability",
                 "mount_probability", "label_probability", "marker_probability",
                 "motion_blur_probability", "vignette_probability",
                 "wrist_side_probability", "wrist_device_probability"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.fov_min <= 0 or args.fov_max < args.fov_min or args.fov_max >= 170:
        parser.error("require 0 < --fov-min <= --fov-max < 170")
    if not 0 < args.min_target_width_fraction < 1:
        parser.error("--min-target-width-fraction must be between 0 and 1")
    if args.sensor_noise_std < 0:
        parser.error("--sensor-noise-std cannot be negative")
    if args.scene_profile == "generic" and not args.cc0textures:
        parser.error("generic scene profile requires a cc0textures path")
    args.camera_model_dir = (os.path.abspath(args.camera_model_dir)
                             if args.camera_model_dir else None)
    if args.camera_model_dir:
        refined_camera = os.path.isfile(os.path.join(args.camera_model_dir, "refined_camera.blend"))
        required_files = ("model_info.json",) if refined_camera else ("base.obj", "texture_diffuse.png", "正面贴图.png")
        for required in required_files:
            if not os.path.isfile(os.path.join(args.camera_model_dir, required)):
                parser.error(f"camera model asset missing: {required}")
    else:
        refined_camera = False

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    current_dir = os.path.abspath(args.dataset_dir or os.getcwd())
    dataset_name = os.path.basename(current_dir)
    bop_parent_path = os.path.dirname(current_dir)
    bop_dataset_path = os.path.join(bop_parent_path, dataset_name)
    preview_dir = os.path.abspath(args.preview_dir) if args.preview_dir else None
    split_path = os.path.join(current_dir, args.output_split)
    if not preview_dir and glob.glob(os.path.join(split_path, "*", "rgb", "*.jpg")):
        config_file = os.path.join(split_path, "render_config.json")
        old_config = {}
        if os.path.isfile(config_file):
            with open(config_file) as f:
                old_config = json.load(f)
        if old_config.get("camera_model_dir") != args.camera_model_dir:
            raise ValueError("Camera asset differs from existing split. Choose a new --output-split.")
    staging_path = None
    if not preview_dir and args.output_split != "train_pbr":
        # BlenderProc's public writer hardcodes train_pbr. Render in an isolated
        # dataset, then publish complete numbered scenes into the requested split.
        staging_path = tempfile.mkdtemp(prefix=".render_staging_", dir=current_dir)
        bop_parent_path, dataset_name = staging_path, "rendered"

    W, H = args.width, args.height
    fx = fy = W / (2.0 * np.tan(np.deg2rad(args.fov) / 2.0))
    cam_json = os.path.join(current_dir, "camera.json")
    if not preview_dir and staging_path is None:
        with open(cam_json, "w") as f:
            json.dump({"cx": W / 2.0, "cy": H / 2.0, "depth_scale": 0.1,
                      "fx": fx, "fy": fy, "height": H, "width": W}, f, indent=2)

    bproc.init()
    if args.render_device == "cpu":
        bproc.renderer.set_render_devices(use_only_cpu=True)

    cc_textures = []
    if args.scene_profile == "generic":
        if os.path.basename(args.cc0textures) == "cc0textures-512":
            cc_textures = bproc.loader.load_512_ccmaterials(args.cc0textures, use_all_materials=True)
        else:
            cc_textures = bproc.loader.load_ccmaterials(args.cc0textures, use_all_materials=True)

    if preview_dir or staging_path:
        intrinsic_matrix = np.array([[fx, 0.0, W / 2.0],
                                     [0.0, fy, H / 2.0],
                                     [0.0, 0.0, 1.0]])
        bproc.camera.set_intrinsics_from_K_matrix(intrinsic_matrix, W, H)
    else:
        bproc.loader.load_bop_intrinsics(bop_dataset_path=bop_dataset_path)

    table_material = bproc.material.create("target_table")
    table_material.set_principled_shader_value("Base Color", [0.58, 0.61, 0.63, 1.0])
    table_material.set_principled_shader_value("Roughness", 0.92)
    if args.scene_profile == "target":
        configure_cloth_grid(table_material)
    environment_objects = []
    if not args.object_layer_preview:
        table = bproc.object.create_primitive(
            "PLANE", scale=[0.45, 0.35, 1.0], location=[0, 0.1, 0])
        table.set_name("table")
        table.replace_materials(table_material)
        environment_objects.append(table)

    area_lights = []
    light_specs = ([([0, -0.05, 1.75], 2.4, 42), ([0, -0.55, 1.05], 1.8, 10)]
                   if args.lighting_profile == "frames_soft" else
                   [([x, y, 1.15], 0.75, 18) for x, y in
                    ((-0.55, -0.10), (0.55, -0.10), (-0.35, 0.50), (0.35, 0.50))])
    for location, size, energy in light_specs:
        light = bproc.types.Light(light_type="AREA")
        light.set_location(location)
        light.blender_obj.data.size = size
        light.set_energy(energy)
        area_lights.append(light)
    bproc.renderer.set_world_background(
        [0.38, 0.41, 0.44] if args.lighting_profile == "frames_soft" else [0.14, 0.15, 0.16],
        strength=0.45 if args.lighting_profile == "frames_soft" else 0.8)

    skin_material = bproc.material.create("forearm_skin")
    skin_material.set_principled_shader_value("Base Color", [0.34, 0.16, 0.08, 1.0])
    skin_material.set_principled_shader_value("Roughness", 0.65)
    strap_material = bproc.material.create("wrist_strap")
    strap_material.set_principled_shader_value("Base Color", [0.025, 0.03, 0.035, 1.0])
    strap_material.set_principled_shader_value("Roughness", 0.55)
    cable_material = bproc.material.create("camera_cable")
    cable_material.set_principled_shader_value("Base Color", [0.008, 0.010, 0.012, 1.0])
    cable_material.set_principled_shader_value("Roughness", 0.45)
    cuff_material = bproc.material.create("non_target_gray_cuff")
    cuff_material.set_principled_shader_value("Base Color", [0.15, 0.18, 0.19, 1.0])
    cuff_material.set_principled_shader_value("Roughness", 0.85)
    wrist_device_material = bproc.material.create("non_target_white_wrist_device")
    wrist_device_material.set_principled_shader_value("Base Color", [0.75, 0.78, 0.77, 1.0])
    wrist_device_material.set_principled_shader_value("Roughness", 0.65)
    cuff_trim_material = bproc.material.create("non_target_cuff_trim")
    cuff_trim_material.set_principled_shader_value("Base Color", [0.25, 0.65, 0.01, 1.0])
    cuff_trim_material.set_principled_shader_value("Roughness", 0.70)
    bezel_material = bproc.material.create("rear_screen_bezel")
    bezel_material.set_principled_shader_value("Base Color", [0.008, 0.010, 0.012, 1.0])
    bezel_material.set_principled_shader_value("Roughness", 0.35)
    display_material = bproc.material.create("rear_screen_display")
    display_material.set_principled_shader_value("Base Color", [0.025, 0.045, 0.052, 1.0])
    display_material.set_principled_shader_value("Roughness", 0.18)
    configure_screen_content(
        display_material,
        (os.path.join(args.camera_model_dir, "正面贴图.png")
         if args.camera_model_dir and not refined_camera else None),
    )
    white_label_material = bproc.material.create("camera_white_label")
    white_label_material.set_principled_shader_value(
        "Base Color", [0.78, 0.80, 0.78, 1.0])
    white_label_material.set_principled_shader_value("Roughness", 0.72)
    marker_black_material = bproc.material.create("camera_marker_black")
    marker_black_material.set_principled_shader_value(
        "Base Color", [0.003, 0.004, 0.005, 1.0])
    marker_black_material.set_principled_shader_value("Roughness", 0.55)
    status_led_material = bproc.material.create("camera_status_led")
    status_led_material.set_principled_shader_value(
        "Base Color", [0.01, 0.65, 0.12, 1.0])
    status_led_material.set_principled_shader_value("Roughness", 0.20)

    garment_objects = []
    if args.scene_profile == "target" and not args.object_layer_preview:
        denim_material = bproc.material.create("denim_garment")
        denim_material.set_principled_shader_value("Base Color", [0.075, 0.13, 0.20, 1.0])
        denim_material.set_principled_shader_value("Roughness", 0.96)
        metal_material = bproc.material.create("garment_buttons")
        metal_material.set_principled_shader_value("Base Color", [0.52, 0.52, 0.48, 1.0])
        metal_material.set_principled_shader_value("Metallic", 0.6)
        metal_material.set_principled_shader_value("Roughness", 0.42)

        pants_material = bproc.material.create("pants_fabric")
        pants_material.set_principled_shader_value("Base Color", [0.025, 0.027, 0.030, 1.0])
        pants_material.set_principled_shader_value("Roughness", 0.84)

        asset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3D资产"))
        target_objects, garment_objects = create_target_scene(
            denim_material, metal_material, pants_material, rng, asset_root)
        environment_objects.extend(target_objects)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    if args.object_layer_preview:
        bproc.renderer.set_output_format(
            file_format="PNG", color_depth=8, enable_transparency=True)

    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
    metadata_path = os.path.join(split_path, "scene_metadata.json")
    scene_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            scene_metadata = json.load(f)
    existing_rgb = glob.glob(os.path.join(split_path, "*", "rgb", "*.jpg"))
    metadata_start_id = len(existing_rgb)

    replacement_bounds_mm = None
    if args.camera_model_dir:
        objs, replacement_bounds_mm = load_replacement_camera_objects(
            args.camera_model_dir, count=2,
        )
    else:
        objs = bproc.loader.load_bop_objs(
            bop_dataset_path=bop_dataset_path, mm2m=True, obj_ids=[1])
        assert len(objs) == 1
        objs += bproc.loader.load_bop_objs(
            bop_dataset_path=bop_dataset_path, mm2m=True, obj_ids=[1])
        for obj in objs:
            obj.set_shading_mode("auto")
            attach_model_pbr_maps(obj, os.path.join(bop_dataset_path, "models"))
    for obj in objs:
        if args.object_layer_preview:
            obj.set_scale(np.asarray(obj.get_scale()) * args.object_layer_scale)

    camera_parts = []
    proxy_scale = args.object_layer_scale if args.object_layer_preview else 1.0
    appearance_face_sign = -1.0 if args.camera_model_dir else 1.0
    for _ in objs:
        if refined_camera:
            camera_parts.append({})
            continue
        parts = create_camera_appearance_parts(
            proxy_scale,
            (bezel_material, display_material, white_label_material,
             marker_black_material, status_led_material),
        )
        set_parts_hidden(all_camera_parts(parts), True)
        camera_parts.append(parts)

    if args.preview_yolo_labels:
        os.makedirs(os.path.join(preview_dir, "labels"), exist_ok=True)
        for index, obj in enumerate(objs):
            obj.set_cp("yolo_target", index + 1)
            for part in all_camera_parts(camera_parts[index]):
                part.set_cp("yolo_target", index + 1)

    # ================== 收集需要进行物理随机化的环境材质 ==================
    dr_materials = [m for m in bproc.material.collect_all() if m.get_name() in 
                    ["target_table", "denim_garment", "white_wall", "wood_floor", 
                     "cardboard_box", "pants_fabric", "black_shirt", "dark_equipment"]]
    # ================== 收集并缓存环境材质的真实属性 ==================
    dr_materials_names = ["target_table", "denim_garment", "white_wall", "wood_floor", 
                          "cardboard_box", "pants_fabric", "black_shirt", "dark_equipment"]
    dr_materials = [m for m in bproc.material.collect_all() if m.get_name() in dr_materials_names]
    # Linked grid color is a node socket, not a constant color to cache/jitter.
    if args.scene_profile == "target":
        dr_materials = [m for m in dr_materials if m.get_name() != "target_table"]
    
    # 用一个字典把真实场景的颜色和粗糙度记录下来，防止被随机化覆盖后找不回来
    original_mat_states = {}
    for mat in dr_materials:
        original_mat_states[mat.get_name()] = {
            # 【关键修复】使用 list() 强制拷贝颜色数组，将其变成固定的死数据
            "Base Color": list(mat.get_principled_shader_value("Base Color")),
            # 同样使用 float() 强转数值，切断粗糙度和金属度的引用
            "Roughness": float(mat.get_principled_shader_value("Roughness")),
            "Metallic": float(mat.get_principled_shader_value("Metallic"))
        }
    # ================== 缓存衣服代理几何体的初始尺寸 ==================
    original_garment_scales = [obj.get_scale() for obj in garment_objects]
    original_garment_locations = [np.array(obj.get_location()).copy() for obj in garment_objects]

    for img_idx in range(args.num_images):
        frame_fov = rng.uniform(args.fov_min, args.fov_max)
        frame_fx = frame_fy = W / (2.0 * np.tan(np.deg2rad(frame_fov) / 2.0))
        frame_intrinsics = np.array([[frame_fx, 0.0, W / 2.0],
                                     [0.0, frame_fy, H / 2.0],
                                     [0.0, 0.0, 1.0]])
        bproc.camera.set_intrinsics_from_K_matrix(frame_intrinsics, W, H)

        if args.object_layer_preview:
            active_indices = set(range(len(objs)))
        elif args.camera_free_preview:
            active_indices = set()
        else:
            if args.min_cameras == args.max_cameras:
                camera_count = args.min_cameras
            elif rng.random() < args.single_camera_probability:
                camera_count = 1
            else:
                camera_count = 2
            active_indices = set(rng.sample(range(len(objs)), camera_count))

        frame_has_arms = (
            args.arms and
            (args.camera_free_preview or rng.random() >= args.context_dropout_probability) and
            not args.object_layer_preview
        )
# ------------------ 物理与光照混合随机化 ------------------
        # 30% 的帧使用受约束的材质与光照扰动，提升抗干扰能力但保持目标域颜色
        if rng.random() < 0.30:
            # 1. 轻微光照扰动
            for light in area_lights:
                light.set_energy(rng.uniform(11.0, 30.0))
                light.set_color([rng.uniform(0.88, 1.0), rng.uniform(0.90, 1.0), 1.0])
                light.set_location([rng.uniform(-0.70, 0.70), rng.uniform(-0.25, 0.70),
                                    rng.uniform(0.95, 1.35)])
                    
            # 2. 保持参考色相，仅改变亮度和很小的通道比例
            for mat in dr_materials:
                reference = original_mat_states[mat.get_name()]
                mat.set_principled_shader_value(
                    "Base Color",
                    jitter_reference_color(reference["Base Color"], rng),
                )
                roughness = float(reference["Roughness"])
                metallic = float(reference["Metallic"])
                mat.set_principled_shader_value("Roughness", np.clip(
                    roughness + rng.uniform(-0.10, 0.10), 0.0, 1.0))
                mat.set_principled_shader_value("Metallic", np.clip(
                    metallic + rng.uniform(-0.08, 0.08), 0.0, 1.0))
        
        # 30% 的帧完美恢复为你精心调整过的真实目标场景，稳住基础指标
        else:
            # 1. 恢复原本的稳定光照 (原代码中的微小合理波动)
            original_light_locs = [(-0.55, -0.10), (0.55, -0.10), (-0.35, 0.50), (0.35, 0.50)]
            for i, light in enumerate(area_lights):
                light.set_energy(rng.uniform(13.0, 24.0))
                light.set_color([rng.uniform(0.88, 1.0), rng.uniform(0.90, 1.0), 1.0])
                light.set_location([original_light_locs[i][0], original_light_locs[i][1], 1.15])

            # 2. 恢复真实的物品材质
            for mat in dr_materials:
                mat_name = mat.get_name()
                mat.set_principled_shader_value("Base Color", original_mat_states[mat_name]["Base Color"])
                mat.set_principled_shader_value("Roughness", original_mat_states[mat_name]["Roughness"])
                mat.set_principled_shader_value("Metallic", original_mat_states[mat_name]["Metallic"])
                
        # ------------------ 保持手臂特征稳定 ------------------
        if args.lighting_profile == "frames_soft":
            for light, (location, size, energy) in zip(area_lights, light_specs):
                light.set_energy(energy * rng.uniform(0.85, 1.15))
                light.set_location(np.asarray(location) +
                                   np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), 0]))
                light.set_color([rng.uniform(0.94, 0.98), rng.uniform(0.97, 1.0), 1.0])
        # 手臂无论在哪个分支，都只进行微小的肤色波动
        skin_material.set_principled_shader_value(
            "Base Color", [float(np.random.uniform(0.31, 0.40)),
                           float(np.random.uniform(0.15, 0.21)),
                           float(np.random.uniform(0.085, 0.125)), 1.0])
        skin_material.set_principled_shader_value("Roughness", np.random.uniform(0.52, 0.78))
        display_tint = rng.choice([
            [0.008, 0.012, 0.015, 1.0],
            [0.018, 0.035, 0.045, 1.0],
            [0.035, 0.060, 0.075, 1.0],
            [0.12, 0.15, 0.16, 1.0],
        ])
        display_material.set_principled_shader_value("Base Color", display_tint)
        display_material.set_principled_shader_value(
            "Roughness", rng.uniform(0.08, 0.32))
        screen_shader = next(n for n in display_material.blender_obj.node_tree.nodes
                             if n.type == "BSDF_PRINCIPLED")
        if screen_shader.inputs.get("Emission Strength") is not None:
            screen_shader.inputs["Emission Strength"].default_value = (
                rng.uniform(0.20, 0.55) if rng.random() < 0.80 else 0.0)
        white_label_material.set_principled_shader_value(
            "Base Color", [rng.uniform(0.65, 0.92), rng.uniform(0.67, 0.94),
                           rng.uniform(0.65, 0.90), 1.0])


        # ------------------ 衣服出现概率与形变随机化 ------------------
        show_garment = rng.random() < args.garment_probability
        
        # 随机生成形变系数：X(宽窄), Y(长短), Z(厚度)
        shape_noise_x = shape_noise_y = shape_noise_z = rng.uniform(0.94, 1.06)
        selected_garment = (rng.choices(range(3), weights=[0.68, 0.19, 0.13])[0]
                            if args.garment_variant == "mixed" else
                            ["main", "second", "third"].index(args.garment_variant))

        for i, garment_object in enumerate(garment_objects):
            garment_object.hide(not show_garment or i != selected_garment)
            if show_garment:
                orig_scale = original_garment_scales[i]
                # 基于缓存的初始尺寸乘以形变系数
                garment_object.set_scale([orig_scale[0] * shape_noise_x, 
                                          orig_scale[1] * shape_noise_y, 
                                          orig_scale[2] * shape_noise_z])
                garment_origin = np.array([0.0, 0.28, 0.004])
                garment_object.set_location(garment_origin +
                    (original_garment_locations[i] - garment_origin) *
                    np.array([shape_noise_x, shape_noise_y, shape_noise_z]))

        composition_band, group_x, target = sample_composition(
            rng, args.lower_middle_probability)
        cam_loc = np.array([rng.uniform(-0.10, 0.10),
                            rng.uniform(-0.44, -0.16),
                            rng.uniform(0.56, 0.96)])

        left_spread = rng.uniform(0.080, 0.235)
        right_spread = rng.uniform(0.080, 0.235)
        wrist_centers = [
            np.array([group_x - left_spread, rng.uniform(0.015, 0.17),
                      rng.uniform(0.050, 0.090)]),
            np.array([group_x + right_spread, rng.uniform(0.015, 0.17),
                      rng.uniform(0.050, 0.090)]),
        ]
        pose_profile = "legacy"
        if args.scene_profile == "target" and frame_has_arms:
            pose_profile, cam_loc, wrist_centers, target = sample_ego_pose(rng, args.pose_profile)
            composition_band = pose_profile
        arm_info = []
        for wrist, side in zip(wrist_centers, (-1.0, 1.0)):
            if pose_profile == "legacy":
                elbow = cam_loc + np.array([side * rng.uniform(0.10, 0.17),
                                            rng.uniform(-0.20, -0.11),
                                            rng.uniform(-0.55, -0.40)])
            else:
                elbow = cam_loc + np.array([side * rng.uniform(0.16, 0.23),
                                            rng.uniform(-0.16, -0.08),
                                            rng.uniform(-0.40, -0.30)])
            axis = elbow - wrist
            axis /= np.linalg.norm(axis)
            mount = wrist + axis * rng.uniform(0.060, 0.100)
            arm_radius = rng.uniform(0.035, 0.043)
            arm_info.append({"wrist": wrist, "elbow": elbow, "axis": axis,
                             "mount": mount, "arm_radius": arm_radius,
                             "band_radius": arm_radius + rng.uniform(0.006, 0.010)})
        placed = [info["mount"].copy() for info in arm_info]
        for obj in objs:
            mat = obj.get_materials()[0]
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.48, 0.68) if refined_camera else np.random.uniform(0.2, 0.9))
        instance_poses = []
        appearance_visibility = []
        for index, (obj, loc) in enumerate(zip(objs, placed)):
            side_probability = max(0.0, 1.0 - args.front_probability - args.back_probability)
            view_mode = rng.choices(
                ["front", "back", "side_plus_y", "side_minus_y"],
                weights=[args.front_probability, args.back_probability,
                         side_probability * 0.5, side_probability * 0.5], k=1)[0]
            rotation_obj = wrist_camera_rotation(
                loc, cam_loc, view_mode=view_mode, rng=rng)
            if frame_has_arms:
                info = arm_info[index]

                # 1. 计算出手臂 12点钟方向 的基准法向量
                base_normal = np.array([0.0, 0.0, 1.0])
                base_normal -= np.dot(base_normal, info["axis"]) * info["axis"]
                base_normal /= np.linalg.norm(base_normal)

                # 2. 计算出 3点钟方向 的侧面法向量
                side_normal = np.cross(info["axis"], base_normal)

                # 3. 引入佩戴角度的随机化 (-2.6 到 2.6 弧度，约等于 -150度 到 150度)
                # 这会让相机在手臂的一大半圆周表面上随机出现
                if rng.random() < args.wrist_side_probability:
                    # Mostly inward-facing mounts, as in the manipulation video.
                    # Keep some outward examples to avoid a fixed layout shortcut.
                    sign = (-1.0 if index == 0 else 1.0)
                    if rng.random() < 0.20:
                        sign *= -1.0
                    roll = sign * rng.uniform(1.15, 1.90)
                else:
                    roll = rng.uniform(-1.0, 1.0)

                # 4. 根据随机角度合成最终的表面法向量
                normal = np.cos(roll) * base_normal + np.sin(roll) * side_normal
                normal /= np.linalg.norm(normal)

                # 5. 更新相机的位置，使其贴在手臂的表面
                loc = loc + normal * (info["band_radius"] + 0.026)
                placed[index] = loc
                info["cam_center"] = loc

                # 6. 生成旋转矩阵（关键：传入 normal 作为相机的 Up 向量，让相机跟着手臂一起侧翻）
                rotation_obj = wrist_camera_rotation(loc, cam_loc, view_mode, rng, world_up=normal)
            else:
                rotation_obj = wrist_camera_rotation(loc, cam_loc, view_mode, rng)
                loc = place_on_surface(loc, rotation_obj)
                placed[index] = loc
            obj.set_location(loc)
            obj.set_rotation_mat(rotation_obj)
            show_screen = args.rear_screen_proxy and (
                bool(args.camera_model_dir) or rng.random() < 0.90)
            show_label = (not args.camera_model_dir
                          and rng.random() < args.label_probability)
            show_tag = (not args.camera_model_dir
                        and rng.random() < args.marker_probability)
            show_led = rng.random() < 0.45
            appearance_visibility.append(
                (show_screen, show_label, show_tag, show_led))
            place_camera_appearance_parts(
                camera_parts[index], loc, rotation_obj, proxy_scale,
                show_screen, show_label, show_tag, show_led,
                face_sign=appearance_face_sign)
            is_active = index in active_indices
            obj.hide(not is_active)
            if not is_active:
                set_parts_hidden(all_camera_parts(camera_parts[index]), True)
            instance_poses.append((loc, rotation_obj, view_mode))

        context_objects = []
        if frame_has_arms:
            for index, (info, (_, rotation_obj, _)) in enumerate(
                    zip(arm_info, instance_poses)):
                context_objects.extend(add_wrist_context(
                    info["wrist"], info["elbow"], info["mount"], info["axis"],
                    info["arm_radius"], info["band_radius"], info["cam_center"],
                    rotation_obj, skin_material, strap_material, cable_material, rng,
                    with_camera=index in active_indices,
                    mount_probability=args.mount_probability,
                    device_materials=(cuff_material, wrist_device_material, cuff_trim_material),
                    device_probability=args.wrist_device_probability))

        occlusion_levels = []
        for instance_index, (loc, _, _) in enumerate(instance_poses):
            if instance_index not in active_indices:
                occlusion_levels.append("absent")
                continue
            if args.object_layer_preview:
                occlusion_levels.append("none")
                continue
            is_occluded = rng.random() < args.occlusion_probability
            if not is_occluded:
                occlusion_levels.append("none")
                continue
            level = rng.choices(["light", "medium", "heavy"],
                                weights=[0.50, 0.35, 0.15], k=1)[0]
            occlusion_levels.append(level)
            toward_camera = cam_loc - loc
            toward_camera /= np.linalg.norm(toward_camera)
            horizontal = np.cross(toward_camera, np.array([0.0, 0.0, 1.0]))
            horizontal /= np.linalg.norm(horizontal)
            vertical = np.cross(horizontal, toward_camera)
            ranges = {
                "light": ((0.004, 0.006), (0.012, 0.020), (0.003, 0.005), (0.018, 0.027)),
                "medium": ((0.006, 0.009), (0.020, 0.032), (0.004, 0.006), (0.010, 0.021)),
                "heavy": ((0.011, 0.016), (0.030, 0.043), (0.006, 0.009), (0.000, 0.010)),
            }
            short_range, long_range, depth_range, offset_range = ranges[level]
            for _ in range(1):
                angle = rng.uniform(-1.05, 1.05)
                long_axis = np.cos(angle) * horizontal + np.sin(angle) * vertical
                normal_axis = toward_camera
                short_axis = np.cross(long_axis, normal_axis)
                finger_rotation = np.column_stack((short_axis, long_axis, normal_axis))
                side = rng.choice([-1.0, 1.0])
                offset = side * rng.uniform(*offset_range)
                finger = bproc.object.create_primitive(
                    "SPHERE", scale=[rng.uniform(*short_range),
                                     rng.uniform(*long_range),
                                     rng.uniform(*depth_range)])
                finger.set_rotation_mat(finger_rotation)
                finger.set_location(loc + toward_camera * rng.uniform(0.020, 0.030)
                                    + horizontal * offset
                                    + vertical * rng.uniform(-0.012, 0.012))
                finger.replace_materials(skin_material)
                context_objects.append(finger)
        # ---- ego 相机位姿：上方俯视 + 随机偏头视线噪声 ----
        look_noise = np.array([
            rng.uniform(-0.01, 0.01),  # 左右随机偏头 (Pan offset)
            rng.uniform(-0.05, 0.08),  # 上下随机抬头/俯视 (Tilt offset)
            0.0
        ])
        if args.scene_profile == "target" and pose_profile in {"work", "fold"}:
            look_noise[1] = rng.uniform(-0.02, 0.02)
        rotation = bproc.camera.rotation_from_forward_vec(
            (target + look_noise) - cam_loc,
            inplane_rot=rng.uniform(-0.12, 0.12)
        )
        cam2world = bproc.math.build_transformation_mat(cam_loc, rotation)
        bproc.camera.add_camera_pose(cam2world, frame=0)
        bpy.context.view_layer.update()
        projected_boxes = [dict(target_id=index + 1, **projected_object_bounds(
            objs[index], cam2world, frame_intrinsics, W, H)) for index in sorted(active_indices)]
        undersized_targets = [box["target_id"] for box in projected_boxes
                              if box.get("valid") and
                              box["width_fraction"] < args.min_target_width_fraction]
        frame_audit = {
            "garment_asset": (["牛仔模型", "牛仔模型_2", "牛仔外套_3"][selected_garment]
                              if garment_objects and show_garment else None),
            "left_clutter_asset": "左侧杂物" if garment_objects else None,
            "left_clutter_projected_bounds": next((projected_object_bounds(
                part, cam2world, frame_intrinsics, W, H)
                for part in environment_objects if part.get_name() == "left_clutter_asset"), None),
            "lighting_profile": args.lighting_profile,
            "lights": [{"location": list(light.get_location()),
                        "energy": float(light.blender_obj.data.energy),
                        "size": float(light.blender_obj.data.size)} for light in area_lights],
            "seed": args.seed, "frame": img_idx, "pose_profile": pose_profile,
            "composition_band": composition_band,
            "camera_location": cam_loc.tolist(), "look_at": (target + look_noise).tolist(),
            "downward_pitch_degrees": float(np.degrees(np.arctan2(
                cam_loc[2] - (target + look_noise)[2],
                np.linalg.norm((target + look_noise - cam_loc)[:2])))),
            "wrists": [point.tolist() for point in wrist_centers],
            "fov_degrees": frame_fov, "projected_boxes": projected_boxes,
            "undersized_targets": undersized_targets,
            "active_indices": sorted(active_indices), "frame_has_arms": frame_has_arms,
        }
        if preview_dir:
            audit_dir = os.path.join(preview_dir, "metadata")
            os.makedirs(audit_dir, exist_ok=True)
            with open(os.path.join(audit_dir, f"{img_idx:06d}.json"), "w") as f:
                json.dump(frame_audit, f, indent=2)

        if args.object_layer_preview:
            for environment_object in environment_objects:
                environment_object.hide(True)
            for context_obj in context_objects:
                context_obj.hide(True)
            for obj in objs:
                obj.hide(True)
            for parts in camera_parts:
                set_parts_hidden(all_camera_parts(parts), True)

            for instance_index, obj in enumerate(objs):
                obj.hide(False)
                loc, rotation_obj, _ = instance_poses[instance_index]
                place_camera_appearance_parts(
                    camera_parts[instance_index], loc, rotation_obj, proxy_scale,
                    *appearance_visibility[instance_index],
                    face_sign=appearance_face_sign)
                layer_data = bproc.renderer.render()
                layer = uint8_color(layer_data["colors"][0])
                if layer.shape[-1] != 4:
                    raise RuntimeError("object-layer render did not produce RGBA output")
                Image.fromarray(layer, mode="RGBA").save(
                    os.path.join(preview_dir, f"{img_idx:06d}_{instance_index}.png"))
                obj.hide(True)
                set_parts_hidden(all_camera_parts(camera_parts[instance_index]), True)

            for environment_object in environment_objects:
                environment_object.hide(False)
            for context_obj in context_objects:
                context_obj.hide(False)
            data = None
        elif args.camera_free_preview:
            for obj in objs:
                obj.hide(True)
            for parts in camera_parts:
                set_parts_hidden(all_camera_parts(parts), True)

            data = bproc.renderer.render()
        else:
            data = bproc.renderer.render()

        if args.object_layer_preview:
            pass
        elif preview_dir:
            preview_color = (process_color(data["colors"][0], rng, args)
                             if args.sensor_effects and args.scene_profile == "target"
                             else uint8_color(data["colors"][0]))
            Image.fromarray(preview_color, mode="RGB").save(
                os.path.join(preview_dir, f"{img_idx:06d}.jpg"),
                quality=rng.randint(70, 94), subsampling=2)
            if args.preview_yolo_labels:
                segmentation = bproc.renderer.render_segmap(
                    map_by=["instance", "cp_yolo_target"],
                    default_values={"cp_yolo_target": 0})
                target_maps = segmentation.get("cp_yolo_target_segmaps")
                target_map = (target_maps[0] if target_maps
                              else np.zeros((H, W), dtype=np.uint8))
                lines, visible_boxes = [], []
                for target_id in range(1, len(objs) + 1):
                    ys, xs = np.where(target_map == target_id)
                    if not len(xs):
                        continue
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    lines.append(
                        f"0 {(x1 + x2) / (2 * W):.6f} {(y1 + y2) / (2 * H):.6f} "
                        f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                    visible_boxes.append({"target_id": target_id,
                                          "xyxy": [x1, y1, x2, y2],
                                          "visible_pixels": int(len(xs))})
                with open(os.path.join(preview_dir, "labels", f"{img_idx:06d}.txt"), "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                with open(os.path.join(preview_dir, "labels", f"{img_idx:06d}.json"), "w") as f:
                    json.dump({"seed": args.seed, "frame": img_idx,
                               "width": W, "height": H,
                               "active_indices": sorted(active_indices),
                               "boxes": visible_boxes,
                               "note": "Visible camera mesh and appearance proxies; excludes wrist/mount."}, f)
        else:
            colors = ([process_color(color, rng, args) for color in data["colors"]]
                      if args.sensor_effects and args.scene_profile == "target" else data["colors"])
            active_target_objects = [objs[index] for index in sorted(active_indices)]
            bproc.writer.write_bop(
                output_dir=bop_parent_path,
                target_objects=active_target_objects,
                dataset=dataset_name,
                depth_scale=0.1,
                depths=data["depth"],
                colors=colors,
                color_file_format="JPEG",
                ignore_dist_thres=10,
                append_to_existing_output=True,
                calc_mask_info_coco=args.bop_masks,
                num_worker=0,
                frames_per_chunk=args.frames_per_scene,
            )
            scene_metadata[str(metadata_start_id + img_idx)] = {
                **frame_audit,
                "view_modes": [pose[2] for pose in instance_poses],
                "occlusion_levels": occlusion_levels,
                "scene_profile": args.scene_profile,
                "composition_band": composition_band,
                "active_indices": sorted(active_indices),
                "camera_count": len(active_indices),
                "frame_has_arms": frame_has_arms,
                "fov_degrees": frame_fov,
            }
            # Keep unfinished custom-split renders recoverable in staging.
            save_metadata_path = (os.path.join(staging_path, "scene_metadata.json")
                                  if staging_path else metadata_path)
            with open(save_metadata_path, "w") as f:
                json.dump(scene_metadata, f, indent=2)

        for context_obj in context_objects:
            bpy.data.objects.remove(context_obj.blender_obj, do_unlink=True)

        print(f"[{img_idx + 1}/{args.num_images}] 完成")

    if staging_path:
        os.makedirs(split_path, exist_ok=True)
        scene_ids = [int(name) for name in os.listdir(split_path)
                     if len(name) == 6 and name.isdigit()]
        next_scene_id = max(scene_ids, default=-1) + 1
        staged_dataset = os.path.join(staging_path, "rendered")
        staged_scenes = sorted(glob.glob(os.path.join(staged_dataset, "train_pbr", "[0-9]" * 6)))
        for offset, scene in enumerate(staged_scenes):
            shutil.move(scene, os.path.join(split_path, f"{next_scene_id + offset:06d}"))
        shutil.copy2(os.path.join(staged_dataset, "camera.json"), os.path.join(split_path, "camera.json"))
        with open(metadata_path, "w") as f:
            json.dump(scene_metadata, f, indent=2)
        # Only remove the task-owned temporary directory after publishing scenes.
        shutil.rmtree(staging_path)
    if not preview_dir and replacement_bounds_mm is not None:
        os.makedirs(split_path, exist_ok=True)
        with open(os.path.join(split_path, "render_config.json"), "w") as f:
            json.dump({
                "camera_model_dir": args.camera_model_dir,
                "corner_bounds_mm": replacement_bounds_mm,
                "screen_texture": "lcd.png" if refined_camera else "正面贴图.png",
                "asset_version": "refined_action_camera_v1" if refined_camera else "external_obj",
            }, f, indent=2, ensure_ascii=False)
    print("全部渲染完成")


if __name__ == "__main__":
    main()
