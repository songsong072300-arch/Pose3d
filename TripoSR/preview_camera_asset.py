import blenderproc as bproc  # Must be imported before bpy.
import argparse
import os

import bpy
import numpy as np
from PIL import Image


def attach_pbr(obj, asset_dir):
    materials = obj.get_materials()
    if not materials:
        material = bproc.material.create("camera_asset")
        obj.replace_materials(material)
        materials = [material]
    material = materials[0].blender_obj
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
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
        normal_map.inputs["Strength"].default_value = 0.45
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()

    asset_dir = os.path.abspath(args.asset_dir)
    os.makedirs(args.out, exist_ok=True)
    bproc.init()
    bproc.renderer.set_render_devices(use_only_cpu=True)
    bproc.renderer.set_max_amount_of_samples(8)
    bproc.camera.set_resolution(args.size, args.size)

    loaded = bproc.loader.load_obj(
        os.path.join(asset_dir, "base.obj"), forward_axis="Y", up_axis="Z",
    )
    if len(loaded) != 1:
        raise RuntimeError(f"expected one object, found {len(loaded)}")
    obj = loaded[0]
    attach_pbr(obj, asset_dir)
    obj.set_shading_mode("auto")

    vertices = np.asarray([list(vertex.co) for vertex in obj.blender_obj.data.vertices])
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    centre = (lo + hi) / 2.0
    for vertex, point in zip(obj.blender_obj.data.vertices, vertices - centre):
        vertex.co = point
    scale = 0.12 / float(np.max(hi - lo))
    obj.set_scale([scale] * 3)
    bpy.context.view_layer.update()

    light = bproc.types.Light(light_type="AREA")
    light.set_location([0.15, -0.18, 0.20])
    light.blender_obj.data.size = 0.25
    light.set_energy(80)
    fill = bproc.types.Light(light_type="AREA")
    fill.set_location([-0.18, 0.12, 0.10])
    fill.blender_obj.data.size = 0.20
    fill.set_energy(45)
    bproc.renderer.set_world_background([0.18, 0.18, 0.18], strength=0.8)

    views = {
        "minus_x": [-0.22, 0.0, 0.0],
        "plus_x": [0.22, 0.0, 0.0],
        "minus_y": [0.0, -0.22, 0.0],
        "plus_y": [0.0, 0.22, 0.0],
        "plus_z": [0.0, 0.0, 0.22],
    }
    for name, location in views.items():
        rotation = bproc.camera.rotation_from_forward_vec(-np.asarray(location))
        pose = bproc.math.build_transformation_mat(location, rotation)
        bproc.camera.add_camera_pose(pose, frame=0)
        data = bproc.renderer.render()
        color = np.asarray(data["colors"][0])
        if color.dtype != np.uint8 and float(color.max()) <= 1.0:
            color = np.clip(color * 255.0, 0, 255).astype(np.uint8)
        else:
            color = np.clip(color, 0, 255).astype(np.uint8)
        Image.fromarray(color[..., :3]).save(os.path.join(args.out, f"{name}.png"))
        bproc.utility.reset_keyframes()

    print(f"bounds={lo.tolist()}..{hi.tolist()} dims={(hi - lo).tolist()}")


if __name__ == "__main__":
    main()
