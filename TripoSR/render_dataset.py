import blenderproc as bproc  # 必须放在最顶端第一行
import os
import json
import argparse
import random
import glob

import numpy as np
import bpy
from PIL import Image, ImageEnhance, ImageFilter


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
    """强制构图将双手压在画面底部边缘"""
    # 让双手在桌面上左右更宽的范围内随机出现
    group_x = rng.uniform(-0.25, 0.25)
    aim_x = group_x + rng.uniform(-0.15, 0.15)

    # 强制固定在 lower_middle 状态
    # Y轴数值增大到 0.45-0.65 (让相机抬头看远处，从而把手压在底边)
    return "lower_middle", group_x, np.array([
        aim_x,
        rng.uniform(0.05, 0.20),
        rng.uniform(0.015, 0.045),
    ])


def set_local_pose(part, object_location, object_rotation, local_location):
    part.set_location(object_location + object_rotation @ np.asarray(local_location))
    part.set_rotation_mat(object_rotation)


def attach_model_pbr_maps(obj, model_dir):
    """Attach optional roughness, metallic, and normal maps beside the BOP PLY."""
    material = obj.get_materials()[0].blender_obj
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")

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
                      cable_material, rng):
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


    cable_start = cam_center + camera_rotation @ np.array([0.0, 0.030, -0.004])
    cable_mid = mount + np.array([rng.uniform(-0.012, 0.012), 0.015,
                                  -(arm_radius + 0.008)])
    cable_end = elbow + np.array([rng.uniform(-0.01, 0.01), 0.03, 0.0])
    for a, b in ((cable_start, cable_mid), (cable_mid, cable_end)):
        cable = create_segment(a, b, 0.0022, cable_material)
        if cable is not None:
            context.append(cable)
    return context


def create_target_scene(grid_material, denim_material, metal_material, pants_material):
    """Create the stable white grid, scalable denim jacket, and pants context."""
    scene_objects = []
    # 桌面网格
    for value in np.arange(-0.45, 0.46, 0.035):
        # 纵向线条：按新桌子的 Y 轴比例缩短长度 (scale 的第二个值改为 0.35)
        vertical = bproc.object.create_primitive(
            "CUBE", scale=[0.0008, 0.35, 0.00035], location=[value, 0.1, 0.001])
            
        # 横向线条：按新桌子的 X 轴比例缩短长度 (scale 的第一个值改为 0.45)
        horizontal = bproc.object.create_primitive(
            "CUBE", scale=[0.45, 0.0008, 0.00035], location=[0.0, value + 0.1, 0.001])
            
        vertical.replace_materials(grid_material)
        horizontal.replace_materials(grid_material)
        scene_objects.extend([vertical, horizontal])

    garment_objects = []

    # === 衣服全局缩放系数 (1.0为默认大小，这里设置为 1.3 放大) ===
    g_scale = 1.3
    base_y = 0.34  # 衣服在桌面上的中心 Y 坐标基准点

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

    # 3. 在相机下方生成黑色裤腿，让低位相机遇到真实的暗色背景
    for side in [-0.08, 0.08]: # 稍微拉开一点左右腿间距
        leg = bproc.object.create_primitive(
            "CUBE", # 【关键修改】改为立方体
            # X(宽度) 给够，Y(厚度) 稍微压扁模拟被裤子包裹的平整感，Z(长度) 拉长
            scale=[0.14, 0.08, 0.35],
            location=[side, -0.22, -0.12]
        )
        # 将长方体放倒并让膝盖朝上倾斜
        leg.set_rotation_euler([np.pi/2 - 0.3, 0, 0])
        leg.replace_materials(pants_material)
        scene_objects.append(leg)
   # ================== 修正的环境干扰构建 ==================

    # === 1. 模拟操作者位于相机下方的大腿/膝盖 ===
    pants_material = bproc.material.create("black_pants")
    pants_material.set_principled_shader_value("Base Color", [0.025, 0.027, 0.030, 1.0])
    pants_material.set_principled_shader_value("Roughness", 0.84)
    
    # 用两个向斜下方倾斜的圆柱体模拟双腿，放在相机后下方形成暗色干扰
    for side in [-0.15, 0.15]: # 左右腿间距
        leg = bproc.object.create_primitive(
            "CYLINDER", 
            scale=[0.12, 0.12, 0.3],
            location=[side, -0.22, -0.12]
        )
        # 将圆柱体放倒并让膝盖朝上倾斜
        leg.set_rotation_euler([np.pi/2 - 0.3, 0, 0])
        leg.replace_materials(pants_material)
        scene_objects.append(leg)

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

    # === 3. 模拟画面左侧的深色设备堆叠与右侧零星杂物 ===
    dark_equipment_mat = bproc.material.create("dark_equipment")
    dark_equipment_mat.set_principled_shader_value("Base Color", [0.05, 0.05, 0.06, 1.0])
    dark_equipment_mat.set_principled_shader_value("Roughness", 0.6)
    
    box_material = bproc.material.create("cardboard_box")
    box_material.set_principled_shader_value("Base Color", [0.4, 0.3, 0.15, 1.0])
    box_material.set_principled_shader_value("Roughness", 0.85)

    import random
    # 左侧：集中堆放 3-4 个深黑色长方体，紧贴桌子左侧边缘 (X= -0.7 到 -0.9)
    for i in range(random.randint(3, 4)):
        left_clutter = bproc.object.create_primitive(
            "CUBE", 
            scale=[random.uniform(0.1, 0.2), random.uniform(0.15, 0.3), random.uniform(0.1, 0.4)], 
            location=[random.uniform(-0.9, -0.7), random.uniform(-0.2, 0.4), -0.5 + i * 0.15]
        )
        left_clutter.replace_materials(dark_equipment_mat)
        scene_objects.append(left_clutter)
        
    # 右侧：象征性放 1 个浅色纸箱，制造非对称感
    right_box = bproc.object.create_primitive(
        "CUBE", 
        scale=[0.15, 0.2, 0.25], 
        location=[random.uniform(0.7, 0.9), random.uniform(0.2, 0.5), -0.4]
    )
    right_box.replace_materials(box_material)
    scene_objects.append(right_box)

    return scene_objects, garment_objects

def uint8_color(color):
    pixels = np.asarray(color)
    if np.issubdtype(pixels.dtype, np.floating) and pixels.max(initial=0.0) <= 1.0:
        pixels = pixels * 255.0
    return np.clip(pixels, 0, 255).astype(np.uint8)


def process_color(color, rng):
    """Approximate the muted exposure and softness of the target video."""
    image = Image.fromarray(uint8_color(color), mode="RGB")
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.52, 0.68))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(1.05, 1.22))
    image = ImageEnhance.Color(image).enhance(rng.uniform(1.00, 1.18))
    if rng.random() < 0.55:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.65)))
    return np.asarray(image)


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
    parser.add_argument("--fov", type=float, default=82.0, help="水平视场角(度)")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--arms", action=argparse.BooleanOptionalAction, default=True,
                        help="生成与左右腕绑定的前臂、手掌、手指、腕带和线缆（默认开启）")
    parser.add_argument("--front-probability", type=float, default=0.10,
                        help="正面视角概率")
    parser.add_argument("--back-probability", type=float, default=0.10,
                        help="背面视角概率；侧面 = 1 - 正面 - 背面")
    parser.add_argument("--lower-middle-probability", type=float, default=0.70,
                        help="保留用于命令兼容；当前构图固定在下方中间区域")
    parser.add_argument("--occlusion-probability", type=float, default=0.20,
                        help="每个实例出现遮挡的总概率；遮挡内部再随机轻/中/重等级")
    parser.add_argument("--samples", type=int, default=8, help="Cycles 每像素采样数")
    parser.add_argument("--scene-profile", choices=("target", "generic"), default="target",
                        help="target 匹配 frames 中的室内腕戴场景；generic 使用 CC0 地面")
    parser.add_argument("--garment-probability", type=float, default=0.8,
                        help="target 场景中出现蓝色衣物上下文的概率（默认 1.0，保证常驻）")
    parser.add_argument("--sensor-effects", action=argparse.BooleanOptionalAction, default=True,
                        help="对 target RGB 应用接近测试视频的曝光、饱和度、模糊和 JPEG 退化")
    parser.add_argument("--rear-screen-proxy", action=argparse.BooleanOptionalAction, default=False,
                        help="为缺少背屏的旧模型添加背屏代理；新模型默认不需要")
    parser.add_argument("--preview-dir", type=str, default=None,
                        help="仅保存 RGB 样图到此目录，不追加或改写 BOP 数据集")
    parser.add_argument("--bop-masks", action="store_true",
                        help="额外生成 BOP mask/info/COCO（较慢，macOS 使用单进程）")
    args = parser.parse_args()

    current_dir = os.path.abspath(args.dataset_dir or os.getcwd())
    dataset_name = os.path.basename(current_dir)
    bop_parent_path = os.path.dirname(current_dir)
    bop_dataset_path = os.path.join(bop_parent_path, dataset_name)
    preview_dir = os.path.abspath(args.preview_dir) if args.preview_dir else None

    W, H = args.width, args.height
    fx = fy = W / (2.0 * np.tan(np.deg2rad(args.fov) / 2.0))
    cam_json = os.path.join(current_dir, "camera.json")
    if not preview_dir:
        with open(cam_json, "w") as f:
            json.dump({"cx": W / 2.0, "cy": H / 2.0, "depth_scale": 0.1,
                      "fx": fx, "fy": fy, "height": H, "width": W}, f, indent=2)

    bproc.init()

    cc_textures = []
    if args.scene_profile == "generic":
        if os.path.basename(args.cc0textures) == "cc0textures-512":
            cc_textures = bproc.loader.load_512_ccmaterials(args.cc0textures, use_all_materials=True)
        else:
            cc_textures = bproc.loader.load_ccmaterials(args.cc0textures, use_all_materials=True)

    if preview_dir:
        intrinsic_matrix = np.array([[fx, 0.0, W / 2.0],
                                     [0.0, fy, H / 2.0],
                                     [0.0, 0.0, 1.0]])
        bproc.camera.set_intrinsics_from_K_matrix(intrinsic_matrix, W, H)
    else:
        bproc.loader.load_bop_intrinsics(bop_dataset_path=bop_dataset_path)

    table = bproc.object.create_primitive("PLANE", scale=[0.45, 0.35, 1.0], location=[0, 0.1, 0])
    table.set_name("table")
    table_material = bproc.material.create("target_table")
    table_material.set_principled_shader_value("Base Color", [0.58, 0.61, 0.63, 1.0])
    table_material.set_principled_shader_value("Roughness", 0.92)
    table.replace_materials(table_material)

    area_lights = []
    for lx, ly in ((-0.55, -0.10), (0.55, -0.10), (-0.35, 0.50), (0.35, 0.50)):
        light = bproc.types.Light(light_type="AREA")
        light.set_location([lx, ly, 1.15])
        light.blender_obj.data.size = 0.75
        light.set_energy(18)
        area_lights.append(light)
    bproc.renderer.set_world_background([0.14, 0.15, 0.16], strength=0.8)

    skin_material = bproc.material.create("forearm_skin")
    skin_material.set_principled_shader_value("Base Color", [0.34, 0.16, 0.08, 1.0])
    skin_material.set_principled_shader_value("Roughness", 0.65)
    strap_material = bproc.material.create("wrist_strap")
    strap_material.set_principled_shader_value("Base Color", [0.025, 0.03, 0.035, 1.0])
    strap_material.set_principled_shader_value("Roughness", 0.55)
    cable_material = bproc.material.create("camera_cable")
    cable_material.set_principled_shader_value("Base Color", [0.008, 0.010, 0.012, 1.0])
    cable_material.set_principled_shader_value("Roughness", 0.45)
    bezel_material = bproc.material.create("rear_screen_bezel")
    bezel_material.set_principled_shader_value("Base Color", [0.008, 0.010, 0.012, 1.0])
    bezel_material.set_principled_shader_value("Roughness", 0.35)
    display_material = bproc.material.create("rear_screen_display")
    display_material.set_principled_shader_value("Base Color", [0.025, 0.045, 0.052, 1.0])
    display_material.set_principled_shader_value("Roughness", 0.18)

    garment_objects = []
    if args.scene_profile == "target":
        grid_material = bproc.material.create("work_surface_grid")
        grid_material.set_principled_shader_value("Base Color", [0.005, 0.005, 0.005, 1.0])
        grid_material.set_principled_shader_value("Roughness", 1.00)
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

        _, garment_objects = create_target_scene(grid_material, denim_material, metal_material, pants_material)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
    metadata_path = os.path.join(current_dir, "train_pbr", "scene_metadata.json")
    scene_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            scene_metadata = json.load(f)
    existing_rgb = glob.glob(os.path.join(current_dir, "train_pbr", "*", "rgb", "*.jpg"))
    metadata_start_id = len(existing_rgb)

    objs = bproc.loader.load_bop_objs(bop_dataset_path=bop_dataset_path, mm2m=True, obj_ids=[1])
    assert len(objs) == 1
    objs += bproc.loader.load_bop_objs(bop_dataset_path=bop_dataset_path, mm2m=True, obj_ids=[1])
    for obj in objs:
        obj.set_shading_mode("auto")
        attach_model_pbr_maps(obj, os.path.join(bop_dataset_path, "models"))

    rear_parts = []
    for _ in objs:
        bezel = display = None
        if args.rear_screen_proxy:
            bezel = bproc.object.create_primitive("CUBE", scale=[0.0010, 0.0310, 0.0188])
            bezel.replace_materials(bezel_material)
            display = bproc.object.create_primitive("CUBE", scale=[0.0006, 0.0287, 0.0165])
            display.replace_materials(display_material)
        rear_parts.append((bezel, display))

    # ================== 收集需要进行物理随机化的环境材质 ==================
    dr_materials = [m for m in bproc.material.collect_all() if m.get_name() in 
                    ["target_table", "denim_garment", "white_wall", "wood_floor", 
                     "cardboard_box", "pants_fabric", "black_shirt", "dark_equipment"]]
    # ================== 收集并缓存环境材质的真实属性 ==================
    dr_materials_names = ["target_table", "denim_garment", "white_wall", "wood_floor", 
                          "cardboard_box", "pants_fabric", "black_shirt", "dark_equipment"]
    dr_materials = [m for m in bproc.material.collect_all() if m.get_name() in dr_materials_names]
    
    # 用一个字典把真实场景的颜色和粗糙度记录下来，防止被随机化覆盖后找不回来
    original_mat_states = {}
    for mat in dr_materials:
        original_mat_states[mat.get_name()] = {
            "Base Color": mat.get_principled_shader_value("Base Color"),
            "Roughness": mat.get_principled_shader_value("Roughness"),
            "Metallic": mat.get_principled_shader_value("Metallic")
        }
    # ================== 缓存衣服代理几何体的初始尺寸 ==================
    original_garment_scales = [obj.get_scale() for obj in garment_objects]

    for img_idx in range(args.num_images):
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
        # 手臂无论在哪个分支，都只进行微小的肤色波动
        skin_material.set_principled_shader_value(
            "Base Color", [float(np.random.uniform(0.31, 0.40)),
                           float(np.random.uniform(0.15, 0.21)),
                           float(np.random.uniform(0.085, 0.125)), 1.0])
        skin_material.set_principled_shader_value("Roughness", np.random.uniform(0.52, 0.78))


        # ------------------ 衣服出现概率与形变随机化 ------------------
        show_garment = rng.random() < args.garment_probability
        
        # 随机生成形变系数：X(宽窄), Y(长短), Z(厚度)
        shape_noise_x = rng.uniform(0.7, 1.5)  
        shape_noise_y = rng.uniform(0.8, 1.3)  
        shape_noise_z = rng.uniform(0.2, 2.0)  

        for i, garment_object in enumerate(garment_objects):
            garment_object.hide(not show_garment) # 根据概率隐藏或显示
            if show_garment:
                orig_scale = original_garment_scales[i]
                # 基于缓存的初始尺寸乘以形变系数
                garment_object.set_scale([orig_scale[0] * shape_noise_x, 
                                          orig_scale[1] * shape_noise_y, 
                                          orig_scale[2] * shape_noise_z])

        composition_band, group_x, target = sample_composition(
            rng, args.lower_middle_probability)
        cam_loc = np.array([rng.uniform(-0.035, 0.035),
                            rng.uniform(-0.43, -0.20),
                            rng.uniform(0.40, 0.72)])

        left_spread = rng.uniform(0.085, 0.160)
        right_spread = rng.uniform(0.085, 0.160)
        wrist_centers = [
            np.array([group_x - left_spread, rng.uniform(0.015, 0.17),
                      rng.uniform(0.050, 0.090)]),
            np.array([group_x + right_spread, rng.uniform(0.015, 0.17),
                      rng.uniform(0.050, 0.090)]),
        ]
        arm_info = []
        for wrist, side in zip(wrist_centers, (-1.0, 1.0)):
            elbow = cam_loc + np.array([side * rng.uniform(0.10, 0.17),
                                        rng.uniform(-0.20, -0.11),
                                        rng.uniform(-0.34, -0.25)])
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
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.2, 0.9))
        instance_poses = []
        for index, (obj, loc) in enumerate(zip(objs, placed)):
            side_probability = max(0.0, 1.0 - args.front_probability - args.back_probability)
            view_mode = rng.choices(
                ["front", "back", "side_plus_y", "side_minus_y"],
                weights=[args.front_probability, args.back_probability,
                         side_probability * 0.5, side_probability * 0.5], k=1)[0]
            rotation_obj = wrist_camera_rotation(
                loc, cam_loc, view_mode=view_mode, rng=rng)
            if args.arms:
                info = arm_info[index]

                # 1. 计算出手臂 12点钟方向 的基准法向量
                base_normal = np.array([0.0, 0.0, 1.0])
                base_normal -= np.dot(base_normal, info["axis"]) * info["axis"]
                base_normal /= np.linalg.norm(base_normal)

                # 2. 计算出 3点钟方向 的侧面法向量
                side_normal = np.cross(info["axis"], base_normal)

                # 3. 引入佩戴角度的随机化 (-2.6 到 2.6 弧度，约等于 -150度 到 150度)
                # 这会让相机在手臂的一大半圆周表面上随机出现
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
            bezel, display = rear_parts[index]
            if bezel is not None:
                set_local_pose(bezel, loc, rotation_obj, [0.0153, 0.0, 0.0])
                set_local_pose(display, loc, rotation_obj, [0.0160, 0.0, 0.0])
            instance_poses.append((loc, rotation_obj, view_mode))

        context_objects = []
        if args.arms:
            for info, (_, rotation_obj, _) in zip(arm_info, instance_poses):
                context_objects.extend(add_wrist_context(
                    info["wrist"], info["elbow"], info["mount"], info["axis"],
                    info["arm_radius"], info["band_radius"], info["cam_center"],
                    rotation_obj, skin_material, strap_material, cable_material, rng))

        occlusion_levels = ["none"] * len(instance_poses)
        """
        occlusion_levels = []
        for loc, _, _ in instance_poses:
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
            """
        # ---- ego 相机位姿：上方俯视 + 随机偏头视线噪声 ----
        look_noise = np.array([
            rng.uniform(-0.20, 0.20),  # 左右随机偏头 (Pan offset)
            rng.uniform(-0.40, 0.60),  # 上下随机抬头/俯视 (Tilt offset)
            0.0
        ])
        rotation = bproc.camera.rotation_from_forward_vec(
            (target + look_noise) - cam_loc,
            inplane_rot=rng.uniform(-0.12, 0.12)
        )
        cam2world = bproc.math.build_transformation_mat(cam_loc, rotation)
        bproc.camera.add_camera_pose(cam2world, frame=0)

        data = bproc.renderer.render()
        if preview_dir:
            preview_color = (process_color(data["colors"][0], rng)
                             if args.sensor_effects and args.scene_profile == "target"
                             else uint8_color(data["colors"][0]))
            Image.fromarray(preview_color, mode="RGB").save(
                os.path.join(preview_dir, f"{img_idx:06d}.jpg"), quality=88, subsampling=2)
        else:
            colors = ([process_color(color, rng) for color in data["colors"]]
                      if args.sensor_effects and args.scene_profile == "target" else data["colors"])
            bproc.writer.write_bop(
                output_dir=bop_parent_path,
                target_objects=objs,
                dataset=dataset_name,
                depth_scale=0.1,
                depths=data["depth"],
                colors=colors,
                color_file_format="JPEG",
                ignore_dist_thres=10,
                append_to_existing_output=True,
                calc_mask_info_coco=args.bop_masks,
                num_worker=0,
            )
            scene_metadata[str(metadata_start_id + img_idx)] = {
                "view_modes": [pose[2] for pose in instance_poses],
                "occlusion_levels": occlusion_levels,
                "scene_profile": args.scene_profile,
                "composition_band": composition_band,
            }
            with open(metadata_path, "w") as f:
                json.dump(scene_metadata, f, indent=2)

        for context_obj in context_objects:
            bpy.data.objects.remove(context_obj.blender_obj, do_unlink=True)

        print(f"[{img_idx + 1}/{args.num_images}] 完成")

    print("全部渲染完成")


if __name__ == "__main__":
    main()
