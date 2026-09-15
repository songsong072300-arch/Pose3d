"""阶段二：为手戴运动相机的位姿估计渲染合成训练数据（BOP 格式）。

基于 HCCEPose 的 s2_p1_gen_pbr_data.py 适配：
- 场景：第一人称（ego）视角
  * 渲染相机位于头部/胸部高度，以 38~48 度俯仰角看向身前桌面
  * 桌面为简单平面 + 高粗糙度纯色漫反射材质（无反光），颜色每帧随机
  * 场景上方 4 个弱强度、颜色微异的区域光，模拟均匀室内顶灯
  * 相机模型绑在小臂距手部几厘米处的护腕（宽短圆柱环）上，底面贴合护腕
  * 左右前臂更粗，肘部位于渲染相机后下方（画面外），手臂从画面底部延伸进来
- 输出：BOP 格式（RGB + 深度 + scene_gt.json 位姿标注）

用法（在 dataset/demo-bin-picking 目录下运行）：
  conda run -n render python ../../render_dataset.py 20 ../../../HCCEPose_src/cc0textures-512
  （参数：渲染图片数 材质库路径）
"""

import os
import json
import argparse
import random
import glob

import numpy as np
import blenderproc as bproc
import bpy


def axis_rotation(axis, angle):
    """Create a 3x3 right-handed rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrist_camera_rotation(obj_location, camera_location, view_mode, rng):
    """Orient the model using the rear/side-heavy distribution in the real video."""
    to_camera = camera_location - obj_location
    to_camera /= np.linalg.norm(to_camera)

    world_up = np.array([0.0, 0.0, 1.0])
    if view_mode in {"front", "back"}:
        # Lens/front normal is local -X; rear-screen normal is local +X.
        local_x_world = -to_camera if view_mode == "front" else to_camera
        local_z_world = world_up - np.dot(world_up, local_x_world) * local_x_world
        local_z_world /= np.linalg.norm(local_z_world)
        local_y_world = np.cross(local_z_world, local_x_world)
    else:
        # Side normals are local +/-Y.
        local_y_world = to_camera if view_mode == "side_plus_y" else -to_camera
        local_z_world = world_up - np.dot(world_up, local_y_world) * local_y_world
        local_z_world /= np.linalg.norm(local_z_world)
        local_x_world = np.cross(local_y_world, local_z_world)
    base = np.column_stack((local_x_world, local_y_world, local_z_world))

    perturbation = (
        axis_rotation("x", rng.uniform(-0.22, 0.22))
        @ axis_rotation("y", rng.uniform(-0.28, 0.28))
        @ axis_rotation("z", rng.uniform(-0.42, 0.42))
    )
    return base @ perturbation


def set_local_pose(part, object_location, object_rotation, local_location):
    part.set_location(object_location + object_rotation @ np.asarray(local_location))
    part.set_rotation_mat(object_rotation)


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
    return segment


def add_wrist_context(wrist, elbow, mount, axis, arm_radius, band_radius,
                      cam_center, camera_rotation, skin_material, strap_material,
                      cable_material, rng):
    """Build one forearm proxy anchored to a wrist-mounted camera.

    Geometry is precomputed in main(): elbow sits behind and below the
    rendering camera so the arm's cut end is never visible -- the arm
    appears to extend in from outside the bottom of the frame. mount is the
    strap point a few centimetres back from the hand; the camera body rests
    on the wristband there (see main() for the exact contact offset).
    """
    wrist = np.asarray(wrist, dtype=float)
    context = []
    # Thicker forearm capsule running from the wrist toward the off-screen
    # elbow, sweeping across the bottom of the first-person view.
    arm = create_segment(wrist, elbow, arm_radius, skin_material)
    if arm is not None:
        context.append(arm)
    # Round cap at the elbow end (normally outside the frame, safety).
    elbow_ball = bproc.object.create_primitive(
        "SPHERE", scale=[arm_radius * 1.02] * 3, location=elbow)
    elbow_ball.replace_materials(skin_material)
    context.append(elbow_ball)

    # Thick velcro-style wristband: a wide short cylinder wrapped around the
    # forearm at the mount point. Solid and slightly larger than the arm, so
    # it reads as a band from any direction.
    band_width = rng.uniform(0.028, 0.038)
    band = create_segment(mount - axis * (band_width * 0.5),
                          mount + axis * (band_width * 0.5),
                          band_radius, strap_material)
    if band is not None:
        context.append(band)

    # Hand proxy at the far end of the forearm (toward the working area).
    palm = bproc.object.create_primitive("SPHERE", scale=[0.040, 0.035, 0.017],
                                         location=wrist + np.array([0.0, 0.012, -0.020]))
    palm.set_rotation_mat(camera_rotation)
    palm.replace_materials(skin_material)
    context.append(palm)

    # Two to three fingers reaching from the palm toward the camera; they may
    # cross the body and act as a natural occluder.
    finger_count = rng.randint(2, 3)
    for finger_index in range(finger_count):
        lateral = (finger_index - (finger_count - 1) / 2.0) * 0.014
        start = wrist + np.array([lateral, 0.004, -0.002 + rng.uniform(-0.004, 0.004)])
        toward_camera = cam_center - start
        toward_camera /= np.linalg.norm(toward_camera)
        end = start + toward_camera * rng.uniform(0.035, 0.060)
        finger = create_segment(start, end, rng.uniform(0.0055, 0.008), skin_material)
        if finger is not None:
            context.append(finger)
            for point in (start, end):
                tip = bproc.object.create_primitive("SPHERE", scale=[0.0065, 0.0065, 0.0065],
                                                    location=point)
                tip.replace_materials(skin_material)
                context.append(tip)

    # Thin cable hanging from the camera's side, under the arm toward the elbow.
    cable_start = cam_center + camera_rotation @ np.array([0.0, 0.030, -0.004])
    cable_mid = mount + np.array([rng.uniform(-0.012, 0.012), rng.uniform(0.000, 0.020),
                                  -(arm_radius + rng.uniform(0.002, 0.008))])
    cable_end = elbow + np.array([rng.uniform(-0.01, 0.01), 0.03, 0.0])
    for a, b in ((cable_start, cable_mid), (cable_mid, cable_end)):
        cable = create_segment(a, b, 0.0022, cable_material)
        if cable is not None:
            context.append(cable)
    return context


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
                        help="（兼容保留）旧版 CC0 材质库路径；背景已改为纯色漫反射，不再使用")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=82.0, help="水平视场角(度)")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--arms", action=argparse.BooleanOptionalAction, default=True,
                        help="生成与左右腕绑定的前臂、手掌、手指、腕带和线缆（默认开启）")
    parser.add_argument("--front-probability", type=float, default=0.20,
                        help="正面视角概率")
    parser.add_argument("--back-probability", type=float, default=0.20,
                        help="背面视角概率；侧面 = 1 - 正面 - 背面，均分给左右两个侧面。"
                             "默认 0.20/0.20/0.60 即 背:正:侧 = 1:1:3")
    parser.add_argument("--occlusion-probability", type=float, default=0.55,
                        help="每个实例出现遮挡的总概率；遮挡内部再随机轻/中/重等级")
    parser.add_argument("--samples", type=int, default=8, help="Cycles 每像素采样数")
    parser.add_argument("--bop-masks", action="store_true",
                        help="额外生成 BOP mask/info/COCO（较慢，macOS 使用单进程）")
    args = parser.parse_args()
    if args.front_probability + args.back_probability > 1.0:
        parser.error("--front-probability 与 --back-probability 之和不能超过 1.0")

    current_dir = os.path.abspath(os.getcwd())
    dataset_name = os.path.basename(current_dir)
    bop_parent_path = os.path.dirname(current_dir)
    bop_dataset_path = os.path.join(bop_parent_path, dataset_name)
    if args.cc0textures:
        args.cc0textures = os.path.abspath(args.cc0textures)

    # 相机内参（与真实视频 3248x2464 的 4:3 比例一致）
    W, H = args.width, args.height
    fx = fy = W / (2.0 * np.tan(np.deg2rad(args.fov) / 2.0))
    cam_json = os.path.join(current_dir, "camera.json")
    with open(cam_json, "w") as f:
        json.dump({"cx": W / 2.0, "cy": H / 2.0, "depth_scale": 0.1,
                  "fx": fx, "fy": fy, "height": H, "width": W}, f, indent=2)

    bproc.init()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    bproc.loader.load_bop_intrinsics(bop_dataset_path=bop_dataset_path)

    # 大面积工作台：简单平面 + 高粗糙度纯色漫反射（无反光），颜色每帧随机化。
    table = bproc.object.create_primitive("PLANE", scale=[8.0, 8.0, 1.0], location=[0, 0, 0])
    table.set_name("table")
    table_material = bproc.material.create("table_surface")
    table_material.set_principled_shader_value("Metallic", 0.0)
    table_material.set_principled_shader_value("Roughness", 0.92)
    table.replace_materials(table_material)

    # 柔和室内顶灯：4 个弱强度、颜色微异的区域光，模拟均匀室内照明。
    area_lights, area_base_energies = [], []
    for lx, ly in ((-0.55, -0.10), (0.55, -0.10), (-0.35, 0.45), (0.35, 0.45)):
        light = bproc.types.Light(light_type="AREA")
        light.set_location(np.array([lx + rng.uniform(-0.10, 0.10),
                                     ly + rng.uniform(-0.10, 0.10),
                                     rng.uniform(1.00, 1.25)]))
        light.blender_obj.data.size = rng.uniform(0.45, 0.90)
        base_energy = rng.uniform(10.0, 18.0)
        light.set_energy(base_energy)
        light.set_color(np.array([rng.uniform(0.95, 1.05), 1.0,
                                  rng.uniform(0.93, 1.05)]))
        area_lights.append(light)
        area_base_energies.append(base_energy)

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

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)

    metadata_path = os.path.join(current_dir, "train_pbr", "scene_metadata.json")
    scene_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            scene_metadata = json.load(f)
    existing_rgb = glob.glob(os.path.join(current_dir, "train_pbr", "*", "rgb", "*.jpg"))
    metadata_start_id = len(existing_rgb)

    # Load the two target instances once. Reusing them avoids stale entries in
    # BlenderProc's BOP object cache after the first frame.
    objs = bproc.loader.load_bop_objs(bop_dataset_path=bop_dataset_path, mm2m=True, obj_ids=[1])
    assert len(objs) == 1
    objs += bproc.loader.load_bop_objs(bop_dataset_path=bop_dataset_path, mm2m=True, obj_ids=[1])
    for obj in objs:
        obj.set_shading_mode("auto")

    # The source image does not show the rear screen. Add a conservative proxy
    # at local +X so rear/oblique views resemble the physical camera.
    rear_parts = []
    for _ in objs:
        bezel = bproc.object.create_primitive("CUBE", scale=[0.0010, 0.0310, 0.0188])
        bezel.replace_materials(bezel_material)
        display = bproc.object.create_primitive("CUBE", scale=[0.0006, 0.0287, 0.0165])
        display.replace_materials(display_material)
        rear_parts.append((bezel, display))

    for img_idx in range(args.num_images):
        # ---- 背景 + 光照随机化 ----
        # 桌面：纯色高粗糙度漫反射（无反光），每帧换柔和底色。
        base_color = rng.uniform(0.22, 0.62)
        table_material.set_principled_shader_value(
            "Base Color", [base_color * rng.uniform(0.90, 1.10),
                           base_color * rng.uniform(0.90, 1.10),
                           base_color * rng.uniform(0.90, 1.10), 1.0])
        table_material.set_principled_shader_value("Roughness", rng.uniform(0.85, 0.98))
        # Vary skin tone and roughness to avoid overfitting to one synthetic hand.
        skin_material.set_principled_shader_value(
            "Base Color", [float(np.random.uniform(0.28, 0.48)),
                           float(np.random.uniform(0.12, 0.25)),
                           float(np.random.uniform(0.07, 0.16)), 1.0])
        skin_material.set_principled_shader_value("Roughness", np.random.uniform(0.52, 0.78))
        # 顶灯轻微扰动：模拟灯泡供电波动与色温差异。
        for light, base_energy in zip(area_lights, area_base_energies):
            light.set_energy(base_energy * rng.uniform(0.70, 1.30))
            light.set_color(np.array([rng.uniform(0.95, 1.05), 1.0,
                                      rng.uniform(0.93, 1.05)]))

        # ---- 第一人称取景：渲染相机模拟头部/胸部，俯视身前桌面 ----
        cam_loc = np.array([rng.uniform(-0.04, 0.04),
                            rng.uniform(-0.40, -0.30),
                            rng.uniform(0.55, 0.75)])

        # 手部锚点在身前近处；相机模型绑在小臂后段的护腕上（见下）。
        x_spread = rng.uniform(0.09, 0.15)
        wrist_centers = [
            np.array([-x_spread, rng.uniform(-0.02, 0.10), rng.uniform(0.055, 0.090)]),
            np.array([ x_spread, rng.uniform(-0.02, 0.10), rng.uniform(0.055, 0.090)]),
        ]

        # ---- 每条手臂的几何：肘部在渲染相机后下方（画面外） ----
        arm_info = []
        for wrist, side in zip(wrist_centers, (-1.0, 1.0)):
            elbow = cam_loc + np.array([side * rng.uniform(0.10, 0.16),
                                        rng.uniform(-0.20, -0.10),
                                        rng.uniform(-0.34, -0.24)])
            axis = elbow - wrist
            axis /= np.linalg.norm(axis)
            # 挂载点：沿小臂从手部向后退几厘米，相机绑在此处的护腕上。
            mount = wrist + axis * rng.uniform(0.045, 0.075)
            arm_radius = rng.uniform(0.036, 0.044)       # 更粗的前臂
            band_radius = arm_radius + rng.uniform(0.005, 0.009)
            arm_info.append(dict(wrist=wrist, elbow=elbow, mount=mount, axis=axis,
                                 arm_radius=arm_radius, band_radius=band_radius))

        for obj in objs:
            mat = obj.get_materials()[0]
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.2, 0.9))
        instance_poses = []
        for index, obj in enumerate(objs):
            info = arm_info[index]
            side_probability = max(0.0, 1.0 - args.front_probability - args.back_probability)
            view_mode = rng.choices(
                ["front", "back", "side_plus_y", "side_minus_y"],
                weights=[args.front_probability, args.back_probability,
                         side_probability * 0.5, side_probability * 0.5], k=1)[0]
            if args.arms:
                # 相机底面贴合护腕：沿垂直于手臂轴的"上"方向偏移（含半包围盒）。
                rotation_obj = wrist_camera_rotation(
                    info["mount"], cam_loc, view_mode=view_mode, rng=rng)
                up = np.array([0.0, 0.0, 1.0])
                normal = up - np.dot(up, info["axis"]) * info["axis"]
                normal /= np.linalg.norm(normal)
                half_extents = np.array([0.0164, 0.03525, 0.0221])
                box_corners = np.array([
                    [sx, sy, sz]
                    for sx in (-half_extents[0], half_extents[0])
                    for sy in (-half_extents[1], half_extents[1])
                    for sz in (-half_extents[2], half_extents[2])])
                half_along_normal = float(np.max(np.abs(box_corners @ normal)))
                loc = info["mount"] + normal * (
                    info["band_radius"] + half_along_normal + rng.uniform(0.0, 0.003))
            else:
                loc = wrist_centers[index] + np.array([rng.uniform(-0.006, 0.006),
                                                       rng.uniform(-0.006, 0.006), 0.03])
                rotation_obj = wrist_camera_rotation(
                    loc, cam_loc, view_mode=view_mode, rng=rng)
                loc = place_on_surface(loc, rotation_obj)
            info["cam_center"] = loc
            obj.set_location(loc)
            obj.set_rotation_mat(rotation_obj)
            bezel, display = rear_parts[index]
            set_local_pose(bezel, loc, rotation_obj, [0.0153, 0.0, 0.0])
            set_local_pose(display, loc, rotation_obj, [0.0160, 0.0, 0.0])
            instance_poses.append((loc, rotation_obj, view_mode))

        # ---- 前臂 + 护腕 + 手部代理；仅作上下文/遮挡，不写入目标标注 ----
        context_objects = []
        if args.arms:
            for info, (_, rotation_obj, _) in zip(arm_info, instance_poses):
                context_objects.extend(add_wrist_context(
                    info["wrist"], info["elbow"], info["mount"], info["axis"],
                    info["arm_radius"], info["band_radius"], info["cam_center"],
                    rotation_obj, skin_material, strap_material, cable_material, rng))

        # Partial foreground occlusion resembling a finger crossing the body.
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

        # ---- ego 相机位姿：头/胸高度，向下俯仰 38~48 度看向身前桌面 ----
        pitch = rng.uniform(np.deg2rad(38.0), np.deg2rad(48.0))
        yaw = rng.uniform(-0.06, 0.06)
        forward = np.array([np.sin(yaw) * np.cos(pitch),
                            np.cos(yaw) * np.cos(pitch),
                            -np.sin(pitch)])
        target = cam_loc + forward * rng.uniform(0.9, 1.3)
        rotation = bproc.camera.rotation_from_forward_vec(target - cam_loc,
                                                          inplane_rot=rng.uniform(-0.08, 0.08))
        cam2world = bproc.math.build_transformation_mat(cam_loc, rotation)
        # Each loop builds and renders one independent scene. Reuse frame 0 so
        # previously added camera keyframes are not rendered again.
        bproc.camera.add_camera_pose(cam2world, frame=0)

        # ---- 渲染并写 BOP ----
        data = bproc.renderer.render()
        bproc.writer.write_bop(
            output_dir=bop_parent_path,
            target_objects=objs,
            dataset=dataset_name,
            depth_scale=0.1,
            depths=data["depth"],
            colors=data["colors"],
            color_file_format="JPEG",
            ignore_dist_thres=10,
            append_to_existing_output=True,
            calc_mask_info_coco=args.bop_masks,
            num_worker=0,
        )
        scene_metadata[str(metadata_start_id + img_idx)] = {
            "view_modes": [pose[2] for pose in instance_poses],
            "occlusion_levels": occlusion_levels,
        }
        with open(metadata_path, "w") as f:
            json.dump(scene_metadata, f, indent=2)

        # 清理物体，下一张图重新布置
        for context_obj in context_objects:
            bpy.data.objects.remove(context_obj.blender_obj, do_unlink=True)

        print(f"[{img_idx + 1}/{args.num_images}] 完成")

    print("全部渲染完成")


if __name__ == "__main__":
    main()
