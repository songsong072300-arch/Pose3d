import trimesh
import numpy as np

# 1. 加载并归中
mesh = trimesh.load("TripoSR/output/camera_v1/0/mesh.obj")
mesh.vertices -= mesh.bounding_box.centroid

# 2. 判断镜头朝向（镜头是突出的，其突起方向的极值点距离中心更远）
min_x, max_x = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
lens_direction = "+X" if abs(max_x) > abs(min_x) else "-X"
print(f"镜头突出方向推测为: {lens_direction}")

# 3. 真实毫米尺寸等比例缩放
# 真实尺寸：X(厚度)=32.8, Y(宽度)=70.5, Z(高度)=44.2
real_dimensions = np.array([32.8, 70.5, 44.2])
scale_factor = (real_dimensions / mesh.bounding_box.extents).mean()
mesh.apply_scale(scale_factor)

# 4. 导出符合 BOP 规范的 .ply 文件
bop_path = "output/dji_action4_bop.ply"
mesh.export(bop_path)
print(f"已导出标准 BOP 模型: {bop_path}")
print(f"最终物理尺寸 (mm): {mesh.bounding_box.extents}")

# 5. 提取并打印 8 个 3D 边界框角点坐标
corners = mesh.bounding_box.vertices
print("\n8 个角点 3D 坐标 (X, Y, Z) mm:")
print(corners)