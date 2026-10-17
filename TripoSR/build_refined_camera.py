import blenderproc as bproc
"""Reference-built action camera. Run with BlenderProc; dimensions are metres.

Canonical axes: +X lens, -X rear LCD, +Y screen-left, +Z top.
The reconstruction uses nominal dimensions, not a measured CAD scan.
"""
import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "refined_action_camera"


def textures(out, reference):
    out.mkdir(parents=True, exist_ok=True)
    photo = Image.open(reference).convert("RGB")
    # Preserve the actual supplied fiducial pattern, with a clean paper border.
    photo.crop((149, 240, 511, 580)).resize((768, 768)).save(out / "marker.png")
    # Only the LCD content is rectified from frame 0; body/background excluded.
    frame = Image.open(ROOT / "frames" / "frame_00000s.jpg").convert("RGB")
    lcd = frame.transform((960, 560), Image.Transform.QUAD,
                          (1178, 594, 1162, 675, 1303, 691, 1332, 604),
                          Image.Resampling.BICUBIC)
    lcd.save(out / "lcd.png")
    label = Image.new("RGB", (1024, 256), (207, 209, 197))
    draw = ImageDraw.Draw(label)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 155)
    except OSError:
        font = ImageFont.load_default(size=155)
    draw.text((65, 30), "dji-004", font=font, fill=(40, 43, 40))
    label.save(out / "label.png")


def material(name, color, roughness, texture=None, emission=0, micro=False, metal=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bs = nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value = (*color, 1)
    bs.inputs["Roughness"].default_value = roughness
    bs.inputs["Metallic"].default_value = metal
    bs.inputs["Specular IOR Level"].default_value = .23 if micro else .3
    if texture:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(texture), check_existing=True)
        links.new(tex.outputs["Color"], bs.inputs["Base Color"])
        if emission:
            links.new(tex.outputs["Color"], bs.inputs["Emission Color"])
            bs.inputs["Emission Strength"].default_value = emission
    if micro:
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 260
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.16
        bump.inputs["Distance"].default_value = 0.000035
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bs.inputs["Normal"])
    return mat


def box(name, size, loc, mat, radius=0.001, segments=6):
    if size[0] < .005 and size[1] > .02 and size[2] > .02:
        # Extruded rounded rectangle: the face radius is independent of thickness.
        r = min(.003, size[1]/4, size[2]/4)
        ring=[]
        for cy,cz,start in [(size[1]/2-r,size[2]/2-r,0),
                            (-size[1]/2+r,size[2]/2-r,90),
                            (-size[1]/2+r,-size[2]/2+r,180),
                            (size[1]/2-r,-size[2]/2+r,270)]:
            for angle in np.linspace(start,start+90,segments+1):
                a=math.radians(angle)
                ring.append((cy+r*math.cos(a),cz+r*math.sin(a)))
        n=len(ring)
        verts=[(x,y,z) for x in [-size[0]/2,size[0]/2] for y,z in ring]
        faces=[tuple(reversed(range(n))),tuple(range(n,2*n))]
        faces.extend((i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n))
        mesh=bpy.data.meshes.new(name)
        mesh.from_pydata(verts,[],faces)
        obj=bpy.data.objects.new(name,mesh)
        bpy.context.collection.objects.link(obj)
        obj.location=loc
        mesh.materials.append(mat)
        for face in list(mesh.polygons)[2:]:
            face.use_smooth=True
        return obj
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if radius:
        bevel = obj.modifiers.new("Manufactured edge radii", "BEVEL")
        bevel.width, bevel.segments = radius, segments
        bpy.ops.object.modifier_apply(modifier=bevel.name)
        for face in obj.data.polygons:
            face.use_smooth = True
        normal = obj.modifiers.new("Face weighted normals", "WEIGHTED_NORMAL")
        bpy.ops.object.modifier_apply(modifier=normal.name)
    return obj


def panel(name, x, y, z, width, height, mat):
    # Explicit UVs: screen-left is +Y when viewed from -X. Never Euler-rotate UVs.
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(x,y+width/2,z-height/2), (x,y-width/2,z-height/2),
                      (x,y-width/2,z+height/2), (x,y+width/2,z+height/2)], [], [(0,3,2,1)])
    mesh.uv_layers.new()
    uv = [(0,0),(1,0),(1,1),(0,1)]
    for loop in mesh.loops:
        mesh.uv_layers.active.data[loop.index].uv = uv[loop.vertex_index]
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mat)
    return obj


def cylinder(name, loc, radius, depth, mat, axis="X"):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=radius, depth=depth,
                                      location=loc, rotation=(0, math.pi/2, 0) if axis=="X" else (0,0,0))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    bevel = obj.modifiers.new("Machined rim", "BEVEL")
    bevel.width, bevel.segments = 0.00025, 3
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    for p in obj.data.polygons:
        p.use_smooth=True
    return obj


def build(out):
    rubber = material("01 charcoal textured body", (.008,.010,.012), .58, micro=True)
    seam = material("02 gasket", (.002,.003,.004), .76)
    glass = material("03 black rear glass", (.003,.005,.006), .22)
    anodized = material("04 anodized lens ring", (.012,.014,.017), .29, metal=.55)
    lens = material("05 optical lens", (.004,.012,.022), .12, metal=.25)
    paper = material("06 real marker paper", (1,1,1), .78, out/"marker.png")
    display = material("07 active LCD", (1,1,1), .38, out/"lcd.png", emission=.65)
    tape = material("08 numbered masking tape", (1,1,1), .82, out/"label.png")
    red = material("09 record accent", (.13,.012,.008), .48)
    # Full body, followed by independently radiused thin rear layers.
    box("Main sealed shell", (.029,.065,.043), (0,0,0), rubber, .0038, 10)
    box("Rear perimeter gasket", (.002,.0626,.0407), (-.0144,0,0), seam, .0009)
    box("Inset rear glass", (.0008,.0608,.0385), (-.0155,0,0), glass, .00039)
    panel("LCD active content", -.01592, 0, -.001, .052, .029, display)
    box("Adhesive marker thickness", (.00018,.0237,.0237), (-.01602,.016,.0048), seam, .00008)
    panel("Actual fiducial", -.01612,.016,.0048,.0236,.0236,paper)
    # Top label is a real thin surface, aligned with the rear viewing direction.
    label = panel("Top identification tape", 0,0,0,.029,.010,tape)
    label.rotation_euler[1] = math.pi/2
    label.location = (-.0005,.006,.02158)
    cylinder("Record button seat", (0,-.022,.0212), .0063,.0012,seam, "Z")
    cylinder("Record button", (0,-.022,.022), .0055,.0007,rubber,"Z")
    cylinder("Record red centre", (0,-.022,.02238), .0022,.0001,red,"Z")
    # Lens face; housing and optical elements are geometric, not photographic.
    box("Lens-side gasket", (.001,.062,.039), (.0143,0,0), seam,.00049)
    box("Lens square housing", (.004,.029,.029), (.016,-.015,.003),rubber,.0019)
    cylinder("Lens outer retaining ring", (.019,-.015,.003),.0126,.003,anodized)
    cylinder("Lens rubber seal", (.0206,-.015,.003),.0109,.0005,seam)
    cylinder("Lens coated glass", (.02092,-.015,.003),.0098,.00025,lens)
    box("Front status display", (.0005,.023,.023), (.0151,.018,.001),glass,.00024)
    # Side door seam and latch, flush with the shell silhouette.
    box("Side battery door seal", (.020,.0005,.031), (0,-.0322,-.001),seam,.00024)
    box("Side battery door", (.0185,.00055,.0295), (0,-.0325,-.001),rubber,.00026)
    box("Door latch", (.006,.0006,.002), (-.002,-.0328,.009),anodized,.00028)
    box("USB door seal", (.016,.0005,.025), (0,.0322,-.002),seam,.00024)
    for z in [-.006,-.003,0,.003]:
        box("Microphone grille", (.0004,.006,.00055),(.01495,.017,z+.012),seam,.00019)
    parts = [obj for obj in bpy.context.scene.objects if obj.type=="MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    obj=bpy.context.object
    obj.name="refined_action_camera"
    bpy.context.scene.cursor.location=(0,0,0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    obj["category_id"]=1
    obj["asset_version"]="refined_action_camera_v1"
    vertices=np.array([v.co[:] for v in obj.data.vertices])
    lo,hi=vertices.min(0)*1000,vertices.max(0)*1000
    bounds={f"{prefix}_{axis}":float(values[i]) for prefix,values in [("min",lo),("max",hi),("size",hi-lo)] for i,axis in enumerate("xyz")}
    (out/"model_info.json").write_text(json.dumps({"asset_version":obj["asset_version"],"corner_bounds_mm":bounds,"units":"metres","reference":"frames/frame_00000s.jpg LCD; supplied 正面贴图.png marker; nominal 65 x 43 mm body"},indent=2),encoding="utf-8")
    return obj


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference",type=Path,default=ROOT.parent/"相机3D模型"/"相机3D模型"/"正面贴图.png")
    args=parser.parse_args()
    out=args.out.resolve()
    textures(out,args.reference)
    bproc.init()
    obj=build(out)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"refined_camera.blend"))
    bpy.ops.export_scene.gltf(filepath=str(out/"refined_camera.glb"),use_selection=True,export_yup=False)
    bproc.renderer.set_render_devices(use_only_cpu=True)
    bproc.renderer.set_max_amount_of_samples(32)
    bproc.camera.set_resolution(900,700)
    bproc.renderer.set_world_background([.22,.22,.22],strength=.35)
    bpy.context.scene.view_settings.view_transform="AgX"
    for loc,energy,size in [((-.10,.07,.14),.16,.16),((-.08,-.12,.04),.055,.12),((.08,.02,.10),.16,.10)]:
        light=bproc.types.Light(light_type="AREA")
        light.set_location(loc)
        light.set_rotation_euler((-Vector(loc)).to_track_quat('-Z','Y').to_euler())
        light.set_energy(energy)
        light.blender_obj.data.shape="DISK"
        light.blender_obj.data.size=size
    views={"rear":(-.135,0,.016),"rear_oblique":(-.120,-.080,.065),"lens":(.135,-.045,.040),"top":(-.075,.095,.12)}
    for name,loc in views.items():
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(loc,bproc.camera.rotation_from_forward_vec(-np.array(loc))),frame=0)
        data=bproc.renderer.render()
        Image.fromarray(np.asarray(data["colors"][0])[...,:3].astype(np.uint8)).save(out/f"preview_{name}.png")
        bproc.utility.reset_keyframes()
    board=Image.new("RGB",(1800,1400))
    for i,name in enumerate(views):
        board.paste(Image.open(out/f"preview_{name}.png"),((i%2)*900,(i//2)*700))
    board.save(out/"preview_board.jpg",quality=95)
    print(f"REFINED CAMERA SAVED: {out}; {len(obj.data.vertices)} vertices")


if __name__=="__main__":
    main()
