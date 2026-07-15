#!/usr/bin/env python3
"""
Blender front-view instrument renders — camera at operational face, transparent bg.
"""
import bpy, mathutils, math, os
from pathlib import Path

GLB = "/root/gpufree-data/multi_scene/output/lab_single_room_s842_v0/lab_single_room_s842_v0.glb"
OUT_DIR = Path("/root/lsat/paper/figures_glb/renders/instruments")

TARGETS = {
    "refrigerator_0":     ("refrigerator",     (0, -1, 0)),
    "incubator_1":        ("incubator",        (-1, 0, 0)),
    "mini_centrifuge_2":  ("mini_centrifuge",  (-1, 0, 0)),
    "pipette_3":          ("pipette",          (0, -1, 0)),
    "waste_bin_4":        ("waste_bin",        (0, 1, 0)),
    "bench_west":         ("lab_bench",        (1, 0, 0)),
    "central_table":      ("lab_table",        (0, -1, 0)),
}

def clear_all():
    for coll in list(bpy.data.collections):
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    for tp in [bpy.data.meshes, bpy.data.materials, bpy.data.images,
               bpy.data.lights, bpy.data.cameras, bpy.data.curves]:
        for block in list(tp): tp.remove(block)

def render_one(obj_name, label, op_face):
    # Hide all except target
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.hide_render = (obj.name != obj_name)

    obj = bpy.data.objects.get(obj_name)
    if obj is None: print(f"  ⚠  {obj_name} not found"); return

    # World-space vertex bounds
    mw = obj.matrix_world
    xs, ys, zs = [], [], []
    for v in obj.data.vertices:
        wc = mw @ v.co
        xs.append(wc.x); ys.append(wc.y); zs.append(wc.z)
    cx, cy, cz = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2
    w, d, h = max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)

    # Rotate op_face by instrument's Z-rotation to get world direction
    rot = obj.rotation_euler
    cos_z, sin_z = math.cos(rot.z), math.sin(rot.z)
    face_world = mathutils.Vector((
        op_face[0]*cos_z - op_face[1]*sin_z,
        op_face[0]*sin_z + op_face[1]*cos_z,
        op_face[2]
    )).normalized()

    # Camera: in front of instrument along operational face
    # face_half ≈ max(half_width half_depth)
    half_face = max(w, d) / 2
    margin = 0.15 if max(w, d) > 0.5 else 0.05  # tight crop for small instruments
    standoff = max(w, d, h) * 0.4 + 0.5
    cam_pos = mathutils.Vector((cx, cy, cz)) + face_world * (half_face + standoff)
    look_at = mathutils.Vector((cx, cy, cz))

    # Ortho camera, face the instrument
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = f"Cam_{label}"
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = max(w, d, h) * (1 + margin) + 0.1
    cam.location = cam_pos
    # Manually orient: -Z = camera forward, +Z = world up
    cam_dir = (look_at - cam_pos).normalized()
    cam.rotation_euler = cam_dir.to_track_quat('-Z', 'Z').to_euler()

    # Lights — SUN from front + fill
    bpy.ops.object.light_add(type='SUN', location=cam_pos + mathutils.Vector((0, 0, 5)))
    bpy.context.object.data.energy = 5.0
    bpy.ops.object.light_add(type='SUN', location=cam_pos + mathutils.Vector((-3, 3, 4)))
    bpy.context.object.data.energy = 2.0

    # Render settings — camera sees horizontal×vertical from face direction
    if abs(face_world.x) > abs(face_world.y):
        view_w, view_h = d, h  # looking along X, sees Y×Z
    else:
        view_w, view_h = w, h  # looking along Y, sees X×Z
    res = 800
    aspect = view_h / max(view_w, 0.01)
    res_h = max(int(res * aspect), 400)
    res_w = res
    if aspect > 2.5:  # very tall → swap so width is the constrained dimension
        res_h = res
        res_w = max(int(res / max(aspect, 0.01)), 400)
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x = res_w
    scene.render.resolution_y = res_h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.filepath = str(OUT_DIR / f"{label}.png")
    scene.view_settings.view_transform = 'Standard'

    bpy.ops.render.render(write_still=True)

    kb = os.path.getsize(OUT_DIR / f"{label}.png") / 1024
    print(f"  ✓ {label}.png  ({w:.2f}×{d:.2f}×{h:.2f}m  {kb:.0f} KB)")

    # Clean camera + lights
    for o in list(bpy.data.objects):
        if o.type in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(o, do_unlink=True)

# ─── Main ──────────────────────────────────────────────────────────
clear_all()
bpy.ops.import_scene.gltf(filepath=GLB)
meshes = [o.name for o in bpy.data.objects if o.type == 'MESH']
print(f"Imported {len(meshes)} meshes\n")
OUT_DIR.mkdir(parents=True, exist_ok=True)
for obj_name, (label, face) in TARGETS.items():
    render_one(obj_name, label, face)
print(f"\nDone → {OUT_DIR}")
