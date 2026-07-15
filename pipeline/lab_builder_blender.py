"""
Lab Builder for Blender — reads lab_spec.json, builds the 3D scene, exports GLB.

Run headless:
  ./blender --background --python lab_builder_blender.py -- \\
      --spec lab_spec.json --out /output/dir/

All geometry creation uses Blender bpy API. The output GLB preserves each
semantic object (floor, walls, furniture, instruments) as a separate mesh.
"""

import bpy
import json
import math
import sys
from pathlib import Path


# ─── Arg parsing ────────────────────────────────────────────────────────────

def _parse_args():
    """Parse --spec and --out from argv after '--'."""
    args = {"spec": None, "out": "."}
    argv = sys.argv
    if "--" in argv:
        tail = argv[argv.index("--") + 1:]
        i = 0
        while i < len(tail):
            if tail[i] == "--spec" and i + 1 < len(tail):
                args["spec"] = tail[i + 1]
                i += 2
            elif tail[i] == "--out" and i + 1 < len(tail):
                args["out"] = tail[i + 1]
                i += 2
            else:
                i += 1
    return args


spec_path = None
output_dir = None

# Parse at import time so we can use early
_spec = None
def _get_spec():
    global _spec, spec_path, output_dir
    if _spec is not None:
        return _spec

    a = _parse_args()
    spec_path = a["spec"]
    output_dir = Path(a["out"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not spec_path:
        print("ERROR: --spec required")
        sys.exit(1)

    with open(spec_path) as f:
        _spec = json.load(f)

    print(f"Spec: {spec_path}  Output: {output_dir}")
    print(f"Scene: {_spec.get('scene_name', 'unnamed')}")
    return _spec


# ─── Scene setup ────────────────────────────────────────────────────────────

def _clear_scene():
    """Remove default cube and all objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Also purge orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def _make_material(name: str, color: tuple):
    """Create a simple material with a diffuse color."""
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = (*color, 1.0)
    # Set Principled BSDF base color via nodes
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    return mat


# ─── Room construction ──────────────────────────────────────────────────────

def _make_box(name: str, location: tuple, dimensions: tuple) -> bpy.types.Object:
    """Create a box primitive and return the object."""
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = (dimensions[0], dimensions[1], dimensions[2])
    # Apply scale so dimensions become baked into the mesh
    bpy.ops.object.transform_apply(scale=True)
    return obj


def _boolean_cut(target_obj: bpy.types.Object,
                 cutter_obj: bpy.types.Object) -> bool:
    """
    Apply boolean DIFFERENCE on target_obj using cutter_obj.
    Returns True on success.
    """
    bpy.context.view_layer.objects.active = target_obj
    mod = target_obj.modifiers.new(name="Boolean_Cut", type='BOOLEAN')
    mod.object = cutter_obj
    mod.operation = 'DIFFERENCE'
    mod.solver = 'FLOAT'  # FASTEST solver, fine for axis-aligned boxes
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except RuntimeError:
        print(f"    ⚠  Boolean failed on {target_obj.name}, skipping cut")
        target_obj.modifiers.remove(mod)
        return False
    return True


def _cut_doorway_in_wall(wall_obj, wall_dir, dw, W, D, t):
    """Cut a single doorway into a wall object. Returns True on success."""
    dw_width = dw.get("width", 1.2)
    dw_height = dw.get("height", 2.1)
    dw_pos_pct = dw.get("position", 0.5)

    loc = wall_obj.location
    if wall_dir in ("north", "south"):
        dw_x = -W/2 + dw_pos_pct * W
        dw_y = loc[1]
        cutter_loc = (dw_x, dw_y, dw_height / 2)
        cutter_dims = (dw_width, t * 3, dw_height)
    else:
        dw_x = loc[0]
        dw_y = -D/2 + dw_pos_pct * D
        cutter_loc = (dw_x, dw_y, dw_height / 2)
        cutter_dims = (t * 3, dw_width, dw_height)

    cutter = _make_box(f"door_cut_temp", cutter_loc, cutter_dims)
    ok = _boolean_cut(wall_obj, cutter)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return ok


def build_room(spec: dict) -> list:
    """
    Build room geometry: floor, walls (with doorways), ceiling.

    Single room: spec["room"] = {width, depth, height, ...}
    Multi-room:  spec["rooms"] = [{width, depth, height, offset}, ...]
                 for two_room/suite templates.

    Returns list of manifest dicts.
    """
    doorways = spec.get("doorways", [])
    t = spec.get("room", {}).get("wall_thickness", 0.15)

    # ── Multi-room path ──
    if "rooms" in spec and len(spec["rooms"]) >= 2:
        return _build_multi_room(spec, doorways, t)

    # ── Single room path ──
    room = spec["room"]
    W = room["width"]
    D = room["depth"]
    H = room["height"]

    return _build_single_room(W, D, H, t, doorways)


def _build_single_room(W, D, H, t, doorways, wall_prefix="") -> list:
    """Build one room: floor + 4 walls + ceiling. Returns manifest entries."""
    manifest = []

    # ── Floor ──
    mat_floor = _make_material("mat_floor", (0.85, 0.83, 0.78))
    floor = _make_box("floor", (0, 0, t / 2), (W, D, t))
    floor.data.materials.append(mat_floor)
    manifest.append({"label": "floor", "category": "floor", "type": "floor"})

    # ── Walls ──
    mat_wall = _make_material("mat_wall", (0.88, 0.87, 0.84))
    wall_configs = [
        ("wall_north", (0,  +D/2, H/2), (W, t, H)),
        ("wall_south", (0,  -D/2, H/2), (W, t, H)),
        ("wall_east",  (+W/2, 0, H/2), (t, D, H)),
        ("wall_west",  (-W/2, 0, H/2), (t, D, H)),
    ]

    for wall_name, loc, dims in wall_configs:
        lbl = f"{wall_prefix}{wall_name}" if wall_prefix else wall_name
        wall = _make_box(lbl, loc, dims)
        wall.data.materials.append(mat_wall)

        wall_dir = wall_name.replace("wall_", "")
        wall_doors = [dw for dw in doorways
                      if dw.get("wall", "").lower() == wall_dir
                      and not dw.get("internal")]  # internal doors handled separately

        for dw in wall_doors:
            _cut_doorway_in_wall(wall, wall_dir, dw, W, D, t)

        manifest.append({
            "label": lbl,
            "category": "wall",
            "type": "wall",
            "wall_name": wall_dir,
        })

    # ── Ceiling ──
    mat_ceil = _make_material("mat_ceiling", (0.92, 0.91, 0.88))
    ceil = _make_box("ceiling", (0, 0, H - t / 2), (W, D, t))
    ceil.data.materials.append(mat_ceil)
    manifest.append({"label": "ceiling", "category": "ceiling", "type": "ceiling"})

    return manifest


def _build_multi_room(spec, doorways, t) -> list:
    """
    Build multi-room geometry (two_room / suite).

    Strategy:
      1. Compute combined bounding box
      2. Build one large outer shell (floor + 4 outer walls + ceiling)
      3. Build internal dividing wall(s) with doorways
      4. Cut doorways in outer walls per room's external doors
    """
    rooms = spec["rooms"]
    layout_type = spec.get("layout_type", "two_room")

    # ── Compute combined extents ──
    min_x = float("inf"); max_x = float("-inf")
    min_y = float("inf"); max_y = float("-inf")
    H = rooms[0].get("height", 3.0)

    for ri, rdef in enumerate(rooms):
        rw = rdef["width"]
        rd = rdef["depth"]
        ox, oy = _resolve_room_offset(rdef, rooms, ri, layout_type)
        rdef["_ox"] = ox; rdef["_oy"] = oy  # store for later use
        min_x = min(min_x, ox - rw/2)
        max_x = max(max_x, ox + rw/2)
        min_y = min(min_y, oy - rd/2)
        max_y = max(max_y, oy + rd/2)

    total_W = max_x - min_x
    total_D = max_y - min_y

    print(f"  Multi-room layout: {total_W:.1f}×{total_D:.1f}m, "
          f"{len(rooms)} rooms")

    # ── Offset everything so outer room is centered at origin ──
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    # ── 1. Outer shell ──
    manifest = []
    mat_floor = _make_material("mat_floor", (0.85, 0.83, 0.78))
    floor = _make_box("floor", (cx, cy, t/2), (total_W, total_D, t))
    floor.data.materials.append(mat_floor)
    manifest.append({"label": "floor", "category": "floor", "type": "floor"})

    mat_wall = _make_material("mat_wall", (0.88, 0.87, 0.84))
    wall_h = H

    # Outer wall names — tracks which walls belong to which room direction
    outer_walls = [
        ("wall_north", (cx, cy + total_D/2, wall_h/2), (total_W, t, wall_h)),
        ("wall_south", (cx, cy - total_D/2, wall_h/2), (total_W, t, wall_h)),
        ("wall_east",  (cx + total_W/2, cy, wall_h/2), (t, total_D, wall_h)),
        ("wall_west",  (cx - total_W/2, cy, wall_h/2), (t, total_D, wall_h)),
    ]

    for wall_name, loc, dims in outer_walls:
        wall = _make_box(wall_name, loc, dims)
        wall.data.materials.append(mat_wall)
        wall_dir = wall_name.replace("wall_", "")
        outer_doors = [dw for dw in doorways
                       if dw.get("wall", "").lower() == wall_dir and not dw.get("internal")]
        for dw in outer_doors:
            _cut_doorway_in_wall(wall, wall_dir, dw, total_W, total_D, t)
        manifest.append({
            "label": wall_name, "category": "wall",
            "type": "wall", "wall_name": wall_dir,
        })

    # ── 2. Internal dividing wall(s) ──
    # Find shared edges between adjacent rooms
    internal_doors = [dw for dw in doorways if dw.get("internal")]
    internal_walls = _compute_internal_walls(rooms, t, wall_h, cx, cy, internal_doors)

    for iw_name, loc, dims, iw_doors in internal_walls:
        iwall = _make_box(iw_name, loc, dims)
        iwall.data.materials.append(mat_wall)

        wall_dir = iw_name.replace("wall_internal_", "")
        # Cut doorways — use total_W/total_D for position calculation
        for dw in iw_doors:
            dw_w = dw.get("width", 1.2)
            dw_h = dw.get("height", 2.1)
            dw_pct = dw.get("position", 0.5)

            if wall_dir in ("north", "south"):
                # Internal north/south wall: X varies, Z is through-wall
                dw_x = -total_W/2 + dw_pct * total_W + cx
                cutter_loc = (dw_x, loc[1], dw_h/2)
                cutter_dims = (dw_w, t * 3, dw_h)
            else:
                # Internal east/west wall: Y varies
                dw_y = -total_D/2 + dw_pct * total_D + cy
                cutter_loc = (loc[0], dw_y, dw_h/2)
                cutter_dims = (t * 3, dw_w, dw_h)

            cutter = _make_box(f"door_cut_{iw_name}", cutter_loc, cutter_dims)
            _boolean_cut(iwall, cutter)
            bpy.data.objects.remove(cutter, do_unlink=True)

        manifest.append({
            "label": iw_name, "category": "wall",
            "type": "wall", "wall_name": f"internal_{wall_dir}",
        })

    # ── 3. Ceiling ──
    mat_ceil = _make_material("mat_ceiling", (0.92, 0.91, 0.88))
    ceil = _make_box("ceiling", (cx, cy, H - t/2), (total_W, total_D, t))
    ceil.data.materials.append(mat_ceil)
    manifest.append({"label": "ceiling", "category": "ceiling", "type": "ceiling"})

    return manifest


def _resolve_room_offset(rdef, all_rooms, idx, layout_type):
    """Resolve a room's offset relative to the first room's origin."""
    offset = rdef.get("offset", [0, 0])
    if offset == "auto" or offset == "auto_east":
        # Place to the east of the previous room
        prev = all_rooms[idx - 1]
        prev_ox = prev.get("_ox", 0)
        prev_oy = prev.get("_oy", 0)
        prev_w = prev["width"]
        gap = 0.15  # wall thickness gap
        ox = prev_ox + prev_w/2 + rdef["width"]/2 + gap
        oy = prev_oy  # align centers north-south
        return ox, oy
    return offset[0], offset[1]


def _compute_internal_walls(rooms, t, wall_h, cx, cy, internal_doors):
    """
    Find shared edges between rooms and create internal wall definitions.

    Returns list of (wall_name, location, dimensions, doorways_for_this_wall).
    """
    walls = []
    wall_idx = 0

    for i, ra in enumerate(rooms):
        for j, rb in enumerate(rooms):
            if j <= i:
                continue

            # Check if rooms share a vertical (north-south) boundary
            ax, ay = ra["_ox"], ra["_oy"]
            bx, by = rb["_ox"], rb["_oy"]
            aw, ad = ra["width"], ra["depth"]
            bw, bd = rb["width"], rb["depth"]

            # East edge of A touches West edge of B
            a_east = ax + aw/2
            b_west = bx - bw/2

            if abs(a_east - b_west) < 0.5:
                # Shared vertical edge — place internal north-south wall
                iw_x = a_east
                iw_y = cy  # centered between rooms
                # Wall spans the overlapping depth
                overlap_top = min(ay + ad/2, by + bd/2)
                overlap_bot = max(ay - ad/2, by - bd/2)
                iw_depth = overlap_top - overlap_bot
                if iw_depth > 0.5:
                    iw_loc = (iw_x, iw_y, wall_h/2)
                    iw_dims = (t, iw_depth, wall_h)
                    doors = [d for d in internal_doors
                            if (d.get("between") in ([i, j], [j, i])
                            or d.get("wall") in (f"internal_{wall_idx}", "interior_east", "interior_west"))]
                    walls.append((f"wall_internal_east_{wall_idx}", iw_loc, iw_dims, doors))
                    wall_idx += 1

            # North edge of A touches South edge of B
            a_north = ay + ad/2
            b_south = by - bd/2
            if abs(a_north - b_south) < 0.5:
                overlap_lr = min(ax + aw/2, bx + bw/2)
                overlap_rl = max(ax - aw/2, bx - bw/2)
                iw_width = overlap_lr - overlap_rl
                if iw_width > 0.5:
                    iw_x = cx
                    iw_y = a_north
                    iw_loc = (iw_x, iw_y, wall_h/2)
                    iw_dims = (iw_width, t, wall_h)
                    doors = [d for d in internal_doors
                            if (d.get("between") in ([i, j], [j, i])
                            or d.get("wall") in (f"internal_{wall_idx}", "interior_north", "interior_south"))]
                    walls.append((f"wall_internal_north_{wall_idx}", iw_loc, iw_dims, doors))
                    wall_idx += 1

    return walls


# ─── Lighting ──────────────────────────────────────────────────────────────

def _add_lights(spec: dict):
    """
    Add lights to the scene for proper illumination.
    - One sun light from above
    - Point lights at room corners for ambient fill
    """
    room = spec["room"]
    W = room["width"]
    D = room["depth"]
    H = room["height"]

    # ── Sun light (main directional) ──
    sun_data = bpy.data.lights.new(name="sun_main", type='SUN')
    sun_data.energy = 50.0
    sun_data.angle = 0.10
    sun_obj = bpy.data.objects.new(name="sun_main", object_data=sun_data)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.location = (W * 0.3, -D * 0.3, H * 1.5)
    sun_obj.rotation_euler = (math.radians(55), math.radians(15), math.radians(-20))

    # ── Area light (ceiling, soft fill) ──
    area_data = bpy.data.lights.new(name="area_ceiling", type='AREA')
    area_data.energy = 400.0
    area_data.size = max(W, D) * 0.85
    area_data.shape = 'RECTANGLE'
    area_data.size_y = min(W, D) * 0.75
    area_obj = bpy.data.objects.new(name="area_ceiling", object_data=area_data)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = (0, 0, H - 0.1)
    area_obj.rotation_euler = (math.radians(180), 0, 0)  # point down

    # ── Point lights at corners (bright) ──
    corners = [
        (-W/2 + 0.3, -D/2 + 0.3),
        (-W/2 + 0.3,  D/2 - 0.3),
        ( W/2 - 0.3, -D/2 + 0.3),
        ( W/2 - 0.3,  D/2 - 0.3),
    ]
    for i, (cx, cy) in enumerate(corners):
        pt_data = bpy.data.lights.new(name=f"point_corner_{i}", type='POINT')
        pt_data.energy = 200.0
        pt_data.shadow_soft_size = 0.25
        pt_obj = bpy.data.objects.new(name=f"point_corner_{i}", object_data=pt_data)
        bpy.context.collection.objects.link(pt_obj)
        pt_obj.location = (cx, cy, H * 0.85)

    # ── Mid-room fill lights ──
    for i, (cx, cy) in enumerate([(0, D*0.3), (0, -D*0.3), (W*0.3, 0), (-W*0.3, 0)]):
        pt_data = bpy.data.lights.new(name=f"point_mid_{i}", type='POINT')
        pt_data.energy = 120.0
        pt_data.shadow_soft_size = 0.35
        pt_obj = bpy.data.objects.new(name=f"point_mid_{i}", object_data=pt_data)
        bpy.context.collection.objects.link(pt_obj)
        pt_obj.location = (cx, cy, H * 0.82)

    # ── Extra overhead points for even coverage ──
    for i, (cx, cy) in enumerate([(-W*0.25, D*0.15), (W*0.25, D*0.15),
                                   (-W*0.25, -D*0.15), (W*0.25, -D*0.15)]):
        pt_data = bpy.data.lights.new(name=f"point_overhead_{i}", type='POINT')
        pt_data.energy = 90.0
        pt_data.shadow_soft_size = 0.40
        pt_obj = bpy.data.objects.new(name=f"point_overhead_{i}", object_data=pt_data)
        bpy.context.collection.objects.link(pt_obj)
        pt_obj.location = (cx, cy, H * 0.92)

    print(f"  ✓ Added lights: 1 sun + 1 area + 12 point")


# ─── Furniture ──────────────────────────────────────────────────────────────

def _make_lab_bench(position: tuple, size: tuple):
    """
    Create a lab bench: top surface + 4 legs, joined into one object.
    position: (x, y, z) — CENTER of the bench in Blender Z-up
    size: (x_len, y_len, z_height_total) — TOTAL bench dimensions
    """
    x_len, y_len, z_total = size
    px, py, pz = position

    top_thick = 0.05  # fixed desk top thickness
    surface_z = pz + z_total / 2  # top surface
    leg_top = surface_z - top_thick

    # Clear selection
    bpy.ops.object.select_all(action='DESELECT')

    # Top surface
    top = _make_box("bench_top_tmp", (px, py, surface_z - top_thick / 2),
                    (x_len, y_len, top_thick))

    # 4 legs from floor (Z=0) to leg_top
    leg_size = 0.06
    leg_z_center = leg_top / 2
    inset = 0.08
    legs = []
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        lx = px + dx * (x_len / 2 - inset)
        ly = py + dy * (y_len / 2 - inset)
        leg = _make_box(f"leg_tmp_{dx}_{dy}", (lx, ly, leg_z_center),
                        (leg_size, leg_size, leg_top))
        legs.append(leg)

    # Join all parts
    bpy.ops.object.select_all(action='DESELECT')
    top.select_set(True)
    for leg in legs:
        leg.select_set(True)
    bpy.context.view_layer.objects.active = top
    bpy.ops.object.join()
    return bpy.context.object


def _make_storage(position, size):
    """Simple box cabinet. position = center, size = (x, y, z) total dims."""
    px, py, pz = position
    x_len, y_len, z_height = size
    obj = _make_box("storage_tmp", (px, py, pz), (x_len, y_len, z_height))
    return obj


def _make_shelf(position, size):
    """Shelf unit: 4 posts + 3 shelves. position = center, size = total dims."""
    px, py, pz = position
    x_len, y_len, z_total = size

    bpy.ops.object.select_all(action='DESELECT')

    post_w = 0.04
    floor_z = pz - z_total / 2  # bottom of shelf unit
    parts = []
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        post = _make_box("post_tmp", (
            px + dx * (x_len / 2 - post_w / 2),
            py + dy * (y_len / 2 - post_w / 2),
            pz,
        ), (post_w, post_w, z_total))
        parts.append(post)

    for pct in [0.3, 0.6, 0.9]:
        shelf_z = floor_z + pct * z_total
        shelf = _make_box("shelf_tmp", (px, py, shelf_z),
                          (x_len - post_w * 2, y_len, 0.03))
        parts.append(shelf)

    bpy.ops.object.select_all(action='DESELECT')
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    return bpy.context.object


FURNITURE_BUILDERS = {
    "lab_bench": _make_lab_bench,
    "storage":   _make_storage,
    "cabinet":   _make_storage,
    "shelf":     _make_shelf,
    "table":     _make_lab_bench,
}


def build_furniture(furniture_specs: list) -> list:
    """Build furniture items. Returns manifest entries."""
    manifest = []
    mat_furn = _make_material("mat_furniture", (0.65, 0.62, 0.55))

    for item in furniture_specs:
        ftype = item.get("type", "storage")
        builder = FURNITURE_BUILDERS.get(ftype, _make_storage)
        pos = tuple(item.get("position", [0, 0, 0]))
        size = tuple(item.get("size", [0.5, 0.5, 1.0]))
        label = item.get("label", f"furniture_{ftype}")

        obj = builder(pos, size)
        obj.name = label
        obj.data.materials.append(mat_furn)

        manifest.append({
            "label": label,
            "category": item.get("category", ftype),
            "type": ftype,
            "position": list(pos),
            "size": list(size),
        })

    return manifest


# ─── Instrument material cleanup ───────────────────────────────────────────

def _cleanup_material_nodes(obj):
    """Simplify material node trees after GLB import to avoid export warnings.

    Blender's glTF importer creates two parallel material representations:
      1. A glTF-specific ShaderNodeGroup (faithful to the original GLB)
      2. A Principled BSDF approximation

    Both contain Image Texture nodes referencing the same images, which
    triggers "more than one shader node tex image" warnings during export.
    We remove the glTF group node and any orphaned Image Textures.
    """
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue
        # Blender 6.0 will remove use_nodes; handle both
        if not getattr(mat, 'use_nodes', True):
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # ── Step 1: Remove glTF-specific ShaderNodeGroup ──
        # These duplicate the PBR textures and cause export warnings
        group_nodes = [n for n in nodes if n.type == 'GROUP']
        material_outputs = [n for n in nodes if n.type == 'OUTPUT_MATERIAL']

        for gn in group_nodes:
            # The glTF group is usually connected to Material Output's Surface
            # Remove it and let the Principled BSDF take over
            for mo in material_outputs:
                for link in list(links):
                    if link.from_node == gn and link.to_node == mo:
                        links.remove(link)
            nodes.remove(gn)

        # ── Step 2: Find Principled BSDF ──
        bsdf = None
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                bsdf = n
                break

        if bsdf is None:
            # If no BSDF found, skip (material might use Emission shader, etc.)
            continue

        # ── Step 3: Ensure BSDF is connected to Material Output ──
        if material_outputs:
            mo = material_outputs[0]
            surface_input = mo.inputs.get('Surface')
            if surface_input and not surface_input.is_linked:
                links.new(bsdf.outputs['BSDF'], surface_input)

        # ── Step 4: Remove ALL Image Textures not reachable from BSDF ──
        reachable_tex = set()

        def follow_socket(socket, visited):
            if not socket.is_linked:
                return
            for link in socket.links:
                node = link.from_node
                if node in visited:
                    continue
                visited.add(node)
                if node.type == 'TEX_IMAGE':
                    reachable_tex.add(node)
                else:
                    for inp in node.inputs:
                        follow_socket(inp, visited)

        tex_inputs = {'Base Color', 'Metallic', 'Roughness', 'Normal',
                      'Emission', 'Specular IOR Level', 'Specular Tint',
                      'Clearcoat', 'Clearcoat Roughness'}
        for inp_name in tex_inputs:
            if inp_name in bsdf.inputs:
                follow_socket(bsdf.inputs[inp_name], set())

        # ── Step 5: Per-channel dedup — keep only the closest Image Texture ──
        for inp_name in tex_inputs:
            if inp_name not in bsdf.inputs:
                continue
            all_tex_for_channel = set()
            closest_tex = None
            closest_dist = 999

            def find_all_tex(socket, visited, depth):
                nonlocal closest_tex, closest_dist
                if not socket.is_linked:
                    return
                for link in socket.links:
                    node = link.from_node
                    if node in visited:
                        continue
                    visited.add(node)
                    if node.type == 'TEX_IMAGE':
                        all_tex_for_channel.add(node)
                        if depth < closest_dist:
                            closest_dist = depth
                            closest_tex = node
                    else:
                        for inp in node.inputs:
                            find_all_tex(inp, visited, depth + 1)

            find_all_tex(bsdf.inputs[inp_name], set(), 0)

            if len(all_tex_for_channel) > 1 and closest_tex is not None:
                # Remove the farther duplicates
                for extra_tex in all_tex_for_channel:
                    if extra_tex != closest_tex:
                        nodes.remove(extra_tex)
                        reachable_tex.discard(extra_tex)

        # ── Step 6: Remove orphan Image Textures ──
        removed = 0
        for n in list(nodes):
            if n.type == 'TEX_IMAGE' and n not in reachable_tex:
                nodes.remove(n)
                removed += 1

        # Also remove orphaned Group nodes that may remain
        for gn in list(nodes):
            if gn.type == 'GROUP':
                nodes.remove(gn)
                removed += 1

        if removed > 0:
            print(f"    ✓ Cleaned {removed} orphan node(s) from '{mat.name}'")


# ─── Instruments ────────────────────────────────────────────────────────────

def build_instruments(instrument_specs: list, instrument_dir: Path) -> list:
    """Import instrument GLB files and place them. Returns manifest entries."""
    manifest = []

    for item in instrument_specs:
        glb_file = instrument_dir / item["file"]
        if not glb_file.exists():
            raise FileNotFoundError(f"Instrument GLB not found: {glb_file}")

        # Import
        bpy.ops.import_scene.gltf(filepath=str(glb_file))
        imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']

        if not imported:
            print(f"    ⚠  No mesh imported from {item['file']}")
            continue

        # Join multiple imported meshes into one object
        if len(imported) >= 2:
            bpy.ops.object.select_all(action='DESELECT')
            for o in imported:
                o.select_set(True)
            bpy.context.view_layer.objects.active = imported[0]
            bpy.ops.object.join()
            obj = bpy.context.view_layer.objects.active
        else:
            obj = imported[0]

        # Clean up material node trees to prevent export warnings
        _cleanup_material_nodes(obj)

        # Hide non-mesh objects (empties, etc.)
        for o in list(bpy.data.objects):
            if o.type != 'MESH' and not o.hide_render:
                o.hide_render = True

        # Transform
        pos = item.get("position", [0, 0, 0])
        rot = item.get("rotation", [0, 0, 0])
        raw_scale = item.get("scale", 1.0)
        if isinstance(raw_scale, (int, float)):
            scale_xyz = (raw_scale, raw_scale, raw_scale)
        elif isinstance(raw_scale, list) and len(raw_scale) == 3:
            scale_xyz = tuple(raw_scale)
        else:
            scale_xyz = (1.0, 1.0, 1.0)

        obj.location = pos
        obj.rotation_euler = [math.radians(r) for r in rot]
        obj.scale = scale_xyz
        # Apply scale so it's baked — ensure only this object is selected
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(scale=True)

        # Get bounding box
        bpy.context.view_layer.update()
        bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
        min_corner = [min(v[i] for v in bbox) for i in range(3)]
        max_corner = [max(v[i] for v in bbox) for i in range(3)]

        label = item.get("label", glb_file.stem)
        obj.name = label

        manifest.append({
            "label": label,
            "category": item.get("category", glb_file.stem),
            "type": "instrument",
            "file": item["file"],
            "position": list(pos),
            "rotation": list(rot),
            "scale": list(scale_xyz),
            "bbox_min": min_corner,
            "bbox_max": max_corner,
            "bbox_size": [max_corner[i] - min_corner[i] for i in range(3)],
        })

    return manifest


# ─── Export ─────────────────────────────────────────────────────────────────

def export_glb(output_path: Path):
    """Export all objects to a single GLB file.

    Applies object transforms before export so vertex data is in world-space.
    This matches the old working pipeline exactly — habitat-sim handles
    Z-up GLBs natively (export_yup=True is effectively a no-op when
    transforms are identity).
    """
    for obj in list(bpy.data.objects):
        if obj.type != 'MESH':
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        loc_nz = obj.location.length > 0.0001
        rot_nz = any(abs(r) > 0.0001 for r in obj.rotation_euler)
        scl_nz = any(abs(s-1) > 0.0001 for s in obj.scale)
        if loc_nz or rot_nz or scl_nz:
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)
    bpy.ops.object.select_all(action='SELECT')

    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format='GLB',
        export_apply=True,
        export_yup=False,         # GLB is already Z-up; no conversion needed
        export_keep_originals=False,
        export_tangents=True,
    )
    file_size = output_path.stat().st_size
    print(f"  ✓ Exported: {output_path} ({file_size:,} bytes)")


# ─── Main ───────────────────────────────────────────────────────────────────

def manifest_to_mesh_indices(manifest: list) -> list:
    """
    After export, determine mesh_index for each manifest entry.
    GLB mesh order matches Blender object creation order.
    We verify by checking object names.
    """
    return manifest  # Order is preserved by creation order


def _import_glb_scene(glb_path: str, label_prefix: str = "") -> list:
    """Import a GLB file and return list of created mesh objects."""
    bpy.ops.import_scene.gltf(filepath=glb_path)
    mesh_objects = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    # Also collect any unselected meshes (some importers don't select all)
    for o in bpy.data.objects:
        if o.type == 'MESH' and o not in mesh_objects and o.name not in [m.name for m in mesh_objects]:
            # Check if it was just created (no users of its data before)
            pass  # safer to only use selected

    # Hide non-mesh objects
    for o in list(bpy.data.objects):
        if o.type != 'MESH' and not o.hide_render:
            o.hide_render = True

    return mesh_objects


def _import_instruments_baked(spec: dict, instrument_dir: Path) -> list:
    """Import instrument GLBs that already have world-space positions baked in.

    Unlike build_instruments() which applies position/rotation/scale from spec,
    this function trusts the GLB's embedded transforms.  Optional spec overrides
    are supported via 'position', 'rotation', 'scale' fields.
    """
    manifest = []

    for item in spec.get("instruments", []):
        glb_file = instrument_dir / item["file"]
        if not glb_file.exists():
            raise FileNotFoundError(f"Instrument GLB not found: {glb_file}")

        bpy.ops.import_scene.gltf(filepath=str(glb_file))
        imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']

        if not imported:
            print(f"    ⚠  No mesh imported from {item['file']}")
            continue

        # Join multiple meshes
        if len(imported) >= 2:
            bpy.ops.object.select_all(action='DESELECT')
            for o in imported:
                o.select_set(True)
            bpy.context.view_layer.objects.active = imported[0]
            bpy.ops.object.join()
            obj = bpy.context.view_layer.objects.active
        else:
            obj = imported[0]

        _cleanup_material_nodes(obj)

        # ── Simplest possible transform: vertex-level edit ──
        bpy.context.view_layer.update()
        ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
        cur = [(min(v[i] for v in ws) + max(v[i] for v in ws)) / 2 for i in range(3)]
        print(f"    [{obj.name}] import center={[round(x,3) for x in cur]}")

        target = item.get('position')
        rot_euler = item.get('rotation')
        new_scale = item.get('scale')

        if target or rot_euler or new_scale:
            # ── Step A: move all vertices so center is at origin ──
            for v in obj.data.vertices:
                v.co.x -= cur[0]; v.co.y -= cur[1]; v.co.z -= cur[2]

            # ── Step B: clear parent and apply so rotation works on centered data ──
            if obj.parent is not None:
                bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            obj.location = (0, 0, 0)
            obj.scale = (1, 1, 1)
            obj.rotation_mode = 'XYZ'
            obj.rotation_euler = (0, 0, 0)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # ── Step C: rotation (Blender applies around origin, which IS center) ──
            if rot_euler:
                obj.rotation_mode = 'XYZ'
                obj.rotation_euler = [math.radians(r) for r in rot_euler]
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
                print(f"    [{obj.name}] rotated by {rot_euler}")

            # ── Step D: scale ──
            if new_scale:
                obj.scale = (new_scale, new_scale, new_scale) if isinstance(new_scale, (int, float)) else tuple(new_scale)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                print(f"    [{obj.name}] scaled by {new_scale}")

            # ── Step E: move all vertices to target position (vertex-level) ──
            final_pos = target if target else cur
            for v in obj.data.vertices:
                v.co.x += final_pos[0]; v.co.y += final_pos[1]; v.co.z += final_pos[2]
            print(f"    [{obj.name}] final pos={final_pos}")

        # Verify
        bpy.context.view_layer.update()
        ws2 = [obj.matrix_world @ v.co for v in obj.data.vertices]
        mn = [min(v[i] for v in ws2) for i in range(3)]
        mx = [max(v[i] for v in ws2) for i in range(3)]
        c2 = [(mn[i]+mx[i])/2 for i in range(3)]
        print(f"    [{obj.name}] AFTER: center={[round(x,3) for x in c2]} bot={mn[2]:.3f}")

        # Hide non-mesh objects
        for o in list(bpy.data.objects):
            if o.type != 'MESH' and not o.hide_render:
                o.hide_render = True

        label = item.get("label", obj.name)
        obj.name = label

        # Read bbox
        bpy.context.view_layer.update()
        bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
        min_corner = [min(v[i] for v in bbox) for i in range(3)]
        max_corner = [max(v[i] for v in bbox) for i in range(3)]

        manifest.append({
            "label": label,
            "category": item.get("category", label),
            "type": "instrument",
            "file": item["file"],
            "bbox_min": min_corner,
            "bbox_max": max_corner,
            "bbox_size": [max_corner[i] - min_corner[i] for i in range(3)],
        })

    return manifest


def _build_internal_walls_for_spec(spec) -> list:
    """
    Build internal dividing walls for multi-room layouts (two_room, suite).
    Called AFTER importing the base room GLB.

    Reads spec["rooms"] to determine where to place internal walls
    and cuts doorways for inter-room connections.
    """
    rooms = spec["rooms"]
    doorways = spec.get("doorways", [])
    layout_type = spec.get("layout_type", "two_room")
    t = rooms[0].get("wall_thickness", 0.15)
    H = rooms[0].get("height", 3.0)

    # Compute room world offsets (same logic as _resolve_room_offset)
    for ri, rdef in enumerate(rooms):
        ow = rdef["width"]; od = rdef["depth"]
        offset = rdef.get("offset", [0, 0])
        if offset == "auto" or offset == "auto_east":
            prev = rooms[ri - 1]
            prev_ox = prev.get("_ox", 0)
            prev_w = prev["width"]
            gap = 0.15
            ox = prev_ox + prev_w/2 + ow/2 + gap
            oy = prev.get("_oy", 0)
        else:
            ox, oy = offset[0], offset[1]
        rdef["_ox"] = ox; rdef["_oy"] = oy

    mat_wall = _make_material("mat_wall_internal", (0.82, 0.81, 0.78))
    manifest = []
    wall_idx = 0

    # Find adjacent room pairs and build internal walls between them
    for i, ra in enumerate(rooms):
        for j, rb in enumerate(rooms):
            if j <= i:
                continue
            ax, ay = ra["_ox"], ra["_oy"]
            bx, by = rb["_ox"], rb["_oy"]
            aw, ad = ra["width"], ra["depth"]
            bw, bd = rb["width"], rb["depth"]

            # East edge of A touches West edge of B
            a_east = ax + aw/2; b_west = bx - bw/2
            if abs(a_east - b_west) < 1.0:
                iw_x = a_east
                overlap_top = min(ay + ad/2, by + bd/2)
                overlap_bot = max(ay - ad/2, by - bd/2)
                iw_depth = overlap_top - overlap_bot
                if iw_depth > 0.5:
                    iw_y = (overlap_top + overlap_bot) / 2
                    iw_name = f"wall_internal_east_{wall_idx}"
                    iwall = _make_box(iw_name, (iw_x, iw_y, H/2), (t, iw_depth, H))
                    iwall.data.materials.append(mat_wall)

                    # Cut doorways
                    internal_doors = [d for d in doorways if d.get("internal")]
                    for dw in internal_doors:
                        dw_pct = dw.get("position", 0.5)
                        dw_w = dw.get("width", 1.2); dw_h = dw.get("height", 2.1)
                        dw_x = iw_x
                        dw_y = overlap_bot + dw_pct * iw_depth
                        cutter = _make_box(f"door_cut_{iw_name}", (dw_x, dw_y, dw_h/2), (t*3, dw_w, dw_h))
                        _boolean_cut(iwall, cutter)
                        bpy.data.objects.remove(cutter, do_unlink=True)

                    manifest.append({"label": iw_name, "category": "wall", "type": "wall",
                                     "wall_name": "internal_east"})
                    wall_idx += 1

            # North edge of A touches South edge of B
            a_north = ay + ad/2; b_south = by - bd/2
            if abs(a_north - b_south) < 1.0:
                iw_y = a_north
                overlap_lr = min(ax + aw/2, bx + bw/2)
                overlap_rl = max(ax - aw/2, bx - bw/2)
                iw_width = overlap_lr - overlap_rl
                if iw_width > 0.5:
                    iw_x = (overlap_lr + overlap_rl) / 2
                    iw_name = f"wall_internal_north_{wall_idx}"
                    iwall = _make_box(iw_name, (iw_x, iw_y, H/2), (iw_width, t, H))
                    iwall.data.materials.append(mat_wall)

                    internal_doors = [d for d in doorways if d.get("internal")]
                    for dw in internal_doors:
                        dw_pct = dw.get("position", 0.5)
                        dw_w = dw.get("width", 1.2); dw_h = dw.get("height", 2.1)
                        dw_x = overlap_rl + dw_pct * iw_width
                        dw_y = iw_y
                        cutter = _make_box(f"door_cut_{iw_name}", (dw_x, dw_y, dw_h/2), (dw_w, t*3, dw_h))
                        _boolean_cut(iwall, cutter)
                        bpy.data.objects.remove(cutter, do_unlink=True)

                    manifest.append({"label": iw_name, "category": "wall", "type": "wall",
                                     "wall_name": "internal_north"})
                    wall_idx += 1

    return manifest


def main():
    spec = _get_spec()

    print(f"\n{'='*60}")
    print(f"Lab Builder (Blender) — {spec.get('scene_name', 'unnamed')}")
    print(f"{'='*60}")

    # 1. Clear scene
    _clear_scene()

    room_manifest = []
    furn_manifest = []
    inst_manifest = []

    # 2. Room — import BIO_GLB_room.glb (once for single-room, twice for two_room)
    room_glb = spec.get("room_glb")
    if not room_glb or not Path(room_glb).exists():
        raise FileNotFoundError(f"room_glb required, got: {room_glb}")

    print(f"\n── Importing room GLB ──")
    room_objs = _import_glb_scene(room_glb)
    for obj in room_objs:
        _cleanup_material_nodes(obj)
        room_manifest.append({"label": obj.name, "category": obj.name.lower(), "type": "room"})
    print(f"  {len(room_objs)} room objects imported")

    # Multi-room: import room GLB a second time, shifted, then connect via doorway
    if "rooms" in spec and len(spec["rooms"]) >= 2:
        second_room = spec["rooms"][1]
        offset = second_room.get("offset", [8.45, 0])
        if offset == "auto" or offset == "auto_east":
            offset = [8.45, 0]

        print(f"\n── Importing second room at offset ({offset[0]:.1f}, {offset[1]:.1f}) ──")
        room2_objs = _import_glb_scene(room_glb)
        for obj in room2_objs:
            _cleanup_material_nodes(obj)
            obj.location = (float(offset[0]), float(offset[1]), 0)
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            obj.name = f"room2_{obj.name}"
            room_manifest.append({"label": obj.name, "category": obj.name.lower(), "type": "room"})
        print(f"  {len(room2_objs)} room objects imported and shifted")

        # Connect rooms: remove adjacent walls + add floor connector
        t = 0.15
        offset_x = float(offset[0])
        removed = 0
        for obj in list(bpy.data.objects):
            if obj.type != 'MESH':
                continue
            name = obj.name
            if name in ('Wall_East', 'Wall_East.001') and 'room2' not in name:
                xs = [v.co[0] for v in obj.data.vertices]
                if abs((max(xs) + min(xs)) / 2 - 4.08) < 0.3:
                    bpy.data.objects.remove(obj, do_unlink=True)
                    removed += 1
                    print(f"  Removed: {name}")
            elif 'room2' in name and 'Wall_West' in name:
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
                print(f"  Removed: {name}")
        print(f"  {removed} wall(s) removed")

        # Floor connector: small box to bridge the gap between room floors
        # Room1 floor ends at X=4.0, room2 floor starts at X=offset_x-4.15=4.30
        # Bridge from X=4.0 to X=4.30 with floor thickness 0.15m at Z=t/2
        gap_left = 4.0
        gap_right = offset_x - 4.15
        gap_width = gap_right - gap_left
        if gap_width > 0.01:
            mat_floor = _make_material("mat_floor_connector", (0.85, 0.83, 0.78))
            floor_conn = _make_box("floor_connector",
                                   ((gap_left + gap_right) / 2, 0, t / 2),
                                   (gap_width, 6.3, t))
            floor_conn.data.materials.append(mat_floor)
            room_manifest.append({"label": "floor_connector", "category": "floor", "type": "floor"})
            print(f"  Floor connector: X={gap_left:.2f}~{gap_right:.2f}")

    # 3. Lighting
    print("\n── Adding lights ──")
    _add_lights(spec)

    # 4. Furniture — import GLB at spec positions (CENTER-origin)
    furniture_items = spec.get("furniture", [])
    if furniture_items:
        furn_dir = Path(spec.get("furniture_dir", spec.get("instrument_dir", ".")))
        print(f"\n── Placing {len(furniture_items)} furniture items ──")
        for item in furniture_items:
            glb_file = furn_dir / item["file"]
            if not glb_file.exists():
                print(f"    ⚠  Furniture GLB not found: {glb_file}")
                continue

            bpy.ops.import_scene.gltf(filepath=str(glb_file))
            imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
            if not imported:
                continue

            if len(imported) >= 2:
                bpy.ops.object.select_all(action='DESELECT')
                for o in imported:
                    o.select_set(True)
                bpy.context.view_layer.objects.active = imported[0]
                bpy.ops.object.join()
                obj = bpy.context.view_layer.objects.active
            else:
                obj = imported[0]

            _cleanup_material_nodes(obj)
            bpy.context.view_layer.update()

            # Clear parent hierarchy (GLB importer may create empties/nodes)
            if obj.parent is not None:
                bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

            # Apply current transform to get clean baseline
            has_xform = any(abs(v) > 0.0001 for v in obj.location) or \
                        any(abs(v) > 0.0001 for v in obj.rotation_euler) or \
                        any(abs(v - 1) > 0.0001 for v in obj.scale)
            if has_xform:
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Get world center, set obj.location to move it to target
            ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
            cur_center = [(min(v[i] for v in ws) + max(v[i] for v in ws)) / 2
                          for i in range(3)]

            target_pos = item.get("position", [0, 0, 0])

            obj.location = tuple(target_pos[i] - cur_center[i] for i in range(3))
            print(f"    [{obj.name}] → ({target_pos[0]:.1f},{target_pos[1]:.1f},{target_pos[2]:.1f})")

            label = item.get("label", obj.name)
            obj.name = label
            furn_manifest.append({
                "label": label,
                "category": item.get("category", label),
                "type": "furniture",
                "file": item["file"],
                "position": list(target_pos),
            })
            print(f"    [{obj.name}] shifted to ({target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f})")
        print(f"  {len(furn_manifest)} furniture items placed")

    # 5. Instruments — import GLB at spec positions (CENTER-origin)
    instruments = spec.get("instruments", [])
    if instruments:
        inst_dir = Path(spec.get("instrument_dir", "."))
        print(f"\n── Placing {len(instruments)} instruments ──")
        inst_manifest = []
        for item in instruments:
            glb_file = inst_dir / item["file"]
            if not glb_file.exists():
                raise FileNotFoundError(f"Instrument GLB not found: {glb_file}")

            bpy.ops.import_scene.gltf(filepath=str(glb_file))
            imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
            if not imported:
                continue

            # Join multiple meshes
            if len(imported) >= 2:
                bpy.ops.object.select_all(action='DESELECT')
                for o in imported: o.select_set(True)
                bpy.context.view_layer.objects.active = imported[0]
                bpy.ops.object.join()
                obj = bpy.context.view_layer.objects.active
            else:
                obj = imported[0]

            _cleanup_material_nodes(obj)
            bpy.context.view_layer.update()

            # Clear parent hierarchy (GLB importer may create empties/nodes)
            if obj.parent is not None:
                bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

            # Apply current transform to get clean mesh data
            has_xform = any(abs(v) > 0.0001 for v in obj.location) or \
                        any(abs(v) > 0.0001 for v in obj.rotation_euler) or \
                        any(abs(v - 1) > 0.0001 for v in obj.scale)
            if has_xform:
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Get world center, set obj.location to move it to target
            ws = [obj.matrix_world @ v.co for v in obj.data.vertices]
            cur_center = [(min(v[i] for v in ws) + max(v[i] for v in ws)) / 2 for i in range(3)]

            target_pos = item.get('position')
            if target_pos:
                obj.location = tuple(target_pos[i] - cur_center[i] for i in range(3))
                # Apply rotation from spec (operational face orientation)
                # GLB importer may set rotation_mode to QUATERNION; force Euler XYZ
                rot = item.get("rotation", [0, 0, 0])
                obj.rotation_mode = 'XYZ'
                obj.rotation_euler = [math.radians(r) for r in rot]
                print(f"    [{obj.name}] → ({target_pos[0]:.1f},{target_pos[1]:.1f},{target_pos[2]:.1f}) rot_z={rot[2]:.1f}°")
            else:
                print(f"    [{obj.name}] baked at ({cur_center[0]:.1f},{cur_center[1]:.1f},{cur_center[2]:.1f})")

            label = item.get("label", obj.name)
            obj.name = label

            bpy.context.view_layer.update()
            bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
            min_corner = [min(v[i] for v in bbox) for i in range(3)]
            max_corner = [max(v[i] for v in bbox) for i in range(3)]

            inst_manifest.append({
                "label": label,
                "category": item.get("category", label),
                "type": "instrument",
                "file": item["file"],
                "bbox_min": min_corner,
                "bbox_max": max_corner,
                "bbox_size": [max_corner[i] - min_corner[i] for i in range(3)],
            })

        print(f"  {len(inst_manifest)} instruments placed")

    # 6. Export
    full_manifest = room_manifest + furn_manifest + inst_manifest
    scene_name = spec.get("scene_name", "generated_lab")

    print(f"\n── Exporting scene ({len(full_manifest)} objects) ──")
    glb_path = output_dir / f"{scene_name}.glb"
    export_glb(glb_path)

    manifest_path = output_dir / f"{scene_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(full_manifest, f, indent=2)
    print(f"  ✓ Manifest: {manifest_path}")

    print(f"\n{'='*60}\nScene Summary\n{'='*60}")
    for i, entry in enumerate(full_manifest):
        extras = f" ← {entry['file']}" if entry.get("type") == "instrument" else ""
        print(f"  [{i:2d}] {entry['label']:20s} ({entry['category']}){extras}")
    print(f"\n✓ Lab Builder complete.")


if __name__ == "__main__":
    main()
