# -*- coding: utf-8 -*-
"""GLB Import Pipeline — two strategies for goal generation.

Strategy 1 (preferred): Read target_* empty node translations from GLTF JSON chunk.
  These are human-placed markers that encode the exact goal position.

Strategy 2 (fallback): Compute goal from equipment centroid + operational face direction.
  Uses known instrument face directions, wall proximity, and half-depth + approach distance.

Works both inside Blender and standalone (via pygltflib + trimesh).
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from math import atan2, sqrt, degrees
from dataclasses import dataclass
from pathlib import Path
import struct, json, os


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GoalPose:
    """A goal position + orientation in world (GLB Z-up) coords."""
    category: str
    position: List[float]     # [x, y, z] in GLB Z-up
    yaw_deg: float            # approach yaw (degrees)
    source: str               # "target_empty" or "op_face"


# ═══════════════════════════════════════════════════════════════════════════
# Name mapping
# ═══════════════════════════════════════════════════════════════════════════

_NAME_TO_CATEGORY: Dict[str, str] = {
    # Direct GLTF node name → category (includes common typos from Blender exports)
    "centrifuge":            "centrifuge",
    "cenreifuge":            "centrifuge",  # typo in GLB
    "centrifuge_2":          "mini_centrifuge",
    "mini_centrifuge":       "mini_centrifuge",
    "incubor":               "incubator",
    "inbubor":               "incubator",   # typo in GLB
    "incubator":             "incubator",
    "refrige":               "refrigerator",
    "refrigerator":          "refrigerator",
    "pipette":               "pipette",
    "Pipette":               "pipette",
    "cabinet":               "cabinet",
    "waste_bin":             "waste_bin",
    "lab_table":             "lab_table",
    "lab_table_1":           "lab_table_1",
    "lab_table_2":           "lab_table_2",
    "lab_bench":             "lab_bench",
}

# Target node name → category ("target_centrifuge" → "centrifuge")
def _target_to_category(target_name: str) -> Optional[str]:
    """Extract category from a target node name like 'target_centrifuge'.

    Handles typos: 'target_cenreifuge' → 'centrifuge', 'target_inbubor' → 'incubator'.
    """
    name_lower = target_name.lower()
    if not name_lower.startswith("target_"):
        return None
    stem = name_lower[len("target_"):]

    # Direct match
    if stem in _NAME_TO_CATEGORY:
        return _NAME_TO_CATEGORY[stem]

    # Fuzzy: levenshtein-like substring match (tolerance ~2 chars)
    for key, cat in sorted(_NAME_TO_CATEGORY.items(), key=lambda x: -len(x[0])):
        # Exact substring
        if key in stem_clean or stem_clean in key:
            return cat
        # Single char difference tolerance (typo like "cenreifuge" ≈ "centrifuge")
        if len(key) >= 5 and len(stem_clean) >= 5:
            if _char_overlap(key, stem_clean) >= max(len(key), len(stem_clean)) - 2:
                return cat

    return None


def _char_overlap(a: str, b: str) -> int:
    """Count matching character bigrams between two strings."""
    bigrams_a = {a[i:i+2] for i in range(len(a)-1)}
    bigrams_b = {b[i:i+2] for i in range(len(b)-1)}
    return len(bigrams_a & bigrams_b)

# Known equipment names (for skipping walls/floors when matching trimesh)
_SKIP_NAMES = {
    "wall", "floor", "ceiling", "door", "trim", "cube", "light", "camera",
    "room", "panel",
}

# Operational face half-depths (meters)
_FACE_HALF_DEPTH: Dict[str, float] = {
    "refrigerator": 0.455, "incubator": 0.673, "centrifuge": 0.245,
    "mini_centrifuge": 0.332, "pipette": 0.159, "waste_bin": 0.183,
    "cabinet": 0.43, "lab_table": 0.80, "lab_table_1": 0.80,
    "lab_table_2": 0.80, "lab_bench": 0.80,
}


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 1: Extract target_* empties from GLTF JSON
# ═══════════════════════════════════════════════════════════════════════════

def _read_gltf_json(glb_path: str) -> dict:
    """Parse the JSON chunk from a GLB file."""
    with open(glb_path, "rb") as f:
        assert f.read(4) == b"glTF"
        f.read(8)  # version + total_length
        json_len = struct.unpack("<I", f.read(4))[0]
        assert f.read(4) == b"JSON"
        return json.loads(f.read(json_len))


def _node_world_position(gltf_data: dict, node_idx: int, parent_map: dict,
                         world_cache: dict) -> List[float]:
    """Compute world position of a GLTF node by accumulating parent transforms."""
    if node_idx in world_cache:
        return world_cache[node_idx]

    node = gltf_data["nodes"][node_idx]

    # Local transform
    mat = None
    if "matrix" in node and node["matrix"]:
        m = node["matrix"]
        mat = [
            [m[0], m[1], m[2], m[3]],
            [m[4], m[5], m[6], m[7]],
            [m[8], m[9], m[10], m[11]],
            [0, 0, 0, 1],
        ]
    else:
        tx = node.get("translation", [0, 0, 0])
        ty = tx[0] if isinstance(tx, list) else 0
        # Build from TRS
        from math import sin, cos
        t = node.get("translation", [0.0, 0.0, 0.0])
        r = node.get("rotation", [0.0, 0.0, 0.0, 1.0])  # x,y,z,w
        s = node.get("scale", [1.0, 1.0, 1.0])

        x, y, z, w = r[0], r[1], r[2], r[3]
        rot = [
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
        ]
        mat = [
            [rot[0][0]*s[0], rot[0][1]*s[0], rot[0][2]*s[0], t[0]],
            [rot[1][0]*s[1], rot[1][1]*s[1], rot[1][2]*s[1], t[1]],
            [rot[2][0]*s[2], rot[2][1]*s[2], rot[2][2]*s[2], t[2]],
            [0, 0, 0, 1],
        ]

    # Apply parent transform
    parent_idx = parent_map.get(node_idx, -1)
    if parent_idx >= 0:
        parent_pos = _node_world_position(gltf_data, parent_idx, parent_map, world_cache)
        # Combine: parent_pos + rotation*translation + scale effects on translation
        px, py, pz = parent_pos
        world_pos = [px + mat[0][3], py + mat[1][3], pz + mat[2][3]]
    else:
        world_pos = [mat[0][3], mat[1][3], mat[2][3]]

    world_cache[node_idx] = world_pos
    return world_pos


def extract_targets_from_glb(glb_path: str) -> Dict[str, GoalPose]:
    """Strategy 1: read target_* empty positions from GLTF JSON chunk.

    Returns {category: GoalPose} for each instrument that has a target marker.
    """
    gltf_data = _read_gltf_json(glb_path)
    nodes = gltf_data.get("nodes", [])

    # Build parent index
    parent_map = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parent_map[child] = i

    # Compute world positions for ALL nodes (cached)
    world_cache = {}
    for i in range(len(nodes)):
        _node_world_position(gltf_data, i, parent_map, world_cache)

    # Find target nodes
    targets: Dict[str, GoalPose] = {}
    for i, node in enumerate(nodes):
        name = node.get("name", "")
        if not name.lower().startswith("target_"):
            continue

        cat = _target_to_category(name)
        if not cat:
            continue

        pos = world_cache.get(i, [0, 0, 0])

        # Check that the target actually has a meaningful position (not 0,0,0)
        if abs(pos[0]) < 0.001 and abs(pos[1]) < 0.001 and abs(pos[2]) < 0.001:
            continue

        # Get child equipment position for yaw computation
        children = node.get("children", [])
        child_pos = None
        if children:
            child_pos = world_cache.get(children[0])
        if child_pos is None:
            child_pos = world_cache.get(node.get("mesh"), None)

        # Compute yaw from target → equipment (agent faces equipment)
        ex, ey = pos[0], pos[1]
        if child_pos:
            cx, cy = child_pos[0], child_pos[1]
            dx = cx - ex
            dy = cy - ey
            length = sqrt(dx * dx + dy * dy)
            if length > 0.01:
                yaw = round(float(degrees(atan2(dx, dy))), 1)
            else:
                yaw = 0.0
        else:
            yaw = 0.0

        targets[cat] = GoalPose(
            category=cat,
            position=[round(pos[0], 2), round(pos[1], 2), 0.0],
            yaw_deg=yaw,
            source="target_empty",
        )

    return targets


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 2: Compute from equipment centroid + operational face
# ═══════════════════════════════════════════════════════════════════════════

# Known operational face directions in world space (unit vector in XY plane)
# Derived from BioVLN config_two_room.py operational faces
_OP_FACE_DIRS: Dict[str, Tuple[float, float]] = {
    "refrigerator":    (0, -1),    # south → -Y
    "incubator":       (-1, 0),    # west → -X
    "centrifuge":      (-1, 0),    # west → -X
    "mini_centrifuge": (-1, 0),    # west → -X
    "pipette":         (0, -1),    # south → -Y
    "waste_bin":       (1, 0),     # east → +X
    "cabinet":         (0, -1),    # south → -Y
    "lab_table":       (0, -1),    # south → -Y
    "lab_table_1":     (1, 0),     # east → +X
    "lab_table_2":     (1, 0),     # east → +X
    "lab_bench":       (0, -1),    # south → -Y
}


def compute_goals_from_equipment(
    equipment_positions: Dict[str, List[float]],
    approach_distance: float = 0.7,
) -> Dict[str, GoalPose]:
    """Strategy 2: compute goals from equipment centroid + operational face direction.

    Args:
        equipment_positions: {category: [x, y, z]} of instrument centroids.
        approach_distance: Standing distance from front face (meters).

    Returns:
        {category: GoalPose}
    """
    goals: Dict[str, GoalPose] = {}
    for cat, pos_3d in equipment_positions.items():
        face_dir = _OP_FACE_DIRS.get(cat, (0, -1))
        fd = _FACE_HALF_DEPTH.get(cat, 0.5)
        total_offset = fd + approach_distance

        gx = pos_3d[0] + face_dir[0] * total_offset
        gy = pos_3d[1] + face_dir[1] * total_offset

        yaw = round(float(degrees(atan2(face_dir[0], face_dir[1]))), 1)

        goals[cat] = GoalPose(
            category=cat,
            position=[round(gx, 2), round(gy, 2), 0.0],
            yaw_deg=yaw,
            source="op_face",
        )

    return goals


# ═══════════════════════════════════════════════════════════════════════════
# Equipment position extraction from trimesh (standalone, no Blender)
# ═══════════════════════════════════════════════════════════════════════════

def _detect_equipment_trimesh(glb_path: str) -> Dict[str, List[float]]:
    """Use trimesh to find equipment by name and return {category: [x,y,z]}."""
    try:
        import numpy as np
        import trimesh as tm
    except ImportError:
        # Fallback: try reading from pygltflib node names + vertex positions
        return _detect_equipment_pygltf(glb_path)

    scene = tm.load(glb_path)
    equipment: Dict[str, List[float]] = {}

    for key, mesh in scene.geometry.items():
        name = str(key)
        name_lower = name.lower()

        # Skip non-equipment
        if any(s in name_lower for s in _SKIP_NAMES):
            continue
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) < 50:
            continue

        # Match name to category
        cat = None
        for key_name, cat_name in sorted(_NAME_TO_CATEGORY.items(),
                                         key=lambda x: -len(x[0])):
            if key_name.lower() in name_lower or name_lower in key_name.lower():
                cat = cat_name
                break
        if cat is None:
            continue

        # Skip duplicates (keep first/largest)
        if cat in equipment:
            continue

        center = mesh.vertices.mean(axis=0)
        equipment[cat] = [round(float(center[0]), 2),
                          round(float(center[1]), 2),
                          round(float(center[2]), 2)]

    return equipment


def _detect_equipment_pygltf(glb_path: str) -> Dict[str, List[float]]:
    """Fallback: detect equipment from pygltflib node + accessor data."""
    try:
        from pygltflib import GLTF2
        import numpy as np
    except ImportError:
        return {}

    gltf = GLTF2().load(glb_path)
    equipment: Dict[str, List[float]] = {}

    for i, node in enumerate(gltf.nodes):
        name = node.name or ""
        name_lower = name.lower()
        if any(s in name_lower for s in _SKIP_NAMES):
            continue
        if node.mesh is None:
            continue

        cat = None
        for key_name, cat_name in sorted(_NAME_TO_CATEGORY.items(),
                                         key=lambda x: -len(x[0])):
            if key_name.lower() in name_lower or name_lower in key_name.lower():
                cat = cat_name
                break
        if cat is None or cat in equipment:
            continue

        # Get vertex positions from accessors
        mesh = gltf.meshes[node.mesh]
        positions = []
        for prim in mesh.primitives:
            if not prim.attributes or prim.attributes.POSITION is None:
                continue
            acc = gltf.accessors[prim.attributes.POSITION]
            bv = gltf.bufferViews[acc.bufferView]
            # Read from bin
            bin_data = gltf.binary_blob()
            offset = bv.byteOffset + acc.byteOffset
            count = acc.count
            data = np.frombuffer(bin_data, dtype=np.float32, count=count*3,
                                 offset=offset).reshape(-1, 3)
            positions.append(data)

        if not positions:
            continue
        all_verts = np.concatenate(positions)
        center = all_verts.mean(axis=0)
        equipment[cat] = [round(float(center[0]), 2),
                          round(float(center[1]), 2),
                          round(float(center[2]), 2)]

    return equipment


# ═══════════════════════════════════════════════════════════════════════════
# Combined pipeline: Strategy 1 → fallback to Strategy 2
# ═══════════════════════════════════════════════════════════════════════════

def generate_goals(
    glb_path: str,
    approach_distance: float = 0.7,
) -> Dict[str, GoalPose]:
    """Full goal generation for a baked GLB.

    1. Try Strategy 1: read target_* empties from GLTF JSON
    2. For any missing categories, fall back to Strategy 2 (operational face)

    Returns:
        {category: GoalPose} for all detected instruments.
    """
    print(f"LSAT: analyzing {os.path.basename(glb_path)}")

    # Strategy 1
    target_goals = extract_targets_from_glb(glb_path)
    covered = set(target_goals.keys())
    if covered:
        print(f"  Strategy 1 (target empties): {len(covered)} goals — {sorted(covered)}")
    else:
        print(f"  Strategy 1 (target empties): no target_* nodes found")

    # Detect equipment positions for fallback
    equipment = _detect_equipment_trimesh(glb_path)
    if not equipment:
        equipment = _detect_equipment_pygltf(glb_path)
    print(f"  Equipment detected: {len(equipment)} — {sorted(equipment.keys())}")

    # Strategy 2 for uncovered equipment
    uncovered = set(equipment.keys()) - covered
    op_goals = {}
    if uncovered:
        uncovered_positions = {c: equipment[c] for c in uncovered}
        op_goals = compute_goals_from_equipment(uncovered_positions, approach_distance)
        print(f"  Strategy 2 (op face): {len(op_goals)} goals — {sorted(op_goals.keys())}")

    # Merge: Strategy 1 takes priority
    all_goals = dict(op_goals)
    all_goals.update(target_goals)  # target_goals override op_goals
    print(f"  Total: {len(all_goals)} goals")
    return all_goals


# ═══════════════════════════════════════════════════════════════════════════
# Export to BioVLN Habitat goals format
# ═══════════════════════════════════════════════════════════════════════════

def export_habitat_goals(
    goals: Dict[str, GoalPose],
    filepath: str,
    scene_name: str = "LabScene",
) -> str:
    """Export goals as BioVLN GOALS_BY_CATEGORY JSON.

    Converts GLB Z-up (x, y, z) → habitat Y-up (x, z, -y).
    """
    output: Dict[str, List[dict]] = {}
    obj_id = 0

    for cat, gp in sorted(goals.items()):
        gx, gy, gz = gp.position
        # GLB Z-up → habitat Y-up
        hpos = [gx, gz, -gy]
        obj_id += 1

        output.setdefault(cat, []).append({
            "position": hpos,
            "object_id": obj_id,
            "object_name": f"{cat}_{obj_id}",
            "object_category": cat,
            "facing_yaw": gp.yaw_deg,
            "source": gp.source,
        })

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "scene": scene_name,
            "format": "biovln_goals_by_category",
            "goals_by_category": output,
        }, f, indent=2)

    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# Blender-specific: import + annotate
# ═══════════════════════════════════════════════════════════════════════════

def import_and_annotate(
    glb_path: str,
    distance: float = 0.7,
    export_habitat: Optional[str] = None,
    create_markers: bool = True,
) -> dict:
    """Full GLB import → goals pipeline (Blender version).

    Imports GLB into Blender, then delegates to generate_goals().
    """
    # Import into Blender
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    import bpy
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # Use standalone pipeline (works whether or not in Blender)
    goals = generate_goals(glb_path, distance)

    # Create markers in Blender
    if create_markers:
        _create_blender_markers(goals)

    # Export
    if export_habitat:
        export_habitat_goals(goals, export_habitat)
        print(f"  Habitat goals: {export_habitat}")

    return {
        "goals": goals,
        "goal_count": len(goals),
    }


def _create_blender_markers(goals: Dict[str, GoalPose],
                            collection_name: str = "LSAT_Targets") -> None:
    """Create small sphere markers in Blender at each goal position."""
    import bpy
    coll = bpy.data.collections.get(collection_name)
    if not coll:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)

    for cat, gp in goals.items():
        name = f"lsat_target_{cat}"
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

        # Small flattened sphere
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.08,
            location=(gp.position[0], gp.position[1], 0.02),
            segments=4, ring_count=3,
        )
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = (1, 1, 0.2)
        obj["lsat_target"] = True
        obj["lsat_category"] = cat
        obj["lsat_yaw"] = gp.yaw_deg

        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)
