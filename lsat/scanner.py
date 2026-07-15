# -*- coding: utf-8 -*-
"""Scene Scanner — scans collections and indexes the scene."""
from __future__ import annotations
from typing import Dict, List, Optional

import bpy
from mathutils import Vector

from .utils import get_objects_in_collection
from .properties import find_equipment_collections, find_room_collections


def scan_equipment(collection_names: Optional[List[str]] = None) -> List[bpy.types.Object]:
    """Return all MESH objects in equipment-tagged collections.

    If collection_names is omitted, auto-discovers collections with lsat_type='equipment'.
    """
    if collection_names:
        result = []
        for name in collection_names:
            result.extend(get_objects_in_collection(name))
        return result

    result = []
    for coll in find_equipment_collections():
        result.extend(get_objects_in_collection(coll.name))
    return result


def scan_room(collection_names: Optional[List[str]] = None) -> List[bpy.types.Object]:
    """Return all MESH objects in room-tagged collections."""
    if collection_names:
        result = []
        for name in collection_names:
            result.extend(get_objects_in_collection(name))
        return result

    result = []
    for coll in find_room_collections():
        result.extend(get_objects_in_collection(coll.name))
    return result


def scan_all() -> Dict[str, List[bpy.types.Object]]:
    """Full scene scan → {equipment, room, obstacle, other}."""
    equips = scan_equipment()
    rooms = scan_room()
    equip_ids = {o.name for o in equips}
    room_ids = {o.name for o in rooms}

    obstacles = []
    other = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.name in equip_ids or obj.name in room_ids:
            continue
        # Heuristic: if it has wall/floor/ceiling in name, treat as room
        name_lower = obj.name.lower()
        if any(k in name_lower for k in ("wall", "floor", "ceiling", "room", "door", "trim")):
            obstacles.append(obj)
        else:
            other.append(obj)

    return {
        "equipment": equips,
        "room": rooms,
        "obstacle": obstacles,
        "other": other,
    }


def get_bbox(obj: bpy.types.Object) -> tuple:
    """Return (min, max) world-space bounding box vectors."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]; ys = [c.y for c in corners]; zs = [c.z for c in corners]
    return (Vector((min(xs), min(ys), min(zs))),
            Vector((max(xs), max(ys), max(zs))))


def get_mesh_center(obj: bpy.types.Object) -> Vector:
    """Vertex-mean centroid in world space."""
    if obj.type != "MESH" or len(obj.data.vertices) == 0:
        return obj.matrix_world.translation.copy()
    total = Vector((0, 0, 0))
    for v in obj.data.vertices:
        total = total + (obj.matrix_world @ v.co)
    return total / len(obj.data.vertices)
