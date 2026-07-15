# -*- coding: utf-8 -*-
"""LSAT utilities — Blender 5.x compatible."""
from __future__ import annotations
from typing import Optional, Tuple, List

import bpy
from mathutils import Vector, Matrix, Euler
from math import radians, degrees, pi


# ═══════════════════════════════════════════════════════════════════════════
# Object traversal
# ═══════════════════════════════════════════════════════════════════════════

def get_objects_in_collection(name: str) -> List[bpy.types.Object]:
    """Return all MESH objects in a named collection (recursively)."""
    coll = bpy.data.collections.get(name)
    if not coll:
        return []
    result = []
    for obj in coll.objects:
        if obj.type == "MESH":
            result.append(obj)
    for child in coll.children:
        for obj in child.objects:
            if obj.type == "MESH":
                result.append(obj)
    return result


def get_world_position(obj: bpy.types.Object) -> Vector:
    """Extract world translation for any object type."""
    if obj.type == "EMPTY":
        return obj.matrix_world.translation.copy()
    if obj.parent and obj.parent.type == "EMPTY":
        # If parented to an empty (e.g., target marker), use empty's position
        return obj.parent.matrix_world.translation.copy()
    if obj.type == "MESH" and len(obj.data.vertices) > 0:
        center = sum((obj.matrix_world @ v.co for v in obj.data.vertices),
                     start=Vector((0, 0, 0))) / len(obj.data.vertices)
        return Vector((center.x, center.y, center.z))
    return obj.matrix_world.translation.copy()


# ═══════════════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════════════

def direction_from_side(side: str) -> Vector:
    """Return unit direction vector for operation_side."""
    return {
        "+Y": Vector((0, 1, 0)),
        "-Y": Vector((0, -1, 0)),
        "+X": Vector((1, 0, 0)),
        "-X": Vector((-1, 0, 0)),
        "+Z": Vector((0, 0, 1)),
    }.get(side, Vector((0, 1, 0)))


def yaw_from_matrix(mat: Matrix) -> float:
    """Extract Z-up yaw angle (radians) from a world matrix.
    Uses local +Y as forward, computes its XZ-plane angle.
    Blender 5.x: V2.angle_signed(V1, up) → V1.angle_signed(up) = -atan2(V1.x, V1.y)
    """
    fwd = (mat.to_3x3() @ Vector((0, 1, 0))).normalized()
    from math import atan2
    return atan2(fwd.x, fwd.y)


def world_forward(obj: bpy.types.Object) -> Vector:
    """World-space local-Y forward of object."""
    return (obj.matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ═══════════════════════════════════════════════════════════════════════════
# Collision utilities (Blender 5.x Object.ray_cast API)
# ═══════════════════════════════════════════════════════════════════════════

def point_in_bbox(point: Vector, bbox_min: Vector, bbox_max: Vector) -> bool:
    return (bbox_min.x <= point.x <= bbox_max.x and
            bbox_min.y <= point.y <= bbox_max.y and
            bbox_min.z <= point.z <= bbox_max.z)


def is_inside_wall(point: Vector, room_coll: Optional[str] = None) -> bool:
    """Crude wall check: is point inside any room mesh's AABB?"""
    room_objs = get_objects_in_collection(room_coll or "Room")
    for obj in room_objs:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        xs = [c.x for c in corners]; ys = [c.y for c in corners]
        zs = [c.z for c in corners]
        mn = Vector((min(xs), min(ys), min(zs)))
        mx = Vector((max(xs), max(ys), max(zs)))
        if point_in_bbox(point, mn, mx):
            return True
    return False


def check_occlusion(origin: Vector, target: Vector,
                    obstacles: Optional[List[bpy.types.Object]] = None) -> bool:
    """Blender 5.x ray_cast: iterate objects, check for occlusion between origin→target.

    Returns True if line-of-sight is blocked.
    """
    direction = (target - origin).normalized()
    distance = (target - origin).length

    # Collect objects to test
    if obstacles:
        test_objs = obstacles
    else:
        test_objs = [o for o in bpy.data.objects if o.type == "MESH"]

    for obj in test_objs:
        # Transform ray to object-local space
        mat_inv = obj.matrix_world.inverted()
        local_origin = mat_inv @ origin
        local_dir = (mat_inv.to_3x3() @ direction).normalized()

        # Blender 5.x: Object.ray_cast(origin, direction)
        hit, location, normal, face_index = obj.ray_cast(local_origin, local_dir)
        if hit:
            world_hit = obj.matrix_world @ location
            hit_dist = (world_hit - origin).length
            if hit_dist < distance - 0.01:
                return True

    return False
