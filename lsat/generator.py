# -*- coding: utf-8 -*-
"""Approach Pose Generator — computes standing positions near equipment."""
from __future__ import annotations
from typing import List, Optional, Tuple

import bpy
from mathutils import Vector, Euler
from math import pi

from . import ApproachPose
from .scanner import get_bbox, get_mesh_center
from .utils import world_forward, direction_from_side, yaw_from_matrix, clip
from .properties import LSAT_SceneProperties


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_approaches(
    obj: bpy.types.Object,
    distance: float = 0.7,
    safety_margin: float = 0.3,
    sides: Optional[List[str]] = None,
) -> List[ApproachPose]:
    """Generate approach poses for an equipment object.

    Args:
        obj: The equipment MESH in Blender.
        distance: Standing distance from front face (default 0.7m).
        safety_margin: Added to bbox_half_depth.
        sides: Which directions to generate (default: ['front']).

    Returns:
        List of ApproachPose dataclass instances.
    """
    if sides is None:
        sides = ["front"]

    # Determine operation side from LSAT property or default to front=+Y
    op_side = getattr(obj, "lsat", None)
    if op_side and hasattr(op_side, "operation_side"):
        base_side = op_side.operation_side
    else:
        base_side = "+Y"

    cat = getattr(op_side, "category", "custom") if op_side else "custom"

    bbox_min, bbox_max = get_bbox(obj)
    size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2
    fwd = world_forward(obj)

    # Select operation_side direction in world
    if base_side in ("+X", "-X"):
        bbox_half = size.x / 2
    elif base_side in ("+Y", "-Y"):
        bbox_half = size.y / 2
    else:
        bbox_half = size.z / 2

    total_dist = bbox_half + safety_margin + distance

    results = []
    for side in sides:
        if side == "front":
            offset = direction_from_side(base_side)
            yaw_offset = 0
        elif side == "left":
            # Left = rotate +Y → -X
            base_dir = direction_from_side(base_side)
            offset = Vector((-base_dir.y, base_dir.x, 0))
            yaw_offset = pi / 2
        elif side == "right":
            base_dir = direction_from_side(base_side)
            offset = Vector((base_dir.y, -base_dir.x, 0))
            yaw_offset = -pi / 2
        elif side == "rear":
            offset = -direction_from_side(base_side)
            yaw_offset = pi
        else:
            continue

        approach_pos = center - offset * total_dist
        approach_pos.z = 0.0  # floor level

        yaw = yaw_from_matrix(obj.matrix_world) + yaw_offset

        results.append(ApproachPose(
            position=approach_pos,
            yaw=yaw,
            side=side,
            object_name=obj.name,
            category=cat,
        ))

    return results


def generate_all_approaches(
    equipment_objects: List[bpy.types.Object],
    distance: float = 0.7,
    safety_margin: float = 0.3,
) -> List[ApproachPose]:
    """Generate approach poses for all equipment in the scene."""
    all_poses = []
    for obj in equipment_objects:
        poses = generate_approaches(obj, distance, safety_margin)
        all_poses.extend(poses)
    return all_poses


def create_target_markers(approaches: List[ApproachPose],
                          collection_name: str = "Targets") -> List[bpy.types.Object]:
    """Create EMPTY objects at each approach pose for visualization/export.

    Each empty stores its metadata as custom properties.
    """
    # Get or create collection
    coll = bpy.data.collections.get(collection_name)
    if not coll:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)

    markers = []
    for i, ap in enumerate(approaches):
        name = f"target_{ap.object_name}_{ap.side}"
        # Remove old
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = "SINGLE_ARROW"
        empty.empty_display_size = 0.3
        empty.location = ap.position
        empty.rotation_euler = Euler((0, 0, ap.yaw), "XYZ")

        # Store metadata as custom properties
        empty["lsat_target"] = True
        empty["lsat_category"] = ap.category
        empty["lsat_side"] = ap.side
        empty["lsat_yaw"] = ap.yaw
        empty["lsat_object"] = ap.object_name

        coll.objects.link(empty)
        markers.append(empty)

    return markers
