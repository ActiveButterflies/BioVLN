# -*- coding: utf-8 -*-
"""Collision Checker — validates approach poses against scene geometry."""
from __future__ import annotations
from typing import List, Optional

import bpy
from mathutils import Vector

from . import ApproachPose
from .utils import check_occlusion, point_in_bbox, get_objects_in_collection


def validate_approach(
    pose: ApproachPose,
    obstacle_objects: Optional[List[bpy.types.Object]] = None,
    room_collection: str = "Room",
) -> bool:
    """Validate a single approach pose against walls, obstacles, and occlusion.

    Returns True if the pose is valid (no collision, no occlusion).
    """
    # 1. Floor check: position.z should be near 0
    if abs(pose.position.z) > 0.5:
        return False

    # 2. Wall check: not inside any room mesh bounding box
    room_objs = get_objects_in_collection(room_collection)
    for obj in room_objs:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        xs = [c.x for c in corners]; ys = [c.y for c in corners]
        zs = [c.z for c in corners]
        mn = Vector((min(xs), min(ys), min(zs)))
        mx = Vector((max(xs), max(ys), max(zs)))
        if point_in_bbox(pose.position, mn, mx):
            return False

    # 3. Obstacle check: not inside equipment bounding boxes
    if obstacle_objects:
        for obj in obstacle_objects:
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            xs = [c.x for c in corners]; ys = [c.y for c in corners]
            zs = [c.z for c in corners]
            mn = Vector((min(xs), min(ys), min(zs)))
            mx = Vector((max(xs), max(ys), max(zs)))
            # expand slightly for clearance
            margin = 0.2
            mn -= Vector((margin, margin, margin))
            mx += Vector((margin, margin, margin))
            if point_in_bbox(pose.position, mn, mx):
                return False

    # 4. Occlusion: line of sight from approach to equipment center
    obj = bpy.data.objects.get(pose.object_name)
    if obj:
        center = sum(
            (obj.matrix_world @ v.co for v in obj.data.vertices),
            start=Vector((0, 0, 0)),
        ) / len(obj.data.vertices)
        # Raise target to instrument mid-height
        target = Vector((center.x, center.y, obj.location.z + 0.5))
        if check_occlusion(pose.position, target, obstacle_objects):
            return False

    return True


def filter_valid_approaches(
    poses: List[ApproachPose],
    obstacle_objects: Optional[List[bpy.types.Object]] = None,
    room_collection: str = "Room",
) -> List[ApproachPose]:
    """Return only valid approaches, with warnings for invalid ones."""
    valid = []
    for pose in poses:
        if validate_approach(pose, obstacle_objects, room_collection):
            valid.append(pose)
    return valid


def resample_invalid(
    pose: ApproachPose,
    obstacle_objects: Optional[List[bpy.types.Object]] = None,
    max_attempts: int = 10,
    step_back: float = 0.3,
) -> Optional[ApproachPose]:
    """Try to fix an invalid pose by stepping back."""
    from dataclasses import replace

    fwd = Vector((Vector((0, 0, 1)).cross(Vector((0, 1, 0))))).normalized()
    direction = Vector((pose.position.x, pose.position.y, 0)).normalized()
    if direction.length < 0.01:
        direction = Vector((0, 1, 0))

    for i in range(1, max_attempts + 1):
        new_pos = pose.position + direction * step_back * i
        new_pose = replace(pose, position=new_pos)
        if validate_approach(new_pose, obstacle_objects):
            return new_pose

    return None
