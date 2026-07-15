# -*- coding: utf-8 -*-
"""Export Manager — exports LSAT data to JSON, Habitat, Isaac formats."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import json
import os
from pathlib import Path
from math import degrees

from . import ApproachPose, InteractionArea, EquipmentData
from .graph import NavGraph, graph_to_dict
from .semantics import to_dict


# ═══════════════════════════════════════════════════════════════════════════
# JSON Export
# ═══════════════════════════════════════════════════════════════════════════

def export_json(
    filepath: str,
    scene_name: str = "LabScene",
    equipment: Optional[List[EquipmentData]] = None,
    approaches: Optional[List[ApproachPose]] = None,
    interaction_areas: Optional[List[InteractionArea]] = None,
    nav_graph: Optional[NavGraph] = None,
) -> str:
    """Export all LSAT data to a JSON file.

    Returns the filepath on success.
    """
    data: Dict[str, Any] = {
        "scene": scene_name,
        "version": "1.0",
        "generator": "LSAT",
    }

    if equipment:
        data["objects"] = [to_dict(eq) for eq in equipment]

    if approaches:
        data["approach_poses"] = [
            {
                "position": list(ap.position),
                "yaw_degrees": round(degrees(ap.yaw), 1),
                "side": ap.side,
                "object": ap.object_name,
                "category": ap.category,
            }
            for ap in approaches
        ]

    if interaction_areas:
        data["interaction_areas"] = [
            {
                "center": list(ia.center),
                "width": ia.width,
                "depth": ia.depth,
                "yaw_degrees": round(degrees(ia.yaw), 1),
                "shape": ia.shape,
                "object": ia.object_name,
            }
            for ia in interaction_areas
        ]

    if nav_graph:
        data["navigation_graph"] = graph_to_dict(nav_graph)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# Habitat Export (goal positions for episodes)
# ═══════════════════════════════════════════════════════════════════════════

def _glb_zup_to_habitat_yup(pos):
    """Convert Blender Z-up (x, y, z) → habitat-sim Y-up (x, z, -y)."""
    x, y, z = pos
    return [x, z, -y]


def export_habitat_goals(
    filepath: str,
    approaches: List[ApproachPose],
    scene_name: str = "LabScene",
) -> str:
    """Export approach poses as habitat-sim goal positions.

    Writes GOALS_BY_CATEGORY format compatible with BioVLN pipeline config.
    """
    goals: Dict[str, List[dict]] = {}
    category_count: Dict[str, int] = {}

    for ap in approaches:
        cat = ap.category
        if cat not in category_count:
            category_count[cat] = 0
        category_count[cat] += 1

        world_pos = _glb_zup_to_habitat_yup(ap.position)

        goals.setdefault(cat, []).append({
            "position": [round(v, 2) for v in world_pos],
            "object_name": f"{cat}_{category_count[cat]}",
            "object_id": 0,  # caller fills in
            "object_category": cat,
            "facing_yaw": round(degrees(ap.yaw), 1),
        })

    output = {
        "scene": scene_name,
        "format": "biovln_goals_by_category",
        "goals_by_category": goals,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# Isaac Sim Export
# ═══════════════════════════════════════════════════════════════════════════

def export_isaac_usd_prim_poses(
    filepath: str,
    approaches: List[ApproachPose],
) -> str:
    """Export approach poses as Isaac Sim USD prim-transform JSON."""
    # Isaac uses Z-up just like Blender, with Y forward
    prims = []
    for i, ap in enumerate(approaches):
        # Rotation: yaw around Z
        quat = _quat_from_yaw(ap.yaw)
        prims.append({
            "prim_path": f"/World/Approach_{ap.object_name}_{ap.side}_{i}",
            "translate": [round(ap.position.x, 3), round(ap.position.y, 3), 0.0],
            "rotate": [round(quat[0], 4), round(quat[1], 4), round(quat[2], 4), round(quat[3], 4)],
            "metadata": {
                "lsat_side": ap.side,
                "lsat_object": ap.object_name,
                "lsat_category": ap.category,
            },
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"prims": prims}, f, indent=2)

    return filepath


def _quat_from_yaw(yaw: float):
    """Z-up yaw → quaternion [x, y, z, w]."""
    from math import sin, cos
    half = yaw / 2
    return [0.0, 0.0, sin(half), cos(half)]
