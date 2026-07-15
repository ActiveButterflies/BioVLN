# -*- coding: utf-8 -*-
"""Semantic Labeler — assigns categories and properties to tagged objects."""
from __future__ import annotations
from typing import Dict, List, Any, Optional

import bpy

from . import EquipmentData
from .scanner import get_bbox, get_mesh_center


def label_from_properties(obj: bpy.types.Object) -> Optional[EquipmentData]:
    """Create EquipmentData from an object's LSAT properties."""
    props = getattr(obj, "lsat", None)
    if not props or not getattr(props, "is_equipment", False):
        return None

    return EquipmentData(
        obj=obj,
        category=getattr(props, "category", "custom"),
        super_category=getattr(props, "super_category", "Instrument"),
        operation_side=getattr(props, "operation_side", "+Y"),
        affordances=_get_affordances(props),
        clearance=getattr(props, "clearance", 0.3),
        operation_height=getattr(props, "operation_height", 0.0),
    )


def label_from_name(obj: bpy.types.Object) -> Optional[EquipmentData]:
    """Heuristic: guess equipment category from object name.

    Used as fallback when LSAT properties are not set.
    """
    name_lower = obj.name.lower()
    category = "custom"

    patterns = {
        "refrig": "refrigerator", "incub": "incubator",
        "centrifuge": "centrifuge", "mini_centrifuge": "mini_centrifuge",
        "pipette": "pipette", "waste": "waste_bin",
        "cabinet": "cabinet", "lab_table": "lab_table",
        "lab_bench": "lab_bench", "pcr": "pcr_machine",
        "balance": "balance", "microscope": "microscope",
    }
    for key, cat in patterns.items():
        if key in name_lower:
            category = cat
            break

    return EquipmentData(obj=obj, category=category)


def label_scene(equipment_objs: List[bpy.types.Object]) -> List[EquipmentData]:
    """Label all equipment in scene. Prefers LSAT properties, falls back to name."""
    results = []
    for obj in equipment_objs:
        data = label_from_properties(obj) or label_from_name(obj)
        results.append(data)
    return results


def _get_affordances(props) -> List[str]:
    """Extract affordance list from object properties."""
    result = []
    if not props:
        return result
    # Check each known affordance flag
    from . import AFFORDANCE_OPTIONS
    for aff in AFFORDANCE_OPTIONS:
        flag_attr = f"affordance_{aff}"
        if hasattr(props, flag_attr) and getattr(props, flag_attr):
            result.append(aff)
    return result


def to_dict(equip: EquipmentData) -> dict:
    """Serialize EquipmentData to a plain dict."""
    bbox_min, bbox_max = equip.bbox
    return {
        "id": equip.obj.name,
        "category": equip.category,
        "super_category": equip.super_category,
        "position": list(equip.center),
        "bbox": {
            "min": list(bbox_min),
            "max": list(bbox_max),
        },
        "operation_side": equip.operation_side,
        "affordances": equip.affordances,
        "clearance": equip.clearance,
        "operation_height": equip.operation_height,
    }
