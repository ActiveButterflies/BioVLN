# -*- coding: utf-8 -*-
"""LSAT properties — Blender 5.x compatible.
Collection metadata uses ID properties (coll['lsat_*']) instead of
PointerProperty on Collection, which has registration issues in 5.x.
"""
from __future__ import annotations
from typing import Any, Dict, Set, Optional

import bpy
from bpy.types import Scene, Object, Collection
from bpy.props import (
    StringProperty, FloatProperty, IntProperty, BoolProperty,
    EnumProperty, PointerProperty,
)
from . import SUPPORTED_CATEGORIES, AFFORDANCE_OPTIONS, OPERATION_SIDES


# ═══════════════════════════════════════════════════════════════════════════
# Enum helpers
# ═══════════════════════════════════════════════════════════════════════════

# Item generator functions (Blender 5.x: must be module-level, not lambdas)
def _category_items(self, context):
    return [(c, c.replace("_", " ").title(), "") for c in SUPPORTED_CATEGORIES]


def _side_items(self, context):
    return [(s, s, "") for s in OPERATION_SIDES]


# ═══════════════════════════════════════════════════════════════════════════
# Per-Object properties
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_ObjectProperties(bpy.types.PropertyGroup):
    is_equipment: BoolProperty(name="Is Equipment", default=False)
    category: EnumProperty(
        name="Category",
        items=_category_items,
        default=12,  # "custom" is last in SUPPORTED_CATEGORIES
    )
    super_category: EnumProperty(
        name="Super Category",
        items=[
            ("Instrument", "Instrument", ""),
            ("Furniture", "Furniture", ""),
            ("Fixture", "Fixture", ""),
            ("Custom", "Custom", ""),
        ],
        default="Instrument",
    )
    operation_side: EnumProperty(
        name="Operation Side",
        items=_side_items,
        default=2,  # "+Y"
    )
    clearance: FloatProperty(
        name="Clearance", default=0.3, min=0.0, max=2.0,
        subtype="DISTANCE", unit="LENGTH",
    )
    operation_height: FloatProperty(
        name="Operation Height", default=0.0,
        min=0.0, max=3.0, subtype="DISTANCE", unit="LENGTH",
    )
    target_marker_name: StringProperty(
        name="Target Marker", default="",
        description="Name of the empty (or mesh) marking the target approach position",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Collection ID property helpers (no PointerProperty on Collection in 5.x)
# ═══════════════════════════════════════════════════════════════════════════

_COLL_KEYS = ("lsat_type",)  # values: "equipment", "room", "obstacle"


def set_collection_type(coll: Collection, type_name: str) -> None:
    """Store collection type as ID property (coll['lsat_type'] = 'equipment')."""
    if not isinstance(type_name, str):
        return
    coll["lsat_type"] = type_name


def get_collection_type(coll: Collection) -> str:
    """Read collection type. Returns empty string if unset."""
    return coll.get("lsat_type", "")


def find_equipment_collections() -> List[Collection]:
    """Return all collections marked as equipment."""
    return [c for c in bpy.data.collections if c.get("lsat_type") == "equipment"]


def find_room_collections() -> List[Collection]:
    return [c for c in bpy.data.collections if c.get("lsat_type") == "room"]


def find_obstacle_collections() -> List[Collection]:
    return [c for c in bpy.data.collections if c.get("lsat_type") == "obstacle"]


# ═══════════════════════════════════════════════════════════════════════════
# Per-Scene global settings
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_SceneProperties(bpy.types.PropertyGroup):
    approach_distance: FloatProperty(
        name="Approach Distance", default=0.7, min=0.2, max=5.0,
        subtype="DISTANCE", unit="LENGTH",
    )
    safety_margin: FloatProperty(
        name="Safety Margin", default=0.3, min=0.0, max=2.0,
        subtype="DISTANCE", unit="LENGTH",
    )
    interaction_width: FloatProperty(
        name="Interaction Width", default=0.8, min=0.2, max=10.0,
        subtype="DISTANCE", unit="LENGTH",
    )
    interaction_depth: FloatProperty(
        name="Interaction Depth", default=0.6, min=0.2, max=5.0,
        subtype="DISTANCE", unit="LENGTH",
    )
    heading_threshold: FloatProperty(
        name="Heading Threshold", default=15.0, min=1.0, max=90.0,
        subtype="ANGLE", unit="ROTATION",
    )
    equipment_collection: StringProperty(name="Equipment Collection", default="Equipment")
    room_collection: StringProperty(name="Room Collection", default="Room")
    obstacle_collection: StringProperty(name="Obstacle Collection", default="Obstacles")
    show_approaches: BoolProperty(name="Show Approaches", default=True)
    show_interaction_areas: BoolProperty(name="Show Interaction Areas", default=True)
    show_nav_graph: BoolProperty(name="Show Nav Graph", default=True)
    show_labels: BoolProperty(name="Show Labels", default=True)


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

_CLASSES = [
    LSAT_ObjectProperties,
    LSAT_SceneProperties,
]


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    Scene.lsat = PointerProperty(type=LSAT_SceneProperties)
    Object.lsat = PointerProperty(type=LSAT_ObjectProperties)


def unregister():
    for attr in ("lsat",):
        if getattr(Scene, attr, None) is not None:
            delattr(Scene, attr)
        if getattr(Object, attr, None) is not None:
            delattr(Object, attr)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
