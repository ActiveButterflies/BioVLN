# -*- coding: utf-8 -*-
"""LabScene Annotation Toolkit (LSAT) — Blender 5.x Add-on.

面向生物实验室零样本 VLN 的场景自动标注工具。
Blender 5.1.2 compatible.

Install: copy this entire addon/ folder into Blender addons directory,
or install via Preferences > Add-ons > Install.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Lazy-import bpy — only available inside Blender
try:
    import bpy
    from mathutils import Vector, Matrix, Euler
    _IN_BLENDER = True
except ModuleNotFoundError:
    _IN_BLENDER = False
    Vector = Matrix = Euler = None


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_EQUIPMENT_COLLECTION = "Equipment"
DEFAULT_ROOM_COLLECTION = "Room"
DEFAULT_APPROACH_DISTANCE = 0.7
DEFAULT_SAFETY_MARGIN = 0.3

SUPPORTED_CATEGORIES = [
    "refrigerator", "incubator", "centrifuge", "mini_centrifuge",
    "pipette", "waste_bin", "cabinet", "lab_table", "lab_bench",
    "pcr_machine", "balance", "microscope", "custom",
]

AFFORDANCE_OPTIONS = [
    "open_lid", "insert_sample", "open_door", "press_button",
    "dispose_waste", "pipette_operation", "access_shelf", "view_display",
]

OPERATION_SIDES = ["+X", "-X", "+Y", "-Y", "+Z"]


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EquipmentData:
    """Per-equipment metadata."""
    obj: Any  # bpy.types.Object when in Blender
    category: str = "custom"
    super_category: str = "Instrument"
    operation_side: str = "+Y"
    affordances: List[str] = field(default_factory=list)
    clearance: float = 0.3
    operation_height: float = 0.0

    @property
    def bbox(self) -> Any:
        """Object-aligned world-space bounding box (min, max). Requires Blender."""
        if not _IN_BLENDER:
            return None
        corners = [self.obj.matrix_world @ Vector(c) for c in self.obj.bound_box]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        zs = [c.z for c in corners]
        return (Vector((min(xs), min(ys), min(zs))),
                Vector((max(xs), max(ys), max(zs))))

    @property
    def center(self) -> Any:
        mn, mx = self.bbox
        return (mn + mx) / 2

    @property
    def bbox_depth(self) -> float:
        mn, mx = self.bbox
        size = mx - mn
        if self.operation_side in ("+X", "-X"):
            return size.x
        elif self.operation_side in ("+Y", "-Y"):
            return size.y
        return size.z


@dataclass
class ApproachPose:
    """A single approach pose near an equipment."""
    position: Any  # Vector in Blender, list outside
    yaw: float
    side: str
    object_name: str
    category: str


@dataclass
class InteractionArea:
    """Interaction area for success checking."""
    center: Vector
    width: float
    depth: float
    yaw: float
    shape: str = "rectangle"
    object_name: str = ""


# NavGraph/NavEdge imported on demand from .graph when inside Blender

# ═══════════════════════════════════════════════════════════════════════════
# Sub-module registration
# ═══════════════════════════════════════════════════════════════════════════

_MODULES = [
    "properties",
    "panel",
    "operators",
]


def register():
    """Register all LSAT sub-modules. Only callable inside Blender."""
    if not _IN_BLENDER:
        raise RuntimeError("LSAT register() requires Blender")
    for mod_name in _MODULES:
        mod = __import__(f"{__name__}.{mod_name}", fromlist=["register"])
        if hasattr(mod, "register"):
            mod.register()


def unregister():
    """Unregister all LSAT sub-modules. Only callable inside Blender."""
    if not _IN_BLENDER:
        return
    for mod_name in reversed(_MODULES):
        mod = __import__(f"{__name__}.{mod_name}", fromlist=["unregister"])
        if hasattr(mod, "unregister"):
            mod.unregister()


# ═══════════════════════════════════════════════════════════════════════════
# Blender add-on metadata
# ═══════════════════════════════════════════════════════════════════════════

bl_info = {
    "name": "LabScene Annotation Toolkit (LSAT)",
    "author": "BioVLN Team",
    "version": (1, 0, 0),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar > LSAT",
    "description": "Auto-generate approach poses, interaction areas, and nav graphs "
                   "for bio-lab VLN simulation scenes.",
    "category": "3D View",
}
