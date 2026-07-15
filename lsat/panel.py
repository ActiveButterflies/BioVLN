# -*- coding: utf-8 -*-
"""LSAT panel — sidebar UI in the 3D Viewport."""
from __future__ import annotations

import bpy
from bpy.types import Panel


class LSAT_PT_main(Panel):
    """Main LSAT panel in the 3D View sidebar."""
    bl_label = "Lab Annotation Toolkit"
    bl_idname = "LSAT_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LSAT"

    def draw(self, context):
        layout = self.layout
        props = context.scene.lsat

        # ── Scene Setup ───────────────────────────────────────────
        box = layout.box()
        box.label(text="Scene Setup", icon="SCENE_DATA")
        box.prop(props, "equipment_collection", text="Equipment")
        box.prop(props, "room_collection", text="Room")
        box.prop(props, "obstacle_collection", text="Obstacles")
        row = box.row()
        row.operator("lsat.scan_scene", text="Scan Scene", icon="VIEWZOOM")
        row.operator("lsat.mark_equipment", text="Tag Selected", icon="RESTRICT_SELECT_OFF")

        layout.separator()

        # ── Approach Pose ─────────────────────────────────────────
        box = layout.box()
        box.label(text="Approach Pose", icon="ORIENTATION_GIMBAL")
        box.prop(props, "approach_distance")
        box.prop(props, "safety_margin")
        row = box.row(align=True)
        row.operator("lsat.generate_approaches", text="Generate", icon="ADD")
        row.operator("lsat.validate", text="Validate", icon="CHECKMARK")
        row = box.row()
        row.operator("lsat.create_markers", text="Create Markers", icon="EMPTY_ARROWS")

        # Show count
        count = context.scene.get("lsat_approach_count", 0)
        invalid = context.scene.get("lsat_invalid_count", 0)
        if count > 0:
            box.label(text=f"  {count} valid approaches ({invalid} rejected)")

        layout.separator()

        # ── Navigation Graph ──────────────────────────────────────
        box = layout.box()
        box.label(text="Navigation Graph", icon="NODETREE")
        row = box.row(align=True)
        op = row.operator("lsat.build_graph", text="KNN")
        op.mode = "knn"
        box.prop(op, "knn_k", text="K")
        row2 = box.row(align=True)
        op2 = row2.operator("lsat.build_graph", text="Radius")
        op2.mode = "radius"
        box.prop(op2, "radius", text="Radius")

        nodes = context.scene.get("lsat_graph_nodes", 0)
        edges = context.scene.get("lsat_graph_edges", 0)
        if nodes > 0:
            box.label(text=f"  {nodes} nodes, {edges} edges")

        layout.separator()

        # ── Export ────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        row = box.row()
        op3 = row.operator("lsat.export", text="JSON")
        op3.export_format = "json"
        op4 = row.operator("lsat.export", text="Habitat")
        op4.export_format = "habitat"
        op5 = row.operator("lsat.export", text="Isaac")
        op5.export_format = "isaac"

        layout.separator()

        # ── Visualization ─────────────────────────────────────────
        row = layout.row()
        row.operator("lsat.toggle_visualization", text="Toggle Viz", icon="HIDE_OFF")
        row2 = layout.row(align=True)
        row2.prop(props, "show_approaches", text="Arrows")
        row2.prop(props, "show_interaction_areas", text="Areas")
        row2.prop(props, "show_nav_graph", text="Graph")
        row2.prop(props, "show_labels", text="Labels")


class LSAT_PT_equipment(Panel):
    """Per-object equipment properties panel."""
    bl_label = "LSAT Equipment"
    bl_idname = "LSAT_PT_equipment"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == "MESH"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        props = obj.lsat

        layout.prop(props, "is_equipment")

        if not props.is_equipment:
            return

        layout.prop(props, "category")
        layout.prop(props, "super_category")
        layout.prop(props, "operation_side")
        layout.prop(props, "clearance")
        layout.prop(props, "operation_height")

        layout.separator()
        layout.label(text="Affordances:")

        from . import AFFORDANCE_OPTIONS
        cols = layout.column_flow(columns=2)
        for aff in AFFORDANCE_OPTIONS:
            cols.prop(obj, f"lsat_affordance_{aff}", text=aff.replace("_", " ").title())

        layout.separator()
        layout.prop(props, "target_marker_name", text="Target Marker")


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

_CLASSES = [
    LSAT_PT_main,
    LSAT_PT_equipment,
]


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
