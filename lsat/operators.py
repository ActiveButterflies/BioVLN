# -*- coding: utf-8 -*-
"""LSAT operators — Blender operators bound to UI buttons."""
from __future__ import annotations
from typing import Dict, List, Optional
import json
from pathlib import Path

import bpy
from bpy.types import Operator, Context
from bpy.props import StringProperty, FloatProperty, EnumProperty

from .scanner import scan_equipment, scan_all
from .generator import generate_all_approaches, create_target_markers
from .semantics import label_scene
from .collision import filter_valid_approaches
from .graph import build_knn, build_radius
from .exporter import export_json, export_habitat_goals
from .visualization import (
    draw_approaches, draw_interaction_areas, draw_nav_graph_edges,
    register_overlay, unregister_overlay,
)
from . import ApproachPose, InteractionArea, EquipmentData
from . import NavGraph


# ═══════════════════════════════════════════════════════════════════════════
# Scene Scanner
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_OT_scan_scene(Operator):
    bl_idname = "lsat.scan_scene"
    bl_label = "Scan Scene"
    bl_description = "Scan collections and index the scene"

    def execute(self, context):
        scan_data = scan_all()
        self.report({"INFO"}, f"Scanned: {len(scan_data['equipment'])} equipment, "
                    f"{len(scan_data['room'])} room, {len(scan_data['obstacle'])} obstacles")

        # Store in scene for later use
        context.scene["lsat_scan_equipment"] = [o.name for o in scan_data["equipment"]]
        context.scene["lsat_scan_room"] = [o.name for o in scan_data["room"]]
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Mark Equipment
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_OT_mark_equipment(Operator):
    bl_idname = "lsat.mark_equipment"
    bl_label = "Mark as Equipment"
    bl_description = "Tag selected objects as equipment"

    def execute(self, context):
        count = 0
        for obj in context.selected_objects:
            if obj.type == "MESH":
                obj.lsat.is_equipment = True
                count += 1
        self.report({"INFO"}, f"Marked {count} objects as equipment")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Generate Approach Poses
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_OT_generate_approaches(Operator):
    bl_idname = "lsat.generate_approaches"
    bl_label = "Generate Approaches"
    bl_description = "Generate approach poses for all equipment"

    def execute(self, context):
        props = context.scene.lsat
        equips = scan_equipment([props.equipment_collection])

        if not equips:
            self.report({"WARNING"}, "No equipment found. Scan or tag equipment first.")
            return {"CANCELLED"}

        # If equipment has lsat properties, apply them before generating
        for obj in equips:
            lsat = getattr(obj, "lsat", None)
            if lsat and not lsat.is_equipment:
                # Auto-detect from name
                name_lower = obj.name.lower()
                for keyword, cat in [
                    ("refrig", "refrigerator"), ("incub", "incubator"),
                    ("centrifuge", "centrifuge"), ("pipette", "pipette"),
                    ("waste", "waste_bin"), ("cabinet", "cabinet"),
                    ("lab_table", "lab_table"), ("lab_bench", "lab_bench"),
                ]:
                    if keyword in name_lower:
                        lsat.category = cat
                        lsat.is_equipment = True
                        break

        approaches = generate_all_approaches(
            equips,
            distance=props.approach_distance,
            safety_margin=props.safety_margin,
        )

        # Validate
        valid = filter_valid_approaches(approaches)
        invalid_count = len(approaches) - len(valid)

        # Store
        context.scene["lsat_approaches"] = _serialize_approaches(valid)
        context.scene["lsat_approach_count"] = len(valid)
        context.scene["lsat_invalid_count"] = invalid_count

        self.report({"INFO"}, f"Generated {len(valid)} valid approaches ({invalid_count} rejected)")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Create Target Markers
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_OT_create_markers(Operator):
    bl_idname = "lsat.create_markers"
    bl_label = "Create Target Markers"
    bl_description = "Create empty arrow markers at every approach pose"

    def execute(self, context):
        approaches = _deserialize_approaches(context.scene)
        if not approaches:
            self.report({"WARNING"}, "No approaches. Generate them first.")
            return {"CANCELLED"}

        markers = create_target_markers(approaches, "Targets")
        self.report({"INFO"}, f"Created {len(markers)} target markers")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Build Nav Graph
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_OT_build_graph(Operator):
    bl_idname = "lsat.build_graph"
    bl_label = "Build Navigation Graph"
    bl_description = "Connect approach poses into a navigation graph"

    mode: EnumProperty(
        name="Mode",
        items=[("knn", "KNN", "K-nearest neighbors"), ("radius", "Radius", "Distance radius")],
        default="knn",
    )
    knn_k: FloatProperty(name="K", default=3, min=1, max=10)
    radius: FloatProperty(name="Radius", default=5.0, min=1.0, max=20.0)

    def execute(self, context):
        approaches = _deserialize_approaches(context.scene)
        if not approaches:
            self.report({"WARNING"}, "No approaches. Generate them first.")
            return {"CANCELLED"}

        if self.mode == "knn":
            graph = build_knn(approaches, k=int(self.knn_k))
        else:
            graph = build_radius(approaches, radius=self.radius)

        context.scene["lsat_graph_nodes"] = len(graph.nodes)
        context.scene["lsat_graph_edges"] = len(graph.edges)
        context.scene["lsat_graph"] = _serialize_graph(graph)

        self.report({"INFO"}, f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_OT_export(Operator):
    bl_idname = "lsat.export"
    bl_label = "Export"
    bl_description = "Export LSAT data to file"

    filepath: StringProperty(subtype="FILE_PATH")
    export_format: EnumProperty(
        name="Format",
        items=[
            ("json", "JSON", "Custom LSAT JSON"),
            ("habitat", "Habitat Goals", "BioVLN habitat-sim goals"),
            ("isaac", "Isaac Sim", "Isaac USD prim poses"),
        ],
        default="json",
    )

    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        approaches = _deserialize_approaches(context.scene)

        equips = scan_equipment([context.scene.lsat.equipment_collection])
        equipment_data = label_scene(equips)

        if self.export_format == "habitat":
            export_habitat_goals(self.filepath, approaches)
        elif self.export_format == "isaac":
            from .exporter import export_isaac_usd_prim_poses
            export_isaac_usd_prim_poses(self.filepath, approaches)
        else:
            graph = _deserialize_graph(context.scene)
            export_json(self.filepath,
                        scene_name=Path(self.filepath).stem,
                        equipment=equipment_data,
                        approaches=approaches,
                        nav_graph=graph)

        self.report({"INFO"}, f"Exported to {self.filepath}")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


# ═══════════════════════════════════════════════════════════════════════════
# Toggle Visualization
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_OT_toggle_visualization(Operator):
    bl_idname = "lsat.toggle_visualization"
    bl_label = "Toggle Visualization"
    bl_description = "Toggle LSAT viewport overlays"

    def execute(self, context):
        props = context.scene.lsat
        approaches = _deserialize_approaches(context.scene)

        if all(not getattr(props, a, False) for a in ("show_approaches", "show_interaction_areas", "show_nav_graph")):
            # Turn all ON
            props.show_approaches = True
            props.show_interaction_areas = True
            props.show_nav_graph = True
            self.report({"INFO"}, "Visualization ON")
        else:
            props.show_approaches = False
            props.show_interaction_areas = False
            props.show_nav_graph = False
            self.report({"INFO"}, "Visualization OFF")

        # Redraw
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Validate All
# ═══════════════════════════════════════════════════════════════════════════

class LSAT_OT_validate(Operator):
    bl_idname = "lsat.validate"
    bl_label = "Validate All Approaches"
    bl_description = "Run collision checks on all approach poses"

    def execute(self, context):
        approaches = _deserialize_approaches(context.scene)
        if not approaches:
            self.report({"WARNING"}, "No approaches to validate")
            return {"CANCELLED"}

        valid = filter_valid_approaches(approaches)
        rejected = len(approaches) - len(valid)
        context.scene["lsat_approaches"] = _serialize_approaches(valid)
        context.scene["lsat_approach_count"] = len(valid)
        context.scene["lsat_invalid_count"] = rejected

        self.report({"INFO"}, f"{len(valid)} valid, {rejected} rejected")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# (De)serialization helpers (store complex types in bpy.types.Scene)
# ═══════════════════════════════════════════════════════════════════════════

def _serialize_approaches(approaches: List[ApproachPose]) -> List[dict]:
    return [
        {
            "position": list(ap.position),
            "yaw": ap.yaw,
            "side": ap.side,
            "object_name": ap.object_name,
            "category": ap.category,
        }
        for ap in approaches
    ]


def _deserialize_approaches(scene: bpy.types.Scene) -> List[ApproachPose]:
    raw = scene.get("lsat_approaches")
    if not raw:
        return []
    from mathutils import Vector
    return [
        ApproachPose(
            position=Vector(r["position"]),
            yaw=r["yaw"],
            side=r["side"],
            object_name=r["object_name"],
            category=r["category"],
        )
        for r in raw
    ]


def _serialize_graph(graph: NavGraph) -> dict:
    from .graph import graph_to_dict
    return graph_to_dict(graph)


def _deserialize_graph(scene: bpy.types.Scene) -> Optional[NavGraph]:
    raw = scene.get("lsat_graph")
    if not raw:
        return None
    from mathutils import Vector
    from .graph import NavEdge
    nodes = [
        ApproachPose(
            position=Vector(n["position"]),
            yaw=n["yaw"],
            object_name=n["object"],
            category=n["category"],
            side=n["side"],
        )
        for n in raw["nodes"]
    ]
    edges = [
        NavEdge(source_idx=int(e["source"]), target_idx=int(e["target"]), distance=e["distance"])
        for e in raw["edges"]
    ]
    return NavGraph(nodes=nodes, edges=edges)


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

_CLASSES = [
    LSAT_OT_scan_scene,
    LSAT_OT_mark_equipment,
    LSAT_OT_generate_approaches,
    LSAT_OT_create_markers,
    LSAT_OT_build_graph,
    LSAT_OT_export,
    LSAT_OT_toggle_visualization,
    LSAT_OT_validate,
]

_keymap_items = []


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
