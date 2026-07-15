# -*- coding: utf-8 -*-
"""Visualization — draws approach arrows, interaction areas, nav graph in viewport."""
from __future__ import annotations
from typing import List, Optional, Set, Tuple

import bpy
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from math import cos, sin, pi

from . import ApproachPose, InteractionArea


def draw_arrow(position: Vector, yaw: float, length: float = 0.5,
               color: Tuple[float, float, float, float] = (0, 1, 0, 1)):
    """Draw a single approach arrow in the 3D viewport."""
    fwd = Vector((0, length, 0))
    rot = Matrix.Rotation(yaw, 4, "Z")
    fwd_rot = rot @ fwd

    tip = position + fwd_rot
    head_len = length * 0.3
    head_w = length * 0.15
    left = rot @ Vector((-head_w, length - head_len, 0))
    right = rot @ Vector((head_w, length - head_len, 0))

    vertices = [
        position, tip,
        tip, position + left,
        tip, position + right,
    ]
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    shader.bind()
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, "LINES", {"pos": vertices})
    batch.draw(shader)


def draw_rect(position: Vector, yaw: float, width: float, depth: float,
              color: Tuple[float, float, float, float] = (0, 0.5, 1, 0.3)):
    """Draw a wireframe rectangle (interaction area) in viewport."""
    hw = width / 2
    hd = depth / 2
    corners_local = [
        Vector((-hw, -hd, 0)),
        Vector((hw, -hd, 0)),
        Vector((hw, hd, 0)),
        Vector((-hw, hd, 0)),
    ]
    rot = Matrix.Rotation(yaw, 4, "Z")
    corners = [position + rot @ c for c in corners_local]

    vertices = [
        corners[0], corners[1],
        corners[1], corners[2],
        corners[2], corners[3],
        corners[3], corners[0],
    ]
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    shader.bind()
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, "LINES", {"pos": vertices})
    batch.draw(shader)


def draw_label(text: str, position: Vector,
               color: Tuple[float, float, float, float] = (1, 1, 1, 1)):
    """Draw a 2D text label anchored at a 3D position."""
    from bpy_extras import view3d_utils
    region = bpy.context.region
    rv3d = bpy.context.region_data
    if not region or not rv3d:
        return

    co_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, position)
    if not co_2d:
        return

    font_id = 0
    blf.position(font_id, int(co_2d.x), int(co_2d.y), 0)
    blf.size(font_id, 12)
    blf.color(font_id, *color)
    blf.draw(font_id, text)


def draw_approaches(approaches: List[ApproachPose],
                    color: Tuple[float, float, float, float] = (0, 1, 0, 1)):
    """Draw arrows for all approach poses."""
    for ap in approaches:
        draw_arrow(ap.position, ap.yaw, color=color)
        draw_label(
            f"{ap.object_name[:15]} ({ap.side})",
            ap.position + Vector((0, 0, 0.3)),
            (1, 1, 1, 1),
        )


def draw_interaction_areas(areas: List[InteractionArea],
                           color: Tuple[float, float, float, float] = (0, 0.5, 1, 0.4)):
    """Draw wireframe interaction areas."""
    for area in areas:
        draw_rect(area.center, area.yaw, area.width, area.depth, color=color)


def draw_nav_graph_edges(graph, color: Tuple[float, float, float, float] = (0.8, 0.8, 0, 0.6)):
    """Draw edges of navigation graph as thin lines."""
    positions = [n.position for n in graph.nodes]
    vertices = []
    for edge in graph.edges:
        if 0 <= edge.source_idx < len(positions) and 0 <= edge.target_idx < len(positions):
            vertices.append(positions[edge.source_idx])
            vertices.append(positions[edge.target_idx])

    if not vertices:
        return

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    shader.bind()
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, "LINES", {"pos": vertices})
    batch.draw(shader)


# ═══════════════════════════════════════════════════════════════════════════
# Viewport overlay handler
# ═══════════════════════════════════════════════════════════════════════════

_draw_handler = None


def overlay_callback(approaches, interaction_areas, nav_graph, scene_props):
    """Combined draw callback for all LSAT overlays."""
    if scene_props.show_approaches:
        draw_approaches(approaches)
    if scene_props.show_interaction_areas:
        draw_interaction_areas(interaction_areas)
    if scene_props.show_nav_graph and nav_graph:
        draw_nav_graph_edges(nav_graph)


def register_overlay(callback_func):
    """Register the 3D viewport overlay draw handler."""
    global _draw_handler
    unregister_overlay()
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        callback_func, (), "WINDOW", "POST_VIEW"
    )


def unregister_overlay():
    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
