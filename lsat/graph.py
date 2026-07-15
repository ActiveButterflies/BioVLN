# -*- coding: utf-8 -*-
"""Navigation Graph Builder — connects approach poses into a reachability graph."""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

from mathutils import Vector

from . import ApproachPose


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NavEdge:
    source_idx: int
    target_idx: int
    distance: float
    direct: bool = True  # False if occluded


@dataclass
class NavGraph:
    nodes: List[ApproachPose]
    edges: List[NavEdge]


# ═══════════════════════════════════════════════════════════════════════════
# Build strategies
# ═══════════════════════════════════════════════════════════════════════════

def build_knn(
    approaches: List[ApproachPose],
    k: int = 3,
    max_distance: float = 15.0,
) -> NavGraph:
    """K-nearest-neighbors: each node connected to its k nearest neighbors.

    Args:
        approaches: List of approach poses.
        k: Number of neighbors to connect to.
        max_distance: Skip edges longer than this (meters).

    Returns:
        NavGraph with nodes and edges.
    """
    n = len(approaches)
    edges: List[NavEdge] = []

    positions = [ap.position for ap in approaches]

    for i in range(n):
        distances = []
        for j in range(n):
            if i == j:
                continue
            d = (positions[i] - positions[j]).length
            if d <= max_distance:
                distances.append((d, j))

        distances.sort()
        for d, j in distances[:k]:
            edges.append(NavEdge(source_idx=i, target_idx=j, distance=d))

    return NavGraph(nodes=approaches, edges=edges)


def build_radius(
    approaches: List[ApproachPose],
    radius: float = 5.0,
) -> NavGraph:
    """Radius-based: connect all nodes within a given distance."""
    n = len(approaches)
    edges: List[NavEdge] = []
    positions = [ap.position for ap in approaches]

    for i in range(n):
        for j in range(i + 1, n):
            d = (positions[i] - positions[j]).length
            if d <= radius:
                edges.append(NavEdge(source_idx=i, target_idx=j, distance=d))
                edges.append(NavEdge(source_idx=j, target_idx=i, distance=d))

    return NavGraph(nodes=approaches, edges=edges)


def add_manual_edges(
    graph: NavGraph,
    manual_pairs: List[Tuple[int, int]],
) -> NavGraph:
    """Add manually specified edges to the graph."""
    for src, tgt in manual_pairs:
        if 0 <= src < len(graph.nodes) and 0 <= tgt < len(graph.nodes):
            d = (graph.nodes[src].position - graph.nodes[tgt].position).length
            # Avoid duplicates
            existing = {(e.source_idx, e.target_idx) for e in graph.edges}
            if (src, tgt) not in existing:
                graph.edges.append(NavEdge(source_idx=src, target_idx=tgt, distance=d))
    return graph


# ═══════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════

def graph_to_dict(graph: NavGraph) -> dict:
    """Export NavGraph as a dict suitable for JSON."""
    return {
        "nodes": [
            {
                "position": list(n.position),
                "yaw": n.yaw,
                "object": n.object_name,
                "category": n.category,
                "side": n.side,
            }
            for n in graph.nodes
        ],
        "edges": [
            {"source": e.source_idx, "target": e.target_idx, "distance": round(e.distance, 3)}
            for e in graph.edges
        ],
    }
