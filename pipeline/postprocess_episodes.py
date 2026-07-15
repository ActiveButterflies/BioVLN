#!/usr/bin/env python3
"""
Post-process episode files to compute geodesic distances and add difficulty labels.

For each scene output directory, loads the navmesh and episodes, computes
shortest-path geodesic distances for every episode, and writes back updated
episode files with:
  - episode["info"]["geodesic_distance"]: float (meters)
  - episode["difficulty"]: "easy" | "medium" | "hard"

Usage:
  # Process a built scene output dir
  python postprocess_episodes.py --scene_dir output/lab_single_room_s150_v0/

  # Process all scenes under a root
  python postprocess_episodes.py --all --scenes_root output/
"""

import gzip
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import habitat_sim
import magnum as mn
import numpy as np


# Difficulty thresholds (geodesic distance in meters)
# Lab rooms are smaller than homes (5-8m max geodesic vs 10-30m),
# so we use tighter thresholds appropriate for lab-scale navigation.
DIFFICULTY_THRESHOLDS = {
    "easy": (0, 2.0),
    "medium": (2.0, 4.0),
    "hard": (4.0, float("inf")),
}


def classify_difficulty(geodesic_dist: float) -> str:
    for label, (lo, hi) in DIFFICULTY_THRESHOLDS.items():
        if lo <= geodesic_dist < hi:
            return label
    return "hard"


def load_simulator(glb_path: Path, navmesh_path: Path) -> habitat_sim.Simulator:
    """Create a habitat-sim instance with the scene and navmesh loaded."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(glb_path)
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.allow_sliding = False
    sim_cfg.load_semantic_mesh = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # Load navmesh
    sim.pathfinder.load_nav_mesh(str(navmesh_path))

    if not sim.pathfinder.is_loaded:
        print(f"  ⚠  NavMesh not loaded for {navmesh_path}")
        sim.close()
        return None

    return sim


def compute_geodesic(sim: habitat_sim.Simulator,
                     start_pos: list,
                     goal_pos: list) -> float:
    """
    Compute shortest-path geodesic distance from start to goal on the NavMesh.
    Both positions are in habitat-sim world coordinates (Y-up).

    Returns geodesic distance in meters, or -1 if no path found.
    """
    pf = sim.pathfinder
    if not pf.is_loaded:
        return -1.0

    start = np.array(start_pos, dtype=np.float32)
    goal = np.array(goal_pos, dtype=np.float32)

    # Snap start to navmesh
    snap_start = pf.snap_point(start)
    if snap_start is not None:
        start = snap_start

    # Snap goal to navmesh (critical: goals may be on tables, above navmesh)
    snap_goal = pf.snap_point(goal)
    if snap_goal is not None:
        goal = snap_goal
    else:
        # Try snapping at ground level
        goal_at_ground = np.array([goal[0], 0.03, goal[2]], dtype=np.float32)
        snap_goal = pf.snap_point(goal_at_ground)
        if snap_goal is not None:
            goal = snap_goal

    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal

    if pf.find_path(path):
        return float(path.geodesic_distance)
    return -1.0


def process_scene(scene_dir: Path) -> dict:
    """
    Process one scene's episodes: compute geodesic distances and add difficulty.

    Returns stats dict.
    """
    scene_name = scene_dir.name
    glb_path = scene_dir / f"{scene_name}.glb"
    ep_path = scene_dir / f"{scene_name}_episodes.json.gz"

    if not glb_path.exists():
        print(f"  ✗ GLB not found: {glb_path}")
        return None
    if not ep_path.exists():
        print(f"  ✗ Episodes not found: {ep_path}")
        return None

    print(f"\n{'─'*50}")
    print(f"Processing: {scene_name}")
    print(f"  GLB: {glb_path}")

    # Load episodes
    with gzip.open(ep_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"]
    print(f"  Episodes: {len(episodes)}")

    # Create simulator with navmesh
    sim = load_simulator(glb_path, scene_dir / f"{scene_name}.navmesh")
    if sim is None:
        return None

    # Process each episode
    n_computed = 0
    n_failed = 0
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    geo_dists = []

    for ep in episodes:
        start = ep["start_position"]
        goals = ep.get("goals", [])

        if not goals:
            ep["info"]["geodesic_distance"] = -1.0
            ep["difficulty"] = "unknown"
            n_failed += 1
            continue

        # Use first goal (single-goal ObjectNav)
        goal = goals[0]["position"]

        geo = compute_geodesic(sim, start, goal)

        if geo > 0:
            ep["info"]["geodesic_distance"] = round(geo, 2)
            geo_dists.append(geo)
            n_computed += 1

            # Difficulty
            diff = classify_difficulty(geo)
            ep["difficulty"] = diff
            difficulty_counts[diff] += 1
        else:
            # Path not found — use euclidean as fallback
            eucl = ep.get("info", {}).get("euclidean_distance", -1)
            if eucl > 0:
                ep["info"]["geodesic_distance"] = round(eucl, 2)
                diff = classify_difficulty(eucl)
                ep["difficulty"] = diff
                difficulty_counts[diff] += 1
                geo_dists.append(eucl)
                n_computed += 1
            else:
                ep["info"]["geodesic_distance"] = -1.0
                ep["difficulty"] = "unknown"
                n_failed += 1

    sim.close()

    # Write back
    with gzip.open(ep_path, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Also write uncompressed for inspection
    json_path = scene_dir / f"{scene_name}_episodes.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    stats = {
        "scene_name": scene_name,
        "total": len(episodes),
        "computed": n_computed,
        "failed": n_failed,
        "difficulty": difficulty_counts,
        "geo_avg": float(np.mean(geo_dists)) if geo_dists else -1,
        "geo_min": float(np.min(geo_dists)) if geo_dists else -1,
        "geo_max": float(np.max(geo_dists)) if geo_dists else -1,
    }

    print(f"  ✓ Geodesic: {n_computed} computed, {n_failed} failed")
    print(f"    avg={stats['geo_avg']:.1f}m min={stats['geo_min']:.1f}m max={stats['geo_max']:.1f}m")
    print(f"    easy={difficulty_counts['easy']} medium={difficulty_counts['medium']} hard={difficulty_counts['hard']}")

    return stats


def process_all(scenes_root: Path) -> list:
    """Process all scene directories under scenes_root."""
    all_stats = []

    for subdir in sorted(scenes_root.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name == "splits":
            continue

        # Check if it's a scene dir (has .glb + _episodes.json.gz)
        scene_name = subdir.name
        if (subdir / f"{scene_name}_episodes.json.gz").exists():
            stats = process_scene(subdir)
            if stats:
                all_stats.append(stats)

    return all_stats


def print_summary(all_stats: list):
    """Print processing summary."""
    print(f"\n{'='*60}")
    print(f"GEODESIC PROCESSING SUMMARY")
    print(f"{'='*60}")

    total_eps = sum(s["total"] for s in all_stats)
    total_computed = sum(s["computed"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)

    print(f"  Scenes processed: {len(all_stats)}")
    print(f"  Episodes total:   {total_eps}")
    print(f"  Geodesic OK:      {total_computed}")
    print(f"  Failed:           {total_failed}")

    print(f"\n  Per-scene:")
    for s in all_stats:
        diff = s["difficulty"]
        print(f"    {s['scene_name']}: "
              f"E={diff['easy']} M={diff['medium']} H={diff['hard']} "
              f"({s['geo_avg']:.1f}m avg)")

    # Overall difficulty distribution
    all_diff = {"easy": 0, "medium": 0, "hard": 0}
    for s in all_stats:
        for k, v in s["difficulty"].items():
            all_diff[k] = all_diff.get(k, 0) + v
    total_labeled = sum(all_diff.values())
    print(f"\n  Overall difficulty distribution:")
    print(f"    Easy:   {all_diff['easy']:4d} ({100*all_diff['easy']/max(total_labeled,1):.0f}%)")
    print(f"    Medium: {all_diff['medium']:4d} ({100*all_diff['medium']/max(total_labeled,1):.0f}%)")
    print(f"    Hard:   {all_diff['hard']:4d} ({100*all_diff['hard']/max(total_labeled,1):.0f}%)")


# ══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process episodes: compute geodesic distances"
    )
    parser.add_argument("--scene_dir", type=str,
                       help="Process a single scene directory")
    parser.add_argument("--all", action="store_true",
                       help="Process all scenes under --scenes_root")
    parser.add_argument("--scenes_root", type=str,
                       default="/root/gpufree-data/multi_scene/output",
                       help="Root directory containing per-scene output dirs")

    args = parser.parse_args()

    if args.scene_dir:
        stats = process_scene(Path(args.scene_dir))
        if stats:
            all_stats = [stats]
        else:
            sys.exit(1)
    elif args.all:
        all_stats = process_all(Path(args.scenes_root))
    else:
        parser.print_help()
        return

    if not all_stats:
        print("No scenes processed.")
        return

    print_summary(all_stats)

    # Save processing manifest
    manifest_path = Path(args.scenes_root) / "geodesic_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
