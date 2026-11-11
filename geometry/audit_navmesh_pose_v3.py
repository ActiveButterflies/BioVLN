"""Check every revised episode for a reachable, collision-free terminal pose.

Requires Habitat-Sim only for PathFinder and ShortestPath. This is a dataset
feasibility audit, not a navigation-policy score. The reported shortest path
is the minimum over a deterministic 0.10 m grid and is labelled approximate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np

from operational_pose_v3 import (PoseConfig, body_clearance, habitat_xy,
                              match_target, region_for, scene_from_spec)


def load_episodes(splits_dir: Path, splits: list[str]) -> list[dict]:
    result = []
    for split in splits:
        path = splits_dir / f"episodes_procedural_full_{split}_clean.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as source:
            rows = json.load(source)["episodes"]
        for row in rows:
            item = dict(row)
            item["_revision_split"] = split
            result.append(item)
    return result


def snap_valid_candidates(pathfinder, scene, region, cfg: PoseConfig) -> tuple[list[np.ndarray], dict]:
    accepted = []
    rejection = Counter()
    for xy in region.sample_centers(cfg):
        if body_clearance(xy, scene, cfg) < -1e-6:
            rejection["asset_collision"] += 1
            continue
        probe = np.array([xy[0], 0.03, -xy[1]], dtype=float)
        snap = pathfinder.snap_point(probe)
        if snap is None or not np.all(np.isfinite(snap)):
            rejection["no_navmesh_snap"] += 1
            continue
        snap = np.asarray(snap, dtype=float)
        if snap[1] < -0.2 or snap[1] >= 0.5:
            rejection["not_floor_level"] += 1
            continue
        if float(np.linalg.norm(snap[[0, 2]] - probe[[0, 2]])) > cfg.candidate_spacing_m + 1e-6:
            rejection["snap_too_far"] += 1
            continue
        if not pathfinder.is_navigable(snap):
            rejection["not_navigable"] += 1
            continue
        edge_clearance = float(pathfinder.distance_to_closest_obstacle(snap))
        additional_radius = max(0.0, cfg.robot_radius_m - cfg.navmesh_baked_agent_radius_m)
        if not math.isfinite(edge_clearance) or edge_clearance < additional_radius - 1e-6:
            rejection["navmesh_boundary_clearance"] += 1
            continue
        actual = habitat_xy(snap)
        forward, lateral = region.coordinates(actual)
        if not (cfg.standoff_min_m - 1e-6 <= forward <= cfg.standoff_max_m + 1e-6
                and abs(lateral) <= region.half_width - cfg.robot_radius_m + 1e-6):
            rejection["snapped_outside_region"] += 1
            continue
        if body_clearance(actual, scene, cfg) < -1e-6:
            rejection["snapped_into_asset"] += 1
            continue
        accepted.append(snap)
    unique = {}
    for snap in accepted:
        key = (round(float(snap[0]), 3), round(float(snap[2]), 3))
        unique[key] = snap
    return list(unique.values()), dict(rejection)


def geodesic(pathfinder, start: np.ndarray, goal: np.ndarray) -> float | None:
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal
    if not pathfinder.find_path(path):
        return None
    distance = float(path.geodesic_distance)
    return distance if math.isfinite(distance) and distance >= 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--navmesh-root", type=Path,
                        help="Separate directory containing <scene>.navmesh files")
    parser.add_argument("--navmesh-baked-radius", type=float, default=0.10)
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    episodes = load_episodes(args.splits_dir, splits)
    by_scene = defaultdict(list)
    for ep in episodes:
        by_scene[ep["scene_name"]].append(ep)

    cfg = PoseConfig(navmesh_baked_agent_radius_m=args.navmesh_baked_radius)
    rows = []
    target_records = []
    for scene_name, scene_episodes in sorted(by_scene.items()):
        scene_dir = args.scenes_root / scene_name
        spec = json.loads((scene_dir / f"{scene_name}_spec.json").read_text(encoding="utf-8"))
        scene = scene_from_spec(spec)
        pathfinder = habitat_sim.PathFinder()
        navmesh_file = ((args.navmesh_root / f"{scene_name}.navmesh")
                        if args.navmesh_root else scene_dir / f"{scene_name}.navmesh")
        if not pathfinder.load_nav_mesh(str(navmesh_file)):
            raise RuntimeError(f"could not load navmesh: {navmesh_file}")
        candidates_by_target = {}
        regions_by_target = {}
        for ep in scene_episodes:
            target = match_target(scene, ep)
            if target.label in candidates_by_target:
                continue
            goal = habitat_xy(ep["goals"][0]["position"])
            region = region_for(scene, target, goal, cfg)
            candidates, rejected = snap_valid_candidates(pathfinder, scene, region, cfg)
            candidates_by_target[target.label] = candidates
            regions_by_target[target.label] = region
            target_records.append({
                "scene_name": scene_name, "target_label": target.label,
                "target_category": target.category,
                "surface_source": region.surface_source,
                "valid_navmesh_candidates": len(candidates),
                "rejection_counts": rejected,
            })
        for ep in scene_episodes:
            target = match_target(scene, ep)
            candidates = candidates_by_target[target.label]
            region = regions_by_target[target.label]
            start = np.asarray(ep["start_position"], dtype=float)
            distances = []
            canonical_distances = []
            for candidate in candidates:
                distance = geodesic(pathfinder, start, candidate)
                if distance is not None:
                    distances.append((distance, candidate))
                    forward, lateral = region.coordinates(habitat_xy(candidate))
                    interior_margin = 0.01
                    if (cfg.standoff_min_m + interior_margin <= forward
                            <= cfg.standoff_max_m - interior_margin
                            and abs(lateral) <= region.half_width - cfg.robot_radius_m - interior_margin):
                        canonical_distances.append((distance, candidate))
            best = min(distances, key=lambda x: x[0]) if distances else None
            canonical = min(canonical_distances, key=lambda x: x[0]) if canonical_distances else None
            rows.append({
                "split": ep["_revision_split"],
                "scene_name": scene_name,
                "episode_id": ep["episode_id"],
                "target_label": target.label,
                "valid_navmesh_candidates": len(candidates),
                "reachable_candidate_count": len(distances),
                "pose_feasible": best is not None,
                "optimal_geo_to_pose_grid_m": round(best[0], 4) if best else None,
                "closest_pose_grid_position_xyz": [float(x) for x in best[1]] if best else None,
                "canonical_interior_pose_feasible": canonical is not None,
                "canonical_interior_pose_position_xyz": [float(x) for x in canonical[1]] if canonical else None,
                "canonical_interior_geo_m": round(canonical[0], 4) if canonical else None,
            })
        print(f"{scene_name}: {len(scene_episodes)} episodes, "
              f"{sum(x['pose_feasible'] for x in rows[-len(scene_episodes):])} feasible",
              flush=True)
    summary = {
        "dataset_name": "BioVLN-Revision-20",
        "annotation_version": "measured-geometry-v3-20261012",
        "source_split": args.split,
        "config": asdict(cfg),
        "shortest_path_note": "minimum of finite NavMesh geodesics to 0.10 m pose grid; orientation is in-place rotation",
        "navmesh_root": str(args.navmesh_root) if args.navmesh_root else str(args.scenes_root),
        "navmesh_margin_note": f"NavMesh baked at {cfg.navmesh_baked_agent_radius_m:.2f} m radius; "
                               f"candidate boundary margin {max(0.0, cfg.robot_radius_m - cfg.navmesh_baked_agent_radius_m):.2f} m",
        "episode_count": len(rows),
        "pose_feasible_episodes": sum(x["pose_feasible"] for x in rows),
        "canonical_interior_pose_feasible_episodes": sum(x["canonical_interior_pose_feasible"] for x in rows),
        "canonical_interior_margin_m": 0.01,
        "target_count": len(target_records),
        "targets_with_valid_navmesh_candidates": sum(x["valid_navmesh_candidates"] > 0 for x in target_records),
        "targets": target_records,
        "per_episode": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}: {summary['pose_feasible_episodes']}/{len(rows)} feasible")


if __name__ == "__main__":
    main()
