"""Equal-budget coordinate-oracle geometry ablation for BioVLN-Revision-20.

Four primary variants: center vs operational-face candidate regions, each with
one or five candidate visits. All receive the true target instance, scene
geometry and NavMesh, so these are *coordinate oracle* methods, not blind
ObjectNav or VLM baselines. Candidate visits use the same oracle pose feedback;
this isolates sampling-region and retry-budget effects under privileged input.

Every row logs the explicit STOP decision, terminal pose, action trajectory,
body clearance, legacy point success and strict operational-pose success.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import time

if not os.environ.get("DISPLAY") and not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import habitat_sim
import magnum as mn
import numpy as np

from audit_navmesh_pose_v3 import geodesic, load_episodes, snap_valid_candidates
from operational_pose_v3 import (PoseConfig, body_clearance, habitat_xy,
                              match_target, norm, region_for, scene_from_spec,
                              score_terminal, sub, trajectory_safety)


LEGACY_RADII = {
    "refrigerator": 1.0, "incubator": 1.0, "cabinet": 1.0,
    "centrifuge": 1.0, "mini_centrifuge": 1.0,
    "pipette": 0.8, "lab_waste_bin": 0.8,
}
ACTION_NAMES = {0: "STOP", 1: "move_forward", 2: "turn_left", 3: "turn_right"}


def heading_from_rotation(rotation) -> tuple[float, float]:
    """Habitat local -Z forward, returned in Blender horizontal XY."""
    x, y, z, w = (float(rotation.x), float(rotation.y),
                  float(rotation.z), float(rotation.w))
    return (-2.0 * (x * z + y * w), 1.0 - 2.0 * (x * x + y * y))


def state_values(agent) -> tuple[np.ndarray, tuple[float, float]]:
    state = agent.get_state()
    return np.array([float(v) for v in state.position]), heading_from_rotation(state.rotation)


def obstacle_edge_clearance(pathfinder, xyz: np.ndarray) -> float:
    if not pathfinder.is_navigable(xyz):
        return -math.inf
    return float(pathfinder.distance_to_closest_obstacle(xyz))


def acceptable_nav_point(pathfinder, scene, xy, cfg: PoseConfig) -> np.ndarray | None:
    if body_clearance(xy, scene, cfg) < -1e-6:
        return None
    probe = np.array([xy[0], 0.03, -xy[1]], dtype=float)
    snap = pathfinder.snap_point(probe)
    if snap is None:
        return None
    snap = np.asarray(snap, dtype=float)
    if not np.all(np.isfinite(snap)) or not -0.2 <= snap[1] < 0.5:
        return None
    if float(np.linalg.norm(snap[[0, 2]] - probe[[0, 2]])) > cfg.candidate_spacing_m + 1e-6:
        return None
    if obstacle_edge_clearance(pathfinder, snap) < cfg.robot_radius_m - cfg.navmesh_baked_agent_radius_m - 1e-6:
        return None
    if body_clearance(habitat_xy(snap), scene, cfg) < -1e-6:
        return None
    return snap


def center_candidates(pathfinder, scene, target, cfg: PoseConfig) -> list[np.ndarray]:
    radii = (0.35, 0.55, 0.75, 0.95, 1.15)
    angles = (0, 45, 90, 135, 180, 225, 270, 315)
    found = {}
    for radius in radii:
        for angle in angles:
            a = math.radians(angle)
            xy = (target.center[0] + radius * math.cos(a),
                  target.center[1] + radius * math.sin(a))
            nav = acceptable_nav_point(pathfinder, scene, xy, cfg)
            if nav is not None:
                found[(round(float(nav[0]), 3), round(float(nav[2]), 3))] = nav
    return list(found.values())


def make_sim(scene_name: str, scenes_root: Path, cfg: PoseConfig,
             forward_m: float, turn_deg: float, navmesh_root: Path | None):
    scene_dir = scenes_root / scene_name
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_dir / f"{scene_name}.glb")
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.allow_sliding = False
    sim_cfg.load_semantic_mesh = False
    sim_cfg.pbr_image_based_lighting = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = cfg.robot_radius_m
    agent_cfg.sensor_specifications = []
    agent_cfg.action_space["move_forward"].actuation.amount = forward_m
    agent_cfg.action_space["turn_left"].actuation.amount = turn_deg
    agent_cfg.action_space["turn_right"].actuation.amount = turn_deg
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    navmesh_path = ((navmesh_root / f"{scene_name}.navmesh")
                    if navmesh_root else scene_dir / f"{scene_name}.navmesh")
    if not sim.pathfinder.load_nav_mesh(str(navmesh_path)):
        sim.close()
        raise RuntimeError(f"NavMesh failed for {scene_name}")
    return sim


def action_toward(current_xyz: np.ndarray, heading_xy: tuple[float, float],
                  destination_xyz: np.ndarray, angle_margin_deg: float) -> int:
    # Blender Y = -Habitat Z; convert heading back to Habitat XZ.
    forward = np.array([heading_xy[0], -heading_xy[1]], dtype=float)
    direction = np.array([destination_xyz[0] - current_xyz[0],
                          destination_xyz[2] - current_xyz[2]], dtype=float)
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return 0
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    direction /= length
    cos_angle = float(np.clip(np.dot(forward, direction), -1.0, 1.0))
    angle = math.degrees(math.acos(cos_angle))
    if angle <= angle_margin_deg:
        return 1
    cross = forward[0] * direction[1] - forward[1] * direction[0]
    return 3 if cross > 0 else 2


def nearest_path_step(pathfinder, current_xyz: np.ndarray,
                      goal_xyz: np.ndarray) -> np.ndarray | None:
    path = habitat_sim.ShortestPath()
    path.requested_start = current_xyz
    path.requested_end = goal_xyz
    if not pathfinder.find_path(path) or not path.points:
        return None
    if len(path.points) == 1:
        return goal_xyz
    return np.asarray(path.points[1], dtype=float)


def evaluated_candidates(pathfinder, scene, target, region,
                         ep: dict, cfg: PoseConfig, mode: str) -> list[tuple[float, np.ndarray]]:
    if mode == "center":
        pool = center_candidates(pathfinder, scene, target, cfg)
    elif mode == "face":
        pool, _ = snap_valid_candidates(pathfinder, scene, region, cfg)
        # The controller has finite 0.10 m movement steps. Keep planned goals
        # away from region edges so a near-goal STOP remains inside the zone.
        margin = cfg.candidate_spacing_m
        pool = [p for p in pool
                if (cfg.standoff_min_m + margin <= region.coordinates(habitat_xy(p))[0]
                    <= cfg.standoff_max_m - margin
                    and abs(region.coordinates(habitat_xy(p))[1])
                    <= region.half_width - cfg.robot_radius_m - margin)]
    else:
        raise ValueError(mode)
    start = np.asarray(ep["start_position"], dtype=float)
    ranked = []
    for candidate in pool:
        distance = geodesic(pathfinder, start, candidate)
        if distance is not None:
            ranked.append((distance, candidate))
    ranked.sort(key=lambda x: (x[0], float(np.linalg.norm(x[1][[0, 2]] - start[[0, 2]])),
                               float(x[1][0]), float(x[1][2])))
    return ranked


def run_episode(ep: dict, scene, sim, cfg: PoseConfig, *, mode: str, k: int,
                max_steps: int, pose_geo: float | None) -> dict:
    pathfinder = sim.pathfinder
    target = match_target(scene, ep)
    legacy_goal = habitat_xy(ep["goals"][0]["position"])
    region = region_for(scene, target, legacy_goal, cfg)
    ranked = evaluated_candidates(pathfinder, scene, target, region, ep, cfg, mode)
    candidates = [xyz for _, xyz in ranked[:k]]
    agent = sim.get_agent(0)
    start = np.asarray(ep["start_position"], dtype=float)
    initial_state = habitat_sim.AgentState()
    initial_state.position = mn.Vector3(float(start[0]), float(start[1]), float(start[2]))
    initial_state.rotation = list(ep["start_rotation"])
    agent.set_state(initial_state)

    trajectory = []
    positions = []
    path_length = 0.0
    last_xyz = start.copy()
    candidate_idx = 0
    no_progress = 0
    best_candidate_dist = math.inf
    attempts = []
    stopped = False
    stop_reason = "step_budget_exhausted"

    for step in range(max_steps):
        xyz, heading = state_values(agent)
        point = habitat_xy(xyz)
        path_length += float(np.linalg.norm(xyz[[0, 2]] - last_xyz[[0, 2]]))
        last_xyz = xyz.copy()
        edge = obstacle_edge_clearance(pathfinder, xyz)
        positions.append(point)
        trajectory.append({
            "step": step, "action": "OBSERVE", "position_xyz": [round(float(v), 4) for v in xyz],
            "heading_blender_xy": [round(float(v), 5) for v in heading],
            "body_clearance_m": round(body_clearance(point, scene, cfg), 4),
            "navmesh_edge_clearance_m": round(edge, 4) if math.isfinite(edge) else None,
        })

        if not candidates or candidate_idx >= len(candidates):
            stopped = True
            stop_reason = "no_more_candidates"
            trajectory.append({"step": step, "action": "STOP"})
            break
        candidate = candidates[candidate_idx]
        distance_to_candidate = float(np.linalg.norm(candidate[[0, 2]] - xyz[[0, 2]]))
        if distance_to_candidate <= 0.08:
            face_destination = np.array([target.center[0], xyz[1], -target.center[1]], dtype=float)
            turn_or_move = action_toward(xyz, heading, face_destination, 10.0)
            if turn_or_move == 1 or turn_or_move == 0:
                hypothetical = score_terminal(
                    point, heading, True, region, scene, cfg,
                    LEGACY_RADII.get(target.category, 1.0), edge)
                attempts.append({"candidate_index": candidate_idx,
                                 "pose_valid": hypothetical["pose_success"],
                                 "failure_reasons": hypothetical["failure_reasons"]})
                if hypothetical["pose_success"] or candidate_idx + 1 >= len(candidates):
                    stopped = True
                    stop_reason = "valid_pose" if hypothetical["pose_success"] else "candidates_exhausted"
                    trajectory.append({"step": step, "action": "STOP"})
                    break
                candidate_idx += 1
                no_progress = 0
                best_candidate_dist = math.inf
                continue
            action = turn_or_move
        else:
            next_step = nearest_path_step(pathfinder, xyz, candidate)
            if next_step is None:
                attempts.append({"candidate_index": candidate_idx,
                                 "pose_valid": False, "failure_reasons": ["no_path_from_current"]})
                candidate_idx += 1
                no_progress = 0
                best_candidate_dist = math.inf
                continue
            action = action_toward(xyz, heading, next_step, 12.0)
            if action == 0:
                action = 1

        if distance_to_candidate < best_candidate_dist - 0.02:
            best_candidate_dist = distance_to_candidate
            no_progress = 0
        else:
            no_progress += 1
        if no_progress >= 40:
            attempts.append({"candidate_index": candidate_idx,
                             "pose_valid": False, "failure_reasons": ["navigation_stuck"]})
            candidate_idx += 1
            no_progress = 0
            best_candidate_dist = math.inf
            continue
        sim.step(ACTION_NAMES[action])
        trajectory.append({"step": step, "action": ACTION_NAMES[action]})

    final_xyz, final_heading = state_values(agent)
    final_point = habitat_xy(final_xyz)
    if not positions or positions[-1] != final_point:
        positions.append(final_point)
    path_length += float(np.linalg.norm(final_xyz[[0, 2]] - last_xyz[[0, 2]]))
    edge = obstacle_edge_clearance(pathfinder, final_xyz)
    outcome = score_terminal(final_point, final_heading, stopped, region, scene, cfg,
                             LEGACY_RADII.get(target.category, 1.0), edge)
    safety = trajectory_safety(positions, scene, cfg)
    spl = 0.0
    if outcome["pose_success"] and pose_geo is not None:
        spl = min(1.0, pose_geo / max(path_length, pose_geo, 1e-6))
    return {
        "episode_id": ep["episode_id"], "scene_name": ep["scene_name"],
        "split": ep["_revision_split"], "target": target.category,
        "target_label": target.label, "method": f"coordinate_oracle_{mode}_k{k}",
        "annotation_version": "measured-geometry-v3-20261012",
        "information_access": "true_target_instance,scene_geometry,full_navmesh,pose_feedback_at_candidates",
        "candidate_pool_reachable": len(ranked), "candidate_budget": k,
        "candidate_attempts": attempts, "stop_reason": stop_reason,
        "did_stop": stopped, "steps": len([x for x in trajectory if x["action"] in ACTION_NAMES.values()]) - int(stopped),
        "final_position_xyz": [round(float(v), 4) for v in final_xyz],
        "final_heading_blender_xy": [round(float(v), 5) for v in final_heading],
        "path_length_m": round(path_length, 4),
        "optimal_geo_to_pose_grid_m": pose_geo,
        "SPL_pose_grid": round(spl, 4),
        **outcome, **safety,
        "trajectory": trajectory,
    }


def load_pose_geodesics(path: Path) -> dict[tuple[str, str], float | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(x["scene_name"], x["episode_id"]): x["optimal_geo_to_pose_grid_m"]
            for x in data["per_episode"]}


def summarize(rows: list[dict], expected: int) -> dict:
    scene_counts = defaultdict(lambda: {"total": 0, "pose_success": 0, "point_success": 0})
    for row in rows:
        item = scene_counts[row["scene_name"]]
        item["total"] += 1
        item["pose_success"] += int(row["pose_success"])
        item["point_success"] += int(row["point_success"])
    return {
        "dataset": "BioVLN-Revision-20", "method": rows[0]["method"] if rows else None,
        "expected_episodes": expected, "completed_episodes": len(rows),
        "pose_SR": sum(x["pose_success"] for x in rows) / len(rows) if rows else 0,
        "legacy_point_SR": sum(x["point_success"] for x in rows) / len(rows) if rows else 0,
        "pose_SPL_grid": sum(x["SPL_pose_grid"] for x in rows) / len(rows) if rows else 0,
        "MCR_body_mean_m": sum(x["MCR_body_m"] for x in rows) / len(rows) if rows else None,
        "VRT_body_025_mean": sum(x["VRT_body_025"] for x in rows) / len(rows) if rows else None,
        "VRT_body_050_mean": sum(x["VRT_body_050"] for x in rows) / len(rows) if rows else None,
        "per_scene": dict(sorted(scene_counts.items())),
        "failure_reasons": dict(Counter(reason for row in rows for reason in row["failure_reasons"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--specs-root", type=Path, required=True, help="Actual-mesh measured-v3 annotation tree")
    parser.add_argument("--navmesh-root", type=Path)
    parser.add_argument("--navmesh-baked-radius", type=float, default=0.10)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--pose-geodesics", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--region", choices=["center", "face"], required=True)
    parser.add_argument("--k", type=int, choices=[1, 5], required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--forward-m", type=float, default=0.10)
    parser.add_argument("--turn-deg", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = PoseConfig(navmesh_baked_agent_radius_m=args.navmesh_baked_radius)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    episodes = load_episodes(args.splits_dir, splits)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    pose_geodesics = load_pose_geodesics(args.pose_geodesics)
    completed = []
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.append(json.loads(line))
    method = f"coordinate_oracle_{args.region}_k{args.k}"
    if any(x["method"] != method for x in completed):
        raise ValueError("checkpoint method mismatch")
    completed_keys = {(x["scene_name"], x["episode_id"]) for x in completed}
    if len(completed_keys) != len(completed):
        raise ValueError("duplicate episode in checkpoint")
    source_keys = {(x["scene_name"], x["episode_id"]) for x in episodes}
    if not completed_keys <= source_keys:
        raise ValueError("checkpoint has episodes outside the source split")
    pending_by_scene = defaultdict(list)
    for ep in episodes:
        key = ep["scene_name"], ep["episode_id"]
        if key not in completed_keys:
            pending_by_scene[ep["scene_name"]].append(ep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with args.output.open("a", encoding="utf-8") as sink:
        for scene_name, scene_episodes in sorted(pending_by_scene.items()):
            spec_path = args.specs_root / scene_name / f"{scene_name}_spec.json"
            scene = scene_from_spec(json.loads(spec_path.read_text(encoding="utf-8")))
            sim = make_sim(scene_name, args.scenes_root, cfg, args.forward_m,
                           args.turn_deg, args.navmesh_root)
            try:
                for ep in scene_episodes:
                    key = scene_name, ep["episode_id"]
                    row = run_episode(ep, scene, sim, cfg,
                                      mode=args.region, k=args.k,
                                      max_steps=args.max_steps,
                                      pose_geo=pose_geodesics.get(key))
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    completed.append(row)
                    print(f"{len(completed)}/{len(episodes)} {scene_name} {ep['episode_id']} "
                          f"pose={int(row['pose_success'])} point={int(row['point_success'])} "
                          f"steps={row['steps']}", flush=True)
            finally:
                sim.close()
    summary = summarize(completed, len(episodes))
    summary["elapsed_this_run_seconds"] = round(time.time() - started, 1)
    summary["runner_config"] = {"max_steps": args.max_steps,
                                "forward_m": args.forward_m, "turn_deg": args.turn_deg,
                                "candidate_budget": args.k,
                                "robot_radius_m": cfg.robot_radius_m,
                                "navmesh_baked_radius_m": cfg.navmesh_baked_agent_radius_m,
                                "navmesh_root": str(args.navmesh_root) if args.navmesh_root else str(args.scenes_root)}
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("completed_episodes", "pose_SR", "legacy_point_SR", "pose_SPL_grid", "elapsed_this_run_seconds")}, indent=2))


if __name__ == "__main__":
    main()
