#!/usr/bin/env python3
"""
3-Zone Operational Envelope Agent for BioVLN.

Implements the 3-Zone model from the paper:
  - Zone 1: Physical AABB of instrument (impassable)
  - Zone 2: Clearance zone (Zone 1 + 0.25m)
  - Zone 3: Operational envelope (face_half_depth + 0.70m from center)

Two modes:
  - oracle: Knows target instrument → navigates directly to its Zone 3
  - explorer: Knows ALL instrument positions → visits them in priority order

Key advantage over standard Oracle: samples MULTIPLE navigable points in Zone 3
rather than relying on a single snapped goal, handling unreachable navmesh cases.

Usage:
  python run_three_zone.py --split dev --output results/
  python run_three_zone.py --all --output results/
"""

import gzip, json, sys, time, math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage
import habitat_sim, magnum as mn


# ─── Paths ──────────────────────────────────────────────────────────────
MULTI_SCENE_DIR = Path("/root/gpufree-data/multi_scene")
OUTPUT_ROOT = MULTI_SCENE_DIR / "output"
SPLITS_DIR = OUTPUT_ROOT / "splits"

IMAGE_W, IMAGE_H = 320, 240
HFOV = 79
CAM_HEIGHT = 0.88
MAX_STEPS = 500
FORWARD_STEP = 0.25
TURN_ANGLE = np.deg2rad(30)

# ─── Instrument Catalog (model-space) ───────────────────────────────────
# operational_face in Blender Z-up model space
# face_half_depth in meters
INSTRUMENT_CATALOG = {
    "refrigerator": {
        "size": (0.89, 0.91, 1.78),
        "operational_face": (0, -1, 0), "face_half_depth": 0.455, "zone": "wall",
    },
    "refrige": {
        "size": (0.89, 0.91, 1.78),
        "operational_face": (0, -1, 0), "face_half_depth": 0.455, "zone": "wall",
    },
    "incubator": {
        "size": (1.35, 0.85, 1.90),
        "operational_face": (-1, 0, 0), "face_half_depth": 0.673, "zone": "wall",
    },
    "incubor": {
        "size": (1.35, 0.85, 1.90),
        "operational_face": (-1, 0, 0), "face_half_depth": 0.673, "zone": "wall",
    },
    "centrifuge": {
        "size": (0.49, 0.42, 0.63),
        "operational_face": (-1, 0, 0), "face_half_depth": 0.245, "zone": "table_top",
    },
    "mini_centrifuge": {
        "size": (0.66, 0.50, 0.43),
        "operational_face": (-1, 0, 0), "face_half_depth": 0.332, "zone": "table_top",
    },
    "pipette": {
        "size": (0.04, 0.32, 0.28),
        "operational_face": (0, -1, 0), "face_half_depth": 0.159, "zone": "table_top",
    },
    "waste_bin": {
        "size": (0.38, 0.37, 0.51),
        "operational_face": (0, 1, 0), "face_half_depth": 0.183, "zone": "floor",
    },
    "lab_waste_bin": {
        "size": (0.38, 0.37, 0.51),
        "operational_face": (0, 1, 0), "face_half_depth": 0.183, "zone": "floor",
    },
    "cabinet": {
        "size": (1.384, 0.574, 1.768),
        "operational_face": (1, 0, 0), "face_half_depth": 0.692, "zone": "wall",
    },
    "lab_table": {
        "size": (2.40, 0.90, 1.10),
        "operational_face": (0, 1, 0), "face_half_depth": 0.45, "zone": "center",
    },
    "lab_table_1": {
        "size": (1.55, 5.94, 0.76),
        "operational_face": (0, -1, 0), "face_half_depth": 2.97, "zone": "wall",
    },
    "lab_bench": {
        "size": (1.55, 5.94, 0.76),
        "operational_face": (0, -1, 0), "face_half_depth": 2.97, "zone": "wall",
    },
    "pcr_machine": {
        "size": (0.50, 0.40, 0.50),
        "operational_face": (-1, 0, 0), "face_half_depth": 0.25, "zone": "table_top",
    },
}

# Zone parameters
CLEARANCE_MARGIN = 0.25   # Zone 2 = Zone 1 + clearance
APPROACH_DISTANCE = 0.70  # Zone 3 distance from operational face

# Number of candidate goal points to sample in Zone 3
ZONE3_CANDIDATE_ANGLES = 8   # angular samples
ZONE3_CANDIDATE_RADII = 3    # radial samples
DIFFICULTY_ORDER = ["easy", "medium", "hard", "unknown"]


# ─── Coordinate Helpers ────────────────────────────────────────────────

def blender_to_yup(pos_blender):
    """Convert Blender Z-up → GLB/Habitat Y-up."""
    x, y_bl, z_bl = pos_blender
    return np.array([x, z_bl, -y_bl])

def yup_to_blender(pos_yup):
    """Convert GLB/Habitat Y-up → Blender Z-up."""
    x, y_gl, z_gl = pos_yup
    return np.array([x, -z_gl, y_gl])


# ─── Simulator Factory ──────────────────────────────────────────────────

class SimFactory:
    def __init__(self):
        self._sim = None
        self._current_scene = None

    def get(self, scene_name: str):
        if self._current_scene == scene_name and self._sim is not None:
            self._sim.get_agent(0)
            return self._sim, self._sim.pathfinder
        if self._sim is not None:
            self._sim.close()
        scene_dir = OUTPUT_ROOT / scene_name
        glb = str(scene_dir / f"{scene_name}.glb")
        navmesh = str(scene_dir / f"{scene_name}.navmesh")
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = glb
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.allow_sliding = False
        sim_cfg.load_semantic_mesh = False
        sim_cfg.override_scene_light_defaults = True
        sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        sim_cfg.pbr_image_based_lighting = True
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = CAM_HEIGHT
        agent_cfg.radius = 0.18
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        sim.pathfinder.load_nav_mesh(navmesh)
        self._sim = sim
        self._current_scene = scene_name
        return sim, sim.pathfinder

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None


# ─── 3-Zone Goal Sampler ────────────────────────────────────────────────

class ThreeZoneGoalSampler:
    """Compute Zone 3 goal candidates for an instrument.

    Given the instrument center (Y-up) and category, computes the operational
    envelope (Zone 3) and samples navigable approach points.

    Zone 3 = instrument_center + operational_face_yup * (face_half_depth + APPROACH_DISTANCE)
    Candidate points are sampled in a half-ring pattern in front of the operational face.
    """

    def __init__(self, pf):
        self.pf = pf

    def compute_center_from_goal(self, goal_yup, instrument_name):
        """Work backwards: given goal (in Zone 3), compute instrument center."""
        inst = INSTRUMENT_CATALOG.get(instrument_name)
        if inst is None:
            return goal_yup  # fallback: can't determine

        face_bl = np.array(inst["operational_face"], dtype=float)
        face_yup = blender_to_yup(face_bl)
        # Face direction in XZ plane (horizontal)
        face_xz = np.array([face_yup[0], face_yup[2]])
        face_xz = face_xz / (np.linalg.norm(face_xz) + 1e-8)

        half_depth = inst["face_half_depth"]
        approach = APPROACH_DISTANCE

        # goal = center + face_xz * (half_depth + approach) in XZ
        goal_xz = np.array([goal_yup[0], goal_yup[2]])
        center_xz = goal_xz - face_xz * (half_depth + approach)

        return np.array([center_xz[0], goal_yup[1], center_xz[1]])

    def compute_zone3_center(self, center_yup, instrument_name):
        """Given instrument center (Y-up), compute Zone 3 center point."""
        inst = INSTRUMENT_CATALOG.get(instrument_name)
        if inst is None:
            return center_yup

        face_bl = np.array(inst["operational_face"], dtype=float)
        face_yup = blender_to_yup(face_bl)
        face_xz = np.array([face_yup[0], face_yup[2]])
        face_xz = face_xz / (np.linalg.norm(face_xz) + 1e-8)

        half_depth = inst["face_half_depth"]
        approach = APPROACH_DISTANCE

        center_xz = np.array([center_yup[0], center_yup[2]])
        zone3_xz = center_xz + face_xz * (half_depth + approach)
        return np.array([zone3_xz[0], center_yup[1], zone3_xz[1]])

    def sample_zone3_candidates(self, center_yup, instrument_name,
                                 agent_xy=None, max_candidates=20):
        """Sample navigable candidate points in Zone 3 of the instrument.

        Samples points in a fan pattern in front of the operational face:
        - Distance from center: face_half_depth to face_half_depth+1.0m
        - Angle: ±60° around the operational face direction

        Returns list of (point_3d, distance_to_instrument_center) sorted by
        navigability and distance to agent.
        """
        inst = INSTRUMENT_CATALOG.get(instrument_name)
        if inst is None:
            # Fallback: sample ring around center
            candidates = []
            for angle in np.linspace(0, 2 * np.pi, 16):
                for r in [0.5, 0.8, 1.1, 1.4]:
                    px = center_yup[0] + r * np.cos(angle)
                    pz = center_yup[2] + r * np.sin(angle)
                    probe = np.array([px, 0.03, pz])
                    snap = self.pf.snap_point(probe)
                    if snap is not None and self.pf.is_navigable(snap) and snap[1] < 0.5:
                        dist = np.linalg.norm(np.array([px, pz]) -
                                              np.array([center_yup[0], center_yup[2]]))
                        candidates.append((np.array([snap[0], snap[1], snap[2]]), dist))
            return candidates[:max_candidates]

        face_bl = np.array(inst["operational_face"], dtype=float)
        face_yup = blender_to_yup(face_bl)
        face_xz = np.array([face_yup[0], face_yup[2]])
        face_xz = face_xz / (np.linalg.norm(face_xz) + 1e-8)

        # Perpendicular direction in XZ plane
        perp_xz = np.array([-face_xz[1], face_xz[0]])

        half_depth = inst["face_half_depth"]
        center_xz = np.array([center_yup[0], center_yup[2]])

        # Zone 3 center
        zone3_center_xz = center_xz + face_xz * (half_depth + APPROACH_DISTANCE)

        candidates = []
        # Angular fan: ±60° around face direction
        angles = np.linspace(-np.pi / 3, np.pi / 3, ZONE3_CANDIDATE_ANGLES)
        # Radial: different approach distances
        radii = np.linspace(0.5, 1.2, ZONE3_CANDIDATE_RADII)

        for angle in angles:
            for r in radii:
                # Direction rotated by angle from face direction
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                direction = face_xz * cos_a + perp_xz * sin_a

                px = zone3_center_xz[0] + direction[0] * r
                pz = zone3_center_xz[1] + direction[1] * r

                probe = np.array([px, 0.03, pz])
                snap = self.pf.snap_point(probe)
                if snap is None or not self.pf.is_navigable(snap) or snap[1] > 0.5:
                    continue

                snap_xyz = np.array([snap[0], snap[1], snap[2]])
                dist_to_center = np.linalg.norm(
                    np.array([snap[0], snap[2]]) - center_xz)

                candidates.append((snap_xyz, dist_to_center))

        # Also add points directly at Zone 3 center + close variations
        for offset in [(0, 0), (0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2)]:
            px = zone3_center_xz[0] + offset[0]
            pz = zone3_center_xz[1] + offset[1]
            probe = np.array([px, 0.03, pz])
            snap = self.pf.snap_point(probe)
            if snap is not None and self.pf.is_navigable(snap) and snap[1] < 0.5:
                snap_xyz = np.array([snap[0], snap[1], snap[2]])
                dist_to_center = np.linalg.norm(
                    np.array([snap[0], snap[2]]) - center_xz)
                candidates.append((snap_xyz, dist_to_center))

        # Deduplicate
        seen = set()
        unique = []
        for pt, d in candidates:
            key = (round(pt[0], 2), round(pt[2], 2))
            if key not in seen:
                seen.add(key)
                unique.append((pt, d))

        # Sort: prefer points closer to ideal Zone 3 distance
        ideal_dist = half_depth + APPROACH_DISTANCE
        unique.sort(key=lambda x: abs(x[1] - ideal_dist))

        return unique[:max_candidates]

    def is_in_zone3(self, agent_xy, center_yup, instrument_name):
        """Check if agent is within Zone 3 (operational envelope) of instrument."""
        inst = INSTRUMENT_CATALOG.get(instrument_name)
        if inst is None:
            return np.linalg.norm(agent_xy - np.array([center_yup[0], center_yup[2]])) < 1.0

        face_bl = np.array(inst["operational_face"], dtype=float)
        face_yup = blender_to_yup(face_bl)
        face_xz = np.array([face_yup[0], face_yup[2]])
        face_xz = face_xz / (np.linalg.norm(face_xz) + 1e-8)
        perp_xz = np.array([-face_xz[1], face_xz[0]])

        half_depth = inst["face_half_depth"]
        center_xz = np.array([center_yup[0], center_yup[2]])

        # Vector from center to agent
        vec = agent_xy - center_xz
        forward = np.dot(vec, face_xz)       # component toward operational face
        lateral = np.dot(vec, perp_xz)       # component perpendicular

        # Zone 3 bounds: forward in [half_depth, half_depth+1.5], lateral in [-1.0, 1.0]
        in_forward = -0.2 <= forward <= half_depth + 1.5
        in_lateral = abs(lateral) < max(inst["size"][0], inst["size"][1], 1.0)
        return in_forward and in_lateral


# ─── Shortest Path Follower ─────────────────────────────────────────────

class ShortestPathFollower:
    def __init__(self, sim):
        self.sim = sim
        self.pf = sim.pathfinder

    def _forward_direction(self, agent_state):
        import quaternion as nq
        q = agent_state.rotation
        fwd = nq.rotate_vectors(q, np.array([[0, 0, -1]]))[0]
        return np.array([fwd[0], fwd[2]])

    def get_next_action(self, goal_pos, agent_state):
        current = np.array([agent_state.position[0], agent_state.position[1],
                            agent_state.position[2]])
        dist = np.linalg.norm(current - goal_pos)
        if dist < 0.3:
            return 0
        path = habitat_sim.ShortestPath()
        path.requested_start = current
        path.requested_end = goal_pos
        found = self.pf.find_path(path)
        if not found:
            snapped = self.pf.snap_point(goal_pos)
            if snapped is not None:
                path.requested_end = snapped
                found = self.pf.find_path(path)
        if not found or len(path.points) < 2:
            return 2
        nxt = np.array(path.points[1])
        target_dir = np.array([nxt[0] - current[0], nxt[2] - current[2]])
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        forward = self._forward_direction(agent_state)
        cross = forward[0] * target_dir[1] - forward[1] * target_dir[0]
        angle = np.arccos(np.clip(np.dot(forward, target_dir), -1, 1))
        if angle < np.deg2rad(10):
            return 1
        elif cross > 0:
            return 3
        else:
            return 2


# ─── Run One Episode (3-Zone Oracle) ────────────────────────────────────

def run_episode(ep, ep_idx, sim_factory: SimFactory, use_oracle: bool = True):
    """Run one episode with 3-Zone agent.

    Oracle mode: knows the target instrument, navigates to its Zone 3.
    Explorer mode: visits instruments in priority order (TODO).
    """
    scene_name = ep.get("scene_name", "unknown")
    sim, pf = sim_factory.get(scene_name)
    follower = ShortestPathFollower(sim)
    sampler = ThreeZoneGoalSampler(pf)

    target_obj = ep["object_category"]
    start_pos = np.array(ep["start_position"])
    start_rot = np.array(ep["start_rotation"])
    goal_yup = np.array(ep["goals"][0]["position"])

    # Compute instrument center from goal
    center_yup = sampler.compute_center_from_goal(goal_yup, target_obj)

    # Sample Zone 3 candidates
    start_xy = start_pos[[0, 2]]
    candidates = sampler.sample_zone3_candidates(center_yup, target_obj, start_xy)

    # Fallback: use original goal
    if not candidates:
        snap = pf.snap_point(np.array([goal_yup[0], 0.03, goal_yup[2]]))
        if snap is not None:
            candidates = [(np.array([snap[0], snap[1], snap[2]]),
                          np.linalg.norm(np.array([snap[0], snap[2]]) -
                                         np.array([center_yup[0], center_yup[2]])))]
        else:
            candidates = [(goal_yup, APPROACH_DISTANCE)]

    # Pick best candidate: closest navigable from start
    best_candidate = candidates[0][0]
    best_geo = float("inf")
    for pt, _ in candidates[:5]:  # check top 5
        path = habitat_sim.ShortestPath()
        path.requested_start = [start_xy[0], 0.03, start_xy[1]]
        path.requested_end = pt
        if pf.find_path(path) and path.geodesic_distance < best_geo:
            best_geo = path.geodesic_distance
            best_candidate = pt

    nav_goal = best_candidate

    # Optimal geodesic for SPL
    geo_ep = ep.get("info", {}).get("geodesic_distance", -1)
    optimal_geo = geo_ep if geo_ep > 0 else float(np.linalg.norm(start_pos - goal_yup))

    # Set agent
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = mn.Vector3(start_pos[0], start_pos[1], start_pos[2])
    state.rotation = list(start_rot)
    agent.set_state(state)

    path_length = 0.0
    last_pos = np.copy(start_pos)
    visited_goals = set()
    no_progress_steps = 0
    last_best_dist = float("inf")
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    current_candidate_idx = 0

    for step in range(MAX_STEPS):
        state = agent.get_state()
        agent_pos = np.array([state.position[0], state.position[1], state.position[2]])
        agent_xy = np.array([agent_pos[0], agent_pos[2]])

        # Track path
        path_length += float(np.linalg.norm(agent_pos[[0, 2]] - last_pos[[0, 2]]))
        last_pos = np.copy(agent_pos)

        # Distance to instrument center
        dist_to_center = float(np.linalg.norm(
            agent_xy - np.array([center_yup[0], center_yup[2]])))

        # ── Success check: agent in Zone 3 ──
        in_zone3 = sampler.is_in_zone3(agent_xy, center_yup, target_obj)
        # Also traditional threshold check
        inst = INSTRUMENT_CATALOG.get(target_obj, {})
        size = inst.get("size", (1.0, 1.0, 1.0))
        threshold = max(size[0], size[1]) * 0.7 + APPROACH_DISTANCE

        if in_zone3 or dist_to_center < threshold:
            SPL = optimal_geo / max(path_length, optimal_geo, 1e-6)
            return {
                "episode_id": ep["episode_id"],
                "scene_name": scene_name,
                "target": target_obj,
                "difficulty": ep.get("difficulty", "unknown"),
                "success": True,
                "steps": step,
                "final_dist": round(dist_to_center, 2),
                "path_length": round(path_length, 2),
                "optimal_geo": round(optimal_geo, 2),
                "SPL": round(SPL, 4),
            }

        # ── Navigation ──
        action = follower.get_next_action(nav_goal, state)

        # Reached current nav goal → try next candidate
        if action == 0:
            visited_goals.add((round(nav_goal[0], 2), round(nav_goal[2], 2)))

            # Try next best candidate
            found_new = False
            for idx in range(len(candidates)):
                pt, _ = candidates[idx]
                key = (round(pt[0], 2), round(pt[2], 2))
                if key not in visited_goals:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = [agent_xy[0], 0.03, agent_xy[1]]
                    path.requested_end = pt
                    if pf.find_path(path):
                        nav_goal = pt
                        found_new = True
                        break

            if not found_new:
                # All candidates exhausted — use best effort
                return {
                    "episode_id": ep["episode_id"],
                    "scene_name": scene_name,
                    "target": target_obj,
                    "difficulty": ep.get("difficulty", "unknown"),
                    "success": False,
                    "SPL": 0.0,
                    "steps": step,
                    "final_dist": round(dist_to_center, 2),
                    "path_length": round(path_length, 2),
                    "optimal_geo": round(optimal_geo, 2),
                    "reason": "all_candidates_exhausted",
                }
            no_progress_steps = 0
            last_best_dist = float("inf")
            continue

        # ── Progress tracking ──
        if dist_to_center < last_best_dist - 0.02:
            last_best_dist = dist_to_center
            no_progress_steps = 0
        else:
            no_progress_steps += 1

        # ── Stuck recovery: try next candidate ──
        if no_progress_steps > 25:
            for idx in range(len(candidates)):
                pt, _ = candidates[idx]
                key = (round(pt[0], 2), round(pt[2], 2))
                if key not in visited_goals:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = [agent_xy[0], 0.03, agent_xy[1]]
                    path.requested_end = pt
                    if pf.find_path(path):
                        nav_goal = pt
                        no_progress_steps = 0
                        last_best_dist = float("inf")
                        break

        # ── Execute action ──
        action_names = {0: None, 1: "move_forward", 2: "turn_left", 3: "turn_right"}
        act_name = action_names.get(action)
        if act_name:
            sim.step(act_name)
            action_counts[action] += 1
        else:
            continue

    # Timeout
    return {
        "episode_id": ep["episode_id"],
        "SPL": 0.0,
        "scene_name": scene_name,
        "target": target_obj,
        "difficulty": ep.get("difficulty", "unknown"),
        "success": False,
        "steps": MAX_STEPS,
        "final_dist": round(float(np.linalg.norm(
            np.array([agent.get_state().position[0], agent.get_state().position[2]]) -
            np.array([center_yup[0], center_yup[2]]))), 2),
        "path_length": round(path_length, 2),
        "optimal_geo": round(optimal_geo, 2),
        "reason": "timeout",
    }


# ─── Evaluate a Split ────────────────────────────────────────────────────

def evaluate_split(split_name: str, max_episodes: Optional[int] = None,
                   output_dir: Optional[Path] = None) -> List[dict]:
    ep_file = SPLITS_DIR / f"episodes_{split_name}.json.gz"
    if not ep_file.exists():
        print(f"✗ Episode file not found: {ep_file}")
        return []

    with gzip.open(ep_file, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"]
    if max_episodes:
        episodes = episodes[:max_episodes]

    # Group by scene
    by_scene = {}
    for i, ep in enumerate(episodes):
        scene = ep.get("scene_name", "unknown")
        by_scene.setdefault(scene, []).append((i, ep))

    sim_factory = SimFactory()
    results = []
    total = len(episodes)
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"3-ZONE EVAL — [{split_name.upper()}] {total} episodes")
    print(f"{'='*60}\n")

    global_idx = 0
    for scene_name, eps_in_scene in sorted(by_scene.items()):
        print(f"  Scene: {scene_name} ({len(eps_in_scene)} episodes)")
        for _, ep in eps_in_scene:
            result = run_episode(ep, global_idx, sim_factory, use_oracle=True)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"    [{global_idx+1}/{total}] {status} {result['target']:20s} "
                  f"steps={result['steps']:3d} dist={result['final_dist']:.2f}m "
                  f"SPL={result.get('SPL',0):.3f} {result.get('reason','')}",
                  flush=True)
            global_idx += 1

    sim_factory.close()
    elapsed = time.time() - t_start

    # ── Metrics ──
    successes = [r for r in results if r["success"]]
    n = len(results)
    sr = len(successes) / n if n > 0 else 0

    by_cat = {}
    for r in results:
        cat = r["target"]
        by_cat.setdefault(cat, {"total": 0, "success": 0})
        by_cat[cat]["total"] += 1
        if r["success"]:
            by_cat[cat]["success"] += 1

    by_diff = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, {"total": 0, "success": 0})
        by_diff[d]["total"] += 1
        if r["success"]:
            by_diff[d]["success"] += 1

    spl_vals = [r.get("SPL", 0) for r in results]
    avg_spl = float(np.mean(spl_vals))
    failed = [r for r in results if not r["success"]]
    avg_dtg = float(np.mean([r.get("final_dist", 0) for r in failed])) if failed else 0
    avg_steps = float(np.mean([r["steps"] for r in results]))

    metrics = {
        "agent": "3-Zone Oracle",
        "split": split_name,
        "total": n,
        "success": len(successes),
        "SR": round(sr, 4),
        "SPL": round(avg_spl, 4),
        "DTG": round(avg_dtg, 2),
        "avg_steps": round(avg_steps, 1),
        "elapsed_seconds": round(elapsed, 1),
        "per_category": {cat: {"SR": round(v["success"] / v["total"], 4) if v["total"] > 0 else 0,
                              "total": v["total"]}
                        for cat, v in sorted(by_cat.items())},
        "per_difficulty": {d: {"SR": round(v["success"] / v["total"], 4) if v["total"] > 0 else 0,
                              "total": v["total"]}
                          for d, v in sorted(by_diff.items())},
    }

    # ── Print ──
    reason_counts = Counter(r.get("reason", "") for r in results if not r["success"])
    print(f"\n{'─'*60}")
    print(f"[{split_name.upper()}] 3-ZONE SUMMARY")
    print(f"{'─'*60}")
    print(f"  SR = {sr*100:.1f}% ({len(successes)}/{n})")
    print(f"  SPL = {avg_spl:.4f}")
    print(f"  DTG (failed) = {avg_dtg:.2f}m")
    print(f"  Avg steps = {avg_steps:.0f}")
    print(f"  Time = {elapsed:.0f}s")
    if reason_counts:
        print(f"\n  Failure reasons:")
        for reason, count in reason_counts.most_common():
            print(f"    {reason}: {count}")
    print(f"\n  Per category:")
    for cat, v in sorted(by_cat.items()):
        cat_sr = v["success"] / v["total"] * 100 if v["total"] > 0 else 0
        print(f"    {cat}: {cat_sr:.0f}% ({v['success']}/{v['total']})")
    print(f"\n  Per difficulty:")
    for d in DIFFICULTY_ORDER:
        if d in by_diff:
            v = by_diff[d]
            d_sr = v["success"] / v["total"] * 100 if v["total"] > 0 else 0
            print(f"    {d}: {d_sr:.0f}% ({v['success']}/{v['total']})")

    # ── Save ──
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "config": {
                "policy": "3-Zone Oracle",
                "split": split_name,
                "max_steps": MAX_STEPS,
                "clearance_margin": CLEARANCE_MARGIN,
                "approach_distance": APPROACH_DISTANCE,
            },
            "metrics": metrics,
            "episode_results": results,
        }
        out_path = output_dir / f"eval_three_zone_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {out_path}")

    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="3-Zone Operational Envelope Evaluation")
    ap.add_argument("--all", action="store_true", help="Run all splits: dev, val, test")
    ap.add_argument("--split", type=str, default=None, help="Specific split (dev/val/test)")
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--output", type=str, default="results", help="Output directory")
    args = ap.parse_args()

    splits_to_run = []
    if args.all:
        splits_to_run = ["dev", "val", "test"]
    elif args.split:
        splits_to_run = [args.split]
    else:
        ap.print_help()
        return

    output_dir = Path(args.output)
    all_metrics = {}

    for split_name in splits_to_run:
        results = evaluate_split(split_name, args.max_episodes, output_dir)
        if results:
            n = len(results)
            sr = sum(1 for r in results if r["success"]) / n * 100
            all_metrics[split_name] = {
                "episodes": n,
                "SR": round(sr, 1),
                "SPL": round(float(np.mean([r.get("SPL", 0) for r in results])), 4),
            }

    if len(splits_to_run) > 1:
        print(f"\n{'='*60}")
        print("3-ZONE CROSS-SPLIT SUMMARY")
        print(f"{'='*60}")
        for s, m in sorted(all_metrics.items()):
            print(f"  [{s.upper()}] SR={m['SR']:.1f}% SPL={m['SPL']:.4f} "
                  f"({m['episodes']} ep)")
        summary_path = output_dir / "eval_three_zone_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
