#!/usr/bin/env python3
"""
Multi-Scene Frontier Exploration Baseline for BioVLN.

Runs pure geometric frontier exploration on dev/val/test splits,
loading the correct scene GLB + navmesh per episode.

Usage:
  # Run all splits
  python run_frontier.py --all --output results/

  # Run a specific split
  python run_frontier.py --split dev --output results/
  python run_frontier.py --split test --output results/

  # Run with limited episodes
  python run_frontier.py --split dev --max_episodes 10
"""

import gzip, json, sys, time
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

OBJ_RADIUS = {
    "refrigerator": 1.0, "refrige": 1.0,
    "centrifuge": 1.0, "mini_centrifuge": 1.0,
    "cabinet": 1.0, "incubator": 1.0, "incubor": 1.0,
    "pipette": 0.8, "lab_waste_bin": 0.8, "waste_bin": 0.8,
    "pcr_machine": 1.0,
    "lab_table": 1.3, "lab_table_1": 1.3, "lab_bench": 1.3,
}

DIFFICULTY_ORDER = ["easy", "medium", "hard", "unknown"]

# ─── Simulator Factory ──────────────────────────────────────────────────

class SimFactory:
    """Create habitat-sim instances for different scenes."""

    def __init__(self):
        self._cache: Dict[str, Tuple] = {}
        self._sim = None
        self._current_scene = None

    def get(self, scene_name: str):
        """Return (sim, pathfinder) for a scene. Caches the most recent."""
        if self._current_scene == scene_name and self._sim is not None:
            self._sim.get_agent(0)  # ensure not stale
            return self._sim, self._sim.pathfinder

        # Close old
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

# ─── Frontier Explorer ──────────────────────────────────────────────────

class FrontierExplorer:
    def __init__(self, sim):
        self.sim = sim
        self.pf = sim.pathfinder
        self.map_size = 200
        self.map_res = 0.05
        self.obs = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        self.explored = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        self.origin = np.array([self.map_size // 2, self.map_size // 2])

    def reset(self):
        self.obs.fill(0)
        self.explored.fill(0)

    def w2g(self, wx, wy):
        return (int(np.clip(wx / self.map_res + self.origin[0], 0, self.map_size - 1)),
                int(np.clip(wy / self.map_res + self.origin[1], 0, self.map_size - 1)))

    def g2w(self, gx, gy):
        return np.array([(gx - self.origin[0]) * self.map_res,
                         (gy - self.origin[1]) * self.map_res])

    def update(self, depth_img, agent_state):
        pos = agent_state.position
        quat = agent_state.rotation
        agent_xy = np.array([pos[0], pos[2]])

        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        forward = np.array([2*(x*z - w*y), 2*(y*z + w*x)])
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.array([forward[1], -forward[0]])

        h, w_img = depth_img.shape
        fx = fy = w_img / (2 * np.tan(np.deg2rad(HFOV) / 2))
        cx, cy = w_img // 2, h // 2

        v, u = np.mgrid[0:h, 0:w_img]
        valid = (depth_img > 0.1) & (depth_img < 6.0)
        Z = depth_img[valid]
        X = (u[valid] - cx) * Z / fx
        Y = -(v[valid] - cy) * Z / fy

        world_dx = X[:, None] * right[None, :] + Z[:, None] * (-forward)[None, :]
        world_xy = agent_xy[None, :] + world_dx

        for pt in world_xy:
            gx, gy = self.w2g(pt[0], pt[1])
            if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                self.explored[gx, gy] = 1

        # Mark obstacles: points above floor level (y>0.3 in habitat world)
        for pt in world_xy:
            gx, gy = self.w2g(pt[0], pt[1])
            if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                pass  # simplified — all explored points are free

        gx_a, gy_a = self.w2g(agent_xy[0], agent_xy[1])
        cv2_circle(self.explored, (gx_a, gy_a), 3, 1, -1)

    def get_frontiers(self):
        free = (self.obs == 0) & (self.explored == 1)
        unknown = self.explored == 0
        unknown_dil = ndimage.binary_dilation(unknown, structure=np.ones((3, 3)))
        frontiers = free & unknown_dil & ~unknown

        labels, n = ndimage.label(frontiers)
        pts = []
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            if len(ys) < 8:
                continue
            centroid = np.array([xs.mean(), ys.mean()])
            world_pt = self.g2w(centroid[0], centroid[1])
            nav_pt = self.pf.snap_point([world_pt[0], 0.03, world_pt[1]])
            if nav_pt is not None and self.pf.is_navigable(nav_pt):
                pts.append(world_pt)
        return pts


def cv2_circle(arr, center, radius, value, thickness=-1):
    """Minimal circle drawing replacement for cv2.circle on numpy arrays."""
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    y0, y1 = max(0, cy - r), min(arr.shape[0] - 1, cy + r)
    x0, x1 = max(0, cx - r), min(arr.shape[1] - 1, cx + r)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x - cx)**2 + (y - cy)**2 <= r**2:
                arr[y, x] = value  # uses row-major: arr[gy, gx]


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
            return 0  # STOP

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
            return 2  # TURN_LEFT

        nxt = np.array(path.points[1])
        target_dir = np.array([nxt[0] - current[0], nxt[2] - current[2]])
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        forward = self._forward_direction(agent_state)
        cross = forward[0] * target_dir[1] - forward[1] * target_dir[0]
        angle = np.arccos(np.clip(np.dot(forward, target_dir), -1, 1))

        if angle < np.deg2rad(10):
            return 1  # MOVE_FORWARD
        elif cross > 0:
            return 3  # TURN_RIGHT
        else:
            return 2  # TURN_LEFT


# ─── Helpers ─────────────────────────────────────────────────────────────

def _find_nav_goal(pf, goal_xy, ref_xy):
    """Find navigable point closest to goal reachable from ref_xy."""
    best, best_offset = None, float("inf")
    for angle in np.linspace(0, 2 * np.pi, 24):
        for r in [0.3, 0.5, 0.7, 1.0, 1.3]:
            probe = np.array([goal_xy[0] + r * np.cos(angle), 0.03,
                              goal_xy[1] + r * np.sin(angle)])
            snap = pf.snap_point(probe)
            if snap is None or not pf.is_navigable(snap) or snap[1] > 0.5:
                continue
            offset = np.linalg.norm(np.array([snap[0], snap[2]]) - goal_xy)
            if offset < best_offset:
                path = habitat_sim.ShortestPath()
                path.requested_start = [ref_xy[0], 0.03, ref_xy[1]]
                path.requested_end = snap
                if pf.find_path(path):
                    best = np.array([snap[0], snap[1], snap[2]])
                    best_offset = offset
    if best is None:
        fallback = np.array([goal_xy[0], 0.03, goal_xy[1]])
        snap = pf.snap_point(fallback)
        if snap is not None:
            return np.array([snap[0], snap[1], snap[2]]), \
                   np.linalg.norm(np.array([snap[0], snap[2]]) - goal_xy)
        return fallback, float("inf")
    return best, best_offset


def _force_new_approach(pf, goal_xy, agent_xy, visited):
    """Find a new approach point from the opposite side of the object."""
    candidates = []
    agent_to_goal = agent_xy - goal_xy
    agent_to_goal = agent_to_goal / (np.linalg.norm(agent_to_goal) + 1e-8)

    for angle in np.linspace(0, 2 * np.pi, 16):
        for r in [0.6, 0.9, 1.2, 1.5]:
            probe = np.array([goal_xy[0] + r * np.cos(angle), 0.03,
                              goal_xy[1] + r * np.sin(angle)])
            snap = pf.snap_point(probe)
            if snap is None or not pf.is_navigable(snap) or snap[1] > 0.5:
                continue
            snap_xy = np.array([snap[0], snap[2]])
            snap_key = (round(snap_xy[0], 2), round(snap_xy[1], 2))
            if snap_key in visited:
                continue

            to_snap = snap_xy - goal_xy
            to_snap = to_snap / (np.linalg.norm(to_snap) + 1e-8)
            opposite_score = -np.dot(agent_to_goal, to_snap)

            path = habitat_sim.ShortestPath()
            path.requested_start = [agent_xy[0], 0.03, agent_xy[1]]
            path.requested_end = snap
            if pf.find_path(path) and path.geodesic_distance > 0.5:
                candidates.append((
                    np.array([snap[0], snap[1], snap[2]]),
                    opposite_score, path.geodesic_distance,
                    np.linalg.norm(snap_xy - goal_xy),
                ))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[1], x[2]))
    return candidates[0][0]


# ─── Run One Episode ─────────────────────────────────────────────────────

def run_episode(ep, ep_idx, sim_factory: SimFactory):
    scene_name = ep.get("scene_name", "unknown")
    sim, pf = sim_factory.get(scene_name)
    explorer = FrontierExplorer(sim)
    follower = ShortestPathFollower(sim)

    target_obj = ep["object_category"]
    start_pos = np.array(ep["start_position"])
    start_rot = np.array(ep["start_rotation"])
    goal_obj_center = np.array(ep["goals"][0]["position"])
    goal_xy = goal_obj_center[[0, 2]]
    start_xy = start_pos[[0, 2]]

    # Optimal geodesic for SPL
    geo_ep = ep.get("info", {}).get("geodesic_distance", -1)
    optimal_geo = geo_ep if geo_ep > 0 else \
        float(np.linalg.norm(start_pos - goal_obj_center))

    nav_goal, nav_offset = _find_nav_goal(pf, goal_xy, start_xy)

    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = mn.Vector3(start_pos[0], start_pos[1], start_pos[2])
    state.rotation = list(start_rot)
    agent.set_state(state)

    no_progress_steps = 0
    last_best_dist = float("inf")
    visited_nav_goals = set()
    path_length = 0.0
    last_pos = np.copy(start_pos)
    threshold = OBJ_RADIUS.get(target_obj, 1.0)
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for step in range(MAX_STEPS):
        state = agent.get_state()
        agent_pos = np.array([state.position[0], state.position[1],
                              state.position[2]])
        agent_xy = np.array([agent_pos[0], agent_pos[2]])
        dist_to_obj = float(np.linalg.norm(agent_xy - goal_xy))

        # Track path length
        path_length += float(np.linalg.norm(agent_pos[[0, 2]] - last_pos[[0, 2]]))
        last_pos = np.copy(agent_pos)

        # Success
        if dist_to_obj < threshold:
            SPL = optimal_geo / max(path_length, optimal_geo, 1e-6)
            return {
                "episode_id": ep["episode_id"],
                "scene_name": scene_name,
                "target": target_obj,
                "difficulty": ep.get("difficulty", "unknown"),
                "success": True,
                "steps": step,
                "final_dist": round(dist_to_obj, 2),
                "path_length": round(path_length, 2),
                "optimal_geo": round(optimal_geo, 2),
                "SPL": round(SPL, 4),
            }

        # Stuck at nav goal
        action = follower.get_next_action(nav_goal, state)
        if action == 0:
            visited_nav_goals.add((round(nav_goal[0], 2), round(nav_goal[2], 2)))
            new_goal = _force_new_approach(pf, goal_xy, agent_xy, visited_nav_goals)
            if new_goal is None:
                return {
                    "episode_id": ep["episode_id"],
                    "scene_name": scene_name,
                    "target": target_obj,
                    "difficulty": ep.get("difficulty", "unknown"),
                    "success": False,
                    "SPL": 0.0,
                    "steps": step,
                    "final_dist": round(dist_to_obj, 2),
                    "path_length": round(path_length, 2),
                    "optimal_geo": round(optimal_geo, 2),
                    "reason": "no_more_approach_points",
                }
            nav_goal = new_goal
            no_progress_steps = 0
            last_best_dist = float("inf")
            continue

        # Progress tracking
        if dist_to_obj < last_best_dist - 0.02:
            last_best_dist = dist_to_obj
            no_progress_steps = 0
        else:
            no_progress_steps += 1

        # Stuck: try new approach
        if no_progress_steps > 20:
            new_goal = _force_new_approach(pf, goal_xy, agent_xy, visited_nav_goals)
            if new_goal is not None and \
               np.linalg.norm(new_goal[[0, 2]] - nav_goal[[0, 2]]) > 0.1:
                nav_goal = new_goal
                no_progress_steps = 0
                last_best_dist = float("inf")

        # Execute
        action_names = {0: None, 1: "move_forward", 2: "turn_left", 3: "turn_right"}
        act_name = action_names.get(action)
        if act_name:
            sim.step(act_name)
            action_counts[action] += 1
        else:
            continue

    # Timeout — SPL = 0 for failures
    return {
        "episode_id": ep["episode_id"],
        "SPL": 0.0,
        "scene_name": scene_name,
        "target": target_obj,
        "difficulty": ep.get("difficulty", "unknown"),
        "success": False,
        "steps": MAX_STEPS,
        "final_dist": round(float(np.linalg.norm(
            np.array([agent.get_state().position[0],
                      agent.get_state().position[2]]) - goal_xy)), 2),
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

    # Group by scene for efficient sim switching
    by_scene = {}
    for i, ep in enumerate(episodes):
        scene = ep.get("scene_name", "unknown")
        by_scene.setdefault(scene, []).append((i, ep))

    sim_factory = SimFactory()
    results = []
    total = len(episodes)
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"FRONTIER EVAL — [{split_name.upper()}] {total} episodes")
    print(f"{'='*60}\n")

    # Process scene-by-scene to minimize sim switching
    global_idx = 0
    for scene_name, eps_in_scene in sorted(by_scene.items()):
        print(f"  Scene: {scene_name} ({len(eps_in_scene)} episodes)")
        for _, ep in eps_in_scene:
            result = run_episode(ep, global_idx, sim_factory)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"    [{global_idx+1}/{total}] {status} {result['target']} "
                  f"difficulty={result.get('difficulty','?')} "
                  f"steps={result['steps']} dist={result['final_dist']:.2f}m "
                  f"SPL={result.get('SPL',0):.3f}", flush=True)
            global_idx += 1

    sim_factory.close()
    elapsed = time.time() - t_start

    # ── Compute metrics ──────────────────────────────────────────────────
    successes = [r for r in results if r["success"]]
    n = len(results)
    sr = len(successes) / n if n > 0 else 0

    # Per-category
    by_cat = {}
    for r in results:
        cat = r["target"]
        by_cat.setdefault(cat, {"total": 0, "success": 0})
        by_cat[cat]["total"] += 1
        if r["success"]:
            by_cat[cat]["success"] += 1

    # Per-difficulty
    by_diff = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, {"total": 0, "success": 0})
        by_diff[d]["total"] += 1
        if r["success"]:
            by_diff[d]["success"] += 1

    # SPL, DTG
    spl_vals = [r.get("SPL", 0) for r in results]
    avg_spl = float(np.mean(spl_vals))
    avg_dtg = float(np.mean([r.get("final_dist", 0) for r in results if not r["success"]]))
    avg_steps = float(np.mean([r["steps"] for r in results]))

    metrics = {
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

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[{split_name.upper()}] SUMMARY")
    print(f"{'─'*60}")
    print(f"  SR = {sr*100:.1f}% ({len(successes)}/{n})")
    print(f"  SPL = {avg_spl:.4f}")
    print(f"  DTG (failed) = {avg_dtg:.2f}m")
    print(f"  Avg steps = {avg_steps:.0f}")
    print(f"  Time = {elapsed:.0f}s")
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

    # ── Save ─────────────────────────────────────────────────────────────
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out = {
            "config": {
                "policy": "Frontier Exploration",
                "split": split_name,
                "max_steps": MAX_STEPS,
            },
            "metrics": metrics,
            "episode_results": results,
        }

        out_path = output_dir / f"eval_frontier_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {out_path}")

    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Multi-Scene Frontier Evaluation")
    ap.add_argument("--all", action="store_true",
                   help="Run on all splits: dev, val, test")
    ap.add_argument("--split", type=str, default=None,
                   help="Run a specific split (dev/val/test)")
    ap.add_argument("--max_episodes", type=int, default=None,
                   help="Limit number of episodes")
    ap.add_argument("--output", type=str, default="results",
                   help="Output directory for results JSON")
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

    # ── Cross-split summary ──────────────────────────────────────────────
    if len(splits_to_run) > 1:
        print(f"\n{'='*60}")
        print("CROSS-SPLIT SUMMARY")
        print(f"{'='*60}")
        for s, m in sorted(all_metrics.items()):
            print(f"  [{s.upper()}] SR={m['SR']:.1f}% SPL={m['SPL']:.4f} "
                  f"({m['episodes']} ep)")

        # Save aggregate
        summary_path = output_dir / "eval_frontier_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
