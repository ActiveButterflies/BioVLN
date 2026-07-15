#!/usr/bin/env python3
"""
Additional baselines for BioVLN: Random agent (lower bound) and Oracle
(shortest-path upper bound).

Usage:
  # Run both baselines on all splits
  python run_baselines.py --all --output results/

  # Run a specific baseline on a specific split
  python run_baselines.py --split dev --agent random --output results/
  python run_baselines.py --split val --agent oracle --output results/
"""

import gzip, json, sys, time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import habitat_sim
import magnum as mn

# ─── Paths ──────────────────────────────────────────────────────────────
MULTI_SCENE_DIR = Path("/root/gpufree-data/multi_scene")
OUTPUT_ROOT = MULTI_SCENE_DIR / "output"
SPLITS_DIR = OUTPUT_ROOT / "splits"

IMAGE_W, IMAGE_H = 320, 240
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

ACTIONS = {
    0: None,           # STOP
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
}

DIFFICULTY_ORDER = ["easy", "medium", "hard", "unknown"]


# ═══════════════════════════════════════════════════════════════════════════
# SimFactory (shared with run_frontier.py)
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Random Agent
# ═══════════════════════════════════════════════════════════════════════════

class RandomAgent:
    """Randomly selects from {forward, turn_left, turn_right} with bias toward forward."""

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    def get_action(self, agent_state, goal_pos):
        # 50% forward, 25% left, 25% right
        p = self.rng.random()
        if p < 0.5:
            return 1  # forward
        elif p < 0.75:
            return 2  # turn left
        else:
            return 3  # turn right


def run_random_episode(ep, ep_idx, sim_factory: SimFactory, rng: np.random.RandomState):
    scene_name = ep.get("scene_name", "unknown")
    sim, pf = sim_factory.get(scene_name)
    agent = RandomAgent(rng)

    target_obj = ep["object_category"]
    start_pos = np.array(ep["start_position"])
    start_rot = np.array(ep["start_rotation"])
    goal_obj_center = np.array(ep["goals"][0]["position"])
    goal_xy = goal_obj_center[[0, 2]]

    geo_ep = ep.get("info", {}).get("geodesic_distance", -1)
    optimal_geo = geo_ep if geo_ep > 0 else \
        float(np.linalg.norm(start_pos - goal_obj_center))

    sim_agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = mn.Vector3(start_pos[0], start_pos[1], start_pos[2])
    state.rotation = list(start_rot)
    sim_agent.set_state(state)

    threshold = OBJ_RADIUS.get(target_obj, 1.0)
    path_length = 0.0
    last_pos = np.copy(start_pos)

    for step in range(MAX_STEPS):
        state = sim_agent.get_state()
        agent_pos = np.array([state.position[0], state.position[1], state.position[2]])
        agent_xy = np.array([agent_pos[0], agent_pos[2]])
        dist_to_obj = float(np.linalg.norm(agent_xy - goal_xy))

        path_length += float(np.linalg.norm(agent_pos[[0, 2]] - last_pos[[0, 2]]))
        last_pos = np.copy(agent_pos)

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
                "agent": "random",
            }

        action = agent.get_action(state, goal_obj_center)
        act_name = ACTIONS.get(action)
        if act_name:
            sim.step(act_name)
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
            np.array([sim_agent.get_state().position[0],
                      sim_agent.get_state().position[2]]) - goal_xy)), 2),
        "path_length": round(path_length, 2),
        "optimal_geo": round(optimal_geo, 2),
        "reason": "timeout",
        "agent": "random",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Oracle Agent (Shortest Path)
# ═══════════════════════════════════════════════════════════════════════════

class OracleAgent:
    """Follows the geodesic shortest path to goal. Upper bound on performance."""

    def __init__(self, sim):
        self.sim = sim
        self.pf = sim.pathfinder
        self._full_path = None
        self._path_idx = 0

    def set_goal(self, start_pos, goal_pos):
        """Pre-compute the full shortest path."""
        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = goal_pos
        found = self.pf.find_path(path)
        if not found:
            snapped = self.pf.snap_point(goal_pos)
            if snapped is not None:
                path.requested_end = snapped
                found = self.pf.find_path(path)
        if found:
            self._full_path = [np.array(p) for p in path.points]
            self._path_idx = 1  # start at second point
        else:
            self._full_path = None

    def get_action(self, agent_state):
        """Return the action to follow the pre-computed path."""
        if self._full_path is None or self._path_idx >= len(self._full_path):
            return 0  # STOP

        current = np.array([agent_state.position[0],
                            agent_state.position[1],
                            agent_state.position[2]])
        target = self._full_path[self._path_idx]

        # If close enough to current waypoint, advance
        dist_to_wp = np.linalg.norm(current - target)
        if dist_to_wp < 0.15:
            self._path_idx += 1
            if self._path_idx >= len(self._full_path):
                return 0  # STOP — reached end
            target = self._full_path[self._path_idx]

        # Compute turn/move
        import quaternion as nq
        q = agent_state.rotation
        forward = nq.rotate_vectors(q, np.array([[0, 0, -1]]))[0]
        forward = np.array([forward[0], forward[2]])
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        target_dir = np.array([target[0] - current[0], target[2] - current[2]])
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)

        cross = forward[0] * target_dir[1] - forward[1] * target_dir[0]
        angle = np.arccos(np.clip(np.dot(forward, target_dir), -1, 1))

        if angle < np.deg2rad(10):
            return 1  # MOVE_FORWARD
        elif cross > 0:
            return 3  # TURN_RIGHT
        else:
            return 2  # TURN_LEFT


def run_oracle_episode(ep, ep_idx, sim_factory: SimFactory):
    scene_name = ep.get("scene_name", "unknown")
    sim, pf = sim_factory.get(scene_name)
    oracle = OracleAgent(sim)

    target_obj = ep["object_category"]
    start_pos = np.array(ep["start_position"])
    start_rot = np.array(ep["start_rotation"])
    goal_obj_center = np.array(ep["goals"][0]["position"])
    goal_xy = goal_obj_center[[0, 2]]
    start_xy = start_pos[[0, 2]]

    geo_ep = ep.get("info", {}).get("geodesic_distance", -1)
    optimal_geo = geo_ep if geo_ep > 0 else \
        float(np.linalg.norm(start_pos - goal_obj_center))

    nav_goal, _ = _find_nav_goal(pf, goal_xy, start_xy)

    sim_agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = mn.Vector3(start_pos[0], start_pos[1], start_pos[2])
    state.rotation = list(start_rot)
    sim_agent.set_state(state)

    oracle.set_goal(start_pos, nav_goal)

    threshold = OBJ_RADIUS.get(target_obj, 1.0)
    path_length = 0.0
    last_pos = np.copy(start_pos)

    for step in range(MAX_STEPS):
        state = sim_agent.get_state()
        agent_pos = np.array([state.position[0], state.position[1], state.position[2]])
        agent_xy = np.array([agent_pos[0], agent_pos[2]])
        dist_to_obj = float(np.linalg.norm(agent_xy - goal_xy))

        path_length += float(np.linalg.norm(agent_pos[[0, 2]] - last_pos[[0, 2]]))
        last_pos = np.copy(agent_pos)

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
                "agent": "oracle",
            }

        action = oracle.get_action(state)
        if action == 0:
            # STOP called but not at goal — snap closer
            nav_goal, _ = _find_nav_goal(pf, goal_xy, agent_xy)
            if np.linalg.norm(nav_goal - agent_pos) < 0.3:
                return {
                    "episode_id": ep["episode_id"],
                    "SPL": 0.0,
                    "scene_name": scene_name,
                    "target": target_obj,
                    "difficulty": ep.get("difficulty", "unknown"),
                    "success": False,
                    "steps": step,
                    "final_dist": round(dist_to_obj, 2),
                    "path_length": round(path_length, 2),
                    "optimal_geo": round(optimal_geo, 2),
                    "reason": "oracle_stuck_next_to_goal",
                    "agent": "oracle",
                }
            oracle.set_goal(agent_pos, nav_goal)
            continue

        act_name = ACTIONS.get(action)
        if act_name:
            sim.step(act_name)
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
            np.array([sim_agent.get_state().position[0],
                      sim_agent.get_state().position[2]]) - goal_xy)), 2),
        "path_length": round(path_length, 2),
        "optimal_geo": round(optimal_geo, 2),
        "reason": "timeout",
        "agent": "oracle",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_split(split_name: str, agent_type: str,
                   max_episodes: Optional[int] = None,
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
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    results = []
    total = len(episodes)
    t_start = time.time()

    agent_label = {"random": "RANDOM", "oracle": "ORACLE"}.get(agent_type, agent_type.upper())

    print(f"\n{'='*60}")
    print(f"{agent_label} EVAL — [{split_name.upper()}] {total} episodes")
    print(f"{'='*60}\n")

    global_idx = 0
    for scene_name, eps_in_scene in sorted(by_scene.items()):
        print(f"  Scene: {scene_name} ({len(eps_in_scene)} episodes)")
        for _, ep in eps_in_scene:
            if agent_type == "random":
                result = run_random_episode(ep, global_idx, sim_factory, rng)
            elif agent_type == "oracle":
                result = run_oracle_episode(ep, global_idx, sim_factory)
            else:
                raise ValueError(f"Unknown agent: {agent_type}")

            results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"    [{global_idx+1}/{total}] {status} {result['target']} "
                  f"difficulty={result.get('difficulty','?')} "
                  f"steps={result['steps']} dist={result['final_dist']:.2f}m "
                  f"SPL={result.get('SPL',0):.3f}", flush=True)
            global_idx += 1

    sim_factory.close()
    elapsed = time.time() - t_start

    # ── Metrics ──────────────────────────────────────────────────────────
    n = len(results)
    successes = [r for r in results if r["success"]]
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

    avg_spl = float(np.mean([r.get("SPL", 0) for r in results]))
    failed_dists = [r.get("final_dist", 0) for r in results if not r["success"]]
    avg_dtg = float(np.mean(failed_dists)) if failed_dists else 0.0
    avg_steps = float(np.mean([r["steps"] for r in results]))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[{split_name.upper()}] {agent_label} SUMMARY")
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
                "policy": f"{agent_type.capitalize()} Agent",
                "split": split_name,
                "max_steps": MAX_STEPS,
            },
            "metrics": {
                "split": split_name,
                "agent": agent_type,
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
            },
            "episode_results": results,
        }
        out_path = output_dir / f"eval_{agent_type}_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="BioVLN Baselines: Random & Oracle")
    ap.add_argument("--all", action="store_true",
                   help="Run random + oracle on all splits")
    ap.add_argument("--split", type=str, default=None,
                   help="Run on a specific split (dev/val/test)")
    ap.add_argument("--agent", type=str, default="all",
                   choices=["random", "oracle", "all"],
                   help="Which agent to run (default: all)")
    ap.add_argument("--max_episodes", type=int, default=None,
                   help="Limit episodes for quick test")
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

    agents_to_run = ["random", "oracle"] if args.agent == "all" else [args.agent]
    output_dir = Path(args.output)

    all_metrics = {}

    for agent_type in agents_to_run:
        for split_name in splits_to_run:
            results = evaluate_split(split_name, agent_type, args.max_episodes, output_dir)
            if results:
                n = len(results)
                successes = sum(1 for r in results if r["success"])
                sr = successes / n * 100 if n > 0 else 0
                key = f"{agent_type}_{split_name}"
                all_metrics[key] = {
                    "agent": agent_type,
                    "split": split_name,
                    "episodes": n,
                    "SR": round(sr, 1),
                    "SPL": round(float(np.mean([r.get("SPL", 0) for r in results])), 4),
                }

    # ── Cross-baseline summary ───────────────────────────────────────────
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("CROSS-BASELINE SUMMARY")
        print(f"{'='*60}")

        # Also load frontier results for comparison
        frontier_path = output_dir / "eval_frontier_summary.json"
        if frontier_path.exists():
            with open(frontier_path) as f:
                frontier = json.load(f)
            for split_name, m in frontier.items():
                all_metrics[f"frontier_{split_name}"] = {
                    "agent": "frontier",
                    "split": split_name,
                    "episodes": m["episodes"],
                    "SR": m["SR"],
                    "SPL": m["SPL"],
                }

        # Print by split
        for split_name in splits_to_run:
            print(f"\n  [{split_name.upper()}]")
            for key, m in sorted(all_metrics.items()):
                if m["split"] == split_name:
                    print(f"    {m['agent']:12s}: SR={m['SR']:5.1f}%  SPL={m['SPL']:.4f}  "
                          f"({m['episodes']} ep)")

        # Save aggregate
        summary_path = output_dir / "eval_baselines_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
