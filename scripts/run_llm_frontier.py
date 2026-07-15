#!/usr/bin/env python3
"""
LLM-Guided Frontier Exploration for BioVLN.

Builds ON the Frontier Exploration baseline (run_frontier.py). The only change:
when stuck and multiple approach points are available around the target object,
the LLM picks which approach angle to try — using semantic knowledge about
lab layouts.

If the LLM is unavailable, falls back to exact Frontier baseline behavior.

Usage:
  python run_llm_frontier.py --split val --max_episodes 10 --output results/
  python run_llm_frontier.py --all --output results/
"""

import gzip, json, sys, time, os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
import habitat_sim, magnum as mn
import requests

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

# ─── LLM Config ─────────────────────────────────────────────────────────
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = "https://api.deepseek.com/anthropic/v1/messages"
LLM_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """You are helping a robot navigate a biology laboratory. The robot is stuck
near a target object and needs to approach it from a different angle. Given the
candidate approach points around the object, pick the best one.

LAB KNOWLEDGE:
- Refrigerators/fridges: tall white appliances, usually against walls or in corners
- Centrifuges: white cylindrical machines, usually on lab benches against walls
- Cabinets: white cabinets with legs, against walls
- Waste bins: small white bins on floor, near benches or in corners

APPROACH STRATEGY:
- For objects against walls: approach from the open side (away from wall)
- For objects in corners: approach from the open quadrant
- For objects in center: approach from any angle
- Prefer the side that is opposite to where you came from

Reply JSON: {"choice": <1-based index>, "reason": "<one sentence>"}"""


def llm_choose_approach(target_obj: str, candidates: List[dict]) -> Optional[int]:
    """Ask LLM to choose which approach point to try next.

    Args:
        target_obj: e.g. "centrifuge"
        candidates: list of {idx, angle_from_agent, angle_from_goal, opposite_score, distance}

    Returns: 1-based index of chosen candidate, or None
    """
    if len(candidates) <= 1:
        return 1 if candidates else None

    lines = []
    for c in candidates:
        lines.append(
            f"  [{c['idx']}] angle_from_agent={c['angle_from_agent']}° "
            f"opposite_score={c['opposite_score']:.2f} distance={c['distance']:.1f}m"
        )
    cand_text = "\n".join(lines)

    prompt = f"""Target: {target_obj}

Candidates (pick ONE):
{cand_text}

Which approach point? Reply JSON: {{"choice": N, "reason": "..."}}"""

    # Bypass SOCKS proxy
    old_proxy = os.environ.pop('ALL_PROXY', None)
    old_all = os.environ.pop('all_proxy', None)
    try:
        resp = requests.post(
            LLM_BASE_URL,
            json={
                "model": LLM_MODEL, "max_tokens": 200,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        result = resp.json()
        text = "".join(b.get("text", "") for b in result.get("content", []))
        if "{" in text and "}" in text:
            parsed = json.loads(text[text.index("{"):text.rindex("}")+1])
            choice = int(parsed.get("choice", 1))
            return max(1, min(choice, len(candidates)))
    except Exception:
        pass
    finally:
        if old_proxy is not None:
            os.environ['ALL_PROXY'] = old_proxy
        if old_all is not None:
            os.environ['all_proxy'] = old_all
    return None


# ─── SimFactory ──────────────────────────────────────────────────────────
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
        sim_cfg.scene_id = glb; sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0; sim_cfg.allow_sliding = False
        sim_cfg.load_semantic_mesh = False
        sim_cfg.override_scene_light_defaults = True
        sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        sim_cfg.pbr_image_based_lighting = True
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = CAM_HEIGHT; agent_cfg.radius = 0.18
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        sim.pathfinder.load_nav_mesh(navmesh)
        self._sim = sim; self._current_scene = scene_name
        return sim, sim.pathfinder

    def close(self):
        if self._sim is not None:
            self._sim.close(); self._sim = None


# ─── Path Follower ───────────────────────────────────────────────────────
class ShortestPathFollower:
    def __init__(self, sim):
        self.sim = sim; self.pf = sim.pathfinder

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
        path.requested_start = current; path.requested_end = goal_pos
        found = self.pf.find_path(path)
        if not found:
            snapped = self.pf.snap_point(goal_pos)
            if snapped is not None:
                path.requested_end = snapped; found = self.pf.find_path(path)
        if not found or len(path.points) < 2:
            return 2
        nxt = np.array(path.points[1])
        target_dir = np.array([nxt[0]-current[0], nxt[2]-current[2]])
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        forward = self._forward_direction(agent_state)
        cross = forward[0]*target_dir[1] - forward[1]*target_dir[0]
        angle = np.arccos(np.clip(np.dot(forward, target_dir), -1, 1))
        if angle < np.deg2rad(10): return 1
        elif cross > 0: return 3
        else: return 2


# ─── Helpers ─────────────────────────────────────────────────────────────
def _find_nav_goal(pf, goal_xy, ref_xy):
    best, best_offset = None, float("inf")
    for angle in np.linspace(0, 2*np.pi, 24):
        for r in [0.3, 0.5, 0.7, 1.0, 1.3]:
            probe = np.array([goal_xy[0]+r*np.cos(angle), 0.03,
                              goal_xy[1]+r*np.sin(angle)])
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


# ─── LLM-Guided Approach Selection ───────────────────────────────────────
def _llm_force_new_approach(pf, goal_xy, agent_xy, visited, target_obj,
                             use_llm=True):
    """Same as _force_new_approach but LLM picks among candidates."""
    candidates = []
    agent_to_goal = agent_xy - goal_xy
    agent_to_goal = agent_to_goal / (np.linalg.norm(agent_to_goal) + 1e-8)

    for angle in np.linspace(0, 2*np.pi, 16):
        for r in [0.6, 0.9, 1.2, 1.5]:
            probe = np.array([goal_xy[0]+r*np.cos(angle), 0.03,
                              goal_xy[1]+r*np.sin(angle)])
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
                # Compute angle from agent to snap (for LLM context)
                to_snap_vec = snap_xy - agent_xy
                angle_from_agent = int(np.rad2deg(np.arctan2(to_snap_vec[1], to_snap_vec[0])))
                candidates.append({
                    "snap": np.array([snap[0], snap[1], snap[2]]),
                    "opposite_score": round(opposite_score, 3),
                    "distance": round(path.geodesic_distance, 1),
                    "angle_from_agent": angle_from_agent,
                    "offset_from_goal": round(np.linalg.norm(snap_xy - goal_xy), 2),
                })

    if not candidates:
        return None

    # LLM selects among candidates
    if use_llm and len(candidates) > 1:
        for i, c in enumerate(candidates):
            c["idx"] = i + 1
        choice = llm_choose_approach(target_obj, candidates)
        if choice is not None:
            return candidates[choice - 1]["snap"]

    # Fallback: baseline behavior (highest opposite score, shortest path)
    candidates.sort(key=lambda x: (-x["opposite_score"], x["distance"]))
    return candidates[0]["snap"]


# ─── Run Episode ─────────────────────────────────────────────────────────
def run_episode(ep, ep_idx, sim_factory: SimFactory, use_llm: bool = True):
    scene_name = ep.get("scene_name", "unknown")
    sim, pf = sim_factory.get(scene_name)
    follower = ShortestPathFollower(sim)

    target_obj = ep["object_category"]
    start_pos = np.array(ep["start_position"])
    start_rot = np.array(ep["start_rotation"])
    goal_obj_center = np.array(ep["goals"][0]["position"])
    goal_xy = goal_obj_center[[0, 2]]
    start_xy = start_pos[[0, 2]]

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
    llm_queries = 0
    last_llm_step = -100  # rate-limit LLM: at most once per 50 steps

    for step in range(MAX_STEPS):
        state = agent.get_state()
        agent_pos = np.array([state.position[0], state.position[1], state.position[2]])
        agent_xy = np.array([agent_pos[0], agent_pos[2]])
        dist_to_obj = float(np.linalg.norm(agent_xy - goal_xy))

        path_length += float(np.linalg.norm(agent_pos[[0, 2]] - last_pos[[0, 2]]))
        last_pos = np.copy(agent_pos)

        if dist_to_obj < threshold:
            SPL = optimal_geo / max(path_length, optimal_geo, 1e-6)
            return {
                "episode_id": ep["episode_id"], "scene_name": scene_name,
                "target": target_obj, "difficulty": ep.get("difficulty", "unknown"),
                "success": True, "steps": step,
                "final_dist": round(dist_to_obj, 2),
                "path_length": round(path_length, 2),
                "optimal_geo": round(optimal_geo, 2),
                "SPL": round(SPL, 4), "llm_queries": llm_queries,
                "agent": "llm_frontier",
            }

        action = follower.get_next_action(nav_goal, state)
        if action == 0:
            visited_nav_goals.add((round(nav_goal[0], 2), round(nav_goal[2], 2)))
            # Rate-limit LLM: only use LLM every 50 steps
            use_llm_now = use_llm and (step - last_llm_step) >= 50
            new_goal = _llm_force_new_approach(
                pf, goal_xy, agent_xy, visited_nav_goals, target_obj,
                use_llm=use_llm_now)
            if use_llm_now:
                llm_queries += 1
                last_llm_step = step
            if new_goal is None:
                return {
                    "episode_id": ep["episode_id"], "scene_name": scene_name,
                    "target": target_obj,
                    "difficulty": ep.get("difficulty", "unknown"),
                    "success": False, "SPL": 0.0, "steps": step,
                    "final_dist": round(dist_to_obj, 2),
                    "path_length": round(path_length, 2),
                    "optimal_geo": round(optimal_geo, 2),
                    "reason": "no_more_approach_points",
                    "llm_queries": llm_queries, "agent": "llm_frontier",
                }
            nav_goal = new_goal
            no_progress_steps = 0
            last_best_dist = float("inf")
            continue

        if dist_to_obj < last_best_dist - 0.02:
            last_best_dist = dist_to_obj
            no_progress_steps = 0
        else:
            no_progress_steps += 1

        if no_progress_steps > 20:
            use_llm_now = use_llm and (step - last_llm_step) >= 50
            new_goal = _llm_force_new_approach(
                pf, goal_xy, agent_xy, visited_nav_goals, target_obj,
                use_llm=use_llm_now)
            if use_llm_now:
                llm_queries += 1
                last_llm_step = step
            if new_goal is not None and \
               np.linalg.norm(new_goal[[0, 2]] - nav_goal[[0, 2]]) > 0.1:
                nav_goal = new_goal
                no_progress_steps = 0
                last_best_dist = float("inf")

        action_names = {0: None, 1: "move_forward", 2: "turn_left", 3: "turn_right"}
        act_name = action_names.get(action)
        if act_name:
            sim.step(act_name)

    # Timeout
    return {
        "episode_id": ep["episode_id"], "SPL": 0.0, "scene_name": scene_name,
        "target": target_obj, "difficulty": ep.get("difficulty", "unknown"),
        "success": False, "steps": MAX_STEPS,
        "final_dist": round(float(np.linalg.norm(
            np.array([agent.get_state().position[0],
                      agent.get_state().position[2]]) - goal_xy)), 2),
        "path_length": round(path_length, 2),
        "optimal_geo": round(optimal_geo, 2),
        "reason": "timeout", "llm_queries": llm_queries,
        "agent": "llm_frontier",
    }


# ─── Evaluate Split ─────────────────────────────────────────────────────
def evaluate_split(split_name: str, max_episodes: Optional[int] = None,
                   use_llm: bool = True, output_dir: Optional[Path] = None):
    ep_file = SPLITS_DIR / f"episodes_{split_name}.json.gz"
    if not ep_file.exists():
        print(f"✗ Episode file not found: {ep_file}")
        return []

    with gzip.open(ep_file, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"]
    if max_episodes:
        episodes = episodes[:max_episodes]

    by_scene = {}
    for i, ep in enumerate(episodes):
        scene = ep.get("scene_name", "unknown")
        by_scene.setdefault(scene, []).append((i, ep))

    sim_factory = SimFactory(); results = []
    total = len(episodes); t_start = time.time()
    agent_label = "LLM-FRONTIER" if use_llm else "FRONTIER-BASELINE"

    print(f"\n{'='*60}")
    print(f"{agent_label} EVAL — [{split_name.upper()}] {total} episodes")
    print(f"{'='*60}\n")

    global_idx = 0
    for scene_name, eps_in_scene in sorted(by_scene.items()):
        print(f"  Scene: {scene_name} ({len(eps_in_scene)} episodes)")
        for _, ep in eps_in_scene:
            result = run_episode(ep, global_idx, sim_factory, use_llm=use_llm)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"    [{global_idx+1}/{total}] {status} {result['target']} "
                  f"d={result.get('difficulty','?')} "
                  f"steps={result['steps']} dist={result['final_dist']:.2f}m "
                  f"SPL={result.get('SPL',0):.3f} llm={result.get('llm_queries',0)}",
                  flush=True)
            global_idx += 1

    sim_factory.close(); elapsed = time.time() - t_start

    n = len(results); successes = [r for r in results if r["success"]]
    sr = len(successes)/n if n > 0 else 0

    by_cat, by_diff = {}, {}
    for r in results:
        cat = r["target"]
        by_cat.setdefault(cat, {"total": 0, "success": 0})
        by_cat[cat]["total"] += 1
        if r["success"]: by_cat[cat]["success"] += 1
    for r in results:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, {"total": 0, "success": 0})
        by_diff[d]["total"] += 1
        if r["success"]: by_diff[d]["success"] += 1

    avg_spl = float(np.mean([r.get("SPL", 0) for r in results]))
    failed_dists = [r.get("final_dist", 0) for r in results if not r["success"]]
    avg_dtg = float(np.mean(failed_dists)) if failed_dists else 0.0
    avg_steps = float(np.mean([r["steps"] for r in results]))

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
        cat_sr = v["success"]/v["total"]*100 if v["total"] > 0 else 0
        print(f"    {cat}: {cat_sr:.0f}% ({v['success']}/{v['total']})")
    print(f"\n  Per difficulty:")
    for d in DIFFICULTY_ORDER:
        if d in by_diff:
            v = by_diff[d]
            d_sr = v["success"]/v["total"]*100 if v["total"] > 0 else 0
            print(f"    {d}: {d_sr:.0f}% ({v['success']}/{v['total']})")

    if output_dir:
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        agent_name = "llm_frontier" if use_llm else "frontier_baseline"
        out = {
            "config": {"policy": agent_label, "split": split_name,
                       "max_steps": MAX_STEPS, "use_llm": use_llm},
            "metrics": {
                "split": split_name, "agent": agent_name,
                "total": n, "success": len(successes),
                "SR": round(sr, 4), "SPL": round(avg_spl, 4),
                "DTG": round(avg_dtg, 2), "avg_steps": round(avg_steps, 1),
                "elapsed_seconds": round(elapsed, 1),
                "per_category": {cat: {"SR": round(v["success"]/v["total"], 4) if v["total"]>0 else 0, "total": v["total"]}
                                for cat, v in sorted(by_cat.items())},
                "per_difficulty": {d: {"SR": round(v["success"]/v["total"], 4) if v["total"]>0 else 0, "total": v["total"]}
                                   for d, v in sorted(by_diff.items())},
            },
            "episode_results": results,
        }
        out_path = output_dir / f"eval_{agent_name}_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {out_path}")

    return results


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="LLM-Guided Frontier Exploration")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true",
                   help="Disable LLM (use baseline frontier behavior)")
    ap.add_argument("--output", type=str, default="results")
    args = ap.parse_args()

    splits_to_run = []
    if args.all:
        splits_to_run = ["dev", "val", "test"]
    elif args.split:
        splits_to_run = [args.split]
    else:
        ap.print_help(); return

    output_dir = Path(args.output)
    all_metrics = {}

    for split_name in splits_to_run:
        results = evaluate_split(split_name, args.max_episodes,
                                 use_llm=not args.no_llm, output_dir=output_dir)
        if results:
            n = len(results)
            sr = sum(1 for r in results if r["success"])/n*100
            agent_name = "llm_frontier" if not args.no_llm else "frontier_baseline"
            all_metrics[f"{agent_name}_{split_name}"] = {
                "agent": agent_name, "split": split_name,
                "episodes": n, "SR": round(sr, 1),
                "SPL": round(float(np.mean([r.get("SPL", 0) for r in results])), 4),
            }

    if len(splits_to_run) > 1:
        print(f"\n{'='*60}")
        print("CROSS-SPLIT SUMMARY")
        print(f"{'='*60}")
        for key, m in sorted(all_metrics.items()):
            print(f"  [{m['split'].upper()}] {m['agent']:20s}: "
                  f"SR={m['SR']:5.1f}%  SPL={m['SPL']:.4f}  ({m['episodes']} ep)")

        summary_path = output_dir / "eval_llm_frontier_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
