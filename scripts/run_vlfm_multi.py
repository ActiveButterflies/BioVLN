#!/usr/bin/env python3
"""
Multi-Scene VLFM Evaluation for BioVLN.

Wraps run_vlfm_v4.py's agent logic and runs it across all dev/val/test splits.
Patches GLB_PATH/NAVMESH_PATH per-episode and replaces create_sim() with
SimFactory-based scene loading.

PREREQUISITE: VLM servers running on ports 12181, 12182, 13183.

Usage:
  python run_vlfm_multi.py --split dev --max_episodes 5 --output results/
  python run_vlfm_multi.py --all --output results/
  python run_vlfm_multi.py --split val --blind --max_episodes 3
"""

import sys
import os
import json
import gzip
import time
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np

# ─── Paths ──────────────────────────────────────────────────────────────
MULTI_SCENE_DIR = Path("/root/gpufree-data/multi_scene")
OUTPUT_ROOT = MULTI_SCENE_DIR / "output"
SPLITS_DIR = OUTPUT_ROOT / "splits"
PIPELINE_DIR = Path("/root/gpufree-data/pipeline")

# Make run_vlfm_v4 importable
sys.path.insert(0, str(PIPELINE_DIR))

# Import the VLFM module — we'll monkey-patch its globals
import run_vlfm_v4 as vlfm

# ═══════════════════════════════════════════════════════════════════════════
# SimFactory — replaces vlfm.create_sim() for multi-scene support
# ═══════════════════════════════════════════════════════════════════════════

import habitat_sim
import magnum as mn

class SimFactory:
    """Create/get habitat-sim instances per scene, caching the most recent."""

    def __init__(self):
        self._sim = None
        self._current_scene = None

    def get(self, scene_name: str):
        if self._current_scene == scene_name and self._sim is not None:
            try:
                self._sim.get_agent(0)
                return self._sim
            except Exception:
                pass

        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:
                pass

        scene_dir = OUTPUT_ROOT / scene_name
        glb = str(scene_dir / f"{scene_name}.glb")
        navmesh = str(scene_dir / f"{scene_name}.navmesh")

        # Use vlfm's create_sim logic but with our paths
        # Temporarily set globals
        vlfm.GLB_PATH = glb
        vlfm.NAVMESH_PATH = navmesh
        sim = vlfm.create_sim()

        self._sim = sim
        self._current_scene = scene_name
        return sim

    def close(self):
        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:
                pass
            self._sim = None


# ═══════════════════════════════════════════════════════════════════════════
# Wrapped run_episode — sets globals before calling the original
# ═══════════════════════════════════════════════════════════════════════════

SIM_FACTORY = None  # Set at evaluation start


def run_episode_wrapped(ep, ep_idx, save_video=False, blind=False, use_vlm_vmap=False):
    """Wrapper that sets GLB_PATH/NAVMESH_PATH before calling vlfm.run_episode."""
    scene_name = ep.get("scene_name", "unknown")
    scene_dir = OUTPUT_ROOT / scene_name
    glb = str(scene_dir / f"{scene_name}.glb")
    navmesh = str(scene_dir / f"{scene_name}.navmesh")

    # Patch globals
    vlfm.GLB_PATH = glb
    vlfm.NAVMESH_PATH = navmesh

    # Extract episode metadata for result enrichment
    geo_dist = ep.get("info", {}).get("geodesic_distance", -1)
    difficulty = ep.get("difficulty", "unknown")
    start_pos = np.array(ep["start_position"])
    goal_pos = np.array(ep["goals"][0]["position"])
    optimal_geo = geo_dist if geo_dist > 0 else float(np.linalg.norm(start_pos - goal_pos))

    # Run the original VLFM run_episode
    try:
        result = vlfm.run_episode(ep, ep_idx, save_video=save_video,
                                  blind=blind, use_vlm_vmap=use_vlm_vmap)
    except Exception as e:
        print(f"      ✗ VLFM crashed: {e}", flush=True)
        result = {"success": False, "steps": 0, "final_dist": float(np.linalg.norm(start_pos[[0,2]] - goal_pos[[0,2]])),
                  "target": ep.get("object_category", "?")}

    # Enrich result with standard fields
    result["scene_name"] = scene_name
    result["difficulty"] = difficulty
    result["target"] = result.get("target", ep.get("object_category", "?"))
    result["optimal_geo"] = round(optimal_geo, 2)

    # Compute SPL: success weighted by path length
    # VLFM doesn't track path_length, so approximate from steps * FORWARD_STEP
    steps = result.get("steps", 0)
    if result.get("success") and steps > 0:
        path_length = steps * vlfm.FORWARD_STEP  # rough estimate
        SPL = optimal_geo / max(path_length, optimal_geo, 1e-6)
    else:
        SPL = 0.0
    result["SPL"] = round(min(SPL, 1.0), 4)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Split Evaluator
# ═══════════════════════════════════════════════════════════════════════════

DIFFICULTY_ORDER = ["easy", "medium", "hard", "unknown"]


def evaluate_split(split_name: str, max_episodes: Optional[int] = None,
                   blind: bool = False, use_vlm_vmap: bool = False,
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

    # Group by scene for logging
    by_scene = {}
    for i, ep in enumerate(episodes):
        scene = ep.get("scene_name", "unknown")
        by_scene.setdefault(scene, []).append((i, ep))

    mode_label = "VLFM-BLIND" if blind else "VLFM"
    if use_vlm_vmap:
        mode_label = "VLFM-VLM"
    total = len(episodes)
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"{mode_label} EVAL — [{split_name.upper()}] {total} episodes")
    print(f"{'='*60}\n")

    results = []
    global_idx = 0

    for scene_name, eps_in_scene in sorted(by_scene.items()):
        print(f"  Scene: {scene_name} ({len(eps_in_scene)} episodes)")
        for _, ep in eps_in_scene:
            t_ep = time.time()
            result = run_episode_wrapped(ep, global_idx, save_video=False,
                                         blind=blind, use_vlm_vmap=use_vlm_vmap)
            elapsed_ep = time.time() - t_ep

            results.append(result)
            status = "✓" if result.get("success") else "✗"
            steps = result.get("steps", 0)
            dist = result.get("final_dist", 0)
            target = result.get("target", ep.get("object_category", "?"))
            print(f"    [{global_idx+1}/{total}] {status} {target} "
                  f"steps={steps} dist={dist:.2f}m "
                  f"t={elapsed_ep:.0f}s", flush=True)
            global_idx += 1

    elapsed = time.time() - t_start

    # ── Compute metrics ──────────────────────────────────────────────────
    n = len(results)
    successes = [r for r in results if r.get("success")]
    sr = len(successes) / n if n > 0 else 0

    by_cat = {}
    for r in results:
        cat = r.get("target", "?")
        by_cat.setdefault(cat, {"total": 0, "success": 0})
        by_cat[cat]["total"] += 1
        if r.get("success"):
            by_cat[cat]["success"] += 1

    by_diff = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, {"total": 0, "success": 0})
        by_diff[d]["total"] += 1
        if r.get("success"):
            by_diff[d]["success"] += 1

    avg_spl = float(np.mean([r.get("SPL", 0) for r in results]))
    failed_dists = [r.get("final_dist", 0) for r in results if not r.get("success")]
    avg_dtg = float(np.mean(failed_dists)) if failed_dists else 0.0
    avg_steps = float(np.mean([r.get("steps", 0) for r in results]))

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[{split_name.upper()}] {mode_label} SUMMARY")
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

        # Convert episode results to standard format
        episode_results = []
        for r in results:
            episode_results.append({
                "episode_id": r.get("episode_id", "?"),
                "scene_name": r.get("scene_name", "?"),
                "target": r.get("target", "?"),
                "difficulty": r.get("difficulty", "unknown"),
                "success": r.get("success", False),
                "steps": r.get("steps", 0),
                "final_dist": r.get("final_dist", 0),
                "SPL": r.get("SPL", 0),
                "agent": "vlfm",
                "blind": blind,
            })

        agent_name = "vlfm_blind" if blind else "vlfm"
        if use_vlm_vmap:
            agent_name = "vlfm_vlm"

        metrics = {
            "split": split_name,
            "agent": agent_name,
            "blind": blind,
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

        out = {
            "config": {"policy": agent_name, "split": split_name,
                       "max_steps": vlfm.MAX_STEPS, "blind": blind},
            "metrics": metrics,
            "episode_results": episode_results,
        }

        out_path = output_dir / f"eval_{agent_name}_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Multi-Scene VLFM Evaluation")
    ap.add_argument("--all", action="store_true",
                   help="Run on all splits: dev, val, test")
    ap.add_argument("--split", type=str, default=None,
                   help="Run a specific split (dev/val/test)")
    ap.add_argument("--max_episodes", type=int, default=None,
                   help="Limit number of episodes per split")
    ap.add_argument("--blind", action="store_true",
                   help="Blind mode: agent does NOT know goal position")
    ap.add_argument("--vlm_vmap", action="store_true",
                   help="Use GPT-4o instead of BLIP-2 for value map")
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

    # Verify VLM servers
    import requests
    for port, name in [(12181, "GroundingDINO"), (12182, "BLIP-2"), (13183, "SAM")]:
        try:
            r = requests.get(f"http://localhost:{port}/", timeout=2)
            print(f"  ✓ {name} server on port {port}")
        except Exception:
            print(f"  ✗ {name} server on port {port} NOT REACHABLE — start it first!")
            return

    output_dir = Path(args.output)
    all_metrics = {}

    for split_name in splits_to_run:
        results = evaluate_split(split_name, args.max_episodes,
                                 blind=args.blind, use_vlm_vmap=args.vlm_vmap,
                                 output_dir=output_dir)
        if results:
            n = len(results)
            sr = sum(1 for r in results if r.get("success")) / n * 100
            agent_name = "vlfm_blind" if args.blind else "vlfm"
            if args.vlm_vmap:
                agent_name = "vlfm_vlm"
            all_metrics[f"{agent_name}_{split_name}"] = {
                "agent": agent_name,
                "split": split_name,
                "episodes": n,
                "SR": round(sr, 1),
                "SPL": round(float(np.mean([r.get("SPL", 0) for r in results])), 4),
            }

    if len(splits_to_run) > 1:
        print(f"\n{'='*60}")
        print("VLFM CROSS-SPLIT SUMMARY")
        print(f"{'='*60}")
        for key, m in sorted(all_metrics.items()):
            print(f"  [{m['split'].upper()}] {m['agent']:12s}: "
                  f"SR={m['SR']:5.1f}%  SPL={m['SPL']:.4f}  ({m['episodes']} ep)")

        summary_path = output_dir / "eval_vlfm_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
