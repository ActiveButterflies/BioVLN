#!/usr/bin/env python3
"""
Unified VLFM evaluation across all BioVLN datasets.

Wraps run_vlfm_v4.py to run on multi_scene, two_room, and lsat_target.
Supports both blind (true VLFM) and vlm_vmap (GPT-4o value maps) modes.

PREREQUISITE: VLM servers on ports 12181 (GDINO), 12182 (BLIP), 13183 (SAM).

Usage:
  # Test with 3 episodes on lsat_target
  python run_vlfm_all.py --dataset lsat_target --split dev --max 3

  # Full run on all datasets
  python run_vlfm_all.py --dataset lsat_target --all
  python run_vlfm_all.py --dataset two_room --all
  python run_vlfm_all.py --dataset multi_scene --all
"""

import sys
import os
import json
import gzip
import time
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np

# ─── Clear proxy env vars (interfere with VLM API calls) ──────────────────
for _key in ["ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(_key, None)

# Headless GL
if not os.environ.get("DISPLAY") and not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

# ─── Dataset Registry ──────────────────────────────────────────────────────

DATASET_ROOTS = {
    "multi_scene": "/root/gpufree-data/multi_scene/output",
    "two_room":    "/root/gpufree-data/two_room_output",
    "lsat_target": "/root/gpufree-data/lsat_test",
}

DATASET_SPLITS = {
    "multi_scene": "/root/gpufree-data/multi_scene/output/splits",
    "two_room":    "/root/gpufree-data/two_room_output/splits",
    "lsat_target": "/root/gpufree-data/lsat_test/splits",
}

DATASET_RESULTS = {
    "multi_scene": "/root/gpufree-data/multi_scene/results",
    "two_room":    "/root/gpufree-data/two_room_output/results",
    "lsat_target": "/root/gpufree-data/lsat_test/results",
}

# ─── Import VLFM module ────────────────────────────────────────────────────

PIPELINE_DIR = Path("/root/gpufree-data/pipeline")
sys.path.insert(0, str(PIPELINE_DIR))
import run_vlfm_v4 as vlfm

# ═══════════════════════════════════════════════════════════════════════════
# SimFactory — handles scene loading for any dataset
# ═══════════════════════════════════════════════════════════════════════════

import habitat_sim
import magnum as mn


class SimFactory:
    """Create/get habitat-sim instances per scene, caching most recent."""

    def __init__(self, scenes_root: str):
        self._scenes_root = Path(scenes_root)
        self._sim = None
        self._current_scene = None

    def get(self, scene_name: str):
        # Check if already loaded
        if self._current_scene == scene_name and self._sim is not None:
            try:
                self._sim.get_agent(0)
                return self._sim
            except Exception:
                pass

        # Close old sim
        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:
                pass
            self._sim = None

        scene_dir = self._scenes_root / scene_name
        glb = str(scene_dir / f"{scene_name}.glb")
        navmesh = str(scene_dir / f"{scene_name}.navmesh")

        if not os.path.exists(glb):
            raise FileNotFoundError(f"GLB not found: {glb}")

        # Use vlfm's create_sim logic with our paths
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
            self._current_scene = None


# ═══════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════

def run_episode_wrapped(ep, ep_idx, sim_factory, scenes_root,
                        blind=True, use_vlm_vmap=True):
    """Run one VLFM episode, setting up scene globals first."""
    scene_name = ep.get("scene_name", "unknown")
    scene_dir = scenes_root / scene_name
    glb = str(scene_dir / f"{scene_name}.glb")
    navmesh = str(scene_dir / f"{scene_name}.navmesh")

    # Patch VLFM globals for create_sim()
    vlfm.GLB_PATH = glb
    vlfm.NAVMESH_PATH = navmesh

    # Episode metadata
    geo_dist = ep.get("info", {}).get("geodesic_distance", -1)
    difficulty = ep.get("difficulty", "unknown")
    start_pos = np.array(ep["start_position"])
    goal_pos = np.array(ep["goals"][0]["position"])
    optimal_geo = geo_dist if geo_dist > 0 else float(np.linalg.norm(start_pos - goal_pos))

    try:
        result = vlfm.run_episode(ep, ep_idx, save_video=False,
                                  blind=blind, use_vlm_vmap=use_vlm_vmap)
    except Exception as e:
        print(f"      ✗ VLFM crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        result = {
            "success": False, "steps": 0,
            "final_dist": float(np.linalg.norm(start_pos[[0, 2]] - goal_pos[[0, 2]])),
            "target": ep.get("object_category", "?")
        }

    # Enrich result
    result["episode_id"] = str(ep.get("episode_id", "?"))
    result["scene_name"] = scene_name
    result["difficulty"] = difficulty
    result["target"] = result.get("target", ep.get("object_category", "?"))
    result["optimal_geo"] = round(optimal_geo, 2)

    # SPL
    steps = result.get("steps", 0)
    if result.get("success") and steps > 0:
        path_length = steps * vlfm.FORWARD_STEP
        SPL = optimal_geo / max(path_length, optimal_geo, 1e-6)
    else:
        SPL = 0.0
    result["SPL"] = round(min(SPL, 1.0), 4)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Split Evaluator
# ═══════════════════════════════════════════════════════════════════════════

DIFFICULTY_ORDER = ["easy", "medium", "hard", "unknown"]


def evaluate_split(dataset: str, split: str,
                   max_episodes: Optional[int] = None,
                   blind: bool = True,
                   use_vlm_vmap: bool = True,
                   output_dir: Optional[Path] = None):
    """Run VLFM on all episodes in a split for a given dataset."""
    scenes_root = Path(DATASET_ROOTS[dataset])
    splits_dir = Path(DATASET_SPLITS[dataset])

    ep_file = splits_dir / f"episodes_{split}.json.gz"
    if not ep_file.exists():
        ep_file = splits_dir / f"episodes_{split}.json"
    if not ep_file.exists():
        print(f"✗ Episode file not found: {splits_dir}/episodes_{split}.json.gz")
        return []

    with (gzip.open(ep_file, "rt", encoding="utf-8") if ep_file.suffix == ".gz"
          else open(ep_file, "r")) as f:
        data = json.load(f)
    episodes = data.get("episodes", data)
    if not isinstance(episodes, list):
        raise ValueError(f"Unexpected format in {ep_file}")

    if max_episodes:
        episodes = episodes[:max_episodes]

    # Group by scene
    by_scene: Dict[str, list] = {}
    for i, ep in enumerate(episodes):
        scene = ep.get("scene_name", "unknown")
        by_scene.setdefault(scene, []).append((i, ep))

    mode_label = "VLFM-VLM" if (blind and use_vlm_vmap) else "VLFM"
    if blind and not use_vlm_vmap:
        mode_label = "VLFM-BLIND"
    total = len(episodes)
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"{mode_label} EVAL — [{dataset}] [{split.upper()}] "
          f"{total} episodes, {len(by_scene)} scenes")
    print(f"  blind={blind}  vlm_vmap={use_vlm_vmap}")
    print(f"{'='*60}\n")

    sim_factory = SimFactory(str(scenes_root))
    results = []
    global_idx = 0

    for scene_name, eps_in_scene in sorted(by_scene.items()):
        print(f"  Scene: {scene_name} ({len(eps_in_scene)} episodes)", flush=True)
        for _, ep in eps_in_scene:
            t_ep = time.time()
            result = run_episode_wrapped(
                ep, global_idx, sim_factory, scenes_root,
                blind=blind, use_vlm_vmap=use_vlm_vmap,
            )
            elapsed_ep = time.time() - t_ep

            results.append(result)
            status = "✓" if result.get("success") else "✗"
            steps = result.get("steps", 0)
            dist = result.get("final_dist", 0)
            target = result.get("target", ep.get("object_category", "?"))
            print(f"    [{global_idx+1}/{total}] {status} {target:<20s} "
                  f"steps={steps:3d} dist={dist:.2f}m "
                  f"t={elapsed_ep:.0f}s", flush=True)
            global_idx += 1

    sim_factory.close()
    elapsed = time.time() - t_start

    # ── Metrics ─────────────────────────────────────────────────────────
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

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[{dataset}] [{split.upper()}] {mode_label} SUMMARY")
    print(f"{'─'*60}")
    print(f"  SR  = {sr*100:.1f}% ({len(successes)}/{n})")
    print(f"  SPL = {avg_spl:.4f}")
    print(f"  DTG (failed) = {avg_dtg:.2f}m")
    print(f"  Avg steps = {avg_steps:.0f}")
    print(f"  Time = {elapsed:.0f}s ({elapsed/60:.1f}min)")
    if by_cat:
        print(f"\n  Per category:")
        for cat, v in sorted(by_cat.items()):
            cat_sr = v["success"] / v["total"] * 100 if v["total"] > 0 else 0
            bar = "█" * int(cat_sr / 5)
            print(f"    {cat:<20s} {cat_sr:5.1f}% {bar} ({v['success']}/{v['total']})")
    if any(d in by_diff for d in DIFFICULTY_ORDER):
        print(f"\n  Per difficulty:")
        for d in DIFFICULTY_ORDER:
            if d in by_diff:
                v = by_diff[d]
                d_sr = v["success"] / v["total"] * 100 if v["total"] > 0 else 0
                print(f"    {d:<10s} {d_sr:5.1f}% ({v['success']}/{v['total']})")
    print(f"{'─'*60}\n")

    # ── Save ────────────────────────────────────────────────────────────
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        agent_name = "vlfm_vlm" if (blind and use_vlm_vmap) else \
                     "vlfm_blind" if blind else "vlfm"

        # Episode results
        episode_results = []
        for r in results:
            episode_results.append({
                "episode_id": r.get("episode_id", "?"),
                "scene_name": r.get("scene_name", "?"),
                "target": r.get("target", "?"),
                "difficulty": r.get("difficulty", "unknown"),
                "success": r.get("success", False),
                "steps": r.get("steps", 0),
                "final_dist": round(r.get("final_dist", 0), 2),
                "path_length": round(r.get("path_length", 0), 2),
                "SPL": r.get("SPL", 0),
            })

        metrics = {
            "agent": f"VLFM ({agent_name})",
            "dataset": dataset,
            "split": split,
            "blind": blind,
            "use_vlm_vmap": use_vlm_vmap,
            "total": n,
            "success": len(successes),
            "SR": round(sr, 4),
            "SPL": round(avg_spl, 4),
            "avg_steps": round(avg_steps, 1),
            "avg_dtg_failed": round(avg_dtg, 2),
            "elapsed": round(elapsed, 1),
            "per_category": {
                cat: {
                    "total": v["total"],
                    "success": v["success"],
                    "SR": round(v["success"] / v["total"], 4) if v["total"] > 0 else 0,
                }
                for cat, v in sorted(by_cat.items())
            },
            "per_difficulty": {
                d: {
                    "total": v["total"],
                    "success": v["success"],
                    "SR": round(v["success"] / v["total"], 4) if v["total"] > 0 else 0,
                }
                for d, v in sorted(by_diff.items())
            },
            "per_episode": episode_results,
        }

        out_path = output_dir / f"eval_{agent_name}_{dataset}_{split}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  Results saved → {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="VLFM evaluation across BioVLN datasets")
    ap.add_argument("--dataset", required=True,
                    choices=["multi_scene", "two_room", "lsat_target"])
    ap.add_argument("--split", default="dev",
                    choices=["dev", "val", "train", "test"])
    ap.add_argument("--all", action="store_true",
                    help="Run all available splits for this dataset")
    ap.add_argument("--max", type=int, default=None,
                    help="Max episodes per split")
    ap.add_argument("--blind", action="store_true", default=True,
                    help="VLFM blind mode (agent doesn't know goal position)")
    ap.add_argument("--vlm-vmap", action="store_true", default=True,
                    help="Use GPT-4o for value maps (recommended)")
    ap.add_argument("--no-blind", dest="blind", action="store_false",
                    help="Non-blind mode (agent knows goal)")
    ap.add_argument("--no-vlm", dest="vlm_vmap", action="store_false",
                    help="Use BLIP-2 instead of GPT-4o for value maps")
    ap.add_argument("--output", type=str, default=None,
                    help="Custom output directory (default: dataset's results/)")

    args = ap.parse_args()

    output_dir = Path(args.output) if args.output else Path(DATASET_RESULTS[args.dataset])

    splits = ["dev", "val", "test"] if args.all else [args.split]
    # Some datasets don't have all splits
    for s in list(splits):
        ep_file = Path(DATASET_SPLITS[args.dataset]) / f"episodes_{s}.json.gz"
        if not ep_file.exists():
            ep_file = Path(DATASET_SPLITS[args.dataset]) / f"episodes_{s}.json"
        if not ep_file.exists():
            print(f"  ⚠ Split '{s}' not found, skipping")
            splits.remove(s)

    print(f"\n{'='*60}")
    print(f"  VLFM Evaluation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Splits:  {splits}")
    print(f"  Mode:    blind={args.blind} vlm_vmap={args.vlm_vmap}")
    print(f"  VLM API: {vlfm.VLM_MODEL} @ {vlfm.VLM_BASE_URL}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}")

    all_results = {}
    for split_name in splits:
        results = evaluate_split(
            args.dataset, split_name,
            max_episodes=args.max,
            blind=args.blind,
            use_vlm_vmap=args.vlm_vmap,
            output_dir=output_dir,
        )
        all_results[split_name] = results

    # Final summary across splits
    if len(splits) > 1:
        print(f"\n{'='*60}")
        print(f"  OVERALL SUMMARY — {args.dataset}")
        print(f"{'='*60}")
        for split_name in splits:
            res = all_results[split_name]
            n_success = sum(1 for r in res if r.get("success"))
            n_total = len(res)
            print(f"  {split_name.upper():<8s} {n_success}/{n_total} "
                  f"({n_success/n_total*100:.1f}%)" if n_total > 0 else f"  {split_name}: 0 eps")
        print(f"{'='*60}\n")
