#!/usr/bin/env python3
"""
Ablation: Centroid-projected goals vs Operational-face goals.

Replaces each episode's operational-face goal with a centroid goal
(instrument center XY projected to floor Z=0), then runs Frontier Exploration.

This tests whether BioVLN's operational-face goal formulation produces
meaningfully different navigation outcomes compared to the standard
ObjectNav centroid-projection approach.
"""

import json, gzip, os, sys, time, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────
MULTI_SCENE = Path("/root/gpufree-data/multi_scene")
SPECS_DIR = MULTI_SCENE / "specs"
SPLITS_DIR = MULTI_SCENE / "output" / "splits"
RESULTS_DIR = MULTI_SCENE / "results" / "ablation_centroid"
OUTPUT_DIR = MULTI_SCENE / "output" / "ablation_centroid"

# Splits to process
SPLITS = {
    "dev": SPLITS_DIR / "episodes_dev.json.gz",
    "val": SPLITS_DIR / "episodes_val.json.gz",
    "test": SPLITS_DIR / "episodes_test.json.gz",
}

# Two-Room and LSAT Target
TWO_ROOM_EP = Path("/root/gpufree-data/two_room_output/splits/episodes_dev.json.gz")
LSAT_EP = None  # LSAT Target episodes are embedded in the scene

# ─── Coordinate conversion ─────────────────────────────────────────────────
def blender_to_glb_yup(blender_pos):
    """Blender Z-up (X, Y_bl, Z_bl) → GLB Y-up (X, Z_bl, -Y_bl)."""
    x, y_bl, z_bl = blender_pos
    return [x, z_bl, -y_bl]

# ─── Load scene spec → instrument positions ─────────────────────────────────
def load_spec_instruments(spec_path):
    """Return {label: {position, category}} from a scene spec."""
    if not os.path.exists(spec_path):
        return {}
    with open(spec_path) as f:
        spec = json.load(f)
    result = {}
    for inst in spec.get("instruments", []):
        label = inst.get("label", "")
        result[label] = {
            "position": inst.get("position", [0, 0, 0]),
            "category": inst.get("category", ""),
            "operational_goal_position": inst.get("operational_goal_position"),
        }
    return result


def generate_centroid_episodes(split_name, episode_path, output_dir):
    """Generate episodes with centroid goals for a Multi-Scene split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(episode_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"]
    new_episodes = []
    spec_cache = {}
    stats = {"total": 0, "matched": 0, "unmatched": 0}

    for ep in episodes:
        scene_id = ep["scene_id"]
        target_cat = ep["object_category"]
        target_name = ep["goals"][0]["object_name"]  # e.g., "refrigerator_0"

        # Load spec (cache by scene_id)
        if scene_id not in spec_cache:
            # scene_id format: "lab_single_room_s150_v0.glb"
            scene_base = scene_id.replace(".glb", "")
            spec_path = SPECS_DIR / f"{scene_base}_spec.json"
            spec_cache[scene_id] = load_spec_instruments(spec_path)

        instruments = spec_cache[scene_id]

        # Find matching instrument
        inst_info = instruments.get(target_name)
        if inst_info is None:
            # Try matching by category only
            for label, info in instruments.items():
                if info["category"] == target_cat:
                    inst_info = info
                    target_name = label
                    break

        if inst_info is None:
            stats["unmatched"] += 1
            new_episodes.append(ep)  # keep original
            continue

        stats["matched"] += 1

        # Compute centroid goal: instrument center XY → floor Z=0
        blender_pos = inst_info["position"]
        centroid_blender = [blender_pos[0], blender_pos[1], 0.0]
        centroid_glb = blender_to_glb_yup(centroid_blender)

        # Create new goal
        new_goal = dict(ep["goals"][0])
        new_goal["position"] = centroid_glb
        new_goal["object_category"] = target_cat
        new_goal["goal_type"] = "centroid"  # mark for tracking

        # Create new episode
        new_ep = dict(ep)
        new_ep["goals"] = [new_goal]
        new_ep["goal_type"] = "centroid"
        new_episodes.append(new_ep)
        stats["total"] += 1

    # Write new episode file
    new_data = dict(data)
    new_data["episodes"] = new_episodes
    new_data["goal_type"] = "centroid"

    out_path = output_dir / f"episodes_{split_name}_centroid.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(new_data, f)

    print(f"  {split_name}: {stats['matched']}/{len(episodes)} matched, "
          f"{stats['unmatched']} unmatched")
    return out_path


def generate_centroid_episodes_two_room(output_dir):
    """Two-Room: use config_two_room instrument positions as centroids."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Two-Room instrument positions (from config_two_room.py / GLB extraction)
    # These are the CENTER positions in GLB Y-up habitat coordinates
    # Format: (cx, cz) in GLB Y-up XZ plane (Y is up)
    two_room_centroids = {
        "cabinet": (5.45, -1.5),
        "centrifuge": (-3.5, -1.7),
        "incubator": (5.45, -2.3),
        "lab_bench": (-3.5, -2.3),
        "lab_table": (0.0, 0.5),
        "lab_table_1": (5.5, 0.0),
        "mini_centrifuge": (-2.8, -2.3),
        "pipette": (-3.0, -2.3),
        "refrigerator": (-3.5, 1.0),
        "waste_bin": (5.4, -1.0),
        "lab_waste_bin": (5.4, -1.0),
    }

    if not TWO_ROOM_EP.exists():
        print("  Two-Room: episode file not found, skipping")
        return None

    with gzip.open(TWO_ROOM_EP, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"]
    new_episodes = []
    matched = 0

    for ep in episodes:
        target_cat = ep["object_category"]

        centroid_xy = two_room_centroids.get(target_cat)
        if centroid_xy is None:
            # Try with underscore variant
            centroid_xy = two_room_centroids.get(target_cat.replace("_", " "))
        if centroid_xy is None:
            new_episodes.append(ep)
            continue

        matched += 1
        centroid_glb = [centroid_xy[0], 0.03, centroid_xy[1]]  # Y=0.03 floor

        new_goal = dict(ep["goals"][0])
        new_goal["position"] = centroid_glb
        new_goal["goal_type"] = "centroid"

        new_ep = dict(ep)
        new_ep["goals"] = [new_goal]
        new_ep["goal_type"] = "centroid"
        new_episodes.append(new_ep)

    new_data = dict(data)
    new_data["episodes"] = new_episodes
    new_data["goal_type"] = "centroid"

    out_path = output_dir / "episodes_two_room_centroid.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(new_data, f)

    print(f"  two_room: {matched}/{len(episodes)} matched")
    return out_path


def generate_centroid_episodes_lsat(output_dir):
    """LSAT Target: use human-annotated target positions as centroids.

    The LSAT target_* empties ARE the instrument centers in the baked GLB.
    For centroid goal, we use those positions projected to floor.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lsat_ep_path = Path("/root/gpufree-data/lsat_test/splits/episodes_dev.json.gz")
    if not lsat_ep_path.exists():
        print("  LSAT Target: episode file not found, skipping")
        return None

    with gzip.open(lsat_ep_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # Load LSAT instrument positions from scene manifest
    lsat_goals_file = Path("/root/lsat/examples/goals_from_target_2.json")
    lsat_positions = {}
    if lsat_goals_file.exists():
        with open(lsat_goals_file) as f:
            lsat_data = json.load(f)
        for cat, goals in lsat_data.items():
            if goals:
                lsat_positions[cat] = goals[0]["position"]

    episodes = data["episodes"]
    new_episodes = []
    matched = 0

    for ep in episodes:
        target_cat = ep["object_category"]

        # Try to get centroid from LSAT positions, or use existing goal with Y=0
        if target_cat in lsat_positions:
            pos = lsat_positions[target_cat]
            centroid_glb = [pos[0], 0.03, pos[2]]
            matched += 1
        else:
            # Fallback: use existing goal but set to floor Y
            old_pos = ep["goals"][0]["position"]
            centroid_glb = [old_pos[0], 0.03, old_pos[2]]

        new_goal = dict(ep["goals"][0])
        new_goal["position"] = centroid_glb
        new_goal["goal_type"] = "centroid"

        new_ep = dict(ep)
        new_ep["goals"] = [new_goal]
        new_ep["goal_type"] = "centroid"
        new_episodes.append(new_ep)

    new_data = dict(data)
    new_data["episodes"] = new_episodes
    new_data["goal_type"] = "centroid"

    out_path = output_dir / "episodes_lsat_centroid.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(new_data, f)

    print(f"  lsat_target: {matched}/{len(episodes)} matched")
    return out_path


# ─── Run Frontier Exploration ──────────────────────────────────────────────
def run_frontier_eval(episode_file, scene_glb_dir, output_path, navmesh_dir=None,
                      max_steps=500):
    """Run Frontier Exploration on centroid-goal episodes."""
    # This is the same as the existing run_frontier.py but with different episode file
    cmd = (
        f"cd {MULTI_SCENE} && "
        f"PYTHONPATH={MULTI_SCENE.parent}:$PYTHONPATH "
        f"/opt/conda/envs/habitat/bin/python3 {MULTI_SCENE}/run_frontier.py "
        f"--episodes {episode_file} "
        f"--scene-dir {scene_glb_dir} "
        f"--output {output_path} "
        f"--max-steps {max_steps}"
    )
    if navmesh_dir:
        pass  # navmesh is auto-discovered from scene dir
    print(f"  Running: {cmd[:150]}...")
    ret = os.system(cmd)
    return ret == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate episodes, don't run evaluation")
    parser.add_argument("--run-only", action="store_true",
                        help="Only run evaluation on pre-generated episodes")
    parser.add_argument("--datasets", nargs="+", default=["dev", "val", "test"],
                        help="Which splits to process")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Generate centroid-goal episodes ────────────────────
    if not args.run_only:
        print("=" * 60)
        print("STEP 1: Generating centroid-goal episodes")
        print("=" * 60)

        episode_files = {}

        for split in args.datasets:
            if split in SPLITS:
                ep_path = generate_centroid_episodes(split, SPLITS[split], OUTPUT_DIR)
                if ep_path:
                    episode_files[f"multi_{split}"] = ep_path

        # Two-Room
        tr_path = generate_centroid_episodes_two_room(OUTPUT_DIR)
        if tr_path:
            episode_files["two_room"] = tr_path

        # LSAT Target
        lsat_path = generate_centroid_episodes_lsat(OUTPUT_DIR)
        if lsat_path:
            episode_files["lsat_target"] = lsat_path

        print(f"\nGenerated {len(episode_files)} episode files")
        for name, path in episode_files.items():
            print(f"  {name}: {path}")
    else:
        # Find pre-generated files
        episode_files = {}
        for f in OUTPUT_DIR.glob("episodes_*_centroid.json.gz"):
            name = f.stem.replace("episodes_", "").replace("_centroid", "")
            if not name.startswith("two_room") and not name.startswith("lsat"):
                name = f"multi_{name}"
            elif name.startswith("two_room"):
                name = "two_room"
            elif name.startswith("lsat"):
                name = "lsat_target"
            episode_files[name] = f

    if args.generate_only:
        print("\nDone (generate-only mode)")
        return

    # ── STEP 2: Run Frontier Exploration ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Running Frontier Exploration on centroid goals")
    print("=" * 60)

    scene_dirs = {
        "multi": str(MULTI_SCENE / "output"),
        "two_room": "/root/gpufree-data/two_room_output",
        "lsat_target": "/root/gpufree-data/lsat_test",
    }

    all_results = {}

    for name, ep_path in sorted(episode_files.items()):
        if name.startswith("multi_"):
            split = name.replace("multi_", "")
            scene_dir = scene_dirs["multi"]
            out_path = str(RESULTS_DIR / f"eval_frontier_centroid_{split}.json")
        elif name == "two_room":
            scene_dir = scene_dirs["two_room"]
            out_path = str(RESULTS_DIR / "eval_frontier_centroid_two_room.json")
        elif name == "lsat_target":
            scene_dir = scene_dirs["lsat_target"]
            out_path = str(RESULTS_DIR / "eval_frontier_centroid_target.json")
        else:
            continue

        print(f"\n--- {name} ---")
        print(f"  Episodes: {ep_path}")
        print(f"  Scenes: {scene_dir}")
        print(f"  Output: {out_path}")

        success = run_frontier_eval(str(ep_path), scene_dir, out_path)
        if success and os.path.exists(out_path):
            with open(out_path) as f:
                data = json.load(f)
            eps = data.get("episode_results", [])
            succ = sum(1 for r in eps if r.get("success"))
            sr = succ / len(eps) * 100 if eps else 0
            print(f"  Result: {succ}/{len(eps)} ({sr:.1f}%)")
            all_results[name] = {"sr": sr, "success": succ, "total": len(eps)}
        else:
            print(f"  FAILED or no output")

    # ── STEP 3: Compare with operational-face results ──────────────
    print("\n" + "=" * 60)
    print("COMPARISON: Centroid vs Operational-Face Goals")
    print("=" * 60)

    op_face_results = {
        "multi_dev": ("/root/gpufree-data/multi_scene/results/eval_frontier_dev.json", "Multi-Scene DEV"),
        "multi_val": ("/root/gpufree-data/multi_scene/results/eval_frontier_val.json", "Multi-Scene VAL"),
        "multi_test": ("/root/gpufree-data/multi_scene/results/eval_frontier_test.json", "Multi-Scene TEST"),
        "two_room": ("/root/gpufree-data/two_room_output/results/eval_frontier_two_room.json", "Two-Room"),
        "lsat_target": ("/root/gpufree-data/lsat_test/results/eval_frontier_target.json", "LSAT Target"),
    }

    print(f"\n{'Dataset':<20} {'Centroid SR':>12} {'Op-Face SR':>12} {'Δ':>10}")
    print("-" * 56)

    for key, (op_path, label) in op_face_results.items():
        c_path = None
        if key.startswith("multi_"):
            split = key.replace("multi_", "")
            c_path = RESULTS_DIR / f"eval_frontier_centroid_{split}.json"
        elif key == "two_room":
            c_path = RESULTS_DIR / "eval_frontier_centroid_two_room.json"
        elif key == "lsat_target":
            c_path = RESULTS_DIR / "eval_frontier_centroid_target.json"

        op_sr = 0
        if os.path.exists(op_path):
            with open(op_path) as f:
                d = json.load(f)
            eps = d.get("episode_results", [])
            op_sr = sum(1 for r in eps if r.get("success")) / len(eps) * 100 if eps else 0

        c_sr = 0
        if c_path and os.path.exists(c_path):
            with open(c_path) as f:
                d = json.load(f)
            eps = d.get("episode_results", [])
            c_sr = sum(1 for r in eps if r.get("success")) / len(eps) * 100 if eps else 0

        delta = c_sr - op_sr
        print(f"{label:<20} {c_sr:>8.1f}%   {op_sr:>8.1f}%   {delta:>+8.1f}pp")

    # ── Summary JSON ───────────────────────────────────────────────
    summary = {
        "ablation": "centroid_vs_operational_face",
        "description": "Frontier Exploration SR with centroid-projected goals vs operational-face goals",
        "results": all_results,
    }
    summary_path = RESULTS_DIR / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
