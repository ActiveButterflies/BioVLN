#!/usr/bin/env python3
"""
Step 3: Generate ObjectNav episodes from NavMesh and semantic objects.

Samples starting positions from the NavMesh, creates episodes for each
target category, and writes compressed + human-readable JSON.

Prerequisites: step2_navmesh.py must be run first.

Usage:
    python3 step3_episodes.py
"""

import json
import gzip
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SCENE_GLB, SCENE_NAME, OUTPUT_DIR,
    TARGET_CATEGORIES, GOALS_BY_CATEGORY,
    EPISODES_PER_CATEGORY, NUM_START_POSITIONS,
    MIN_START_SPACING, RANDOM_SEED,
    derived_paths,
)


def glb_to_world(pos):
    """Convert GLB Z-up coords to habitat-sim Y-up world coords.
    FromTwoVectors(-Z, -Y): (x, y, z)_glb → (x, z, -y)_world
    """
    x, y, z = pos
    return [x, z, -y]


def random_yaw_quat(rng):
    """Random rotation quaternion around Y/up axis (x, y, z, w)."""
    angle = rng.uniform(0, 2 * np.pi)
    half = angle / 2
    return [0.0, float(np.sin(half)), 0.0, float(np.cos(half))]


def main():
    paths = derived_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Step 3: Episode Generation")
    print(f"  Scene:      {SCENE_GLB}")
    print(f"  Categories: {len(TARGET_CATEGORIES)}")
    print(f"  Ep/cat:     {EPISODES_PER_CATEGORY}")
    print(f"  Starts:     {NUM_START_POSITIONS} (min spacing {MIN_START_SPACING}m)")
    print()

    rng = np.random.RandomState(RANDOM_SEED)

    import habitat_sim
    import os
    os.chdir(OUTPUT_DIR)

    # ── Load simulator ────────────────────────────────────────────────────
    print("Loading simulator with NavMesh...", flush=True)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(SCENE_GLB.name)  # we chdir'd to OUTPUT_DIR
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = 1.5
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    pf = sim.pathfinder
    pf.seed(RANDOM_SEED)  # Deterministic per-scene start positions

    print(f"  NavMesh loaded: {pf.is_loaded} (seed={RANDOM_SEED})")
    if not pf.is_loaded:
        print("  ❌ NavMesh not loaded. Run step2_navmesh.py first.")
        sim.close()
        sys.exit(1)

    # ── Optional custom config ──────────────────────────────────────────
    try:
        from config import CUSTOM_STARTS, CUSTOM_GOALS
    except ImportError:
        CUSTOM_STARTS = None
        CUSTOM_GOALS = None

    # ── Convert goals to world coords ─────────────────────────────────────
    goals_world = {}

    if CUSTOM_GOALS and len(CUSTOM_GOALS) > 0:
        # ── Custom goals (GLB Z-up) override auto-derived goals ─────────
        print(f"\n  Using custom goals for categories: {list(CUSTOM_GOALS.keys())}")
        for cat, cgoals in CUSTOM_GOALS.items():
            goals_world[cat] = []
            for g in cgoals:
                gw = dict(g)
                gw["position"] = glb_to_world(g["position"])
                # Use object_id from annotation if not provided
                if "object_id" not in gw:
                    gw["object_id"] = TARGET_CATEGORIES.get(cat, 0)
                if "object_name" not in gw:
                    gw["object_name"] = f"{cat}_custom"
                if "object_category" not in gw:
                    gw["object_category"] = cat
                goals_world[cat].append(gw)
        # For categories NOT in custom goals, fall back to auto
        for cat, goals in GOALS_BY_CATEGORY.items():
            if cat not in goals_world:
                goals_world[cat] = []
                for g in goals:
                    gw = dict(g)
                    # GOALS_BY_CATEGORY is now in GLB Y-up (matches scene), no conversion needed
                    gw["position"] = g["position"]
                    goals_world[cat].append(gw)
    else:
        # ── Auto-derived goals (already in GLB Y-up, no glb_to_world) ──
        for cat, goals in GOALS_BY_CATEGORY.items():
            goals_world[cat] = []
            for g in goals:
                gw = dict(g)
                # GLB is already Y-up (export_yup=True), goal position matches scene
                gw["position"] = list(g["position"])
                goals_world[cat].append(gw)

    # ── Snap all goals to nearest navigable NavMesh point ───────────────
    print(f"\n  Snapping goals to NavMesh...")
    total_snapped = 0
    for cat, goals in goals_world.items():
        for g in goals:
            pos = g["position"]
            snapped = pf.snap_point(pos)
            # Only accept floor-level snaps (Y < 0.5) — reject furniture-top islands
            if snapped is not None and pf.is_navigable(snapped) and snapped[1] < 0.5:
                g["position"] = [float(v) for v in snapped]
                total_snapped += 1
            else:
                # Try to find nearest FLOOR-LEVEL navigable point via search
                best = None
                best_dist = float('inf')
                for _ in range(500):
                    rp = pf.get_random_navigable_point()
                    if rp[1] > 0.5:  # skip furniture-top islands
                        continue
                    d = sum((rp[i] - pos[i])**2 for i in range(3))
                    if d < best_dist:
                        best_dist = d
                        best = rp
                if best is not None:
                    g["position"] = [float(v) for v in best]
                    total_snapped += 1
    print(f"    {total_snapped}/{sum(len(v) for v in goals_world.values())} goals snapped")

    # ── Resolve start positions ───────────────────────────────────────────

    start_positions = []
    start_rotations = []
    start_mode = "auto"

    if CUSTOM_STARTS and len(CUSTOM_STARTS) > 0:
        # ── Custom start positions (GLB Z-up, converted below) ──────────
        start_mode = "custom"
        print(f"\n  Using {len(CUSTOM_STARTS)} custom start positions...")
        for cs in CUSTOM_STARTS:
            if isinstance(cs, dict):
                pos = cs.get("position", cs.get("pos", [0, 0, 0]))
                # Rotation: optional Euler angles in degrees (GLB Z-up)
                rot_deg = cs.get("rotation", cs.get("rot", None))
            elif isinstance(cs, (list, tuple)) and len(cs) >= 3:
                pos = list(cs[:3])
                rot_deg = list(cs[3:6]) if len(cs) >= 6 else None
            else:
                continue
            wpos = glb_to_world(pos)
            if pf.is_navigable(wpos):
                start_positions.append(wpos)
                start_rotations.append(rot_deg)
            else:
                print(f"    ⚠  Custom start not navigable: GLB {pos} → world {wpos}, skipping")
    else:
        # ── Auto-sample navigable start positions ───────────────────────
        print(f"\n  Sampling {NUM_START_POSITIONS} start positions...")

        attempts = 0
        while len(start_positions) < NUM_START_POSITIONS and attempts < 2000:
            pt = pf.get_random_navigable_point()
            pt = [float(v) for v in pt]

            # Floor-level only
            if pt[1] > 0.5 or pt[1] < -0.2:
                attempts += 1
                continue

            # Minimum spacing between starts
            too_close = False
            for existing in start_positions:
                dx = pt[0] - existing[0]
                dz = pt[2] - existing[2]
                if (dx * dx + dz * dz) < (MIN_START_SPACING ** 2):
                    too_close = True
                    break

            if not too_close:
                start_positions.append(pt)
                start_rotations.append(None)  # random

            attempts += 1

    print(f"  Got {len(start_positions)} positions (mode: {start_mode})")

    if not start_positions:
        print("  ❌ No valid start positions found. Check NavMesh or CUSTOM_STARTS.")
        sim.close()
        sys.exit(1)

    # ── Generate episodes ─────────────────────────────────────────────────
    print(f"\n  Generating episodes...")

    category_mapping = {name: i for i, name in enumerate(sorted(TARGET_CATEGORIES.keys()))}

    episodes = []
    episode_id = 0

    for category in TARGET_CATEGORIES:
        goals = goals_world.get(category, [])
        if not goals:
            print(f"    ⚠  No goals for '{category}', skipping")
            continue

        goals_with_viewpoints = []
        for g in goals:
            # Build view_point from facing_yaw (computed by auto_config)
            yaw = g.get("facing_yaw", 0.0)
            half = yaw / 2.0
            vp = {
                "agent_state": {
                    "position": list(g["position"]),
                    "rotation": [0.0, float(np.sin(half)), 0.0, float(np.cos(half))],
                }
            }
            goals_with_viewpoints.append({
                "position": g["position"],
                "object_id": g["object_id"],
                "object_name": g["object_name"],
                "object_category": g["object_category"],
                "view_points": [vp],
            })

        for i, start_pos in enumerate(start_positions):
            if i >= EPISODES_PER_CATEGORY:
                break

            goal_pos = np.array(goals[0]["position"])
            start = np.array(start_pos)
            euclidean_dist = float(np.linalg.norm(goal_pos - start))

            # Rotation: custom or random
            if i < len(start_rotations) and start_rotations[i] is not None:
                rot = start_rotations[i]
                # Convert Euler degrees (rx, ry, rz around GLB Z-up) → world Y-up quaternion
                angle_deg = rot[2] if len(rot) > 2 else rot[0]  # z-axis rotation in Z-up → y-axis in Y-up
                half = np.radians(angle_deg) / 2
                start_rot = [0.0, float(np.sin(half)), 0.0, float(np.cos(half))]
            else:
                start_rot = random_yaw_quat(rng)

            ep = {
                "episode_id": str(episode_id),
                "scene_id": f"{SCENE_NAME}.glb",
                "scene_dataset_config": paths['scene_config'].name,
                "start_position": start_pos,
                "start_rotation": start_rot,
                "info": {
                    "geodesic_distance": -1.0,
                    "euclidean_distance": euclidean_dist,
                },
                "goals": goals_with_viewpoints,
                "start_room": None,
                "shortest_paths": None,
                "object_category": category,
            }
            episodes.append(ep)
            episode_id += 1

    # ── Build dataset ─────────────────────────────────────────────────────
    dataset = {
        "episodes": episodes,
        "category_to_task_category_id": category_mapping,
        "category_to_scene_annotation_category_id": {},
    }

    # Write compressed
    with gzip.open(str(paths['episodes_gz']), 'wt', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    print(f"  ✓ {paths['episodes_gz']}")

    # Write uncompressed for inspection
    with open(paths['episodes_json'], 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"  ✓ {paths['episodes_json']}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generated Episodes")
    print(f"{'='*60}")
    print(f"  Total: {len(episodes)}")
    for cat in TARGET_CATEGORIES:
        count = sum(1 for ep in episodes if ep["object_category"] == cat)
        goal_list = GOALS_BY_CATEGORY.get(cat, [{}])
        goal = goal_list[0].get("object_name", "?") if goal_list else "?"
        print(f"  {cat:20}  {count:3d} ep  → {goal}")

    # Verify navigability
    all_nav = all(pf.is_navigable(ep["start_position"]) for ep in episodes)
    print(f"\n  All starts navigable: {all_nav}")

    sim.close()
    print(f"\n✓ Step 3 complete.")


if __name__ == "__main__":
    main()
