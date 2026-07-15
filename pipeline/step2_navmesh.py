#!/usr/bin/env python3
"""
Step 2: Generate NavMesh for the scene using habitat-sim Recast/Detour.

The NavMesh is saved to a .navmesh file and can be used by step3 and step4.

Prerequisites: step1_semantic.py must be run first.

Usage:
    python3 step2_navmesh.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NAVMESH_SETTINGS, SCENE_GLB, OUTPUT_DIR, SCENE_NAME,
    derived_paths,
)


def main():
    paths = derived_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Step 2: NavMesh Generation")
    print(f"  Scene:  {SCENE_GLB}")
    print(f"  Output: {paths['navmesh']}")
    print()

    import habitat_sim

    # ── Create simulator with scene config ────────────────────────────────
    print("Creating simulator...", flush=True)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(SCENE_GLB)
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = 1.5
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # ── Get scene info ───────────────────────────────────────────────────
    sg = sim.get_active_scene_graph()
    root = sg.get_root_node()
    scene_bb = root.compute_cumulative_bb()
    print(f"  Scene bounds: min={scene_bb.min}, max={scene_bb.max}")
    print(f"  Scene size:   {scene_bb.size()}")

    # ── Configure NavMesh settings ───────────────────────────────────────
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()

    for key, value in NAVMESH_SETTINGS.items():
        setattr(ns, key, value)

    print(f"\n  NavMesh parameters:")
    for key, value in NAVMESH_SETTINGS.items():
        print(f"    {key}: {value}")

    # ── Recompute NavMesh ────────────────────────────────────────────────
    print("\n  Recomputing NavMesh...", flush=True)
    sim.recompute_navmesh(sim.pathfinder, ns)
    pf = sim.pathfinder

    # ── Analyze NavMesh islands ──────────────────────────────────────────
    print(f"\n  NavMesh islands: {pf.num_islands}")

    islands = []
    for i in range(pf.num_islands):
        v = np.array(pf.build_navmesh_vertices(i)).reshape(-1, 3)
        area = (v[:, 0].max() - v[:, 0].min()) * (v[:, 2].max() - v[:, 2].min())
        islands.append({
            'id': i,
            'y_mean': float(v[:, 1].mean()),
            'y_min': float(v[:, 1].min()),
            'y_max': float(v[:, 1].max()),
            'area': float(area),
            'verts': len(v),
        })

    # Floor = large island(s) at low Y
    low_y = min(d['y_mean'] for d in islands)

    for d in sorted(islands, key=lambda x: x['y_mean']):
        is_floor = d['y_mean'] < low_y + 0.5 and d['area'] > 5
        label = "  ← FLOOR" if is_floor else ""
        print(f"    Island {d['id']}: Y={d['y_mean']:.2f} "
              f"[{d['y_min']:.2f}–{d['y_max']:.2f}] "
              f"area={d['area']:.1f} m²  {d['verts']} verts{label}")

    floor_verts = sum(d['verts'] for d in islands
                      if d['y_mean'] < low_y + 0.5 and d['area'] > 5)

    # ── Save NavMesh ─────────────────────────────────────────────────────
    pf.save_nav_mesh(str(paths['navmesh']))
    print(f"\n  ✓ NavMesh saved: {paths['navmesh']}")
    print(f"    Floor vertices: {floor_verts}")
    print(f"    NavMesh loaded: {pf.is_loaded}")

    # Quick sanity check
    bounds = pf.get_bounds()
    print(f"    NavMesh bounds: {bounds}")

    sim.close()
    print(f"\n✓ Step 2 complete.")


if __name__ == "__main__":
    main()
