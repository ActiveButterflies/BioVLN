"""Rebuild scene NavMeshes for the revised 0.18 m circular robot base.

Original 0.10 m NavMeshes are left untouched. Recast settings match the
supplement's config.py except for ``agent_radius``. Output files and hashes
are kept in a separate revision directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

if not os.environ.get("DISPLAY") and not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import habitat_sim


NAVMESH_SETTINGS = {
    "cell_size": 0.02,
    "cell_height": 0.03,
    "agent_radius": 0.18,
    "agent_height": 1.7,
    "agent_max_climb": 0.2,
    "agent_max_slope": 45.0,
    "region_min_size": 2,
    "region_merge_size": 5,
    "edge_max_len": 6.0,
    "edge_max_error": 0.5,
    "verts_per_poly": 6.0,
    "detail_sample_dist": 1.0,
    "detail_sample_max_error": 0.1,
    "filter_low_hanging_obstacles": True,
    "filter_ledge_spans": True,
    "filter_walkable_low_height_spans": True,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_names = set()
    for split in ("train", "val", "test"):
        source = args.splits_dir / f"episodes_procedural_full_{split}_clean.json.gz"
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            scene_names.update(ep["scene_name"] for ep in json.load(handle)["episodes"])
    spec_files = [args.scenes_root / name / f"{name}_spec.json"
                  for name in sorted(scene_names)]
    missing = [str(path) for path in spec_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing scene specs: {missing}")
    if args.max_scenes:
        spec_files = spec_files[:args.max_scenes]
    records = []
    for spec_path in spec_files:
        scene_name = spec_path.parent.name
        source_glb = spec_path.parent / f"{scene_name}.glb"
        output = args.output_dir / f"{scene_name}.navmesh"
        if output.exists():
            data = output.read_bytes()
            records.append({"scene": scene_name, "navmesh": str(output),
                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "status": "existing"})
            print(f"{scene_name}: existing", flush=True)
            continue
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(source_glb)
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.load_semantic_mesh = False
        sim_cfg.pbr_image_based_lighting = False
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 1.7
        agent_cfg.radius = NAVMESH_SETTINGS["agent_radius"]
        agent_cfg.sensor_specifications = []
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        try:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            for key, value in NAVMESH_SETTINGS.items():
                setattr(settings, key, value)
            ok = sim.recompute_navmesh(sim.pathfinder, settings)
            if not ok or not sim.pathfinder.is_loaded:
                raise RuntimeError(f"NavMesh rebuild failed: {scene_name}")
            sim.pathfinder.save_nav_mesh(str(output))
            islands = int(sim.pathfinder.num_islands)
        finally:
            sim.close()
        data = output.read_bytes()
        records.append({"scene": scene_name, "navmesh": str(output),
                        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                        "num_islands": islands, "status": "created"})
        print(f"{scene_name}: {len(data)} bytes, {islands} islands", flush=True)
    manifest = {"source": "BioVLN-Revision-20",
                "scene_names_from_splits": sorted(scene_names),
                "settings": NAVMESH_SETTINGS,
                "scene_count": len(records), "files": records}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(records)} NavMeshes to {args.output_dir}")


if __name__ == "__main__":
    main()
