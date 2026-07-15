#!/usr/bin/env python3
"""
Auto-config generator for the BioVLN pipeline.

Reads a lab_spec.json and mesh_manifest.json (from LabBuilder),
generates the runtime configuration needed by the existing pipeline steps.

Usage:
  python auto_config.py --spec lab_spec.json --manifest scene_manifest.json \\
      --scene /path/to/scene.glb --out /path/to/output/
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def glb_to_world(pos):
    """Convert GLB Z-up coords to habitat-sim Y-up world coords."""
    x, y, z = pos
    return [x, z, -y]


def _make_semantic_annotations(mesh_manifest: List[dict]) -> Dict[int, tuple]:
    """
    Build SEMANTIC_ANNOTATIONS dict from mesh manifest.
    Assigns object_id sequentially (1-based).

    Returns: {mesh_idx: (object_id, label, region_id)}
    """
    annotations = {}
    for idx, entry in enumerate(mesh_manifest):
        obj_id = idx + 1
        label = entry.get("category", entry.get("label", f"obj{obj_id}"))
        region_id = 1   # all in one region
        annotations[idx] = (obj_id, label, region_id)
    return annotations


def _make_target_categories(mesh_manifest: List[dict],
                            target_cats: List[str]) -> Dict[str, int]:
    """
    Map target category names to their object_id from SEMANTIC_ANNOTATIONS.

    target_cats: list of category strings (e.g., ["centrifuge", "refrigerator"])
    """
    annotations = _make_semantic_annotations(mesh_manifest)
    targets = {}
    for cat in target_cats:
        # Find all mesh entries with this category
        matches = [(idx, obj_id, label) for idx, (obj_id, label, region)
                   in annotations.items() if label == cat]
        if not matches:
            # Try case-insensitive match
            matches = [(idx, obj_id, label) for idx, (obj_id, label, region)
                       in annotations.items() if label.lower() == cat.lower()]
        if matches:
            targets[cat] = matches[0][1]  # first match's object_id
        else:
            print(f"  ⚠  Target category '{cat}' not found in mesh manifest")
    return targets


# Floor height in GLB Y-up coordinates (same as habitat-sim world Y)
FLOOR_Y = 0.0


def _blender_to_glb_yup(blender_pos):
    """Convert Blender Z-up coordinate to GLB Y-up.

    Blender  (Z-up):   (X, Y_bl, Z_bl)  where Z_bl = up
    GLB Y-up after export_yup=True:  (X, Z_bl, -Y_bl)  where Y = up
    """
    x, y_bl, z_bl = blender_pos
    return [x, z_bl, -y_bl]


def _make_goals_by_category(instrument_specs: List[dict],
                            mesh_manifest: List[dict],
                            target_cats: List[str]) -> Dict[str, List[dict]]:
    """
    Generate goal positions from instrument spec.
    Prefers spec["operational_goal_position"] (Blender Z-up, floor level);
    falls back to bbox center from mesh_manifest.
    Converts to GLB Y-up and forces Y to FLOOR_Y for robot navigation.

    Each goal includes facing_yaw (radians) — the yaw the agent should
    face to look toward the instrument from the goal position.
    """
    import math
    goals = {}
    for cat in target_cats:
        goals[cat] = []
        for item in instrument_specs:
            if item.get("category", "").lower() != cat.lower():
                continue
            item_label = item.get("label", "")

            # ── Goal position (Blender Z-up → GLB Y-up) ──
            goal_blender = item.get("operational_goal_position")
            if goal_blender is not None:
                # Use pre-computed operational goal from scene_generator
                pos = _blender_to_glb_yup(goal_blender)
                pos[1] = FLOOR_Y  # ensure floor level

                # Compute facing yaw: from goal toward instrument center
                inst_ctr = item.get("position", [0, 0, 0])
                dx = inst_ctr[0] - goal_blender[0]
                dy = inst_ctr[1] - goal_blender[1]
                facing_yaw = math.atan2(dx, -dy)  # Blender XY → habitat Y-up yaw
            else:
                # Fallback: bbox center from manifest
                pos = [0, 0, 0]
                facing_yaw = 0.0
                for entry in mesh_manifest:
                    if entry.get("label") == item_label:
                        bmin = entry.get("bbox_min", [0, 0, 0])
                        bmax = entry.get("bbox_max", [0, 0, 0])
                        center_blender = [(bmin[i] + bmax[i]) / 2 for i in range(3)]
                        pos = _blender_to_glb_yup(center_blender)
                        pos[1] = FLOOR_Y
                        break

            goals[cat].append({
                "position": pos,
                "object_id": _find_object_id(cat, mesh_manifest),
                "object_name": f"{cat}_{len(goals[cat]) + 1}",
                "object_category": cat,
                "facing_yaw": facing_yaw,
            })
    return goals


def _find_object_id(category: str, mesh_manifest: List[dict]) -> int:
    """Find object_id for a category in the mesh manifest."""
    annotations = _make_semantic_annotations(mesh_manifest)
    for idx, (obj_id, label, region) in annotations.items():
        if label.lower() == category.lower():
            return obj_id
    return 0


def _make_instruments_viz(instrument_specs: List[dict],
                          target_cats: List[str],
                          mesh_manifest: List[dict] = None) -> List[tuple]:
    """
    Generate INSTRUMENTS list for step4 visualization.
    Format: (name, glb_x, glb_y, glb_z, radius, hex_color)
    Falls back to bbox center if spec has no 'position'.
    """
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
              "#e67e22", "#1abc9c", "#e91e63", "#00bcd4", "#ff5722"]

    instruments = []
    for i, item in enumerate(instrument_specs):
        name = item.get("label", item.get("file", f"inst_{i}"))
        item_label = item.get("label", "")
        # Use bbox center from manifest (reflects actual world position after overrides)
        pos = [0, 0, 0]
        if mesh_manifest:
            for entry in mesh_manifest:
                if entry.get("label") == item_label:
                    if "bbox_min" in entry and "bbox_max" in entry:
                        pos = [(entry["bbox_min"][j] + entry["bbox_max"][j]) / 2 for j in range(3)]
                    break
        # Estimate radius from spec or use default
        size = item.get("size", [0.5, 0.5, 0.5])
        radius = max(size) / 2 if isinstance(size, list) else 0.3
        color = item.get("color_hex", colors[i % len(colors)])
        instruments.append((name, pos[0], pos[1], pos[2], radius, color))

    return instruments


def _safe_repr(value):
    """Convert a Python value to a valid Python literal string."""
    if value is None:
        return "None"
    s = json.dumps(value, indent=4, ensure_ascii=False)
    s = s.replace("true", "True").replace("false", "False").replace("null", "None")
    return s


def make_navmesh_settings(overrides: Optional[dict] = None) -> dict:
    """Default NavMesh settings, optionally overridden."""
    settings = {
        "cell_size": 0.02,
        "cell_height": 0.03,
        "agent_radius": 0.1,
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
    if overrides:
        settings.update(overrides)
    return settings


def make_scene_dataset_config(scene_name: str) -> dict:
    """Generate habitat-sim scene dataset config with Z-up orientation."""
    return {
        "stages": {
            "paths": {
                ".glb": [f"{scene_name}.glb"]
            },
            "default_attributes": {
                "shader_type": "pbr",
                "semantic_descriptor_filename": f"{scene_name}.semantic.txt",
                "semantic_asset": f"{scene_name}.semantic.glb",
                "has_semantic_textures": True,
            }
        },
        "objects": {},
        "light_setups": {}
    }


def generate_runtime_config(
    spec: dict,
    mesh_manifest: List[dict],
    scene_glb: Path,
    output_dir: Path,
) -> str:
    """
    Generate the content of _runtime_config.py.

    Returns the Python source as a string.
    """
    scene_name = spec.get("scene_name", "generated_lab")
    target_cats = spec.get("target_categories", [])
    navmesh_overrides = spec.get("navmesh_settings", {})
    furniture_specs = spec.get("furniture", [])
    instrument_specs = spec.get("instruments", [])

    # Build config values
    semantic = _make_semantic_annotations(mesh_manifest)
    targets = _make_target_categories(mesh_manifest, target_cats)
    goals = _make_goals_by_category(instrument_specs, mesh_manifest, target_cats)
    instruments = _make_instruments_viz(instrument_specs, target_cats, mesh_manifest)
    navmesh = make_navmesh_settings(navmesh_overrides)
    scene_config = make_scene_dataset_config(scene_name)

    # Fragile objects = those with small size (equipment)
    fragile = {name for name, *_ in instruments if "table" not in name.lower()
               and "bench" not in name.lower()}

    # Convert SEMANTIC_ANNOTATIONS to Python dict literal
    semantic_lines = []
    for idx in sorted(semantic):
        obj_id, label, region = semantic[idx]
        semantic_lines.append(f"    {idx}: ({obj_id}, \"{label}\", {region}),")
    semantic_str = "{\n" + "\n".join(semantic_lines) + "\n}"

    # Convert TARGET_CATEGORIES
    targets_lines = []
    for cat, obj_id in sorted(targets.items()):
        targets_lines.append(f"    \"{cat}\": {obj_id},")
    targets_str = "{\n" + "\n".join(targets_lines) + "\n}"

    # Convert GOALS_BY_CATEGORY
    goals_str = json.dumps(goals, indent=4, ensure_ascii=False)
    goals_str = goals_str.replace("true", "True").replace("false", "False").replace("null", "None")

    # Convert INSTRUMENTS
    inst_lines = []
    for name, x, y, z, radius, color in instruments:
        inst_lines.append(f"    (\"{name}\", {x:.4f}, {y:.4f}, {z:.4f}, {radius:.3f}, \"{color}\"),")
    inst_str = "[\n" + "\n".join(inst_lines) + "\n]"

    # Convert FRAGILE_OBJECTS
    fragile_str = "{" + ", ".join(f'"{f}"' for f in fragile) + "}"

    # NavMesh settings (must be valid Python, not JSON)
    navmesh_str = json.dumps(navmesh, indent=4)
    navmesh_str = navmesh_str.replace("true", "True").replace("false", "False").replace("null", "None")

    # Scene config (must be valid Python, not JSON)
    scene_cfg_str = json.dumps(scene_config, indent=4)
    scene_cfg_str = scene_cfg_str.replace("true", "True").replace("false", "False").replace("null", "None")

    output_abs = str(output_dir.resolve())

    code = f'''# Auto-generated by auto_config.py — DO NOT EDIT MANUALLY
# This file is loaded by config.py at import time.

from pathlib import Path as _Path

SCENE_NAME = "{scene_name}"
SCENE_GLB = _Path("{scene_glb.resolve()}")
OUTPUT_DIR = _Path("{output_abs}")

SCENE_CONFIG_FILE = OUTPUT_DIR / "scene_dataset_config.json"

SEMANTIC_ANNOTATIONS = {semantic_str}

TARGET_CATEGORIES = {targets_str}

GOALS_BY_CATEGORY = {goals_str}

EPISODES_PER_CATEGORY = {spec.get("episodes_per_category", 10)}
NUM_START_POSITIONS = {spec.get("num_start_positions", 20)}
MIN_START_SPACING = {spec.get("min_start_spacing", 0.8)}
RANDOM_SEED = {spec.get("random_seed", 42)}

CUSTOM_STARTS = {_safe_repr(spec.get("custom_starts"))}
CUSTOM_GOALS = {_safe_repr(spec.get("custom_goals"))}

INSTRUMENTS = {inst_str}

SAFETY_RADIUS = {spec.get("safety_radius", 1.3)}
FRAGILE_OBJECTS = {fragile_str}

NAVMESH_SETTINGS = {navmesh_str}

SCENE_DATASET_CONFIG = {scene_cfg_str}
'''
    return code


def write_runtime_config(
    spec_path: Path,
    manifest_path: Path,
    scene_glb: Path,
    output_dir: Path,
    runtime_path: Optional[Path] = None,
) -> Path:
    """
    Read spec and manifest, write _runtime_config.py.

    Args:
        spec_path: Path to lab_spec.json
        manifest_path: Path to mesh_manifest.json
        scene_glb: Path to the generated scene GLB
        output_dir: Pipeline output directory
        runtime_path: Where to write _runtime_config.py (default: pipeline dir)

    Returns: Path to the written config file.
    """
    with open(spec_path) as f:
        spec = json.load(f)

    with open(manifest_path) as f:
        manifest = json.load(f)

    code = generate_runtime_config(spec, manifest, scene_glb, output_dir)

    if runtime_path is None:
        runtime_path = Path(__file__).parent / "_runtime_config.py"

    runtime_path.write_text(code)
    print(f"  ✓ Runtime config: {runtime_path}")
    return runtime_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate pipeline runtime config")
    ap.add_argument("--spec", required=True, help="lab_spec.json")
    ap.add_argument("--manifest", required=True, help="mesh_manifest.json")
    ap.add_argument("--scene", required=True, help="Path to generated scene.glb")
    ap.add_argument("--out", required=True, help="Output directory for pipeline artifacts")
    ap.add_argument("--runtime-path", default=None,
                    help="Where to write _runtime_config.py (default: pipeline dir)")
    args = ap.parse_args()

    runtime = Path(args.runtime_path) if args.runtime_path else None
    write_runtime_config(
        Path(args.spec), Path(args.manifest),
        Path(args.scene), Path(args.out),
        runtime,
    )
    print("✓ Auto-config complete.")
