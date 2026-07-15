"""
Config for BIO_GLB_operate_target_2.glb — user-authored single-room lab scene
with 8 target_* empty objects for goal positions (LSAT Strategy 1).

Generated from GLTF node inspection + LSAT target extraction on 2026-07-02.
Reconstructed config on 2026-07-08.
"""
from pathlib import Path

SCENE_NAME = "lab_single_room_target"
SCENE_GLB = Path("/root/gpufree-data/BIO_GLB_operate_target_2.glb")
OUTPUT_DIR = Path("/root/gpufree-data/lsat_test")

SCENE_CONFIG_FILE = OUTPUT_DIR / "scene_dataset_config.json"

# ─── Semantic Annotations (mesh_idx → (object_id, label, region_id)) ──
# Based on GLTF node names from pygltflib inspection.
# All 15 meshes covered. Room structure assigned IDs 9–14.
# Object IDs 1–8 match LSAT target_* extraction order.

SEMANTIC_ANNOTATIONS = {
    # ─ Room structure ─
    0:  (9,  "floor",             1),   # Floor
    1:  (10, "ceiling",           1),   # Ceiling
    2:  (11, "wall",              1),   # Wall_North
    3:  (12, "wall",              1),   # Wall_South
    4:  (13, "wall",              1),   # Wall_East
    5:  (14, "wall",              1),   # Wall_West

    # ─ Instruments (navigation targets) ─
    6:  (16, "centrifuge",        1),   # Centrifuge (full-size, non-target décor)
    7:  (5,  "mini_centrifuge",   1),   # centrifuge_2 (mini, TARGET)
    8:  (7,  "refrigerator",      1),   # refrige
    9:  (2,  "incubator",         1),   # incubor
    10: (6,  "pipette",           1),   # Pipette
    11: (3,  "lab_table",         1),   # lab_table
    12: (1,  "cabinet",           1),   # cabinet
    13: (8,  "waste_bin",         1),   # waste_bin
    14: (4,  "lab_table_1",       1),   # lab_table_1
}

# ─── NavMesh Settings ─────────────────────────────────────────────────────
NAVMESH_SETTINGS = {
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

# ─── Goal Positions (GLB Z-up coords from LSAT target_* extraction) ──────
# Source: /root/lsat/examples/goals_from_target_2.json
# These get Z-up → Y-up converted by step3_episodes.py.
GOALS_BY_CATEGORY = {
    "cabinet": [{
        "position": [2.77, 1.91, 0.0],
        "object_id": 1,
        "object_name": "cabinet_1",
        "object_category": "cabinet",
        "facing_yaw": -55.5,
    }],
    "incubator": [{
        "position": [1.89, -2.04, 0.0],
        "object_id": 2,
        "object_name": "incubator_2",
        "object_category": "incubator",
        "facing_yaw": -90.0,
    }],
    "lab_table": [{
        "position": [0.82, 0.29, 0.0],
        "object_id": 3,
        "object_name": "lab_table_3",
        "object_category": "lab_table",
        "facing_yaw": -70.2,
    }],
    "lab_table_1": [{
        "position": [-2.75, 0.0, 0.0],
        "object_id": 4,
        "object_name": "lab_table_1_4",
        "object_category": "lab_table_1",
        "facing_yaw": 90.0,
    }],
    "mini_centrifuge": [{
        "position": [-2.82, -1.83, 0.0],
        "object_id": 5,
        "object_name": "mini_centrifuge_5",
        "object_category": "mini_centrifuge",
        "facing_yaw": 118.5,
    }],
    "pipette": [{
        "position": [-0.22, 0.30, 0.0],
        "object_id": 6,
        "object_name": "pipette_6",
        "object_category": "pipette",
        "facing_yaw": 35.8,
    }],
    "refrigerator": [{
        "position": [-0.09, -1.62, 0.0],
        "object_id": 7,
        "object_name": "refrigerator_7",
        "object_category": "refrigerator",
        "facing_yaw": 176.7,
    }],
    "waste_bin": [{
        "position": [-1.63, 2.21, 0.0],
        "object_id": 8,
        "object_name": "waste_bin_8",
        "object_category": "waste_bin",
        "facing_yaw": 36.4,
    }],
}

TARGET_CATEGORIES = {
    "cabinet":         1,
    "incubator":       2,
    "lab_table":       3,
    "lab_table_1":     4,
    "mini_centrifuge": 5,
    "pipette":         6,
    "refrigerator":    7,
    "waste_bin":       8,
}

CUSTOM_STARTS = None
CUSTOM_GOALS = GOALS_BY_CATEGORY  # Use GOALS_BY_CATEGORY as custom goals (Z-up → Y-up converted)

EPISODES_PER_CATEGORY = 5
NUM_START_POSITIONS = 10
MIN_START_SPACING = 1.5
RANDOM_SEED = 42

# ─── Visualization ────────────────────────────────────────────────────────
# (label, x_glb, y_glb, z_floor, radius, hex_color)
INSTRUMENTS = [
    ("cabinet",          2.77,  1.91, 0.0, 0.55, "#2ecc71"),
    ("incubator",        1.89, -2.04, 0.0, 0.68, "#3498db"),
    ("lab_table",        0.82,  0.29, 0.0, 0.80, "#ecf0f1"),
    ("lab_table_1",     -2.75,  0.00, 0.0, 0.80, "#bdc3c7"),
    ("mini_centrifuge", -2.82, -1.83, 0.0, 0.40, "#e67e22"),
    ("pipette",         -0.22,  0.30, 0.0, 0.15, "#9b59b6"),
    ("refrigerator",    -0.09, -1.62, 0.0, 0.68, "#1abc9c"),
    ("waste_bin",       -1.63,  2.21, 0.0, 0.30, "#f39c12"),
]

SAFETY_RADIUS = 1.3
FRAGILE_OBJECTS = {"centrifuge", "refrigerator", "incubator", "cabinet"}

# ─── Scene Dataset Config ─────────────────────────────────────────────────
SCENE_DATASET_CONFIG = {
    "stages": {
        "paths": {
            ".glb": [f"{SCENE_NAME}.glb"]
        },
        "default_attributes": {
            "shader_type": "phong",
            "semantic_descriptor_filename": f"{SCENE_NAME}.semantic.txt",
            "semantic_asset": f"{SCENE_NAME}.semantic.glb",
            "has_semantic_textures": True,
        }
    },
    "objects": {},
    "light_setups": {},
    "scene_instances": {
        "default_attributes": {
            "default_lighting": "no_lights"
        },
        "paths": {
            ".json": []
        }
    }
}

def derived_paths():
    base = OUTPUT_DIR / SCENE_NAME
    return {
        "semantic_glb":  OUTPUT_DIR / f"{SCENE_NAME}.semantic.glb",
        "semantic_txt":  OUTPUT_DIR / f"{SCENE_NAME}.semantic.txt",
        "navmesh":       OUTPUT_DIR / f"{SCENE_NAME}.navmesh",
        "episodes_gz":   OUTPUT_DIR / f"{SCENE_NAME}_episodes.json.gz",
        "episodes_json": OUTPUT_DIR / f"{SCENE_NAME}_episodes.json",
        "viz_png":       OUTPUT_DIR / f"{SCENE_NAME}_navmesh_map.png",
        "viz_svg":       OUTPUT_DIR / f"{SCENE_NAME}_navmesh_map.svg",
        "scene_config":  OUTPUT_DIR / "scene_dataset_config.json",
    }

def validate():
    errors = []
    if not SCENE_GLB.exists():
        errors.append(f"Scene GLB not found: {SCENE_GLB}")
    obj_ids = [obj_id for obj_id, _, _ in SEMANTIC_ANNOTATIONS.values()]
    if len(obj_ids) != len(set(obj_ids)):
        errors.append("Duplicate object_id in SEMANTIC_ANNOTATIONS")
    # Verify all 15 meshes covered
    if len(SEMANTIC_ANNOTATIONS) != 15:
        errors.append(f"Expected 15 mesh annotations, got {len(SEMANTIC_ANNOTATIONS)}")
    if errors:
        print("❌ Config validation FAILED:")
        for e in errors:
            print(f"   - {e}")
        return False
    print(f"✓ Config validated. {len(SEMANTIC_ANNOTATIONS)} meshes, {len(TARGET_CATEGORIES)} targets")
    return True

if __name__ == "__main__":
    validate()
