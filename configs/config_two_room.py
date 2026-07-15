"""
Config for BIO_GLB_two_room.glb — baked two-room lab scene with 35 meshes.
Generated from GLTF node inspection on 2026-07-01.
"""
from pathlib import Path

SCENE_NAME = "BIO_GLB_two_room"
SCENE_GLB = Path("/root/gpufree-data/BIO_GLB_two_room.glb")
OUTPUT_DIR = Path("/root/gpufree-data/two_room_output")

SCENE_CONFIG_FILE = OUTPUT_DIR / "scene_dataset_config.json"

# ─── Semantic Annotations (mesh_idx → (object_id, label, region_id)) ──
# Based on GLTF node names from pygltflib inspection.
# All 35 meshes covered. Floor/Ceiling/Walls share category labels.

SEMANTIC_ANNOTATIONS = {
    # ─ Room structure ─
    0:  (1,  "wall",              1),   # Wall_BottomRoom_Right
    11: (2,  "floor",             1),   # Floor_RoomA
    12: (3,  "floor",             1),   # Floor_RoomB
    13: (4,  "ceiling",           1),   # Ceiling_RoomA
    14: (5,  "ceiling",           1),   # Ceiling_RoomB
    15: (6,  "wall",              1),   # WallA_South_Left
    16: (7,  "wall",              1),   # WallA_South_Right
    17: (8,  "wall",              1),   # DoorA_Lintel
    18: (9,  "wall",              1),   # WallA_West
    19: (10, "wall",              1),   # WallA_North
    20: (11, "wall",              1),   # WallA_East_Upper
    21: (12, "wall",              1),   # WallA_East_Lower
    22: (13, "wall",              1),   # DoorAB_Lintel
    23: (14, "wall",              1),   # WallB_North
    24: (15, "wall",              1),   # WallB_South
    25: (16, "wall",              1),   # WallB_East
    26: (17, "wall",              1),   # TrimA_North
    27: (18, "wall",              1),   # TrimA_West
    28: (19, "wall",              1),   # TrimA_South
    29: (20, "wall",              1),   # TrimB_North
    30: (21, "wall",              1),   # TrimB_South
    31: (22, "wall",              1),   # TrimB_East
    32: (23, "wall",              1),   # TwoRoom_Panel_1
    33: (24, "wall",              1),   # TwoRoom_Panel_2
    34: (25, "wall",              1),   # TwoRoom_Panel_3

    # ─ Furniture ─
    1:  (26, "lab_table",         1),   # lab_table (Room A center)
    3:  (27, "lab_table_1",       1),   # lab_table_1.001 (Room B south)
    5:  (28, "lab_bench",         1),   # model (Room A north wall, floor)

    # ─ Instruments ─
    2:  (29, "cabinet",           1),   # cabinet.001 (Room B north)
    4:  (30, "pipette",           1),   # Pipette.001 (Room B south, large verts)
    6:  (31, "centrifuge",        1),   # Centrifuge.001 (Room B south)
    7:  (32, "refrigerator",       1),   # refrige.001 (Room A north)
    8:  (33, "incubator",         1),   # incubor.001 (Room A south)
    9:  (34, "waste_bin",         1),   # waste_bin.001 (connector area)
    10: (35, "mini_centrifuge",   1),   # centrifuge_2.001 (Room A north, high-poly)
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

# ─── Goal Positions (GLB Z-up coords from GLTF world positions) ──────
# These are approximate centers; step3 will snap to floor-level navmesh.
GOALS_BY_CATEGORY = {
    "cabinet": [{
        "position": [9.87, 1.87, 0.0],
        "object_id": 29,
        "object_name": "cabinet_29",
        "object_category": "cabinet",
    }],
    "centrifuge": [{
        "position": [7.01, -1.80, 0.0],
        "object_id": 31,
        "object_name": "centrifuge_31",
        "object_category": "centrifuge",
    }],
    "incubator": [{
        "position": [-2.11, -0.89, 0.0],
        "object_id": 33,
        "object_name": "incubator_33",
        "object_category": "incubator",
    }],
    "mini_centrifuge": [{
        "position": [-2.61, 2.00, 0.0],
        "object_id": 35,
        "object_name": "mini_centrifuge_35",
        "object_category": "mini_centrifuge",
    }],
    "pipette": [{
        "position": [8.51, -1.68, 0.0],
        "object_id": 30,
        "object_name": "pipette_30",
        "object_category": "pipette",
    }],
    "refrigerator": [{
        "position": [0.28, 1.07, 0.0],
        "object_id": 32,
        "object_name": "refrigerator_32",
        "object_category": "refrigerator",
    }],
    "waste_bin": [{
        "position": [5.72, 1.92, 0.0],
        "object_id": 34,
        "object_name": "waste_bin_34",
        "object_category": "waste_bin",
    }],
    "lab_table": [{
        "position": [0.0, -1.50, 0.0],
        "object_id": 26,
        "object_name": "lab_table_26",
        "object_category": "lab_table",
    }],
    "lab_table_1": [{
        "position": [8.83, -1.18, 0.0],
        "object_id": 27,
        "object_name": "lab_table_1_27",
        "object_category": "lab_table_1",
    }],
    "lab_bench": [{
        "position": [-1.61, 1.30, 0.0],
        "object_id": 28,
        "object_name": "lab_bench_28",
        "object_category": "lab_bench",
    }],
}

TARGET_CATEGORIES = {
    "cabinet":         29,
    "centrifuge":      31,
    "incubator":       33,
    "mini_centrifuge": 35,
    "pipette":         30,
    "refrigerator":    32,
    "waste_bin":       34,
    "lab_table":       26,
    "lab_table_1":     27,
    "lab_bench":       28,
}

CUSTOM_STARTS = None
CUSTOM_GOALS = GOALS_BY_CATEGORY  # Use GOALS_BY_CATEGORY as custom goals (will be Z-up → Y-up converted)

EPISODES_PER_CATEGORY = 5
NUM_START_POSITIONS = 10
MIN_START_SPACING = 1.5
RANDOM_SEED = 42

# ─── Visualization ────────────────────────────────────────────────────────
INSTRUMENTS = [
    ("cabinet",         10.67,  2.67, 0.0, 0.43, "#2ecc71"),
    ("centrifuge",       6.55, -2.63, 0.0, 0.58, "#e74c3c"),
    ("incubator",       -3.38, -1.42, 0.0, 0.68, "#3498db"),
    ("mini_centrifuge", -3.43,  2.63, 0.0, 0.40, "#e67e22"),
    ("pipette",          8.76, -2.50, 0.0, 0.15, "#9b59b6"),
    ("refrigerator",     0.57,  2.19, 0.0, 0.68, "#1abc9c"),
    ("waste_bin",        5.05,  2.49, 0.0, 0.30, "#f39c12"),
    ("lab_table",        0.00,  0.00, 0.0, 0.80, "#ecf0f1"),
    ("lab_table_1",      9.70, -2.40, 0.0, 0.80, "#bdc3c7"),
    ("lab_bench",       -2.77,  2.25, 0.0, 0.80, "#95a5a6"),
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
    if errors:
        print("❌ Config validation FAILED:")
        for e in errors:
            print(f"   - {e}")
        return False
    print(f"✓ Config validated. {len(SEMANTIC_ANNOTATIONS)} meshes, {len(TARGET_CATEGORIES)} targets")
    return True

if __name__ == "__main__":
    validate()
