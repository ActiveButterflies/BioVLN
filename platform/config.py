"""
Shared configuration for multi-scene generation.

- Instrument catalog (GLB files, sizes, categories) — CENTER-origin
- Furniture catalog (GLB files) — CENTER-origin
- Layout archetypes with placement rules
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────

PIPELINE_DIR = Path("/root/gpufree-data/pipeline")
INSTRUMENT_DIR = Path("/root/gpufree-data/new_instructment")
FURNITURE_DIR = Path("/root/gpufree-data/new_instructment")

# Room GLB: 8.30×6.30×3.24m, CENTER-origin (floor at approximately Z=0)
ROOM_GLB = INSTRUMENT_DIR / "room_zero.glb"
ROOM_GLB_DIMS = (8.00, 6.00, 3.24)  # INNER floor size (walls +0.15 each side = 8.3×6.3 outer)
BLENDER_PATH = Path("/root/gpufree-data/blender-5.1.2-linux-x64/blender")
GENERATE_LAB = PIPELINE_DIR / "generate_lab.py"

# ─── Instrument Catalog ───────────────────────────────────────────────────
# All instruments are CENTER-origin (geometric center at ~0,0,0).
# Sizes are (X, Y, Z) in Blender Z-up: X=east-west, Y=north-south, Z=up.
# Zone "wall" = floor-standing against a wall.
# Zone "table_top" = placed on a table/bench surface.
# Zone "floor" = floor-standing anywhere.

INSTRUMENT_CATALOG = {
    # size = (width_X, depth_Y, height_Z) — bbox extent after Blender import
    # bb_center_z = Z of bbox center in model space
    # bottom_z = Z of the effective bottom in model space
    # Placement: spec_Z = target + bb_center_z - bottom_z (puts bottom at target)
    # operational_face = (dx, dy, dz) model-space normal of the face the operator stands in front of
    # face_half_depth = half the model extent in the operational_face direction
    "refrigerator": {
        "file": "refrige_zero.glb", "label": "refrigerator",
        "category": "refrigerator", "size": (0.89, 0.91, 1.78),
        "bb_center_z": 0.0147, "bottom_z": -0.8776, "zone": "wall",
        "operational_face": (0, -1, 0), "face_half_depth": 0.455,   # door faces -Y (south), front with handles
    },
    "incubator": {
        "file": "incubor_zero.glb", "label": "incubator",
        "category": "incubator", "size": (1.35, 0.85, 1.90),
        "bb_center_z": -0.0113, "bottom_z": -0.9613, "zone": "wall",
        "operational_face": (-1, 0, 0), "face_half_depth": 0.673,   # door on -X face (wide side), faces west
    },
    "centrifuge": {
        "file": "centrifuge_zero.glb", "label": "centrifuge",
        "category": "centrifuge", "size": (0.49, 0.42, 0.63),
        "bb_center_z": 0.1259, "bottom_z": -0.1894, "zone": "table_top",
        "operational_face": (-1, 0, 0), "face_half_depth": 0.245,   # control panel faces -X (west)
    },
    "mini_centrifuge": {
        "file": "mini_centrifuge_zero.glb", "label": "mini_centrifuge",
        "category": "mini_centrifuge", "size": (0.66, 0.50, 0.43),
        "bb_center_z": 0.0685, "bottom_z": -0.1463, "zone": "table_top",
        "operational_face": (-1, 0, 0), "face_half_depth": 0.332,   # lid hinge / button panel faces -X (west)
    },
    "pipette": {
        "file": "pipette_zero.glb", "label": "pipette",
        "category": "pipette", "size": (0.04, 0.32, 0.28),
        "bb_center_z": -0.0508, "bottom_z": -0.1908, "zone": "table_top",
        "operational_face": (0, -1, 0), "face_half_depth": 0.159,   # wider face -Y (south), button/display side
    },
    "waste_bin": {
        "file": "waste_bin_zero.glb", "label": "waste_bin",
        "category": "lab_waste_bin", "size": (0.38, 0.37, 0.51),
        "bb_center_z": -0.0958, "bottom_z": -0.3495, "zone": "floor",
        "operational_face": (0, 1, 0), "face_half_depth": 0.183,    # pedal/opening faces +Y (north)
    },
    "cabinet": {
        "file": "cabinet_zero.glb", "label": "cabinet",
        "category": "cabinet", "size": (1.384, 0.574, 1.768),
        "bb_center_z": 0.1512, "bottom_z": -0.7328, "zone": "wall",
        "operational_face": (1, 0, 0), "face_half_depth": 0.692,    # doors face +X (east), wide side
    },
}

# ─── Furniture Catalog ────────────────────────────────────────────────────

FURNITURE_CATALOG = {
    "lab_bench": {
        "file": "lab_bench_zero.glb", "size": (1.55, 5.94, 0.76),
        "bb_center_z": -0.0213, "bottom_z": -0.4007,
        "table_top_z": 0.76,   # world-space bench top (model 0.36, bench placed at Z=0.3794)
    },
    "lab_table": {
        "file": "lab_table_zero.glb", "size": (2.40, 0.90, 1.10),
        "bb_center_z": 0.550, "bottom_z": 0.000,
        "table_top_z": 0.78,   # actual flat work surface (Z≈0.78 per vertex analysis, max upward area)
    },
}

# ─── Layout Archetypes ────────────────────────────────────────────────────

# Room inner dimensions (walls are in the GLB, sit outside floor edge):
#   Floor: X=[-4.0, 4.0], Y=[-3.0, 3.0]
#   Wall inner face = floor edge; no deduction needed.
INNER_X_HALF = 4.0
INNER_Y_HALF = 3.0
WALL_T = 0.0   # walls extend outward from floor edge, don't reduce interior

LAYOUT_TEMPLATES = {
    # === Single Room: 1 open space, 4 walls, 1 door ===
    "single_room": {
        "description": (
            "Single lab room (8.3×6.3m) with bench along west wall, central table, "
            "wall instruments (refrigerator/cabinet/incubator), table-top instruments "
            "(centrifuge/mini_centrifuge/pipette), and floor waste_bin."
        ),
        "room": {
            "width": ROOM_GLB_DIMS[0],
            "depth": ROOM_GLB_DIMS[1],
            "height": ROOM_GLB_DIMS[2],
            "wall_thickness": WALL_T,
        },
        "doorways": [
            {"wall": "south", "position": 0.5, "width": 1.2, "height": 2.1},
        ],
        "furniture": {
            "bench_west": {
                "catalog_key": "lab_bench",
                "wall": "west",
                "offset": 0.08,          # gap from inner wall face
            },
            "central_table": {
                "catalog_key": "lab_table",
                "zone": "center",
            },
        },
        "instrument_slots": {
            "refrigerator_north": {
                "allowed": ["refrigerator"], "count": 1,
                "zone": "wall", "wall": "north",
                "position_hint": 0.25,    # fraction along wall
            },
            "cabinet_east": {
                "allowed": ["cabinet", "incubator"], "count": 1,
                "zone": "wall", "wall": "east",
                "position_hint": 0.30,
            },
            "incubator_west": {
                "allowed": ["incubator", "refrigerator"], "count": (0, 1),
                "zone": "wall", "wall": "west",
                "position_hint": 0.70,    # away from bench area
            },
            "table_equip_1": {
                "allowed": ["centrifuge", "mini_centrifuge"], "count": 1,
                "zone": "table_top", "on_furniture": "central_table",
            },
            "table_equip_2": {
                "allowed": ["mini_centrifuge", "pipette"], "count": (1, 2),
                "zone": "table_top", "on_furniture": "central_table",
            },
            "waste_bin_south": {
                "allowed": ["waste_bin"], "count": (0, 1),
                "zone": "wall", "wall": "south",
                "position_hint": 0.80,
            },
        },
    },

    # === Two Room: main lab + instrument room ===
    "two_room": {
        "description": (
            "Two rooms connected by passage. Main lab (6×5m) has bench, wall instruments. "
            "Instrument room (3.5×3.5m) has centrifuge and incubator."
        ),
        "rooms": [
            {
                "name": "main_lab",
                "width": 6.0,
                "depth": 5.0,
                "height": ROOM_GLB_DIMS[2],
                "wall_thickness": WALL_T,
                "offset": [0, 0],
            },
            {
                "name": "instrument_room",
                "width": 3.5,
                "depth": 3.5,
                "height": ROOM_GLB_DIMS[2],
                "wall_thickness": WALL_T,
                "offset": "auto_east",
            },
        ],
        "doorways": [
            {"wall": "south", "room": "main_lab", "position": 0.5,
             "width": 1.2, "height": 2.1},
            {"from_room": "main_lab", "to_room": "instrument_room",
             "from_wall": "east", "to_wall": "west",
             "position": 0.5, "width": 1.2, "height": 2.1, "internal": True},
        ],
        "furniture": {
            "bench_main_west": {
                "catalog_key": "lab_bench",
                "room": "main_lab",
                "wall": "west",
                "offset": 0.08,
            },
        },
        "instrument_slots": {
            "main_refrigerator": {
                "allowed": ["refrigerator"], "count": 1,
                "zone": "wall", "room": "main_lab", "wall": "north",
                "position_hint": 0.50,
            },
            "main_cabinet": {
                "allowed": ["cabinet", "incubator", "waste_bin"], "count": 1,
                "zone": "wall", "room": "main_lab", "wall": "east",
                "position_hint": 0.70,
            },
            "inst_centrifuge": {
                "allowed": ["centrifuge", "mini_centrifuge"], "count": 1,
                "zone": "center", "room": "instrument_room",
            },
            "inst_incubator": {
                "allowed": ["incubator", "refrigerator"], "count": (0, 1),
                "zone": "wall", "room": "instrument_room", "wall": "north",
                "position_hint": 0.50,
            },
        },
    },

    # === Suite: 3 rooms (cold + main + culture) ===
    "suite": {
        "description": (
            "Three-room suite: cold room (NW, 3×3m), main lab (center, 6×5m), "
            "tissue culture (E, 3×3m). Connected by doorways."
        ),
        "rooms": [
            {
                "name": "cold_room",
                "width": 3.0, "depth": 3.0,
                "height": ROOM_GLB_DIMS[2],
                "wall_thickness": WALL_T,
                "offset": [-2.5, 1.65],
            },
            {
                "name": "main_lab",
                "width": 6.0, "depth": 5.0,
                "height": ROOM_GLB_DIMS[2],
                "wall_thickness": WALL_T,
                "offset": [0, -0.65],
            },
            {
                "name": "culture_room",
                "width": 3.0, "depth": 3.0,
                "height": ROOM_GLB_DIMS[2],
                "wall_thickness": WALL_T,
                "offset": "auto_east",
            },
        ],
        "doorways": [
            {"wall": "south", "room": "main_lab", "position": 0.5,
             "width": 1.2, "height": 2.1},
            {"from_room": "cold_room", "to_room": "main_lab",
             "from_wall": "south", "to_wall": "north",
             "position": 0.5, "width": 1.2, "height": 2.1, "internal": True},
            {"from_room": "culture_room", "to_room": "main_lab",
             "from_wall": "west", "to_wall": "east",
             "position": 0.5, "width": 1.2, "height": 2.1, "internal": True},
        ],
        "furniture": {
            "table_main": {
                "catalog_key": "lab_table",
                "room": "main_lab",
                "zone": "center",
            },
        },
        "instrument_slots": {
            "cold_refrigerator": {
                "allowed": ["refrigerator", "incubator"], "count": 1,
                "zone": "wall", "room": "cold_room", "wall": "north",
                "position_hint": 0.50,
            },
            "main_refrigerator": {
                "allowed": ["refrigerator"], "count": 1,
                "zone": "wall", "room": "main_lab", "wall": "north",
                "position_hint": 0.50,
            },
            "main_cabinet": {
                "allowed": ["cabinet", "incubator"], "count": 1,
                "zone": "wall", "room": "main_lab", "wall": "east",
                "position_hint": 0.70,
            },
            "main_table_equip": {
                "allowed": ["centrifuge", "mini_centrifuge", "pipette"],
                "count": (1, 2),
                "zone": "table_top", "on_furniture": "table_main",
            },
            "culture_incubator": {
                "allowed": ["incubator", "refrigerator"], "count": 1,
                "zone": "wall", "room": "culture_room", "wall": "north",
                "position_hint": 0.50,
            },
            "culture_waste": {
                "allowed": ["waste_bin"], "count": (0, 1),
                "zone": "wall", "room": "culture_room", "wall": "south",
                "position_hint": 0.50,
            },
        },
    },

    # === Open Plan: large room with bench rows ===
    "open_plan": {
        "description": (
            "Large open-plan lab with bench rows and central table. "
            "Rich instrument set with all 7 types possible."
        ),
        "room": {
            "width": ROOM_GLB_DIMS[0],
            "depth": ROOM_GLB_DIMS[1],
            "height": ROOM_GLB_DIMS[2],
            "wall_thickness": WALL_T,
        },
        "doorways": [
            {"wall": "south", "position": 0.5, "width": 1.5, "height": 2.1},
        ],
        "furniture": {
            "bench_west": {
                "catalog_key": "lab_bench",
                "wall": "west",
                "offset": 0.08,
            },
            "bench_east": {
                "catalog_key": "lab_bench",
                "wall": "east",
                "offset": 0.08,
            },
            "central_table": {
                "catalog_key": "lab_table",
                "zone": "center",
            },
        },
        "instrument_slots": {
            "refrigerator_north": {
                "allowed": ["refrigerator"], "count": 1,
                "zone": "wall", "wall": "north", "position_hint": 0.30,
            },
            "incubator_north": {
                "allowed": ["incubator", "refrigerator"], "count": (0, 1),
                "zone": "wall", "wall": "north", "position_hint": 0.70,
            },
            "cabinet_east": {
                "allowed": ["cabinet", "incubator", "refrigerator"], "count": 1,
                "zone": "wall", "wall": "east", "position_hint": 0.25,
            },
            "table_equip_1": {
                "allowed": ["centrifuge"], "count": 1,
                "zone": "table_top", "on_furniture": "central_table",
            },
            "table_equip_2": {
                "allowed": ["mini_centrifuge", "pipette"], "count": (1, 2),
                "zone": "table_top", "on_furniture": "central_table",
            },
            "waste_bin_south": {
                "allowed": ["waste_bin"], "count": (0, 1),
                "zone": "wall", "wall": "south", "position_hint": 0.20,
            },
        },
    },
}

# ─── Split Assignment ─────────────────────────────────────────────────────

SPLIT_DISTRIBUTION = {
    "train": {"single_room": 3, "two_room": 3, "suite": 3, "open_plan": 3},
    "val":   {"single_room": 1, "two_room": 1, "suite": 1, "open_plan": 1},
    "test":  {"single_room": 1, "two_room": 1, "suite": 1, "open_plan": 1},
}

SPLIT_DISTRIBUTION_MVP = {
    "train": {"single_room": 2, "two_room": 1},
    "val":   {"single_room": 1},
    "test":  {"single_room": 1},
}

# ─── Episode Generation Defaults ───────────────────────────────────────────

EPISODES_PER_CATEGORY = 10
NUM_START_POSITIONS = 25
MIN_START_SPACING = 0.5

# ─── NavMesh Settings ──────────────────────────────────────────────────────

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
