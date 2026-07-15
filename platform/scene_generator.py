#!/usr/bin/env python3
"""
Scene Generator: produce lab_spec.json files from layout archetypes.

Each call produces one scene spec with randomized:
  - Room dimensions (fixed to room GLB size)
  - Equipment selection (subset of available instruments)
  - Equipment positions (CENTER-origin, rule-constrained)
  - Furniture placement (GLB import)

Usage:
  # List available templates
  python scene_generator.py --list

  # Generate one scene
  python scene_generator.py --template single_room --seed 42 --output spec.json

  # Generate multiple variants
  python scene_generator.py --template single_room --count 5 --output_dir specs/

  # Generate all templates for a split
  python scene_generator.py --all --split train --output_dir specs/
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import shared config
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    INSTRUMENT_CATALOG,
    FURNITURE_CATALOG,
    LAYOUT_TEMPLATES,
    EPISODES_PER_CATEGORY,
    NUM_START_POSITIONS,
    MIN_START_SPACING,
    NAVMESH_SETTINGS,
    INSTRUMENT_DIR,
    FURNITURE_DIR,
    ROOM_GLB,
    ROOM_GLB_DIMS,
    SPLIT_DISTRIBUTION,
    SPLIT_DISTRIBUTION_MVP,
)


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate System
# ═══════════════════════════════════════════════════════════════════════════
#
# lab_spec.json uses Blender Z-up coordinates:
#   X = east-west, Y = north-south, Z = up
#   Floor at Z = 0
#   Room center at (0, 0, 0)
#
# Instruments are rotated -90° X by the Blender builder, so in the final
# scene: size[0]=X(width), size[1]=Y(depth), size[2]=Z(height).
# ═══════════════════════════════════════════════════════════════════════════


class InstrumentPlacer:
    """Place instruments in a room following zone constraints.

    Zones:
      - wall:       Against a specific wall, floor-standing
      - corner:     In a room corner, floor-standing
      - center:     Room center area, floor-standing
      - table_top:  On top of a furniture piece
      - floor:      On the floor (anywhere navigable)

    All positions are CENTER-origin aware: Z = size_z/2 for floor items.
    """

    def __init__(self, rng: np.random.RandomState, room_w: float, room_d: float):
        self.rng = rng
        self._occupied: List[Tuple[float, float, float, float]] = []  # (x, y, hx, hy)
        self._room_w = room_w
        self._room_d = room_d

    # Walls are in the room GLB, sit outside floor. No thickness deduction.
    WALL_THICKNESS = 0.0
    WALL_GAP = 0.02  # tiny gap between instrument back and inner wall

    def _inner_bounds(self, rw: float, rd: float, margin: float = 0.3) -> Dict:
        """Return valid placement rectangle in Blender XY (inner wall faces)."""
        iw2 = rw / 2 - self.WALL_THICKNESS
        id2 = rd / 2 - self.WALL_THICKNESS
        return {
            "x_min": -iw2 + margin,
            "x_max":  iw2 - margin,
            "y_min": -id2 + margin,
            "y_max":  id2 - margin,
        }

    def _wall_limits(self, wall: str, rw: float, rd: float) -> Dict:
        """Return the inner wall face coordinate and the valid range along it."""
        wt = self.WALL_THICKNESS
        ix_max =  rw / 2 - wt
        ix_min = -rw / 2 + wt
        iy_max =  rd / 2 - wt
        iy_min = -rd / 2 + wt
        return {"ix_min": ix_min, "ix_max": ix_max,
                "iy_min": iy_min, "iy_max": iy_max}

    def _wall_position(self, wall: str, rw: float, rd: float,
                       size_xy: Tuple[float, float],
                       position_hint: Optional[float] = None
                       ) -> Tuple[float, float]:
        """Position flush against inner wall face. Returns (x, y) center.

        For CENTER-origin, the instrument's center is placed at a distance of
        half its depth from the wall, so the back face touches the wall+gap.
        """
        hx, hy = size_xy[0] / 2, size_xy[1] / 2
        L = self._wall_limits(wall, rw, rd)

        if position_hint is None:
            position_hint = self.rng.uniform(0.2, 0.8)

        if wall == "north":
            y = L["iy_max"] - hy - self.WALL_GAP
            x_range = L["ix_min"] + hx, L["ix_max"] - hx
            x = x_range[0] + position_hint * (x_range[1] - x_range[0])
        elif wall == "south":
            y = L["iy_min"] + hy + self.WALL_GAP
            x_range = L["ix_min"] + hx, L["ix_max"] - hx
            x = x_range[0] + position_hint * (x_range[1] - x_range[0])
        elif wall == "east":
            x = L["ix_max"] - hx - self.WALL_GAP
            y_range = L["iy_min"] + hy, L["iy_max"] - hy
            y = y_range[0] + position_hint * (y_range[1] - y_range[0])
        elif wall == "west":
            x = L["ix_min"] + hx + self.WALL_GAP
            y_range = L["iy_min"] + hy, L["iy_max"] - hy
            y = y_range[0] + position_hint * (y_range[1] - y_range[0])
        else:
            x, y = 0.0, 0.0
        return float(x), float(y)

    def _corner_position(self, corner: str, rw: float, rd: float,
                         size_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Position in a room corner."""
        hx, hy = size_xy[0] / 2, size_xy[1] / 2
        L = self._wall_limits(corner, rw, rd)  # placeholder; we use direct calc
        wt = self.WALL_THICKNESS
        gap = 0.05
        ix_max =  rw / 2 - wt
        ix_min = -rw / 2 + wt
        iy_max =  rd / 2 - wt
        iy_min = -rd / 2 + wt

        corners = {
            "northwest": (ix_min + hx + gap, iy_max - hy - gap),
            "northeast": (ix_max - hx - gap, iy_max - hy - gap),
            "southwest": (ix_min + hx + gap, iy_min + hy + gap),
            "southeast": (ix_max - hx - gap, iy_min + hy + gap),
        }
        return corners.get(corner, (0.0, 0.0))

    def _center_position(self, bounds: Dict,
                         size_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Position in room center with small random offset."""
        hx, hy = size_xy[0] / 2, size_xy[1] / 2
        cx = (bounds["x_min"] + bounds["x_max"]) / 2
        cy = (bounds["y_min"] + bounds["y_max"]) / 2
        ox = self.rng.uniform(-0.8, 0.8)
        oy = self.rng.uniform(-0.8, 0.8)
        x = float(np.clip(cx + ox, bounds["x_min"] + hx, bounds["x_max"] - hx))
        y = float(np.clip(cy + oy, bounds["y_min"] + hy, bounds["y_max"] - hy))
        return x, y

    def _place_z(self, zone: str, size_xyz: Tuple[float, float, float],
                 bottom_z: float, bb_center_z: float = 0.0,
                 table_top: float = 0.0) -> float:
        """Compute spec Z so the model's effective bottom sits at the target height.

        Builder moves bbox center to spec_Z. So:
          spec_Z = target_height + bb_center_z - bottom_z
        where bottom_z is the effective bottom in model space,
        bb_center_z = size[2]/2 + z_min (bbox center in model space).
        When bottom_z ≈ z_min (flat base): spec_Z = target + size[2]/2.
        """
        target = table_top if zone == "table_top" else 0.0
        return target + bb_center_z - bottom_z

    def _check_collision(self, x: float, y: float,
                         hx: float, hy: float,
                         clearance: float = 0.25) -> bool:
        """Check AABB collision with clearance."""
        for ox, oy, ohx, ohy in self._occupied:
            if (abs(x - ox) < hx + ohx + clearance and
                    abs(y - oy) < hy + ohy + clearance):
                return True
        return False

    def _reserve(self, x: float, y: float, hx: float, hy: float):
        self._occupied.append((x, y, hx, hy))

    def place(self, zone: str, size_xyz: Tuple[float, float, float],
              bottom_z: float = None, bb_center_z: float = 0.0,
              rw: float = None, rd: float = None,
              wall: Optional[str] = None,
              corner: Optional[str] = None,
              position_hint: Optional[float] = None,
              furniture_info: Optional[Dict] = None,
              ) -> Optional[Tuple[float, float, float]]:
        """
        Place an instrument. Returns (x, y, z) in Blender Z-up (CENTER-origin),
        or None if placement failed after retries.

        size_xyz = (sx, sy, sz) — full extent X,Y,Z after Blender import.
        bottom_z = the Z of the effective bottom in model space (from catalog).
        """
        if rw is None:
            rw = self._room_w
        if rd is None:
            rd = self._room_d

        bounds = self._inner_bounds(rw, rd)
        # Ground-plane half-extents: X=width, Y=depth (north-south)
        hx = size_xyz[0] / 2         # X width
        hy = size_xyz[1] / 2         # Y depth

        if bottom_z is None:
            bottom_z = -(size_xyz[2] / 2)  # fallback: bbox bottom

        max_attempts = 30

        for _ in range(max_attempts):
            if zone == "wall" and wall:
                x, y = self._wall_position(wall, rw, rd, (hx * 2, hy * 2),
                                           position_hint or self.rng.uniform(0.1, 0.9))
                z = self._place_z(zone, size_xyz, bottom_z, bb_center_z)
            elif zone == "corner" and corner:
                x, y = self._corner_position(corner, rw, rd, (hx * 2, hy * 2))
                z = self._place_z(zone, size_xyz, bottom_z, bb_center_z)
            elif zone == "center":
                x, y = self._center_position(bounds, (hx * 2, hy * 2))
                z = self._place_z(zone, size_xyz, bottom_z, bb_center_z)
            elif zone == "table_top" and furniture_info:
                fx, fy = furniture_info["center_xy"]
                fsize_x, fsize_y = furniture_info["size_xy"]
                ftop = furniture_info["table_top_z"]
                fx_min = fx - fsize_x / 2 + hx + 0.04
                fx_max = fx + fsize_x / 2 - hx - 0.04
                fy_min = fy - fsize_y / 2 + hy + 0.04
                fy_max = fy + fsize_y / 2 - hy - 0.04
                # Avoid faucet on central_table (faucet at ~(-0.55,-0.31), 0.18×0.12m)
                faucet_exclusions = furniture_info.get("faucet_exclusions", [])
                if fx_min >= fx_max or fy_min >= fy_max:
                    return None
                # Try multiple times to avoid faucet
                for _ in range(5):
                    x = self.rng.uniform(fx_min, fx_max)
                    y = self.rng.uniform(fy_min, fy_max)
                    # Check faucet collision
                    hit_faucet = False
                    for (fz_x, fz_y, fz_hx, fz_hy) in faucet_exclusions:
                        if abs(x - fz_x) < hx + fz_hx + 0.03 and abs(y - fz_y) < hy + fz_hy + 0.03:
                            hit_faucet = True
                            break
                    if not hit_faucet:
                        break
                z = self._place_z(zone, size_xyz, bottom_z, bb_center_z, table_top=ftop)
            elif zone == "floor":
                ox = self.rng.choice([-1, 1]) * self.rng.uniform(0.5, 1.8)
                oy = self.rng.choice([-1, 1]) * self.rng.uniform(0.5, 1.8)
                cx = (bounds["x_min"] + bounds["x_max"]) / 2
                cy = (bounds["y_min"] + bounds["y_max"]) / 2
                x = float(np.clip(cx + ox, bounds["x_min"] + hx, bounds["x_max"] - hx))
                y = float(np.clip(cy + oy, bounds["y_min"] + hy, bounds["y_max"] - hy))
                z = self._place_z(zone, size_xyz, bottom_z, bb_center_z)
            else:
                x, y = 0.0, 0.0
                z = self._place_z(zone, size_xyz, bottom_z, bb_center_z)

            if not self._check_collision(x, y, hx, hy):
                self._reserve(x, y, hx, hy)
                return (float(x), float(y), float(z))

        return None


# ═══════════════════════════════════════════════════════════════════════════
# Rotation & Operational Goal Computation
# ═══════════════════════════════════════════════════════════════════════════

# Wall → desired world-space operational face direction (into room, away from wall)
_WALL_FACE_DIR = {
    "north": (0, -1),   # instrument back to +Y, face south (-Y) into room
    "south": (0, 1),    # instrument back to -Y, face north (+Y) into room
    "east":  (-1, 0),   # instrument back to +X, face west (-X) into room
    "west":  (1, 0),    # instrument back to -X, face east (+X) into room
}


def _compute_rotation(cat: dict, zone: str, wall: str,
                      pos: Tuple[float, float, float],
                      furn_info: dict = None) -> list:
    """Compute Blender Z-rotation [0, 0, deg] so operational face points
    into the room / away from the wall.

    Returns rounded Euler rotation [rx, ry, rz] in degrees.
    """
    import math

    op_face = cat.get("operational_face")
    if op_face is None or (op_face[0] == 0 and op_face[1] == 0):
        # No horizontal operational face (e.g. waste_bin top opening) → no rotation
        return [0, 0, 0]

    mdx, mdy = op_face[0], op_face[1]   # model-space face direction (XY plane)

    # Determine desired world-space direction
    if zone == "wall" and wall and wall in _WALL_FACE_DIR:
        wdx, wdy = _WALL_FACE_DIR[wall]
    elif zone == "table_top" and furn_info and "center_xy" in furn_info:
        tx, ty = furn_info["center_xy"]
        dx, dy = pos[0] - tx, pos[1] - ty
        length = math.hypot(dx, dy)
        if length < 0.01:
            wdx, wdy = 1.0, 0.0  # degenerate: default east
        else:
            wdx, wdy = dx / length, dy / length
    elif zone in ("floor", "center"):
        # Face toward room center (origin)
        dx, dy = -pos[0], -pos[1]
        length = math.hypot(dx, dy)
        if length < 0.01:
            wdx, wdy = 1.0, 0.0
        else:
            wdx, wdy = dx / length, dy / length
    else:
        # Default: face east (+X)
        wdx, wdy = 1.0, 0.0

    # Angle from model_dir to world_dir via Z-axis rotation
    angle_rad = math.atan2(mdx * wdy - mdy * wdx, mdx * wdx + mdy * wdy)
    angle_deg = round(math.degrees(angle_rad), 1)

    # Normalize to [-180, 180]
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg <= -180:
        angle_deg += 360

    return [0, 0, angle_deg]


def _compute_operational_goal(cat: dict, pos: Tuple[float, float, float],
                              rotation: list) -> list:
    """Compute operational goal position (Blender Z-up, floor level).

    Place the standing point directly in front of the instrument's
    operational face, at ground height so the robot can navigate there.

    Returns [gx, gy, gz] in Blender Z-up, with gz ≈ 0.0 (floor).
    """
    import math

    op_face = cat.get("operational_face")
    if op_face is None:
        # Fallback: bbox center + offset
        return [round(pos[0] + 0.5, 3), round(pos[1], 3), 0.0]

    if op_face[2] == 1:
        # Top opening (waste_bin): put goal beside the instrument
        gx = pos[0] + cat["size"][0] / 2 + 0.5
        gy = pos[1]
        return [round(gx, 3), round(gy, 3), 0.0]

    # Rotate model-space face direction to world space
    mdx, mdy = op_face[0], op_face[1]
    angle_rad = math.radians(rotation[2])
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    face_wx = cos_a * mdx - sin_a * mdy
    face_wy = sin_a * mdx + cos_a * mdy

    half_d = cat.get("face_half_depth", 0.3)
    offset = cat.get("operational_offset", 0.5)
    gx = pos[0] + face_wx * (half_d + offset)
    gy = pos[1] + face_wy * (half_d + offset)

    return [round(gx, 3), round(gy, 3), 0.0]


# ═══════════════════════════════════════════════════════════════════════════
# Furniture Placement (GLB-based)
# ═══════════════════════════════════════════════════════════════════════════

def place_furniture(template_def: Dict, room_w: float, room_d: float,
                    rng: np.random.RandomState) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Place furniture items from GLB catalog.

    Returns:
      - furniture_specs: list of {file, position, label, category, rotation, scale}
      - furniture_info:   {name: {center_xy, size_xy, table_top_z}} for instrument placement
    """
    furniture_specs = []
    furniture_info = {}

    WT = 0.0   # walls are in the GLB, extend outward from floor edge

    def _inner(rw, rd):
        return {
            "x_min": -rw / 2 + WT, "x_max": rw / 2 - WT,
            "y_min": -rd / 2 + WT, "y_max": rd / 2 - WT,
        }

    for fname, fdef in template_def.get("furniture", {}).items():
        fcat = FURNITURE_CATALOG[fdef["catalog_key"]]
        size = fcat["size"]          # (X, Y, Z) total extent
        sz_x, sz_y, sz_z = size

        bounds = _inner(room_w, room_d)
        table_top_z = fcat["table_top_z"]

        if "zone" in fdef and fdef["zone"] == "center":
            fx = (bounds["x_min"] + bounds["x_max"]) / 2
            fy = (bounds["y_min"] + bounds["y_max"]) / 2
        elif "wall" in fdef:
            wall = fdef["wall"]
            offset = fdef.get("offset", 0.08)
            # Ground-plane half-extents: X=width, Z=depth → Y spread
            hx, hy = sz_x / 2, sz_z / 2

            if wall == "north":
                fx = 0.0
                fy = bounds["y_max"] - hy - offset
            elif wall == "south":
                fx = 0.0
                fy = bounds["y_min"] + hy + offset
            elif wall == "east":
                fx = bounds["x_max"] - hx - offset
                fy = 0.0
            elif wall == "west":
                fx = bounds["x_min"] + hx + offset
                fy = 0.0
            else:
                fx, fy = 0, 0
        else:
            fx = rng.uniform(bounds["x_min"], bounds["x_max"])
            fy = rng.uniform(bounds["y_min"], bounds["y_max"])

        # spec_Z = target(0) + bb_center_z - bottom_z (puts bottom at floor)
        bb_center_z = fcat.get("bb_center_z", sz_z / 2.0)
        fz = bb_center_z - fcat.get("bottom_z", -(sz_z / 2.0))

        furniture_specs.append({
            "file": fcat["file"],
            "label": fname,
            "category": fdef["catalog_key"],
            "type": fdef["catalog_key"],
            "position": [float(fx), float(fy), float(fz)],
            "rotation": [0, 0, 0],
            "scale": 1.0,
            "size": list(size),
        })

        # Faucet exclusion zone on central_table (at local ~(-0.55,-0.31), 0.18×0.12m)
        faucet_exclusions = []
        if fdef.get("catalog_key") == "lab_table":
            faucet_exclusions = [( -0.55, -0.31, 0.09, 0.06 )]  # x, y, half_width, half_height

        furniture_info[fname] = {
            "center_xy": (fx, fy),
            "size_xy": (sz_x, sz_y),
            "table_top_z": table_top_z,
            "faucet_exclusions": faucet_exclusions,
        }

    return furniture_specs, furniture_info


# ═══════════════════════════════════════════════════════════════════════════
# Main Spec Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_spec(template_name: str, seed: int,
                  variation: int = 0, split: str = "train") -> dict:
    """
    Generate a complete lab_spec.json from a layout template.

    Args:
        template_name: key in LAYOUT_TEMPLATES
        seed: random seed for reproducibility
        variation: variation index (0, 1, 2, ...)
        split: train/val/test

    Returns:
        lab_spec dict ready to be saved as JSON
    """
    if template_name not in LAYOUT_TEMPLATES:
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Available: {list(LAYOUT_TEMPLATES.keys())}"
        )

    template = LAYOUT_TEMPLATES[template_name]
    rng = np.random.RandomState(seed * 100 + variation)

    # ── 1. Room dimensions ──────────────────────────────────────────────
    room_w, room_d, room_h = ROOM_GLB_DIMS
    rooms_out = None

    if "rooms" in template:
        rooms_out = []
        for rdef in template["rooms"]:
            rooms_out.append({
                "name": rdef["name"],
                "width": rdef.get("width", room_w),
                "depth": rdef.get("depth", room_d),
                "height": rdef.get("height", room_h),
                "offset": rdef.get("offset", [0, 0]),
                "wall_thickness": rdef.get("wall_thickness", 0.15),
            })

    # ── 2. Doorways ─────────────────────────────────────────────────────
    doorways = []
    for dw in template.get("doorways", []):
        dw_copy = dict(dw)
        if "position" in dw_copy and isinstance(dw_copy["position"], (int, float)):
            dw_copy["position"] = round(float(rng.uniform(0.25, 0.75)), 2)
        doorways.append(dw_copy)

    # ── 3. Furniture ────────────────────────────────────────────────────
    furniture_specs, furniture_info = place_furniture(
        template, room_w, room_d, rng
    )

    # ── 4. Instruments ──────────────────────────────────────────────────
    instrument_specs = []
    target_categories: List[str] = []
    used_files = set()

    # Placer for main room
    placer = InstrumentPlacer(rng, room_w, room_d)

    # Per-room placers for multi-room templates
    room_placers = {}
    if rooms_out:
        for rdef in rooms_out:
            room_placers[rdef["name"]] = InstrumentPlacer(
                rng, rdef["width"], rdef["depth"]
            )

    for slot_name, slot_def in template.get("instrument_slots", {}).items():
        count_def = slot_def["count"]
        if isinstance(count_def, tuple):
            lo, hi = count_def
            count = rng.randint(lo, hi + 1)
        else:
            count = count_def

        room_ref = slot_def.get("room")
        active_placer = room_placers.get(room_ref, placer) if room_ref else placer
        active_rw = room_w
        active_rd = room_d
        if room_ref and rooms_out:
            for rdef in rooms_out:
                if rdef["name"] == room_ref:
                    active_rw = rdef["width"]
                    active_rd = rdef["depth"]
                    break

        zone = slot_def.get("zone", "wall")
        wall = slot_def.get("wall")
        corner = slot_def.get("corner")
        pos_hint = slot_def.get("position_hint")
        on_furn = slot_def.get("on_furniture")

        for _ in range(count):
            allowed = [a for a in slot_def["allowed"]
                       if INSTRUMENT_CATALOG[a]["file"] not in used_files]
            if not allowed:
                continue
            choice = rng.choice(allowed)
            cat = INSTRUMENT_CATALOG[choice]
            used_files.add(cat["file"])

            # Resolve furniture info for table_top
            furn_info = None
            if zone == "table_top" and on_furn:
                furn_info = furniture_info.get(on_furn)

            pos = active_placer.place(
                zone=zone,
                size_xyz=cat["size"],
                bottom_z=cat.get("bottom_z"),
                bb_center_z=cat.get("bb_center_z", cat["size"][2] / 2.0),
                rw=active_rw,
                rd=active_rd,
                wall=wall,
                corner=corner,
                position_hint=pos_hint,
                furniture_info=furn_info,
            )

            if pos is None:
                # Placement failed, skip this instrument
                used_files.discard(cat["file"])
                continue

            # ── Rotation: point operational face away from wall / into room ──
            rotation = _compute_rotation(cat, zone, wall, pos, furn_info)

            # ── Operational goal position (Blender Z-up, on floor) ──
            goal_pos = _compute_operational_goal(cat, pos, rotation)

            entry = {
                "file": cat["file"],
                "label": f"{choice}_{len(instrument_specs)}",
                "category": cat["category"],
                "position": [round(float(pos[0]), 3),
                            round(float(pos[1]), 3),
                            round(float(pos[2]), 3)],
                "rotation": rotation,
                "scale": 1.0,
                "operational_goal_position": goal_pos,  # Blender Z-up, floor level
            }
            instrument_specs.append(entry)

            if cat["category"] not in target_categories:
                target_categories.append(cat["category"])

    # ── 5. Build spec ──────────────────────────────────────────────────
    scene_name = f"lab_{template_name}_s{seed}_v{variation}"

    spec = {
        "scene_name": scene_name,
        "layout_type": template_name,
        "split": split,
        "room_glb": str(ROOM_GLB),
        "room": {
            "width": round(room_w, 2),
            "depth": round(room_d, 2),
            "height": room_h,
            "wall_thickness": WALL_THICKNESS,
        },
        "doorways": doorways,
        "furniture": furniture_specs,
        "furniture_dir": str(FURNITURE_DIR),
        "instrument_dir": str(INSTRUMENT_DIR),
        "instruments": instrument_specs,
        "target_categories": target_categories,
        "episodes_per_category": EPISODES_PER_CATEGORY,
        "num_start_positions": NUM_START_POSITIONS,
        "min_start_spacing": MIN_START_SPACING,
        "random_seed": seed * 100 + variation,
        "safety_radius": 1.3,
        "navmesh_settings": NAVMESH_SETTINGS,
    }
    if rooms_out:
        spec["rooms"] = rooms_out

    return spec


# ═══════════════════════════════════════════════════════════════════════════
# Batch Generation
# ═══════════════════════════════════════════════════════════════════════════

def _split_seed(template_name: str, i: int, split: str) -> int:
    """Deterministic seed that varies by template, variant index, AND split."""
    base = hash(template_name) % 10000
    split_offset = hash(split) % 1000
    return (base + split_offset + i * 137) % 10000


def generate_batch(template_name: str, count: int,
                   start_variation: int = 0,
                   split: str = "train") -> List[dict]:
    """Generate multiple variants of the same template."""
    specs = []
    for i in range(count):
        seed = _split_seed(template_name, i, split)
        spec = generate_spec(template_name, seed, start_variation + i, split)
        specs.append(spec)
    return specs


def generate_all_for_split(split: str, mvp: bool = False) -> List[dict]:
    """Generate all scenes for a given split according to SPLIT_DISTRIBUTION."""
    dist = SPLIT_DISTRIBUTION_MVP if mvp else SPLIT_DISTRIBUTION
    template_counts = dist.get(split, {})
    specs = []
    for template_name, count in template_counts.items():
        specs.extend(generate_batch(template_name, count, split=split))
    return specs


def generate_all(mvp: bool = False) -> List[dict]:
    """Generate all scenes for all splits."""
    specs = []
    for split in ["train", "val", "test"]:
        specs.extend(generate_all_for_split(split, mvp=mvp))
    return specs


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

WALL_THICKNESS = 0.15

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate BioVLN lab_spec.json from layout templates"
    )
    parser.add_argument("--list", action="store_true",
                       help="List available templates")
    parser.add_argument("--template", type=str,
                       help="Template name (single_room, two_room, etc.)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--variation", type=int, default=0,
                       help="Variation index")
    parser.add_argument("--count", type=int, default=1,
                       help="Number of variants to generate")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split (train/val/test)")
    parser.add_argument("--all", action="store_true",
                       help="Generate all scenes for all splits")
    parser.add_argument("--mvp", action="store_true",
                       help="Use MVP distribution (5 scenes)")
    parser.add_argument("--output", type=str,
                       help="Output JSON file (single scene)")
    parser.add_argument("--output_dir", type=str, default="specs",
                       help="Output directory for multiple scenes")

    args = parser.parse_args()

    if args.list:
        print("Available templates:")
        for name, tpl in LAYOUT_TEMPLATES.items():
            print(f"  {name}: {tpl['description']}")
        print(f"\nAvailable instruments (CENTER-origin):")
        for name, cat in INSTRUMENT_CATALOG.items():
            print(f"  {name}: {cat['file']} "
                  f"({cat['size'][0]:.2f}×{cat['size'][1]:.2f}×{cat['size'][2]:.2f}m) "
                  f"zone={cat['zone']}")
        print(f"\nAvailable furniture (CENTER-origin):")
        for name, cat in FURNITURE_CATALOG.items():
            print(f"  {name}: {cat['file']} "
                  f"({cat['size'][0]:.2f}×{cat['size'][1]:.2f}×{cat['size'][2]:.2f}m)")
        return

    if args.all:
        specs = generate_all(mvp=args.mvp)
    elif args.template:
        specs = generate_batch(args.template, args.count,
                               start_variation=args.variation,
                               split=args.split)
    else:
        parser.print_help()
        return

    if not specs:
        print("No specs generated.")
        return

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output and len(specs) == 1:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(specs[0], f, indent=2)
        print(f"✓ {out_path}")
    else:
        for spec in specs:
            name = spec["scene_name"]
            out_path = output_dir / f"{name}_spec.json"
            with open(out_path, "w") as f:
                json.dump(spec, f, indent=2)
            print(f"✓ {out_path}  [{spec['split']}] "
                  f"{spec['room']['width']:.1f}×{spec['room']['depth']:.1f}m "
                  f"{len(spec['target_categories'])} categories "
                  f"{len(spec['instruments'])} instruments")

    # Summary
    by_split = {}
    for spec in specs:
        s = spec["split"]
        by_split.setdefault(s, []).append(spec["scene_name"])
    print(f"\nGenerated {len(specs)} specs:")
    for split, names in sorted(by_split.items()):
        print(f"  [{split}] {len(names)} scenes: {', '.join(names)}")


if __name__ == "__main__":
    main()
