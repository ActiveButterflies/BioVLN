"""Geometry and outcome contract for BioVLN-Revision-20.

Coordinates in this module are Blender horizontal (X, Y). Habitat positions
are converted with ``(x, y, z) -> (x, -z)``. This evaluator is independent of
the navigation policy: all agents must be scored against the same pose set.

The paper does not specify a robot manipulator. The region is consequently a
*navigation access proxy*, not a verified manipulation pose. Main settings are
0.50--0.90 m standoff from the accessible physical surface, a 0.18 m circular
base and a 30 degree target-facing tolerance. Sensitivity analyses must vary
these settings; they are not properties established by the original paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


ANNOTATION_VERSION = "measured-geometry-v3-20261012"
Vec2 = tuple[float, float]


@dataclass(frozen=True)
class PoseConfig:
    robot_radius_m: float = 0.18
    navmesh_baked_agent_radius_m: float = 0.10
    standoff_min_m: float = 0.50
    standoff_max_m: float = 0.90
    half_work_width_m: float = 0.50
    heading_tolerance_deg: float = 30.0
    placement_buffer_m: float = 0.25
    risk_threshold_m: float = 0.50
    candidate_spacing_m: float = 0.10

    def __post_init__(self) -> None:
        if self.robot_radius_m <= 0 or self.standoff_min_m <= 0:
            raise ValueError("robot radius and standoff must be positive")
        if self.navmesh_baked_agent_radius_m < 0:
            raise ValueError("navmesh baked radius cannot be negative")
        if self.standoff_max_m <= self.standoff_min_m:
            raise ValueError("standoff max must exceed min")
        if self.half_work_width_m <= self.robot_radius_m:
            raise ValueError("work half-width must exceed robot radius")
        if not 0 < self.heading_tolerance_deg <= 90:
            raise ValueError("heading tolerance must be in (0, 90]")


@dataclass(frozen=True)
class Asset:
    label: str
    category: str
    center: Vec2
    half_x: float
    half_y: float
    rotation_deg: float
    bottom_z: float
    top_z: float
    operational_goal: Vec2 | None = None
    local_face: Vec2 | None = None

    def to_local_vector(self, vector: Vec2) -> Vec2:
        return rotate(vector, -self.rotation_deg)

    def to_local_point(self, point: Vec2) -> Vec2:
        return self.to_local_vector(sub(point, self.center))

    def signed_distance(self, point: Vec2) -> float:
        """Signed 2-D distance to the oriented footprint; negative inside."""
        u, v = self.to_local_point(point)
        dx, dy = abs(u) - self.half_x, abs(v) - self.half_y
        outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
        inside = min(max(dx, dy), 0.0)
        return outside + inside

    def ray_exit_distance(self, point: Vec2, direction: Vec2) -> float | None:
        """Distance from an interior point to the first footprint edge."""
        px, py = self.to_local_point(point)
        dx, dy = self.to_local_vector(direction)
        if abs(px) > self.half_x + 1e-6 or abs(py) > self.half_y + 1e-6:
            return None
        times = []
        if dx > 1e-9:
            times.append((self.half_x - px) / dx)
        elif dx < -1e-9:
            times.append((-self.half_x - px) / dx)
        if dy > 1e-9:
            times.append((self.half_y - py) / dy)
        elif dy < -1e-9:
            times.append((-self.half_y - py) / dy)
        return min((t for t in times if t >= 0), default=None)


@dataclass(frozen=True)
class SceneGeometry:
    name: str
    instruments: tuple[Asset, ...]
    furniture: tuple[Asset, ...]

    @property
    def obstacles(self) -> tuple[Asset, ...]:
        return self.instruments + self.furniture


@dataclass(frozen=True)
class OperationalRegion:
    target_label: str
    target_category: str
    target_center: Vec2
    face_outward: Vec2
    surface_origin: Vec2
    half_width: float
    surface_source: str
    legacy_goal: Vec2

    def coordinates(self, point: Vec2) -> tuple[float, float]:
        delta = sub(point, self.surface_origin)
        normal = dot(delta, self.face_outward)
        lateral = cross(self.face_outward, delta)
        return normal, lateral

    def sample_centers(self, cfg: PoseConfig) -> list[Vec2]:
        """Deterministic grid of base centers inside the body-eroded region."""
        lo = cfg.standoff_min_m
        hi = cfg.standoff_max_m
        lat = self.half_width - cfg.robot_radius_m
        normals = linspace_by_step(lo, hi, cfg.candidate_spacing_m)
        laterals = linspace_by_step(-lat, lat, cfg.candidate_spacing_m)
        perp = (-self.face_outward[1], self.face_outward[0])
        return [add(self.surface_origin,
                    add(mul(self.face_outward, d), mul(perp, w)))
                for d in normals for w in laterals]


def rotate(vector: Vec2, degrees: float) -> Vec2:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1])


def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Vec2, k: float) -> Vec2:
    return (a[0] * k, a[1] * k)


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def unit(a: Vec2) -> Vec2:
    length = norm(a)
    if length < 1e-9:
        raise ValueError("zero-length direction")
    return (a[0] / length, a[1] / length)


def linspace_by_step(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("grid step must be positive")
    count = max(1, math.ceil((end - start) / step))
    return [start + (end - start) * i / count for i in range(count + 1)]


# Dimensions: supplement_source/code/config.py. Face-v2 normals audited against
# four cardinal source-asset views, with the two ambiguous single-side choices
# explicitly confirmed by the user on 2026-10-12. Original scorer is preserved.
# Historical source: supplement_source/code/config.py, INSTRUMENT_CATALOG and
# FURNITURE_CATALOG. Tuple = (half X, half Y, height, local outward XY).
CATALOG: dict[str, tuple[float, float, float, Vec2 | None]] = {
    "refrigerator": (0.445, 0.455, 1.780, (0.0, -1.0)),
    "incubator": (0.675, 0.425, 1.900, (-1.0, 0.0)),
    "centrifuge": (0.245, 0.210, 0.630, (1.0, 0.0)),
    "mini_centrifuge": (0.330, 0.250, 0.430, (1.0, 0.0)),
    "pipette": (0.020, 0.160, 0.280, (1.0, 0.0)),
    "lab_waste_bin": (0.190, 0.185, 0.510, (0.0, 1.0)),
    "cabinet": (0.692, 0.287, 1.768, (0.0, -1.0)),
    "lab_bench": (0.775, 2.970, 0.760, None),
    "lab_table": (1.200, 0.450, 1.100, None),
}
ALIASES = {"refrige": "refrigerator", "incubor": "incubator",
           "waste_bin": "lab_waste_bin", "lab_table_1": "lab_bench"}


def _category(name: str) -> str:
    return ALIASES.get(name, name)


def _parse_asset(item: dict[str, Any]) -> Asset:
    category = _category(str(item["category"]))
    if category not in CATALOG:
        raise ValueError(f"unknown asset geometry: {category}")
    hx, hy, catalog_height, local_face = CATALOG[category]
    # Face-v2 accepts only an explicit local XY unit normal. Never infer from
    # historical operational_goal_position: those points use face-v1.
    if "operational_face_local_xy" in item:
        override = item["operational_face_local_xy"]
        if not isinstance(override, (list, tuple)) or len(override) != 2:
            raise ValueError("operational_face_local_xy must have two components")
        override = tuple(float(v) for v in override)
        if not all(math.isfinite(v) for v in override) or not math.isclose(norm(override), 1.0, abs_tol=1e-6):
            raise ValueError("operational_face_local_xy must be a finite unit vector")
        if local_face is None:
            raise ValueError("Cannot assign an operational face to furniture")
        local_face = override
    scale = float(item.get("scale", 1.0))
    if scale <= 0:
        raise ValueError(f"invalid scale for {item.get('label')}")
    pos = item["position"]
    rotation = float(item.get("rotation", [0, 0, 0])[2])
    height = float(item.get("size", (2 * hx, 2 * hy, catalog_height))[2])
    center_z = float(pos[2])
    goal = item.get("operational_goal_position")
    center = (float(pos[0]), float(pos[1]))
    half_x, half_y = hx * scale, hy * scale
    bottom_z, top_z = center_z - height * scale / 2, center_z + height * scale / 2
    measured = item.get("measured_geometry")
    if measured is not None:
        if measured.get("coordinate_frame") != "blender_z_up":
            raise ValueError("Measured geometry must use this pipeline's Blender Z-up export")
        source_hash = measured.get("source_scene_glb_sha256", "")
        if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise ValueError("Missing measured scene geometry provenance")
        center = tuple(float(v) for v in measured["center_xy"])
        halves = tuple(float(v) for v in measured["half_extents_local_xy"])
        bounds = tuple(float(v) for v in measured["vertical_bounds_z"])
        if len(center) != 2 or len(halves) != 2 or len(bounds) != 2:
            raise ValueError("Malformed measured oriented footprint")
        if not all(math.isfinite(v) for v in center + halves + bounds):
            raise ValueError("Nonfinite measured footprint")
        if min(halves) <= 0 or bounds[1] <= bounds[0]:
            raise ValueError("Invalid measured dimensions")
        half_x, half_y = halves  # Already includes actual instance scale.
        bottom_z, top_z = bounds
    return Asset(
        label=str(item.get("label", category)), category=category,
        center=center,
        half_x=half_x, half_y=half_y,
        rotation_deg=rotation,
        bottom_z=bottom_z,
        top_z=top_z,
        operational_goal=(float(goal[0]), float(goal[1])) if goal else None,
        local_face=local_face,
    )


def scene_from_spec(spec: dict[str, Any]) -> SceneGeometry:
    return SceneGeometry(
        name=str(spec["scene_name"]),
        instruments=tuple(_parse_asset(x) for x in spec.get("instruments", [])),
        furniture=tuple(_parse_asset(x) for x in spec.get("furniture", [])),
    )


def habitat_xy(position_xyz: Iterable[float]) -> Vec2:
    p = tuple(position_xyz)
    return (float(p[0]), -float(p[2]))


def match_target(scene: SceneGeometry, episode: dict[str, Any]) -> Asset:
    category = _category(str(episode["object_category"]))
    object_name = str(episode["goals"][0].get("object_name", ""))
    named = [x for x in scene.instruments if x.label == object_name]
    if named:
        if named[0].category != category:
            raise ValueError(f"object name/category conflict: {object_name}, {category}")
        return named[0]
    goal_xy = habitat_xy(episode["goals"][0]["position"])
    matches = [x for x in scene.instruments if x.category == category]
    if not matches:
        raise ValueError(f"no {category} in scene {scene.name}")
    return min(matches, key=lambda x: norm(sub(x.operational_goal or x.center, goal_xy)))


def _support_for(target: Asset, scene: SceneGeometry) -> Asset | None:
    if target.bottom_z < 0.45:
        return None
    matches = [x for x in scene.furniture
               if x.signed_distance(target.center) <= 0.05
               and x.top_z <= target.top_z + 0.10]
    return min(matches, key=lambda x: abs(x.top_z - target.bottom_z), default=None)


def region_for(scene: SceneGeometry, target: Asset,
               legacy_goal: Vec2, cfg: PoseConfig) -> OperationalRegion:
    if target.local_face is None:
        raise ValueError(f"target {target.label} has no operational face")
    outward = unit(rotate(target.local_face, target.rotation_deg))
    support = _support_for(target, scene)
    source = support or target
    # For table-top objects, project to the accessible support edge. This
    # avoids treating the small object's footprint as the entire work area.
    exit_distance = source.ray_exit_distance(target.center, outward)
    if exit_distance is None:
        if support is not None:
            raise ValueError(f"target {target.label} lies outside support")
        exit_distance = source.half_x if abs(target.local_face[0]) > 0.5 else source.half_y
    surface = add(target.center, mul(outward, exit_distance))
    return OperationalRegion(
        target_label=target.label, target_category=target.category,
        target_center=target.center, face_outward=outward,
        surface_origin=surface,
        half_width=cfg.half_work_width_m,
        surface_source=source.label,
        legacy_goal=legacy_goal,
    )


def body_clearance(point: Vec2, scene: SceneGeometry, cfg: PoseConfig) -> float:
    """Minimum signed distance between circular base and physical assets."""
    if not scene.obstacles:
        return math.inf
    return min(asset.signed_distance(point) - cfg.robot_radius_m
               for asset in scene.obstacles)


def score_terminal(point: Vec2, heading: Vec2, did_stop: bool,
                   region: OperationalRegion, scene: SceneGeometry,
                   cfg: PoseConfig, legacy_radius_m: float,
                   navmesh_edge_clearance_m: float | None = None) -> dict[str, Any]:
    normal, lateral = region.coordinates(point)
    in_zone = (cfg.standoff_min_m <= normal <= cfg.standoff_max_m
               and abs(lateral) <= region.half_width - cfg.robot_radius_m)
    clearance = body_clearance(point, scene, cfg)
    target_delta = sub(region.target_center, point)
    to_target = unit(target_delta) if norm(target_delta) > 1e-9 else mul(region.face_outward, -1.0)
    heading_unit = unit(heading)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot(heading_unit, to_target)))))
    facing = angle <= cfg.heading_tolerance_deg
    collision_free = clearance >= -1e-6
    edge_required = max(0.0, cfg.robot_radius_m - cfg.navmesh_baked_agent_radius_m)
    navmesh_body_clear = (navmesh_edge_clearance_m is None
                          or navmesh_edge_clearance_m >= edge_required - 1e-6)
    point_success = did_stop and norm(sub(point, region.legacy_goal)) <= legacy_radius_m
    pose_success = did_stop and in_zone and facing and collision_free and navmesh_body_clear
    reasons = []
    if not did_stop:
        reasons.append("no_stop")
    if not in_zone:
        reasons.append("outside_operation_zone")
    if not facing:
        reasons.append("wrong_heading")
    if not collision_free:
        reasons.append("body_collision")
    if not navmesh_body_clear:
        reasons.append("navmesh_boundary_clearance")
    return {
        "pose_success": pose_success,
        "point_success": point_success,
        "in_operation_zone": in_zone,
        "facing_target": facing,
        "body_collision_free": collision_free,
        "navmesh_body_clear": navmesh_body_clear,
        "navmesh_edge_clearance_m": navmesh_edge_clearance_m,
        "body_clearance_m": clearance,
        "heading_error_deg": angle,
        "surface_distance_m": normal,
        "surface_lateral_m": lateral,
        "failure_reasons": reasons,
        "metric_config": asdict(cfg),
    }


def trajectory_safety(points: Iterable[Vec2], scene: SceneGeometry,
                      cfg: PoseConfig) -> dict[str, float]:
    clearances = [body_clearance(p, scene, cfg) for p in points]
    if not clearances:
        raise ValueError("trajectory contains no positions")
    return {
        "MCR_body_m": min(clearances),
        "VRT_body_025": sum(c < cfg.placement_buffer_m for c in clearances) / len(clearances),
        "VRT_body_050": sum(c < cfg.risk_threshold_m for c in clearances) / len(clearances),
    }
