"""Create immutable, target-facing view points for the verified 20-scene set.

The old goal position is retained for diagnostic comparison. Only the single
goal view point is replaced by a reachable pose from the 0.18 m NavMesh audit.
Original compressed splits are never modified.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

from operational_pose_v2 import (PoseConfig, habitat_xy, match_target, region_for,
                              scene_from_spec, score_terminal)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def facing_quaternion(start_xyz: list[float], target_xy: tuple[float, float]
                      ) -> tuple[list[float], tuple[float, float]]:
    start_xy = habitat_xy(start_xyz)
    dx, dy = target_xy[0] - start_xy[0], target_xy[1] - start_xy[1]
    if math.hypot(dx, dy) <= 1e-8:
        raise ValueError("view point coincides with target center")
    yaw = math.atan2(-dx, dy)
    quaternion = [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]
    heading = (-math.sin(yaw), math.cos(yaw))
    return quaternion, heading


def deterministic_gzip(payload: dict) -> bytes:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as sink:
        sink.write(raw)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--navmesh-audit", required=True, type=Path)
    parser.add_argument("--navmesh-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    audit = json.loads(args.navmesh_audit.read_text(encoding="utf-8"))
    if audit["episode_count"] != 930 or audit["pose_feasible_episodes"] != 930:
        raise ValueError("0.18 m NavMesh audit must verify all 930 episodes")
    if audit.get("canonical_interior_pose_feasible_episodes") != 930:
        raise ValueError("all episodes need interior canonical poses")
    if audit["config"]["navmesh_baked_agent_radius_m"] != 0.18:
        raise ValueError("wrong NavMesh agent radius")
    manifest = json.loads(args.navmesh_manifest.read_text(encoding="utf-8"))
    if manifest["scene_count"] != 20 or manifest["settings"]["agent_radius"] != 0.18:
        raise ValueError("NavMesh manifest does not describe 20 radius-0.18 scenes")
    audit_rows = {(row["scene_name"], row["episode_id"]): row
                  for row in audit["per_episode"]}
    if len(audit_rows) != 930:
        raise ValueError("duplicate audit episode")
    cfg = PoseConfig(navmesh_baked_agent_radius_m=0.18)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_records = []
    all_keys = set()
    split_scenes = {}
    for split in ("train", "val", "test"):
        source = args.metadata_dir / f"episodes_procedural_full_{split}_clean.json.gz"
        result = args.output_dir / f"episodes_procedural_full_{split}_pose_v1.json.gz"
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            original = json.load(handle)
        derived = copy.deepcopy(original)
        scene_names = set()
        annotations = []
        for episode in derived["episodes"]:
            scene_name = episode["scene_name"]
            key = (scene_name, episode["episode_id"])
            if key in all_keys:
                raise ValueError(f"duplicate source episode: {key}")
            all_keys.add(key)
            scene_names.add(scene_name)
            row = audit_rows.get(key)
            if row is None or not row["pose_feasible"]:
                raise ValueError(f"missing reachable pose: {key}")
            spec_path = args.metadata_dir / f"{scene_name}_spec.json"
            scene = scene_from_spec(json.loads(spec_path.read_text(encoding="utf-8")))
            target = match_target(scene, episode)
            xyz = row["canonical_interior_pose_position_xyz"]
            quaternion, heading = facing_quaternion(xyz, target.center)
            goal = episode["goals"][0]
            region = region_for(scene, target, habitat_xy(goal["position"]), cfg)
            outcome = score_terminal(habitat_xy(xyz), heading, True,
                                     region, scene, cfg, 0.0)
            if not outcome["pose_success"]:
                raise ValueError(f"derived viewpoint invalid: {key}: {outcome['failure_reasons']}")
            goal["view_points"] = [{"agent_state": {
                "position": xyz, "rotation": quaternion}}]
            annotations.append({
                "scene_name": scene_name, "episode_id": episode["episode_id"],
                "target_label": target.label, "view_point_xyz": xyz,
                "rotation_xyzw": quaternion,
                "source_goal_position_xyz": goal["position"],
                "surface_source": region.surface_source,
                "heading_error_deg": outcome["heading_error_deg"],
                "surface_distance_m": outcome["surface_distance_m"],
                "surface_lateral_m": outcome["surface_lateral_m"],
                "body_clearance_m": outcome["body_clearance_m"],
                "approx_geodesic_m": row["canonical_interior_geo_m"],
            })
        split_scenes[split] = scene_names
        result.write_bytes(deterministic_gzip(derived))
        sidecar = args.output_dir / f"pose_annotations_{split}.jsonl"
        sidecar.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n"
                                   for item in annotations), encoding="utf-8")
        split_records.append({
            "split": split, "episodes": len(annotations),
            "scenes": sorted(scene_names),
            "source": str(source), "source_sha256": sha256(source),
            "derived": str(result), "derived_sha256": sha256(result),
            "annotations": str(sidecar), "annotations_sha256": sha256(sidecar),
        })
    if len(all_keys) != 930:
        raise ValueError(f"expected 930 derived episodes, got {len(all_keys)}")
    if any(split_scenes[a] & split_scenes[b]
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("scene leakage across splits")
    metadata = {
        "name": "BioVLN-Revision-20-pose-v1",
        "view_point_change": "one shortest-reachable interior-grid face-facing pose per episode",
        "legacy_goal_position": "retained unchanged for point-metric diagnostics",
        "navmesh_audit_sha256": sha256(args.navmesh_audit),
        "navmesh_manifest_sha256": sha256(args.navmesh_manifest),
        "pose_config": vars(cfg),
        "splits": split_records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(all_keys),
                      "scenes": len(set().union(*split_scenes.values())),
                      "split_counts": {r["split"]: r["episodes"] for r in split_records}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
