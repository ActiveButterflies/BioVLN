#!/usr/bin/env python
"""Collect single-frame RGB-D views around instruments with geometry-derived labels.

Blender horizontal (x, y) maps to Habitat (x, z) as x -> x, y -> -z.
The operational outward normal f is the unit vector from the instrument centre to its
operational goal position (both in Blender xy).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

SCENES = Path("data/multi_scene/output")
SPECS = Path("data/BioVLN/review_revision/dataset_metadata_measured_v3")
NAV = Path("data/BioVLN/review_revision/navmesh_r018")
STANDOFF_LO, STANDOFF_HI, LATERAL_MAX, HEAD_TOL = 0.50, 0.90, 0.50, 30.0


def make_sim(scene, size, hfov):
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = str(SCENES / scene / f"{scene}.glb")
    sc.enable_physics = False
    sc.gpu_device_id = 0
    sc.load_semantic_mesh = False
    sc.pbr_image_based_lighting = False
    ac = habitat_sim.agent.AgentConfiguration()
    ac.height = 0.0
    ac.radius = 0.18
    specs = []
    for uuid, stype in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [size, size]
        s.position = [0.0, 0.88, 0.0]
        s.hfov = hfov
        specs.append(s)
    ac.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [ac]))
    if not sim.pathfinder.load_nav_mesh(str(NAV / f"{scene}.navmesh")):
        raise RuntimeError(scene)
    return sim


def frame_of(inst):
    g = inst["measured_geometry"]
    cb = np.array(g["center_xy"], float)
    gb = np.array(inst["operational_goal_position"][:2], float)
    d = gb - cb
    if np.linalg.norm(d) < 1e-6:
        return None
    fb = d / np.linalg.norm(d)
    hx, hy = g["half_extents_local_xy"]
    th = float(inst.get("rotation", [0, 0, 0])[2])
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-math.sin(th), math.cos(th)])
    half = abs(hx * float(np.dot(fb, u))) + abs(hy * float(np.dot(fb, v)))
    c_hab = np.array([cb[0], -cb[1]])
    f_hab = np.array([fb[0], -fb[1]])
    return c_hab, f_hab, half


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    names = args.scenes or sorted(p.name for p in SPECS.iterdir() if p.is_dir())
    if args.limit_scenes:
        names = names[: args.limit_scenes]
    rgbs, deps, rows = [], [], []
    for si, scene in enumerate(names):
        spec = json.loads((SPECS / scene / f"{scene}_spec.json").read_text())
        sim = make_sim(scene, args.size, args.hfov)
        agent = sim.get_agent(0)
        for inst in spec.get("instruments", []):
            fo = frame_of(inst)
            if fo is None:
                continue
            c, f, half = fo
            cand = []
            for r in (0.6, 0.9, 1.2, 1.5):
                for deg in range(0, 360, 30):
                    cand.append((inst["measured_geometry"]["center_xy"][0] + r * math.cos(math.radians(deg)),
                                 inst["measured_geometry"]["center_xy"][1] + r * math.sin(math.radians(deg))))
            # GOAL_JITTER: poses around the designated operating position, mostly positive
            gb0 = np.array(inst["operational_goal_position"][:2], float)
            for dr, dh in ((0.0, 0.0), (0.10, 0.0), (-0.10, 0.0), (0.0, 8.0), (0.0, -8.0),
                           (0.06, 5.0), (-0.06, -5.0)):
                th = math.radians(dh)
                off = np.array([math.cos(th) * dr, math.sin(th) * dr])
                cand.append((gb0[0] + off[0], gb0[1] + off[1]))
            if True:
                for pb0 in cand:
                    pb = np.array(pb0)
                    a = math.radians(deg)
                    probe = np.array([pb[0], 0.03, -pb[1]])
                    snap = sim.pathfinder.snap_point(probe)
                    if snap is None or not np.all(np.isfinite(snap)):
                        continue
                    h = np.array([float(snap[0]), float(snap[2])])
                    if float(np.linalg.norm(h - np.array([probe[0], probe[2]]))) > 0.15:
                        continue
                    if float(sim.pathfinder.distance_to_closest_obstacle([h[0], 0.03, h[1]])) < 0.16:
                        continue
                    rel = h - c
                    standoff = float(np.dot(rel, f) - half)
                    lateral = abs(cross(f, rel))
                    d_face = c - h
                    if np.linalg.norm(d_face) < 1e-6:
                        continue
                    d_face = d_face / np.linalg.norm(d_face)
                    yaw = math.atan2(-d_face[0], -d_face[1])
                    ang = math.degrees(math.atan2(cross(f, rel), float(np.dot(f, rel))))
                    side = 0 if abs(ang) <= 45 else (1 if abs(ang) >= 135 else (2 if ang > 0 else 3))
                    for jit in (-12.0, 0.0, 12.0):
                        y = yaw + math.radians(jit)
                        fwd = np.array([-math.sin(y), -math.cos(y)])
                        head = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(fwd, -f))))))
                        in_zone = int(STANDOFF_LO <= standoff <= STANDOFF_HI and lateral <= LATERAL_MAX
                                      and head <= HEAD_TOL)
                        state = agent.get_state()
                        state.position = np.array([h[0], 0.0, h[1]], dtype=np.float32)
                        state.rotation = quat_from_angle_axis(y, np.array([0.0, 1.0, 0.0]))
                        agent.set_state(state)
                        obs = sim.get_sensor_observations()
                        rgbs.append(obs["rgb"].astype(np.uint8))
                        deps.append(obs["depth"].astype(np.float16))
                        rows.append(dict(scene=scene, inst=inst["label"], cat=inst["category"],
                                         standoff=round(standoff, 4), lateral=round(lateral, 4),
                                         head=round(head, 2), in_zone=in_zone, side=side))
                        if args.debug and len(rows) <= 6:
                            print({k: rows[-1][k] for k in ("standoff", "lateral", "head", "in_zone", "side")})
        sim.close()
        print(f"[{si+1}/{len(names)}] {scene}: frames={len(rows)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, rgb=np.stack(rgbs), depth=np.stack(deps), meta=json.dumps(rows))
    pos = sum(r["in_zone"] for r in rows)
    print(f"saved frames={len(rows)} in_zone={pos} ({100*pos/max(1,len(rows)):.1f}%)")


if __name__ == "__main__":
    main()
