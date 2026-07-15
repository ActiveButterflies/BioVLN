#!/usr/bin/env python3
"""
VLFM evaluation with safety metrics (Min Clearance + Zone 2 Violation Rate).

Strategy: monkey-patch habitat_sim.Simulator.step to record agent trajectories,
then run run_vlfm_all.py normally. Compute safety after each episode.
"""

import sys, os, json, gzip, time
from pathlib import Path
import numpy as np

# Clear proxy
for k in ["ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)
if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import habitat_sim, magnum as mn

ZONE2_DANGER = 0.5

INSTRUMENT_PARAMS = {
    "refrigerator": (0.455, 0.7, 0.45), "refrige": (0.455, 0.7, 0.45),
    "incubator": (0.673, 0.7, 0.68), "incubor": (0.673, 0.7, 0.68),
    "centrifuge": (0.245, 0.7, 0.25), "mini_centrifuge": (0.332, 0.7, 0.33),
    "pipette": (0.159, 0.7, 0.06), "waste_bin": (0.183, 0.7, 0.19),
    "lab_waste_bin": (0.183, 0.7, 0.19), "cabinet": (0.692, 0.7, 0.69),
    "lab_table": (0.45, 0.7, 1.20), "lab_table_1": (0.45, 0.7, 1.20),
    "lab_bench": (2.97, 0.7, 0.78),
}

# ─── Global step hook ───────────────────────────────────────────────────────

_ORIG_STEP = habitat_sim.Simulator.step
_TRAJ = {}; _CUR = [None]; _REC = [False]

def _hooked_step(self, action, *a, **kw):
    _ORIG_STEP(self, action, *a, **kw)
    if _REC[0]:
        try:
            st = self.get_agent(0).get_state()
            _TRAJ.setdefault(_CUR[0], []).append(
                (float(st.position[0]), float(st.position[2])))
        except: pass

habitat_sim.Simulator.step = _hooked_step

# ─── Safety ──────────────────────────────────────────────────────────────────

def extract_facing(ep):
    try:
        vp = ep["goals"][0].get("view_points")
        if vp and len(vp) > 0:
            r = vp[0].get("agent_state", {}).get("rotation", [])
            if len(r) == 4:
                q = mn.Quaternion(mn.Vector3(float(r[0]),float(r[1]),float(r[2])),float(r[3]))
                f = q.transform_vector(mn.Vector3(0,0,-1))
                return np.array([f[0], f[2]])
    except: pass
    return np.array([0.0, -1.0])

def build_all_insts(episodes):
    insts = {}
    for ep in episodes:
        g = ep["goals"][0]; cat = g.get("object_category", ep.get("object_category","?"))
        if cat in insts: continue
        gx = np.array([g["position"][0], g["position"][2]], dtype=np.float64)
        f = extract_facing(ep); fn = f/(np.linalg.norm(f) or 1.0)
        fh, ap, hw = INSTRUMENT_PARAMS.get(cat, (0.5,0.7,0.5))
        insts[cat] = (gx - fn*(fh+ap), max(fh, hw))
    return [(c,r) for c,r in insts.values()]

def compute_safety(traj, goal_xy, facing, target_cat, all_insts=None):
    if not traj: return {"min_clear": -1, "zone2_rate": 0}
    if all_insts:
        inst_list = list(all_insts)
    else:
        p = INSTRUMENT_PARAMS.get(target_cat, (0.5, 0.7, 0.5))
        fn = facing/(np.linalg.norm(facing) or 1.0)
        inst_list = [(np.array(goal_xy)-fn*(p[0]+p[1]), max(p[0],p[2]))]
    ds = []
    for pt in traj:
        md = min(max(0.0, float(np.linalg.norm(np.array(pt)-c))-r) for c,r in inst_list)
        ds.append(md)
    return {"min_clear": round(min(ds),3), "zone2_rate": round(sum(1 for d in ds if d<ZONE2_DANGER)/len(ds),4)}

# ─── Dataset registry ───────────────────────────────────────────────────────

REG = {
    "lsat_target": {"scenes":"/root/gpufree-data/lsat_test","splits":"/root/gpufree-data/lsat_test/splits","results":"/root/gpufree-data/lsat_test/results"},
    "two_room":    {"scenes":"/root/gpufree-data/two_room_output","splits":"/root/gpufree-data/two_room_output/splits","results":"/root/gpufree-data/two_room_output/results"},
    "multi_scene": {"scenes":"/root/gpufree-data/multi_scene/output","splits":"/root/gpufree-data/multi_scene/output/splits","results":"/root/gpufree-data/multi_scene/results"},
}

# ─── Import VLFM ────────────────────────────────────────────────────────────

PIPE = Path("/root/gpufree-data/pipeline")
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(Path("/root/gpufree-data/multi_scene")))
import run_vlfm_v4 as vlfm
import run_vlfm_all

# Fix: remove PBR lighting that crashes EGL
_orig_create_sim = vlfm.create_sim
def _safe_create_sim():
    from habitat_sim import agent as am
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = vlfm.GLB_PATH; sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0; sim_cfg.allow_sliding = False
    specs = []
    for name, stype in [("rgb", habitat_sim.SensorType.COLOR),
                         ("depth", habitat_sim.SensorType.DEPTH)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = name; spec.sensor_type = stype
        spec.resolution = [vlfm.IMAGE_H, vlfm.IMAGE_W]
        spec.position = mn.Vector3(0, vlfm.CAM_HEIGHT, 0)
        spec.hfov = vlfm.HFOV
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        specs.append(spec)
    agent_cfg = am.AgentConfiguration()
    agent_cfg.height = vlfm.CAM_HEIGHT; agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    if os.path.exists(vlfm.NAVMESH_PATH):
        sim.pathfinder.load_nav_mesh(vlfm.NAVMESH_PATH)
    return sim
vlfm.create_sim = _safe_create_sim

# ─── Run one dataset ────────────────────────────────────────────────────────

def run_vlfm_dataset(dataset, split="dev", max_eps=None):
    cfg = REG[dataset]
    ep_file = Path(cfg["splits"]) / f"episodes_{split}.json.gz"
    scenes_root = Path(cfg["scenes"])
    result_dir = Path(cfg["results"])

    with gzip.open(ep_file, "rt") as f:
        eps = json.load(f)["episodes"]
    if max_eps: eps = eps[:max_eps]
    all_insts = build_all_insts(eps)

    total = len(eps)
    results = []
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  VLFM + SAFETY [{dataset}] [{split.upper()}] — {total} eps")
    print(f"  blind=True  vlm_vmap=True")
    print(f"{'='*60}\n")

    sims = {}  # scene_name → sim (cached per scene)

    for i, ep in enumerate(eps):
        ep_id = str(ep.get("episode_id", i))
        scene_name = ep.get("scene_name", "?")
        target_cat = ep.get("object_category", "?")
        goal = ep["goals"][0]
        goal_xy = np.array([goal["position"][0], goal["position"][2]])
        facing = extract_facing(ep)

        # Set up scene
        scene_dir = scenes_root / scene_name
        glb = str(scene_dir / f"{scene_name}.glb")
        navmesh = str(scene_dir / f"{scene_name}.navmesh")
        vlfm.GLB_PATH = glb
        vlfm.NAVMESH_PATH = navmesh

        # Start recording
        _CUR[0] = ep_id; _REC[0] = True; _TRAJ.pop(ep_id, None)

        t_ep = time.time()
        try:
            result = vlfm.run_episode(ep, i, save_video=False, blind=True, use_vlm_vmap=True)
        except Exception as e:
            result = {"success": False, "steps": 0, "final_dist": -1, "target": target_cat}
        elapsed = time.time() - t_ep

        _REC[0] = False

        # Safety
        safety = compute_safety(_TRAJ.get(ep_id, []), goal_xy, facing, target_cat, all_insts)
        result["safety_min_clear"] = safety["min_clear"]
        result["safety_zone2_rate"] = safety["zone2_rate"]
        result["episode_id"] = ep_id
        result["scene_name"] = scene_name
        result["target"] = target_cat
        results.append(result)

        status = "✓" if result.get("success") else "✗"
        print(f"  [{i+1}/{total}] {status} {target_cat:<20s} "
              f"steps={result.get('steps',0):3d} "
              f"clr={safety['min_clear']:.2f}m z2={safety['zone2_rate']:.1%} "
              f"t={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    n = len(results); n_ok = sum(1 for r in results if r.get("success"))
    sr = n_ok/n if n>0 else 0
    clears = [r.get("safety_min_clear",-1) for r in results if r.get("safety_min_clear",-1)>=0]
    z2rs = [r.get("safety_zone2_rate",-1) for r in results if r.get("safety_zone2_rate",-1)>=0]

    metrics = {
        "method": "vlfm_vlm", "dataset": dataset, "split": split,
        "total": n, "success": n_ok,
        "SR": round(sr,4), "SPL": 0,
        "elapsed": round(elapsed,1),
        "safety": {
            "avg_min_clearance": round(np.mean(clears),3) if clears else 0,
            "avg_zone2_violation_rate": round(np.mean(z2rs),4) if z2rs else 0,
        },
        "per_episode": results,
    }

    out = result_dir / f"eval_vlfm_vlm_{dataset}_{split}_safety.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  SR={sr:.1%} minClear={metrics['safety']['avg_min_clearance']:.3f}m "
          f"zone2={metrics['safety']['avg_zone2_violation_rate']:.1%} t={elapsed:.0f}s")
    print(f"  ✓ {out.name}")
    return metrics


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lsat_target", choices=["lsat_target","two_room","multi_scene"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    if args.all:
        for ds in ["lsat_target", "two_room"]:
            run_vlfm_dataset(ds, args.split, max_eps=args.max)
        # Summary
        print(f"\n{'='*72}")
        print(f"  VLFM SAFETY SUMMARY")
        print(f"  {'Dataset':<14s} {'SR':>6s} {'SPL':>6s} {'MinClear':>8s} {'Zone2%':>8s}")
        print(f"  {'-'*50}")
        for ds in ["lsat_target", "two_room"]:
            f = Path(REG[ds]["results"]) / f"eval_vlfm_vlm_{ds}_{args.split}_safety.json"
            if f.exists():
                d = json.load(open(f)); s = d["safety"]
                print(f"  {ds:<14s} {d['SR']:>5.1%} {d['SPL']:>6.3f} {s['avg_min_clearance']:>7.3f}m {s['avg_zone2_violation_rate']:>7.1%}")
        print(f"{'='*72}")
    else:
        run_vlfm_dataset(args.dataset, args.split, max_eps=args.max)
