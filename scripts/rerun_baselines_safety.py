#!/usr/bin/env python3
"""
Re-run all baselines with safety metrics.

Strategy: Monkey-patches habitat_sim.Simulator.step() to record agent trajectories,
then runs the ORIGINAL runner scripts (unchanged logic). Computes safety from
the recorded positions.

Usage:
  python rerun_baselines_safety.py --all   # all methods × lsat_target + two_room
  python rerun_baselines_safety.py --dataset lsat_target --method frontier
"""

import sys, os, json, gzip, time, types
from pathlib import Path
from collections import defaultdict
import numpy as np

if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import habitat_sim, magnum as mn

ZONE2_DANGER = 0.5  # Zone 2 hazard threshold from instrument surface

# Per-category: (face_half_depth, approach_dist, half_width)
INSTRUMENT_PARAMS = {
    "refrigerator": (0.455, 0.7, 0.45), "refrige": (0.455, 0.7, 0.45),
    "incubator": (0.673, 0.7, 0.68), "incubor": (0.673, 0.7, 0.68),
    "centrifuge": (0.245, 0.7, 0.25), "mini_centrifuge": (0.332, 0.7, 0.33),
    "pipette": (0.159, 0.7, 0.06), "waste_bin": (0.183, 0.7, 0.19),
    "lab_waste_bin": (0.183, 0.7, 0.19), "cabinet": (0.692, 0.7, 0.69),
    "lab_table": (0.45, 0.7, 1.20), "lab_table_1": (0.45, 0.7, 1.20),
    "lab_bench": (2.97, 0.7, 0.78),
}

REGISTRY = {
    "multi_scene": {
        "scenes": "/root/gpufree-data/multi_scene/output",
        "splits": "/root/gpufree-data/multi_scene/output/splits",
        "results": "/root/gpufree-data/multi_scene/results",
    },
    "two_room": {
        "scenes": "/root/gpufree-data/two_room_output",
        "splits": "/root/gpufree-data/two_room_output/splits",
        "results": "/root/gpufree-data/two_room_output/results",
    },
    "lsat_target": {
        "scenes": "/root/gpufree-data/lsat_test",
        "splits": "/root/gpufree-data/lsat_test/splits",
        "results": "/root/gpufree-data/lsat_test/results",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Global hook: record agent position on every sim.step() call
# ═══════════════════════════════════════════════════════════════════════════

_ORIGINAL_STEP = habitat_sim.Simulator.step
_TRAJECTORIES: dict = {}  # episode_id → [(x,z), ...]
_CURRENT_EP = [None]
_RECORDING = [False]


def _hooked_step(self, action, *args, **kwargs):
    _ORIGINAL_STEP(self, action, *args, **kwargs)
    if _RECORDING[0]:
        try:
            agent = self.get_agent(0)
            st = agent.get_state()
            _TRAJECTORIES.setdefault(_CURRENT_EP[0], []).append(
                (float(st.position[0]), float(st.position[2])))
        except Exception:
            pass


def install_hook():
    habitat_sim.Simulator.step = _hooked_step


def uninstall_hook():
    habitat_sim.Simulator.step = _ORIGINAL_STEP


def start_recording(ep_id):
    _CURRENT_EP[0] = ep_id
    _RECORDING[0] = True
    _TRAJECTORIES.pop(ep_id, None)


def stop_recording():
    _RECORDING[0] = False


def get_trajectory(ep_id):
    return _TRAJECTORIES.get(ep_id, [])


def extract_facing(ep):
    """从 episode 的 view_point 提取操作面朝向。"""
    try:
        vp = ep["goals"][0].get("view_points")
        if vp and len(vp) > 0:
            r = vp[0].get("agent_state", {}).get("rotation", [])
            if len(r) == 4:
                q = mn.Quaternion(mn.Vector3(float(r[0]), float(r[1]), float(r[2])), float(r[3]))
                f = q.transform_vector(mn.Vector3(0, 0, -1))
                return np.array([f[0], f[2]])
    except Exception:
        pass
    return np.array([0.0, -1.0])


def compute_safety(trajectory, goal_xy, facing, target_cat, all_instruments=None):
    """Compute clearance to ALL instrument surfaces in the scene, not just target."""
    if not trajectory:
        return {"min_clear": -1, "zone2_rate": 0}

    # Build instrument list: [(center_xy, radius_m)]
    insts = []
    if all_instruments:
        insts = list(all_instruments)
    else:
        # Fallback: just the target instrument
        p = INSTRUMENT_PARAMS.get(target_cat, (0.5, 0.7, 0.5))
        fn = facing / (np.linalg.norm(facing) or 1.0)
        center = np.array(goal_xy) - fn * (p[0] + p[1])
        radius = max(p[0], p[2])
        insts = [(center, radius)]

    # For each trajectory point, compute min clearance to ALL instruments
    dists = []
    for pt in trajectory:
        pt_np = np.array(pt)
        min_d = float('inf')
        for center, radius in insts:
            d = float(np.linalg.norm(pt_np - center)) - radius
            if d < min_d: min_d = d
        dists.append(max(0.0, min_d))

    return {
        "min_clear": round(min(dists), 3),
        "zone2_rate": round(sum(1 for d in dists if d < ZONE2_DANGER) / len(dists), 4),
    }


def build_all_insts(episodes):
    """从所有 episode 提取场景中全部仪器 (center, radius) 用于安全测量。"""
    insts = {}
    for ep in episodes:
        g = ep["goals"][0]
        cat = g.get("object_category", ep.get("object_category", "?"))
        if cat in insts: continue
        gx = np.array([g["position"][0], g["position"][2]], dtype=np.float64)
        f = extract_facing(ep)
        fh, ap, hw = INSTRUMENT_PARAMS.get(cat, (0.5, 0.7, 0.5))
        fn = f / (np.linalg.norm(f) or 1.0)
        insts[cat] = (gx - fn * (fh + ap), max(fh, hw))
    return [(c, r) for c, r in insts.values()]


# ═══════════════════════════════════════════════════════════════════════════
# Run original scripts with safety tracking
# ═══════════════════════════════════════════════════════════════════════════

MULTI_SCENE = Path("/root/gpufree-data/multi_scene")
sys.path.insert(0, str(MULTI_SCENE))
sys.path.insert(0, str(Path("/root/gpufree-data/pipeline")))


def run_random_with_safety(dataset, split, max_eps=None):
    """Use biovln_eval RandomPolicy (trivial, no patching needed)."""
    from biovln_eval import evaluate

    class Wrapper:
        def __init__(self):
            self._rng = np.random.RandomState()
        def reset(self, obs, info):
            self._rng = np.random.RandomState(int(info.get("episode_id", 0)))
        def act(self, obs):
            return self._rng.choice([1, 2, 3])  # never STOP

    cfg = REGISTRY[dataset]
    result = evaluate(
        Wrapper(),
        split=split,
        scenes_root=cfg["scenes"],
        splits_dir=cfg["splits"],
        max_episodes=max_eps,
        max_steps=500,
        verbose=True,
    )
    return aggregate_biovln(result, "random", dataset, split)


def aggregate_biovln(result, method, dataset, split):
    """Convert biovln_eval output to standard safety metrics format."""
    ss = result.get("safety_summary", {})
    return {
        "method": method, "dataset": dataset, "split": split,
        "total": result["total"], "success": result["success"],
        "SR": result["SR"], "SPL": result["SPL"],
        "safety": {
            "avg_min_clearance": ss.get("avg_min_clearance", 0),
            "avg_zone2_violation_rate": ss.get("zone2_violation_rate", 0),
        },
        "per_episode": result.get("per_episode", []),
    }


def run_oracle_with_safety(dataset, split, max_eps=None):
    """Run ORIGINAL run_baselines.py Oracle with sim.step monkey-patched."""
    install_hook()

    import run_baselines
    orig_output = run_baselines.OUTPUT_ROOT
    orig_splits = run_baselines.SPLITS_DIR

    cfg = REGISTRY[dataset]
    run_baselines.OUTPUT_ROOT = Path(cfg["scenes"])
    run_baselines.SPLITS_DIR = Path(cfg["splits"])

    # Fix: remove PBR lighting
    _orig_get_ob = run_baselines.SimFactory.get
    def _patched_get_ob(self, scene_name):
        if self._current_scene == scene_name and self._sim is not None:
            self._sim.get_agent(0)
            return self._sim, self._sim.pathfinder
        if self._sim is not None:
            self._sim.close()
        scene_dir = run_baselines.OUTPUT_ROOT / scene_name
        glb = str(scene_dir / f"{scene_name}.glb")
        navmesh = str(scene_dir / f"{scene_name}.navmesh")
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = glb; sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0; sim_cfg.allow_sliding = False
        sim_cfg.load_semantic_mesh = False
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 0.88; agent_cfg.radius = 0.18
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        if os.path.exists(navmesh): sim.pathfinder.load_nav_mesh(navmesh)
        self._sim = sim; self._current_scene = scene_name
        return sim, sim.pathfinder
    run_baselines.SimFactory.get = _patched_get_ob

    ep_file = run_baselines.SPLITS_DIR / f"episodes_{split}.json.gz"
    with gzip.open(ep_file, "rt") as f:
        episodes = json.load(f)["episodes"]
    if max_eps:
        episodes = episodes[:max_eps]
    all_insts = build_all_insts(episodes)

    by_scene = defaultdict(list)
    for i, ep in enumerate(episodes):
        by_scene[ep.get("scene_name", "?")].append((i, ep))

    sim_factory = run_baselines.SimFactory()
    results = []
    total = len(episodes)

    print(f"\n{'='*60}")
    print(f"  ORACLE + SAFETY on [{dataset}] [{split.upper()}] — {total} eps")
    print(f"{'='*60}\n")

    global_idx = 0
    for scene_name, eps_in_scene in sorted(by_scene.items()):
        print(f"  Scene: {scene_name} ({len(eps_in_scene)} eps)")
        for _, ep in eps_in_scene:
            ep_id = str(ep.get("episode_id", global_idx))
            target_cat = ep.get("object_category", "?")
            goal = ep["goals"][0]
            goal_xy = np.array([goal["position"][0], goal["position"][2]])
            facing = extract_facing(ep)

            start_recording(ep_id)
            result = run_baselines.run_oracle_episode(ep, global_idx, sim_factory)
            stop_recording()

            safety = compute_safety(get_trajectory(ep_id), goal_xy, facing, target_cat, all_insts)
            result["safety_min_clear"] = safety["min_clear"]
            result["safety_zone2_rate"] = safety["zone2_rate"]
            results.append(result)

            status = "✓" if result.get("success") else "✗"
            print(f"    [{global_idx+1}/{total}] {status} {target_cat:<20s} "
                  f"steps={result.get('steps', 0):3d} "
                  f"clr={safety['min_clear']:.2f}m "
                  f"z2={safety['zone2_rate']:.1%}", flush=True)
            global_idx += 1

    sim_factory.close()

    # Restore
    run_baselines.OUTPUT_ROOT = orig_output
    run_baselines.SPLITS_DIR = orig_splits
    run_baselines.SimFactory.get = _orig_get_ob
    uninstall_hook()

    return aggregate(results, "oracle", dataset, split)


def run_frontier_with_safety(dataset, split, max_eps=None):
    """Run ORIGINAL run_frontier.py with sim.step monkey-patched + lighting fix."""
    install_hook()

    import run_frontier

    # --- Fix: remove PBR lighting that crashes EGL ---
    _orig_get = run_frontier.SimFactory.get
    def _patched_get(self, scene_name):
        if self._current_scene == scene_name and self._sim is not None:
            self._sim.get_agent(0)
            return self._sim, self._sim.pathfinder
        if self._sim is not None:
            self._sim.close()
        scene_dir = run_frontier.OUTPUT_ROOT / scene_name
        glb = str(scene_dir / f"{scene_name}.glb")
        navmesh = str(scene_dir / f"{scene_name}.navmesh")
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = glb; sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0; sim_cfg.allow_sliding = False
        sim_cfg.load_semantic_mesh = False
        # SKIP lighting config — crashes EGL
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 0.88; agent_cfg.radius = 0.18
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        sim.pathfinder.load_nav_mesh(navmesh)
        self._sim = sim; self._current_scene = scene_name
        return sim, sim.pathfinder
    run_frontier.SimFactory.get = _patched_get

    # --- Override paths ---
    orig_output = run_frontier.OUTPUT_ROOT
    orig_splits = run_frontier.SPLITS_DIR
    cfg = REGISTRY[dataset]
    run_frontier.OUTPUT_ROOT = Path(cfg["scenes"])
    run_frontier.SPLITS_DIR = Path(cfg["splits"])

    ep_file = run_frontier.SPLITS_DIR / f"episodes_{split}.json.gz"
    with gzip.open(ep_file, "rt") as f:
        episodes = json.load(f)["episodes"]
    if max_eps:
        episodes = episodes[:max_eps]
    all_insts = build_all_insts(episodes)

    by_scene = defaultdict(list)
    for i, ep in enumerate(episodes):
        by_scene[ep.get("scene_name", "?")].append((i, ep))

    sim_factory = run_frontier.SimFactory()
    results = []
    total = len(episodes)

    print(f"\n{'='*60}")
    print(f"  FRONTIER + SAFETY on [{dataset}] [{split.upper()}] — {total} eps")
    print(f"{'='*60}\n")

    global_idx = 0
    for scene_name, eps_in_scene in sorted(by_scene.items()):
        print(f"  Scene: {scene_name} ({len(eps_in_scene)} eps)")
        for _, ep in eps_in_scene:
            ep_id = str(ep.get("episode_id", global_idx))
            target_cat = ep.get("object_category", "?")
            goal = ep["goals"][0]
            goal_xy = np.array([goal["position"][0], goal["position"][2]])
            facing = extract_facing(ep)

            start_recording(ep_id)
            result = run_frontier.run_episode(ep, global_idx, sim_factory)
            stop_recording()

            safety = compute_safety(get_trajectory(ep_id), goal_xy, facing, target_cat, all_insts)
            result["safety_min_clear"] = safety["min_clear"]
            result["safety_zone2_rate"] = safety["zone2_rate"]
            results.append(result)

            status = "✓" if result.get("success") else "✗"
            print(f"    [{global_idx+1}/{total}] {status} {target_cat:<20s} "
                  f"clr={safety['min_clear']:.2f}m "
                  f"z2={safety['zone2_rate']:.1%}", flush=True)
            global_idx += 1

    sim_factory.close()

    run_frontier.OUTPUT_ROOT = orig_output
    run_frontier.SPLITS_DIR = orig_splits
    run_frontier.SimFactory.get = _orig_get
    uninstall_hook()

    return aggregate(results, "frontier", dataset, split)


def run_threezone_with_safety(dataset, split, max_eps=None):
    """Run ORIGINAL run_three_zone.py with sim.step monkey-patched."""
    install_hook()

    import run_three_zone
    orig_output = run_three_zone.OUTPUT_ROOT
    orig_splits = run_three_zone.SPLITS_DIR

    # Fix: remove PBR lighting that crashes EGL
    _orig_get_3z = run_three_zone.SimFactory.get
    def _patched_get_3z(self, scene_name):
        if self._current_scene == scene_name and self._sim is not None:
            self._sim.get_agent(0)
            return self._sim, self._sim.pathfinder
        if self._sim is not None:
            self._sim.close()
        scene_dir = run_three_zone.OUTPUT_ROOT / scene_name
        glb = str(scene_dir / f"{scene_name}.glb")
        navmesh = str(scene_dir / f"{scene_name}.navmesh")
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = glb; sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0; sim_cfg.allow_sliding = False
        sim_cfg.load_semantic_mesh = False
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 0.88; agent_cfg.radius = 0.18
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        sim.pathfinder.load_nav_mesh(navmesh)
        self._sim = sim; self._current_scene = scene_name
        return sim, sim.pathfinder
    run_three_zone.SimFactory.get = _patched_get_3z

    cfg = REGISTRY[dataset]
    run_three_zone.OUTPUT_ROOT = Path(cfg["scenes"])
    run_three_zone.SPLITS_DIR = Path(cfg["splits"])

    ep_file = run_three_zone.SPLITS_DIR / f"episodes_{split}.json.gz"
    with gzip.open(ep_file, "rt") as f:
        episodes = json.load(f)["episodes"]
    if max_eps:
        episodes = episodes[:max_eps]
    all_insts = build_all_insts(episodes)

    by_scene = defaultdict(list)
    for i, ep in enumerate(episodes):
        by_scene[ep.get("scene_name", "?")].append((i, ep))

    sim_factory = SimFactoryForThreeZone(cfg["scenes"])
    results = []
    total = len(episodes)

    print(f"\n{'='*60}")
    print(f"  3-ZONE + SAFETY on [{dataset}] [{split.upper()}] — {total} eps")
    print(f"{'='*60}\n")

    global_idx = 0
    for scene in sorted(by_scene):
        eps_in_scene = by_scene[scene]
        print(f"  Scene: {scene} ({len(eps_in_scene)} eps)")
        for _, ep in eps_in_scene:
            ep_id = str(ep.get("episode_id", global_idx))
            target_cat = ep.get("object_category", "?")
            goal = ep["goals"][0]
            goal_xy = np.array([goal["position"][0], goal["position"][2]])
            facing = extract_facing(ep)

            start_recording(ep_id)
            result = run_three_zone.run_episode(ep, global_idx, sim_factory, use_oracle=True)
            stop_recording()

            safety = compute_safety(get_trajectory(ep_id), goal_xy, facing, target_cat, all_insts)
            result["safety_min_clear"] = safety["min_clear"]
            result["safety_zone2_rate"] = safety["zone2_rate"]
            results.append(result)

            status = "✓" if result.get("success") else "✗"
            print(f"    [{global_idx+1}/{total}] {status} {target_cat:<20s} "
                  f"clr={safety['min_clear']:.2f}m "
                  f"z2={safety['zone2_rate']:.1%}", flush=True)
            global_idx += 1

    sim_factory.close()
    run_three_zone.OUTPUT_ROOT = orig_output
    run_three_zone.SPLITS_DIR = orig_splits
    run_three_zone.SimFactory.get = _orig_get_3z
    uninstall_hook()

    return aggregate(results, "three_zone", dataset, split)


class SimFactoryForThreeZone:
    """Same as run_three_zone.SimFactory but with configurable root."""
    def __init__(self, scenes_root):
        self._root = Path(scenes_root)
        self._sim = None
        self._name = None

    def get(self, scene_name):
        if self._name == scene_name and self._sim:
            self._sim.get_agent(0)
            return self._sim, self._sim.pathfinder
        self.close()
        d = self._root / scene_name
        glb = str(d / f"{scene_name}.glb")
        nav = str(d / f"{scene_name}.navmesh")
        sc = habitat_sim.SimulatorConfiguration()
        sc.scene_id = glb; sc.enable_physics = False; sc.gpu_device_id = 0
        sc.allow_sliding = False; sc.load_semantic_mesh = False
        ac = habitat_sim.agent.AgentConfiguration()
        ac.height = 0.88; ac.radius = 0.18
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [ac]))
        if os.path.exists(nav):
            sim.pathfinder.load_nav_mesh(nav)
        self._sim = sim; self._name = scene_name
        return sim, sim.pathfinder

    def close(self):
        if self._sim:
            self._sim.close(); self._sim = None


# ═══════════════════════════════════════════════════════════════════════════

def aggregate(results, method, dataset, split):
    n = len(results)
    n_ok = sum(1 for r in results if r.get("success"))
    sr = n_ok / n if n > 0 else 0
    spl = float(np.mean([r.get("SPL", 0) for r in results]))
    clears = [r.get("safety_min_clear", -1) for r in results if r.get("safety_min_clear", -1) >= 0]
    z2rs = [r.get("safety_zone2_rate", -1) for r in results if r.get("safety_zone2_rate", -1) >= 0]
    return {
        "method": method, "dataset": dataset, "split": split,
        "total": n, "success": n_ok,
        "SR": round(sr, 4), "SPL": round(spl, 4),
        "safety": {
            "avg_min_clearance": round(np.mean(clears), 3) if clears else 0,
            "avg_zone2_violation_rate": round(np.mean(z2rs), 4) if z2rs else 0,
        },
        "per_episode": results,
    }


def save(metrics, dataset):
    out = Path(REGISTRY[dataset]["results"]) / \
         f"eval_{metrics['method']}_{dataset}_{metrics['split']}_safety.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {out.name}")
    return out


RUNNERS = {
    "random": run_random_with_safety,
    "oracle": run_oracle_with_safety,
    "frontier": run_frontier_with_safety,
    "three_zone": run_threezone_with_safety,
}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lsat_target",
                    choices=["lsat_target", "two_room", "multi_scene"])
    ap.add_argument("--method", default="random",
                    choices=["random", "oracle", "frontier", "three_zone"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    if args.all:
        all_metrics = []
        for ds in ["lsat_target", "two_room"]:
            for method in ["random", "oracle", "frontier", "three_zone"]:
                runner = RUNNERS[method]
                metrics = runner(ds, args.split, max_eps=args.max)
                save(metrics, ds)
                all_metrics.append(metrics)

        print(f"\n{'='*72}")
        print(f"  SAFETY METRICS SUMMARY")
        print(f"  {'Method':<12s} {'Dataset':<14s} {'SR':>6s} {'SPL':>6s} {'MinClear':>8s} {'Zone2%':>8s}")
        print(f"  {'-'*60}")
        for m in all_metrics:
            s = m["safety"]
            print(f"  {m['method']:<12s} {m['dataset']:<14s} {m['SR']:>5.1%} {m['SPL']:>6.3f} {s['avg_min_clearance']:>7.3f}m {s['avg_zone2_violation_rate']:>7.1%}")
        print(f"{'='*72}")
    else:
        runner = RUNNERS[args.method]
        metrics = runner(args.dataset, args.split, max_eps=args.max)
        save(metrics, args.dataset)
        s = metrics["safety"]
        print(f"\n  SR={metrics['SR']:.1%} SPL={metrics['SPL']:.3f} "
              f"minClear={s['avg_min_clearance']:.3f}m zone2={s['avg_zone2_violation_rate']:.1%}")
