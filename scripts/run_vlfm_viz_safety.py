#!/usr/bin/env python3
"""Run VLFM on one episode, record trajectory, render top-down with Zone 2 overlay."""

import sys, os, json, gzip, time
import numpy as np
from pathlib import Path

for k in ["ALL_PROXY","HTTP_PROXY","HTTPS_PROXY","all_proxy","http_proxy","https_proxy"]:
    os.environ.pop(k, None)
if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import habitat_sim, magnum as mn
import quaternion as nq

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

GLB = "/root/gpufree-data/lsat_test/lab_single_room_target/lab_single_room_target.glb"
NAVMESH = "/root/gpufree-data/lsat_test/lab_single_room_target/lab_single_room_target.navmesh"

# ─── Setup: import VLFM, monkey-patch step, select episode ──────────────────

PIPE = Path("/root/gpufree-data/pipeline")
sys.path.insert(0, str(PIPE))
import run_vlfm_v4 as vlfm

vlfm.GLB_PATH = GLB
vlfm.NAVMESH_PATH = NAVMESH

# Fix PBR lighting
_orig_create = vlfm.create_sim
def _safe_create():
    from habitat_sim import agent as am
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = GLB; sc.enable_physics = False; sc.gpu_device_id = 0; sc.allow_sliding = False
    specs = []
    for n, t in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = n; s.sensor_type = t; s.resolution = [vlfm.IMAGE_H, vlfm.IMAGE_W]
        s.position = mn.Vector3(0, vlfm.CAM_HEIGHT, 0); s.hfov = vlfm.HFOV
        s.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        specs.append(s)
    ac = am.AgentConfiguration(); ac.height = vlfm.CAM_HEIGHT; ac.radius = 0.18
    ac.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [ac]))
    if os.path.exists(NAVMESH): sim.pathfinder.load_nav_mesh(NAVMESH)
    return sim
vlfm.create_sim = _safe_create

# Monkey-patch step to record trajectory
_ORIG_STEP = habitat_sim.Simulator.step
_TRAJ = []
def _hooked(self, action, *a, **kw):
    _ORIG_STEP(self, action, *a, **kw)
    try:
        st = self.get_agent(0).get_state()
        _TRAJ.append((float(st.position[0]), float(st.position[2])))
    except: pass
habitat_sim.Simulator.step = _hooked

# Load episode 14 (lab_table, z2=100%)
with gzip.open("/root/gpufree-data/lsat_test/splits/episodes_dev.json.gz", "rt") as f:
    ep = json.load(f)["episodes"][14]
print(f"Running VLFM on {ep['object_category']} (ep {ep['episode_id']})...")
print(f"Start: {ep['start_position']}")

_TRAJ.clear()
t0 = time.time()
result = vlfm.run_episode(ep, 14, save_video=False, blind=True, use_vlm_vmap=True)
elapsed = time.time() - t0
habitat_sim.Simulator.step = _ORIG_STEP

print(f"Result: {'SUCCESS' if result.get('success') else 'FAILURE'} "
      f"steps={result.get('steps',0)} dist={result.get('final_dist',0):.2f}m t={elapsed:.0f}s")
print(f"Trajectory: {len(_TRAJ)} points")

# ─── Compute safety ────────────────────────────────────────────────────────

goal = ep["goals"][0]
goal_xy = np.array([goal["position"][0], goal["position"][2]], dtype=np.float64)

# Instrument face direction from view_points
facing = np.array([0.0, -1.0])
try:
    vp = goal.get("view_points", [])
    if vp and len(vp) > 0:
        r = vp[0].get("agent_state", {}).get("rotation", [])
        if len(r) == 4:
            q = mn.Quaternion(mn.Vector3(float(r[0]), float(r[1]), float(r[2])), float(r[3]))
            f = q.transform_vector(mn.Vector3(0, 0, -1))
            facing = np.array([f[0], f[2]])
except: pass

p = INSTRUMENT_PARAMS.get(ep.get("object_category","?"), (0.5, 0.7, 0.5))
fh, ap, hw = p
fn = facing / (np.linalg.norm(facing) or 1.0)
inst_center = goal_xy - fn * (fh + ap)
inst_radius = max(fh, hw)

# Per-point safety
traj_np = np.array(_TRAJ)
dists_to_center = np.linalg.norm(traj_np - inst_center, axis=1)
clearances = np.maximum(0, dists_to_center - inst_radius)
unsafe_mask = clearances < ZONE2_DANGER

print(f"\nSafety: instrument_center={inst_center.tolist()} radius={inst_radius:.2f}m")
print(f"  Unsafe points: {unsafe_mask.sum()}/{len(clearances)} ({unsafe_mask.sum()/len(clearances)*100:.1f}%)")
print(f"  Min clearance: {clearances.min():.3f}m")

# ─── Render top-down scene ──────────────────────────────────────────────────

# Get scene bounds
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = GLB; cfg.enable_physics = False; cfg.gpu_device_id = 0
sim_tmp = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.agent.AgentConfiguration()]))
bb = sim_tmp.get_active_scene_graph().get_root_node().compute_cumulative_bb()
sim_tmp.close()

# Set up top-down camera with depth sensor
IMG_W = 1200
cx = float(bb.center()[0]); cz = float(bb.center()[2])
scene_w = float(bb.max[0] - bb.min[0]); scene_d = float(bb.max[2] - bb.min[2])
hfov = 120
min_h = max(scene_w, scene_d) * 1.05 / (2 * np.tan(np.deg2rad(hfov / 2)))
cam_y = max(float(bb.max[1]), float(bb.max[1]) - 0.3) + min_h
img_h = max(int(IMG_W * scene_d / scene_w), 600)

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.height = cam_y
rgb_spec = habitat_sim.CameraSensorSpec()
rgb_spec.uuid = "c"; rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_spec.resolution = [img_h, IMG_W]; rgb_spec.hfov = hfov
agent_cfg.sensor_specifications = [rgb_spec]

sim_cfg2 = habitat_sim.SimulatorConfiguration()
sim_cfg2.scene_id = GLB; sim_cfg2.enable_physics = False; sim_cfg2.gpu_device_id = 0

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg2, [agent_cfg]))
cam_pos = mn.Vector3(cx, cam_y, cz)
look_at = mn.Vector3(cx, float(bb.min[1]) + 0.01, cz)
mat = mn.Matrix4.look_at(cam_pos, look_at, mn.Vector3(0, 0, 1))
rq = mn.Quaternion.from_matrix(mat.rotation())
agent = sim.initialize_agent(0)
state = habitat_sim.agent.AgentState()
state.position = cam_pos; state.rotation = [rq.vector.x, rq.vector.y, rq.vector.z, rq.scalar]
agent.set_state(state, reset_sensors=False)
for _ in range(10): sim.get_sensor_observations()
rgb = sim.get_sensor_observations()["c"]
sim.close()

# ─── Compute world→pixel transform ──────────────────────────────────────────

eff_h = cam_y - float(bb.min[1])
vis_w = 2 * eff_h * np.tan(np.deg2rad(hfov / 2))
vfov = 2 * np.arctan(np.tan(np.deg2rad(hfov / 2)) * img_h / IMG_W)
vis_h = 2 * eff_h * np.tan(vfov / 2)

def world_to_px(wx, wz):
    """Convert world (X, Z) to pixel (col, row)."""
    nx = 0.5 - (wx - cx) / vis_w  # X-flip from look_at with up=(0,0,1)
    ny = 0.5 - (wz - cz) / vis_h
    return int(nx * IMG_W), int(ny * img_h)

# ─── Draw ───────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 9))

# Scene image
ax.imshow(rgb)

# Plot trajectory
pxs, pys = [], []
for wx, wz in _TRAJ:
    px, py = world_to_px(wx, wz)
    pxs.append(px); pys.append(py)
pxs = np.array(pxs); pys = np.array(pys)

# Color segments: green=safe, red=unsafe
for i in range(len(pxs) - 1):
    color = '#ef4444' if unsafe_mask[i] else '#22c55e'
    ax.plot(pxs[i:i+2], pys[i:i+2], color=color, linewidth=1.5, alpha=0.9)

# Start marker
ax.scatter(pxs[0], pys[0], c='white', s=120, edgecolors='#3b82f6', linewidths=2, zorder=5)
ax.scatter(pxs[-1], pys[-1], c='#3b82f6', s=120, edgecolors='white', linewidths=2, zorder=5, marker='s')

# Instrument center and Zone 2 boundary
icx, icy = world_to_px(inst_center[0], inst_center[1])
zone2_radius_px = inst_radius / vis_w * IMG_W  # approximate
ax.scatter(icx, icy, c='#ef4444', s=60, marker='X', linewidths=2, zorder=5, edgecolors='white')

# Zone 2 danger circle (instrument surface + ZONE2_DANGER)
danger_radius_px = (inst_radius + ZONE2_DANGER) / vis_w * IMG_W
circle = plt.Circle((icx, icy), danger_radius_px, fill=False, color='#ef4444',
                     linewidth=2, linestyle='--', alpha=0.8, zorder=4)
ax.add_patch(circle)
# Instrument surface circle
inst_circle = plt.Circle((icx, icy), zone2_radius_px, fill=True, color='#ef4444',
                          alpha=0.15, zorder=3, linewidth=1, edgecolor='#ef4444')
ax.add_patch(inst_circle)

# Goal position
gx, gy = world_to_px(goal_xy[0], goal_xy[1])
ax.scatter(gx, gy, c='#fbbf24', s=100, marker='D', linewidths=1.5, edgecolors='white', zorder=5)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#22c55e', lw=2, label='Safe trajectory'),
    Line2D([0], [0], color='#ef4444', lw=2, label='Zone 2 violation (<0.5m)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='#3b82f6',
           markersize=10, label='Start'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#3b82f6', markeredgecolor='white',
           markersize=10, label='End'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#fbbf24', markeredgecolor='white',
           markersize=10, label='Goal'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#ef4444', markeredgecolor='white',
           markersize=10, label='Instrument center'),
    Line2D([0], [0], linestyle='--', color='#ef4444', lw=2, label=f'Zone 2 boundary ({ZONE2_DANGER}m)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

# Title
safety_pct = unsafe_mask.sum() / max(len(unsafe_mask), 1) * 100
target = ep.get("object_category", "?")
success = "SUCCESS" if result.get("success") else "FAILURE"
title_line1 = f'VLFM Trajectory — {target} | {success} | Zone 2 violation: {safety_pct:.0f}%'
title_line2 = f'Steps: {result.get("steps",0)}  Min Clearance: {clearances.min():.3f}m  SR threshold: {inst_radius:.1f}m'
ax.set_title(title_line1 + '\n' + title_line2, fontsize=12)
ax.axis('off')

# Save
out = "/root/lsat/paper/figures_glb/renders/vlfm_safety_trajectory.png"
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {out}")
