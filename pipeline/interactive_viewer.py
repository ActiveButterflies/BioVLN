#!/usr/bin/env python3
"""
Interactive Lab Scene Viewer — keyboard-controlled agent in habitat-sim.

Displays RGB, depth, and semantic sensor observations in an OpenCV window.
Agent movement is constrained to the navmesh.

Usage:
    DISPLAY=:1 conda run -n habitat python3 pipeline/interactive_viewer.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector


# ─── Default paths ────────────────────────────────────────────────────────────

DEFAULT_SCENE = "/root/gpufree-data/test_instructment_out/biovln_lab_v4.glb"
DEFAULT_NAVMESH = "/root/gpufree-data/test_instructment_out/biovln_lab_v4.navmesh"

RESOLUTION_W, RESOLUTION_H = 1280, 720   # 16:9 (W,H for window/sizing)
SENSOR_H, SENSOR_W = RESOLUTION_H, RESOLUTION_W  # habitat-sim uses [H,W] in CameraSensorSpec
FOV = 90                   # degrees, hfov=90 gives vfov≈59° at 16:9
SENSOR_HEIGHT = 1.5       # meters above ground
MOVE_SPEED = 0.15          # meters per frame (at 30fps)
ROTATE_SPEED = 3.0         # degrees per frame


# ─── Quaternion helpers ───────────────────────────────────────────────────────

def yaw_quaternion(degrees: float) -> np.ndarray:
    """Quaternion for yaw rotation around up-axis, as [x, y, z, w]."""
    q = quat_from_angle_axis(np.radians(degrees), np.array([0, 1, 0]))
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)


def quat_to_yaw(q) -> float:
    """Extract yaw angle (degrees) from quaternion (numpy array or mn.Quaternion)."""
    # Handle magnum Quaternion objects
    if hasattr(q, 'x'):
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    else:
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
    return np.degrees(np.arctan2(siny_cosp, cosy_cosp))


# ─── Simulator setup ──────────────────────────────────────────────────────────

def create_simulator(scene_glb: str, navmesh_path: str):
    """Create habitat-sim simulator with RGB, depth, and semantic sensors."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_glb
    sim_cfg.enable_physics = False
    sim_cfg.override_scene_light_defaults = False

    sensors = {}
    for uuid, stype in [
        ("color_sensor", habitat_sim.SensorType.COLOR),
        ("depth_sensor", habitat_sim.SensorType.DEPTH),
        ("semantic_sensor", habitat_sim.SensorType.SEMANTIC),
    ]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = stype
        spec.resolution = [SENSOR_H, SENSOR_W]  # habitat-sim uses [height, width] order
        spec.position = [0.0, SENSOR_HEIGHT, 0.0]
        spec.hfov = FOV
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensors[uuid] = spec

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = list(sensors.values())

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    print(f"  Simulator OK | Scene: {Path(scene_glb).name}")

    # NavMesh
    pf = sim.pathfinder
    if Path(navmesh_path).exists():
        pf.load_nav_mesh(navmesh_path)
        print(f"  NavMesh loaded | islands: {pf.num_islands}, navigable: {pf.is_loaded}")
    else:
        print(f"  ⚠  NavMesh not found: {navmesh_path}")

    return sim


# ─── Rendering helpers ────────────────────────────────────────────────────────

def render_observations(sim) -> dict:
    """Get all sensor observations from the simulator."""
    return sim.get_sensor_observations()


def depth_to_color(depth: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
    """Convert single-channel depth to RGB colormap."""
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    depth_clip = np.clip(depth, 0, max_depth)
    normalized = depth_clip / max_depth
    # Jet-inspired colormap
    r = np.clip((normalized - 0.5) * 3.0, 0, 1)
    g = np.clip(1.0 - np.abs(normalized - 0.5) * 2.5, 0, 1)
    b = np.clip(1.0 - normalized * 1.5, 0, 1)
    return (np.stack([b, g, r], axis=-1) * 255).astype(np.uint8)


def semantic_to_color(semantic: np.ndarray) -> np.ndarray:
    """Convert semantic IDs to colored image."""
    unique_ids = np.unique(semantic)
    color_map = {}
    for uid in unique_ids:
        if uid == 0:
            color_map[uid] = [0, 0, 0]
        else:
            color_map[uid] = [
                (uid * 67 + 13) % 256,
                (uid * 113 + 47) % 256,
                (uid * 173 + 89) % 256,
            ]
    h, w = semantic.shape[:2] if semantic.ndim == 3 else semantic.shape
    color_sem = np.zeros((h, w, 3), dtype=np.uint8)
    for uid, clr in color_map.items():
        mask = (semantic.squeeze() == uid)
        color_sem[mask] = clr
    return color_sem


def prepare_display_frame(obs: dict, view_mode: str) -> np.ndarray:
    """Build the display frame from sensor observations."""
    rgb = obs["color_sensor"]         # (H, W, 4) RGBA
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    depth = obs["depth_sensor"]       # (H, W, 1)
    semantic = obs["semantic_sensor"] # (H, W)

    if view_mode == "rgb":
        return rgb.copy()
    elif view_mode == "depth":
        return depth_to_color(depth)
    elif view_mode == "semantic":
        return semantic_to_color(semantic)
    elif view_mode == "all":
        h, w = rgb.shape[:2]
        h2, w2 = h // 2, w // 2
        depth_rgb = depth_to_color(depth)
        sem_rgb = semantic_to_color(semantic)
        # 2x2 grid: RGB | Depth / Semantic | ?
        top = np.hstack([rgb[:, :w2], depth_rgb[:, :w2]])
        # Resize semantic to fit
        sem_resized = cv2.resize(sem_rgb, (w2, h2))
        bottom_left = cv2.resize(rgb, (w2, h2))
        # Show RGB again in bottom-left for reference
        bottom = np.hstack([bottom_left, sem_resized])
        return np.vstack([top, bottom])
    else:
        return rgb.copy()


def draw_hud(frame: np.ndarray, sim, view_mode: str, fps: float,
             nav_snap_enabled: bool):
    """Draw HUD text overlay on the frame."""
    agent = sim.get_agent(0)
    state = agent.state
    pos = state.position
    yaw = quat_to_yaw(state.rotation)

    lines = [
        f"View: {view_mode} | FPS: {fps:.0f} | NavSnap: {'ON' if nav_snap_enabled else 'OFF'}",
        f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  Yaw: {yaw:.1f} deg",
        f"5=Entrance 6=NW 7=SE 8=SW | WASD=Move Q/E=Rotate 1-4=Views R=Reset M=SnapNav ESC=Quit",
    ]

    # Semi-transparent overlay at bottom
    overlay = frame.copy()
    h = frame.shape[0]
    bar_h = 70
    cv2.rectangle(overlay, (0, h - bar_h), (frame.shape[1], h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    for i, line in enumerate(lines):
        y = h - bar_h + 18 + i * 20
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)


# ─── Agent movement ────────────────────────────────────────────────────────────

def move_agent(sim, forward: float, right: float, nav_snap_enabled: bool):
    """Move agent forward/backward and left/right relative to heading."""
    agent = sim.get_agent(0)
    state = agent.state
    rot = state.rotation
    forward_vec = quat_rotate_vector(rot, np.array([0, 0, -1.0]))
    right_vec = quat_rotate_vector(rot, np.array([1.0, 0, 0.0]))

    new_pos = (
        np.array(state.position)
        + forward_vec * forward * MOVE_SPEED
        + right_vec * right * MOVE_SPEED
    )

    # Snap to navmesh
    if nav_snap_enabled:
        pf = sim.pathfinder
        if pf.is_loaded:
            snapped = pf.snap_point(new_pos)
            if snapped is not None and np.isfinite(snapped).all():
                new_pos = snapped

    state.position = new_pos
    agent.set_state(state)


def rotate_agent(sim, delta_yaw: float):
    """Rotate agent by delta_yaw degrees."""
    agent = sim.get_agent(0)
    state = agent.state
    current_yaw = quat_to_yaw(state.rotation)
    new_yaw = current_yaw + delta_yaw
    state.rotation = yaw_quaternion(new_yaw)
    agent.set_state(state)


def reset_agent(sim):
    """Reset agent to a navigable position on the floor."""
    agent = sim.get_agent(0)
    pf = sim.pathfinder

    # Get random navigable point, but filter out ceiling/top-of-furniture islands
    # NavMesh islands above Y=2.0 are ceiling or furniture-top, not walkable floor
    pos = None
    if pf.is_loaded:
        for _ in range(100):
            rp = pf.get_random_navigable_point()
            if rp is not None and rp[1] < 2.0:  # floor level
                pos = np.array(rp, dtype=np.float32)
                break

    if pos is None:
        # Fallback: snap to open area near center
        pos = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        if pf.is_loaded:
            snapped = pf.snap_point(pos)
            if snapped is not None and np.isfinite(snapped).all():
                pos = snapped

    state = agent.state
    state.position = pos
    state.rotation = yaw_quaternion(0)
    agent.set_state(state)
    print(f"  Agent reset to: {pos}")


def teleport_agent(sim, position: list, rotation_deg: float = 0):
    """Teleport agent to a specific position and rotation.
    Forces floor-level Y to avoid ceiling-island snaps."""
    agent = sim.get_agent(0)
    pf = sim.pathfinder
    # Start with a floor-level guess (Z=0.5 in habitat Y-up = near floor)
    pos = np.array([position[0], 0.15, position[2]], dtype=np.float32)
    if pf.is_loaded:
        snapped = pf.snap_point(pos)
        if snapped is not None and np.isfinite(snapped).all() and snapped[1] < 2.0:
            pos = snapped
    state = agent.state
    state.position = pos
    state.rotation = yaw_quaternion(rotation_deg)
    agent.set_state(state)
    print(f"  Teleported to: {pos}  yaw={rotation_deg}")


# ─── Main interactive loop ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive Lab Scene Viewer")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="Path to scene GLB")
    parser.add_argument("--navmesh", default=DEFAULT_NAVMESH, help="Path to navmesh")
    parser.add_argument("--start-pos", nargs=3, type=float, default=None,
                        help="Start position x y z")
    parser.add_argument("--window-w", type=int, default=1280, help="Window width")
    parser.add_argument("--window-h", type=int, default=720, help="Window height")
    args = parser.parse_args()

    # ── Create simulator ──
    print("=" * 60)
    print("Interactive Lab Scene Viewer")
    print("=" * 60)
    print("Creating simulator...")
    sim = create_simulator(args.scene, args.navmesh)

    agent = sim.get_agent(0)
    print(f"  Sensors: {list(agent._sensors.keys())}")

    # ── Set initial position ──
    if args.start_pos:
        teleport_agent(sim, args.start_pos)
    else:
        reset_agent(sim)

    # ── OpenCV window ──
    window_name = "BioVLN Lab Viewer — Interactive Agent Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_w, args.window_h)

    view_mode = "rgb"
    nav_snap_enabled = True
    running = True
    frame_count = 0
    last_time = time.time()
    fps = 0.0

    print("\nControls:")
    print("  WASD      — Move forward/back/strafe")
    print("  Q/E       — Rotate left/right (Q=left, E=right)")
    print("  1/2/3/4   — View: RGB / Depth / Semantic / All")
    print("  5/6/7/8   — Teleport: South Entrance / NW Area / SE Area / SW Corner")
    print("  R         — Reset to navigable point")
    print("  M         — Toggle navmesh snap")
    print("  ESC       — Quit")
    print()

    while running:
        # ── Render ──
        obs = render_observations(sim)
        frame = prepare_display_frame(obs, view_mode)

        # Resize to window
        display = cv2.resize(frame, (args.window_w, args.window_h))

        # Draw HUD
        draw_hud(display, sim, view_mode, fps, nav_snap_enabled)

        cv2.imshow(window_name, display)

        # ── Keyboard input (30ms = ~33fps) ──
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            running = False

        # View mode
        elif key == ord('1'):
            view_mode = "rgb"
        elif key == ord('2'):
            view_mode = "depth"
        elif key == ord('3'):
            view_mode = "semantic"
        elif key == ord('4'):
            view_mode = "all"

        # Teleport presets (v4 room: 8m x 6m)
        elif key == ord('5'):
            teleport_agent(sim, [0.0, 0.5, -2.5], 0)
            print("  → South area")
        elif key == ord('6'):
            teleport_agent(sim, [-3.0, 0.5, 1.5], 60)
            print("  → NW area (near lab_table_1)")
        elif key == ord('7'):
            teleport_agent(sim, [3.0, 0.5, -1.0], -90)
            print("  → SE area (near cabinet)")
        elif key == ord('8'):
            teleport_agent(sim, [-3.0, 0.5, -2.0], 30)
            print("  → SW corner")

        # Reset
        elif key == ord('r'):
            reset_agent(sim)

        # Toggle navmesh snap
        elif key == ord('m'):
            nav_snap_enabled = not nav_snap_enabled
            print(f"  NavMesh snap: {'ON' if nav_snap_enabled else 'OFF'}")

        # ── Movement (continuous via polling) ──
        # Use cv2.waitKey with a short delay, then poll the key state
        # For held keys, we use a workaround: check repeatedly
        # Actually cv2.waitKey returns -1 when no key pressed, but for held keys
        # it depends on the OS key repeat settings.
        # Better approach: use the key code we just got to set velocity,
        # and maintain momentum between frames.
        forward = 0.0
        right = 0.0
        rotate_delta = 0.0

        # Check the specific key we just received (handles repeated keys)
        # This won't handle simultaneous WASD well, so we supplement with
        # getWindowProperty to check if any key is down.
        if key == ord('w'):
            forward += 1.0
        elif key == ord('s'):
            forward -= 1.0
        elif key == ord('a'):
            right -= 1.0
        elif key == ord('d'):
            right += 1.0
        elif key == ord('q'):
            rotate_delta += ROTATE_SPEED
        elif key == ord('e'):
            rotate_delta -= ROTATE_SPEED

        if forward != 0.0 or right != 0.0:
            move_agent(sim, forward, right, nav_snap_enabled)

        if rotate_delta != 0.0:
            rotate_agent(sim, rotate_delta)

        # ── FPS counter ──
        frame_count += 1
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - last_time + 0.001)
            last_time = now

    # ── Cleanup ──
    cv2.destroyAllWindows()
    sim.close()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
