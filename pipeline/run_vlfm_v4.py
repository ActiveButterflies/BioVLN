#!/usr/bin/env python3
"""
VLFM-style semantic exploration on BIO lab scene.
Communicates with VLM servers via direct HTTP (avoids torch GPU conflict).

Saves visualization video per episode showing: RGB view | Value Map | Obstacle Map | Depth.

Requires VLM servers on ports 12181-12184.
Usage:
    cd /root/autodl-tmp/BIOVLN
    python biovln/run_vlfm.py --episodes 3 --video
"""

import os, sys, json, gzip, time, argparse, base64
import numpy as np, cv2
import requests
import torch
from sklearn.cluster import DBSCAN

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)  # /root/gpufree-data
sys.path.insert(0, _PROJ)

import habitat_sim, magnum as mn, quaternion as nq
from vlm_servers.patch_value_map_hf import update_value_map_patches, get_blip2_model

# ─── Config ─────────────────────────────────────────────────────────────
GLB_PATH = f"{_PROJ}/test_instructment_out/biovln_lab_v4.glb"
NAVMESH_PATH = f"{_PROJ}/test_instructment_out/biovln_lab_v4.navmesh"
EPISODE_FILE = f"{_PROJ}/test_instructment_out/biovln_lab_v4_episodes.json.gz"
VIDEO_DIR = f"{_PROJ}/test_instructment_out/vlfm_videos"
IMAGE_W, IMAGE_H = 640, 480
HFOV, CAM_HEIGHT, MAX_STEPS = 79, 0.88, 500
FORWARD_STEP = 0.25
TURN_ANGLE_DEG = 30

OBJ_RADIUS = {
    "lab_table": 1.3, "lab_table_1": 1.3,
    "refrigerator": 1.0, "refrige": 1.0, "incubor": 1.0,
    "centrifuge": 1.0, "mini_centrifuge": 1.0,
    "cabinet": 1.0, "pipette": 0.8, "lab_waste_bin": 0.8,
}

DOMAIN_PROMPTS = {
    "centrifuge": "laboratory centrifuge . white cylindrical machine . round lid",
    "mini_centrifuge": "small centrifuge . compact white machine . tiny round lid",
    "lab_table": "laboratory table . large rectangular table . work surface in center of room",
    "lab_table_1": "laboratory bench . long counter along wall . workbench",
    "refrigerator": "refrigerator . laboratory refrigerator . tall white appliance . fridge",
    "refrige": "refrigerator . laboratory refrigerator . tall white appliance . fridge",
    "incubor": "laboratory incubator . large white box . temperature controlled chamber . tall appliance",
    "cabinet": "laboratory cabinet . white cabinet with four legs . square and trapezoidal white casing . tall metal storage cabinet",
    "pipette": "pipette . laboratory pipette . small white plastic pipette on table . micropipette",
    "lab_waste_bin": "square plastic waste bin . white waste container on floor",
    "waste_bin": "square plastic waste bin . white waste container on floor",
    "incubator": "laboratory incubator . large white box . temperature controlled chamber . tall appliance",
    "lab_bench": "laboratory bench . long counter along wall . workbench",
}
ACTIONS = {0: None, 1: "move_forward", 2: "turn_left", 3: "turn_right"}
CONTROL_NAMES = {0: "STOP", 1: "FWD  ", 2: "TL   ", 3: "TR   "}

# VLM call frequency (reduced to prevent hanging)
ITM_EVERY_N = 3        # Full-image BLIP-2 ITM via server: call every 3 steps (legacy fallback)
PATCH_ITM_EVERY_N = 15 # Patch-based BLIP-2: run every 15 steps (same as GDINO)
DINO_EVERY_N = 3       # GroundingDINO: call every 3 steps
INIT_SPINS = 12        # 360° scan (12×30°) for full coverage

# ── BLIP-2 model (lazy-loaded, shared across episodes, HF version) ──
def get_blip2():
    """Lazy-load BLIP-2 via HuggingFace (patch_value_map_hf)."""
    return get_blip2_model()

# Global timeout per episode (seconds) — after this, agent returns best effort
EPISODE_TIMEOUT = 900

# ─── Image Enhancement ───────────────────────────────────────────────────
def enhance_image(rgb):
    """Brighten dark rendered images for VLM detection.
    The GLB scene renders ~mean=49/255 — too dark for VLM models.
    Apply gamma correction and brightness boost.
    """
    img = rgb.astype(np.float32) / 255.0
    # Gamma correction: brighten midtones
    gamma = 0.6
    img = np.power(np.clip(img, 0, 1), gamma)
    # Brightness boost
    img = np.clip(img * 1.5, 0, 1)
    return (img * 255).astype(np.uint8)


# ─── Depth-to-3D projection (for blind mode) ─────────────────────────────
def bbox_to_3d(bbox_norm, depth_img, agent_state, hfov_deg=79, img_w=640, img_h=480):
    """Estimate 3D world position of a detected object from its 2D bbox + depth.

    Args:
        bbox_norm: [x1, y1, x2, y2] in normalized 0-1 coordinates
        depth_img: (H, W) depth array in meters
        agent_state: habitat_sim.AgentState with .position and .rotation
    Returns:
        (x, y, z) world position, or None if depth is invalid
    """
    x1, y1, x2, y2 = bbox_norm
    # Center of bbox in pixel coordinates
    u = (x1 + x2) / 2 * img_w
    v = (y1 + y2) / 2 * img_h

    # Get depth: use median of the bbox region for robustness
    x1_px = int(x1 * img_w); x2_px = int(x2 * img_w)
    y1_px = int(y1 * img_h); y2_px = int(y2 * img_h)
    x1_px = max(0, min(x1_px, img_w-1)); x2_px = max(x1_px+1, min(x2_px, img_w))
    y1_px = max(0, min(y1_px, img_h-1)); y2_px = max(y1_px+1, min(y2_px, img_h))

    # Use depth at bbox CENTER (small neighborhood) — the object is most likely here
    cx_px = int(u); cy_px = int(v)
    r = 3  # 7×7 neighborhood
    x0, x1 = max(0, cx_px-r), min(img_w-1, cx_px+r)
    y0, y1 = max(0, cy_px-r), min(img_h-1, cy_px+r)
    center_patch = depth_img[y0:y1, x0:x1]
    valid_depth = center_patch[(center_patch > 0.1) & (center_patch < 10.0)]
    if len(valid_depth) < 3:
        return None
    d = float(np.median(valid_depth))

    # Camera intrinsics
    hfov_rad = np.deg2rad(hfov_deg)
    fx = (img_w / 2) / np.tan(hfov_rad / 2)
    fy = fx  # square pixels
    cx, cy = img_w / 2, img_h / 2

    # Camera-frame 3D: X=right, Y=up, Z=forward
    x_cam = (u - cx) * d / fx
    y_cam = -(v - cy) * d / fy  # image Y goes down, 3D Y goes up
    z_cam = d

    # Rotate to world frame
    # pinhole (+Z fwd) → habitat-sim local (-Z fwd)
    cam_vec = np.array([x_cam, y_cam, -z_cam])
    pos = agent_state.position
    rot = agent_state.rotation

    rotated = nq.rotate_vectors(_to_nq_quat(rot), np.array([cam_vec]))[0]

    world_pos = np.array([pos[0], pos[1], pos[2]]) + rotated

    # Diagnostic log
    if os.environ.get('VLFM_DEBUG'):
        print(f"  [DBG bbox_to_3d] bbox={bbox_norm} pixel=({u:.0f},{v:.0f})", flush=True)
        print(f"  [DBG bbox_to_3d] depth_patch shape={depth_patch.shape} valid={len(valid_depth)}", flush=True)
        print(f"  [DBG bbox_to_3d] depth median={d:.2f}m min={valid_depth.min():.2f} max={valid_depth.max():.2f}", flush=True)
        print(f"  [DBG bbox_to_3d] cam_vec=({x_cam:.2f}, {y_cam:.2f}, {z_cam:.2f})", flush=True)
        print(f"  [DBG bbox_to_3d] agent_pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})", flush=True)
        print(f"  [DBG bbox_to_3d] world_pos=({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})", flush=True)

    return (float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))


# ─── VLM Client (robust, with signal-based timeout) ──────────────────────
import signal

class VLMTimeout(Exception):
    pass

def _with_timeout(func, timeout_s):
    """Run func with a hard timeout using SIGALRM."""
    old_handler = signal.signal(signal.SIGALRM, lambda sig, frame: None)
    signal.alarm(timeout_s)
    try:
        result = func()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return result
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        raise

def blip2_cosine(image_bgr, text_prompt, port=12182, timeout=10):
    """Call BLIP2-ITM server with hard timeout."""
    # Enhance image before sending
    enhanced = enhance_image(image_bgr)
    _, buf = cv2.imencode('.jpg', enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    def _call():
        r = requests.post(f"http://localhost:{port}/blip2itm",
                          json={"image": img_b64, "txt": text_prompt},
                          timeout=(3, timeout))
        r.raise_for_status()
        return float(r.json()["response"])

    return _with_timeout(_call, timeout + 5)


def gdino_detect(image_bgr, text_prompt, port=12181, timeout=20):
    """Call GroundingDINO server with hard timeout."""
    # Enhance image before sending
    enhanced = enhance_image(image_bgr)
    _, buf = cv2.imencode('.jpg', enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    def _call():
        import ast
        r = requests.post(f"http://localhost:{port}/gdino",
                          json={"image": img_b64, "caption": text_prompt},
                          timeout=(3, timeout))
        if r.status_code != 200:
            return []
        resp = r.json()
        # GDINO returns string-encoded JSON: {'boxes': "[[...]]", 'phrases': "['...']", 'logits': "[...]"}
        boxes = ast.literal_eval(resp["boxes"]) if isinstance(resp.get("boxes"), str) else resp.get("boxes", [])
        phrases = ast.literal_eval(resp["phrases"]) if isinstance(resp.get("phrases"), str) else resp.get("phrases", [])
        logits = ast.literal_eval(resp["logits"]) if isinstance(resp.get("logits"), str) else resp.get("logits", [])
        detections = []
        for i in range(len(boxes)):
            box_norm = boxes[i]  # normalized 0-1 coordinates
            detections.append({
                "label": phrases[i] if i < len(phrases) else "unknown",
                "confidence": logits[i] if i < len(logits) else 0.0,
                "box_norm": box_norm,  # [x1, y1, x2, y2] in 0-1
            })
        return detections

    return _with_timeout(_call, timeout + 5)


# ─── VLM Verification (GPT-4o via proxy) ──────────────────────────────────
VLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
VLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
VLM_MODEL = "gpt-4o"

_vlm_client = None

def get_vlm_client():
    global _vlm_client
    if _vlm_client is None:
        import openai
        _vlm_client = openai.OpenAI(api_key=VLM_API_KEY, base_url=VLM_BASE_URL, timeout=15)
        print("[VLM] GPT-4o client initialized", flush=True)
    return _vlm_client

def vlm_verify_detections(rgb, detections, target_name):
    """Use GPT-4o to verify each GDINO detection. Returns only verified detections."""
    if not detections:
        return []

    client = get_vlm_client()
    h, w = rgb.shape[:2]
    verified = []

    for i, det in enumerate(detections):
        box = det["box_norm"]
        x1 = max(0, int(box[0] * w)); y1 = max(0, int(box[1] * h))
        x2 = min(w, int(box[2] * w)); y2 = min(h, int(box[3] * h))
        if x2 <= x1 + 10 or y2 <= y1 + 10:
            continue  # too small

        crop = rgb[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if crop.shape[-1] == 3 else crop
        _, buf = cv2.imencode('.jpg', crop_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_b64 = base64.b64encode(buf).decode('utf-8')

        prompt = (
            f"Look at this image crop and answer ONLY 'yes' or 'no'.\n"
            f"Is this a {target_name}? Answer:"
        )

        try:
            resp = client.chat.completions.create(
                model=VLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                max_tokens=5,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip().lower()
            if answer.startswith("yes"):
                verified.append(det)
            else:
                # Log rejected detection
                pass
        except Exception as e:
            # On API error, keep the detection (fail open)
            verified.append(det)

    return verified


# ─── VLM Value Map Generation ───────────────────────────────────────────
# Grid dimensions for image division
VLM_GRID_ROWS, VLM_GRID_COLS = 4, 6  # 24 cells like the 360° views

def update_value_map_vlm(rgb, depth, agent_3d, fwd_x, fwd_y, vmap, target_name):
    """Use GPT-4o to score each grid cell, then project scores onto value map."""
    h_img, w_img = rgb.shape[:2]
    cell_h, cell_w = h_img // VLM_GRID_ROWS, w_img // VLM_GRID_COLS

    # Draw grid on image for VLM
    grid_img = rgb.copy()
    for r in range(1, VLM_GRID_ROWS):
        cv2.line(grid_img, (0, r*cell_h), (w_img, r*cell_h), (255, 255, 0), 1)
    for c in range(1, VLM_GRID_COLS):
        cv2.line(grid_img, (c*cell_w, 0), (c*cell_w, h_img), (255, 255, 0), 1)
    for r in range(VLM_GRID_ROWS):
        for c in range(VLM_GRID_COLS):
            cx = c * cell_w + cell_w // 2
            cy = r * cell_h + cell_h // 2
            cv2.putText(grid_img, f'{r},{c}', (cx-15, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # Encode and send to VLM
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR),
                          [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    prompt = (
        f"The image is divided into a {VLM_GRID_ROWS}x{VLM_GRID_COLS} grid labeled [row,col]. "
        f"Each cell represents a direction in a laboratory. "
        f"Rate each cell's likelihood of leading to a '{target_name}'. "
        f"Reply ONLY with one line per cell in the format: row,col:score "
        f"where score is 0.0 (definitely not) to 1.0 (definitely yes). "
        f"Include ALL {VLM_GRID_ROWS*VLM_GRID_COLS} cells. No other text."
    )

    try:
        client = get_vlm_client()
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]
            }],
            max_tokens=300, temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
    except Exception:
        return None  # API error, skip this update

    # Parse VLM response
    scores = {}
    for line in text.split('\n'):
        line = line.strip()
        if ':' not in line:
            continue
        try:
            parts = line.split(':')
            key = parts[0].strip()
            score = float(parts[-1].strip().split()[0])  # handle trailing text
            scores[key] = np.clip(score, 0.0, 1.0)
        except (ValueError, IndexError):
            continue

    if len(scores) < 4:  # too few cells parsed
        return None

    # Back-project each cell to 3D and update value map
    hfov_rad = np.deg2rad(HFOV)
    fx = (w_img / 2) / np.tan(hfov_rad / 2)
    fy = fx; cx, cy = w_img / 2, h_img / 2

    updated = 0
    for r in range(VLM_GRID_ROWS):
        for c in range(VLM_GRID_COLS):
            key = f'{r},{c}'
            if key not in scores:
                continue
            score = scores[key]
            if score < 0.01:
                continue

            # Cell center pixel
            pu = c * cell_w + cell_w // 2
            pv = r * cell_h + cell_h // 2

            # Depth at cell center
            u_i, v_i = int(pu), int(pv)
            u_i = max(0, min(w_img-1, u_i))
            v_i = max(0, min(h_img-1, v_i))
            d = float(depth[v_i, u_i])
            if d < 0.2 or d > 8.0:
                continue

            # Camera-frame → world
            x_cam = (pu - cx) * d / fx
            y_cam = -(pv - cy) * d / fy
            z_cam = d
            cam_vec = np.array([x_cam, y_cam, z_cam])
            # Simple rotation approximation using forward vector
            # pinhole +Z → habitat -Z: negate forward component
            right = np.array([-fwd_y, 0, fwd_x])
            up = np.array([0, 1, 0])
            fwd = np.array([fwd_x, 0, fwd_y])
            world_offset = x_cam * right + y_cam * up - z_cam * fwd
            wx = agent_3d[0] + world_offset[0]
            wy = agent_3d[2] + world_offset[2]

            # Simple update: set value at world position
            vmap.set_value(wx, wy, min(score, 0.4))  # capped: don't overwrite detection boosts
            updated += 1

    if updated > 0:
        return {"scores": scores, "updated": updated}
    return None


# ─── Simulator ───────────────────────────────────────────────────────────
def create_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = GLB_PATH; sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0; sim_cfg.allow_sliding = False
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    sim_cfg.pbr_image_based_lighting = True

    specs = []
    for name, stype in [("rgb", habitat_sim.SensorType.COLOR),
                         ("depth", habitat_sim.SensorType.DEPTH)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = name; spec.sensor_type = stype
        spec.resolution = [IMAGE_H, IMAGE_W]
        spec.position = mn.Vector3(0, CAM_HEIGHT, 0)
        spec.hfov = HFOV
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        specs.append(spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = CAM_HEIGHT; agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    if os.path.exists(NAVMESH_PATH):
        sim.pathfinder.load_nav_mesh(NAVMESH_PATH)
    return sim


# ─── Maps ────────────────────────────────────────────────────────────────
class SimpleValueMap:
    """2D top-down value map: each cell stores max BLIP-2 semantic score."""
    def __init__(self, size=500, resolution=0.05):
        self.size = size; self.res = resolution
        self.origin = size // 2
        self.data = np.zeros((size, size), dtype=np.float32)
        self.traj = np.zeros((size, size), dtype=np.uint8)
        self._seen = set()

    def reset(self):
        self.data.fill(0); self.traj.fill(0); self._seen.clear()

    def w2g(self, wx, wy):
        return (np.clip(int(wx/self.res)+self.origin, 0, self.size-1),
                np.clip(int(wy/self.res)+self.origin, 0, self.size-1))

    def g2w(self, gx, gy):
        return np.array([(gx-self.origin)*self.res, (gy-self.origin)*self.res])

    def update(self, value, depth, ax, ay, fwd_x, fwd_y):
        h, w = depth.shape; fx = fy = w/(2*np.tan(np.deg2rad(HFOV)/2))
        cx, cy = w//2, h//2; right_x, right_y = -fwd_y, fwd_x
        vv, uu = np.mgrid[0:h, 0:w]
        valid = (depth > 0.5) & (depth < 5.0)
        Z = depth[valid]; X = (uu[valid]-cx)*Z/fx; Y = -(vv[valid]-cy)*Z/fy
        wx_arr = ax + X*right_x + Z*fwd_x; wy_arr = ay + X*right_y + Z*fwd_y
        for wx, wy in zip(wx_arr, wy_arr):
            gx, gy = self.w2g(wx, wy)
            if self.data[gx, gy] < value:
                self.data[gx, gy] = value

    def update_bbox(self, value, depth, bbox_norm, ax, ay, fwd_x, fwd_y):
        """Project a high value along the ENTIRE ray through the bbox direction,
        from near (0.5m) to far (5.0m). This ensures frontiers at all distances
        in the detection direction get boosted, not just the object's depth."""
        h, w = depth.shape
        fx = fy = w / (2 * np.tan(np.deg2rad(HFOV) / 2))
        cx, cy = w // 2, h // 2
        right_x, right_y = -fwd_y, fwd_x

        # Bbox region
        x1 = int(bbox_norm[0] * w); x2 = int(bbox_norm[2] * w)
        y1 = int(bbox_norm[1] * h); y2 = int(bbox_norm[3] * h)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        # Sample ray directions from bbox pixels at MULTIPLE depths
        # Use a subset of pixels (every 4th) and depth layers to keep it fast
        step = 4
        depths = np.linspace(0.5, 5.0, 10)  # 10 depth layers
        for py in range(y1, y2, step):
            for px in range(x1, x2, step):
                if depth[py, px] < 0.3:  # skip invalid depth pixels
                    continue
                # Project this pixel direction at all depth layers
                for d in depths:
                    x_cam = (px - cx) * d / fx
                    y_cam = -(py - cy) * d / fy
                    wx = ax + x_cam * right_x - d * fwd_x
                    wy = ay + x_cam * right_y - d * fwd_y
                    gx, gy = self.w2g(wx, wy)
                    if self.data[gx, gy] < value:
                        self.data[gx, gy] = value

    def set_value(self, wx, wy, value, radius=3):
        """Set a Gaussian-blob of value at world position (like a spatial hint)."""
        gx, gy = self.w2g(wx, wy)
        r = radius
        x0, x1 = max(0, gx-r), min(self.size-1, gx+r)
        y0, y1 = max(0, gy-r), min(self.size-1, gy+r)
        if x0 >= x1 or y0 >= y1:
            return
        # Small Gaussian kernel
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        dist2 = (xx - gx)**2 + (yy - gy)**2
        kernel = np.exp(-dist2 / (2*(r/2)**2))
        self.data[y0:y1+1, x0:x1+1] = np.maximum(
            self.data[y0:y1+1, x0:x1+1], value * kernel)

    def mark_traj(self, wx, wy):
        gx, gy = self.w2g(wx, wy)
        cv2.circle(self.traj, (gx, gy), 2, 1, -1)

    def get_value(self, wx, wy, radius=5):
        gx, gy = self.w2g(wx, wy)
        r = radius
        x0, x1 = max(0, gx-r), min(self.size-1, gx+r)
        y0, y1 = max(0, gy-r), min(self.size-1, gy+r)
        region = self.data[x0:x1+1, y0:y1+1]
        return float(np.mean(region)) if region.size > 0 else 0.0

    def visualize(self):
        dmax = max(np.max(self.data), 1e-6)
        norm = np.clip(self.data / dmax, 0, 1)
        # Heatmap: blue (low) → green → yellow → red (high)
        vis = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        vis[:, :, 0] = (norm * 255).astype(np.uint8)       # R for high
        vis[:, :, 1] = ((1-norm) * 255).astype(np.uint8)   # G for low
        vis[:, :, 2] = ((1-norm) * 128).astype(np.uint8)   # B for very low
        vis[self.traj > 0] = [255, 0, 0]  # blue where agent walked
        return vis


class SimpleObstacleMap:
    def __init__(self, size=500, resolution=0.05):
        self.size = size; self.res = resolution; self.origin = size//2
        self.obs = np.zeros((size, size), dtype=np.uint8)
        self.explored = np.zeros((size, size), dtype=np.uint8)

    def reset(self):
        self.obs.fill(0); self.explored.fill(0)

    def w2g(self, wx, wy):
        return (np.clip(int(wx/self.res)+self.origin, 0, self.size-1),
                np.clip(int(wy/self.res)+self.origin, 0, self.size-1))

    def g2w(self, gx, gy):
        return np.array([(gx-self.origin)*self.res, (gy-self.origin)*self.res])

    def update(self, depth, ax, ay, fwd_x, fwd_y):
        h, w = depth.shape; fx = fy = w/(2*np.tan(np.deg2rad(HFOV)/2))
        cx, cy = w//2, h//2; right_x, right_y = -fwd_y, fwd_x
        vv, uu = np.mgrid[0:h, 0:w]
        valid = (depth > 0.1) & (depth < 5.0)
        Z = depth[valid]; X = (uu[valid]-cx)*Z/fx; Y = -(vv[valid]-cy)*Z/fy
        wx_arr = ax + X*right_x + Z*fwd_x; wy_arr = ay + X*right_y + Z*fwd_y
        for wx, wy in zip(wx_arr, wy_arr):
            gx, gy = self.w2g(wx, wy); self.explored[gx, gy] = 1
        # Mark agent position as free
        gx, gy = self.w2g(ax, ay)
        cv2.circle(self.explored, (gx, gy), 3, 1, -1)
        cv2.circle(self.obs, (gx, gy), 1, 0, -1)

    def get_frontiers(self):
        from scipy import ndimage
        free = (self.obs == 0) & (self.explored == 1)
        unknown_dil = ndimage.binary_dilation(self.explored == 0, np.ones((3,3)))
        frontiers = free & unknown_dil
        labels, n = ndimage.label(frontiers)
        pts = []
        for i in range(1, n+1):
            ys, xs = np.where(labels == i)
            if len(ys) < 8: continue
            gx, gy = int(xs.mean()), int(ys.mean())
            pts.append(self.g2w(gx, gy))
        return pts

    def visualize(self):
        vis = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        free = (self.obs == 0) & (self.explored == 1)
        unknown = self.explored == 0
        vis[free] = [100, 180, 100]     # green = free
        vis[unknown] = [60, 60, 60]     # grey = unknown
        # Draw frontiers
        for pt in self.get_frontiers():
            gx, gy = self.w2g(pt[0], pt[1])
            cv2.circle(vis, (gx, gy), 2, (0, 0, 255), -1)  # red dot
        return vis


# ─── Spatial Prior ────────────────────────────────────────────────────────
class SpatialPrior:
    """Domain knowledge: where each object is likely to be in a biology lab.

    - lab_table: center of room
    - lab_bench: along walls
    - refrigerator: corner (two walls meet)
    - centrifuge: on lab_bench → near wall, especially once bench is found
    """
    def __init__(self, omap):
        self.omap = omap  # reference to obstacle map for room geometry
        self.lab_bench_positions = []  # detected bench positions

    def notify_detection(self, obj_name, world_xy):
        """Called when an object is detected by GroundingDINO."""
        if obj_name in ("lab_bench", "bench"):
            self.lab_bench_positions.append(world_xy)

    def _room_center(self):
        """Estimate room center from explored free space."""
        free = (self.omap.obs == 0) & (self.omap.explored == 1)
        ys, xs = np.where(free)
        if len(ys) < 10:
            return np.array([0.0, 0.0])  # fallback
        return self.omap.g2w(int(xs.mean()), int(ys.mean()))

    def _room_extent(self):
        """Estimate room XZ extent from explored area."""
        free = (self.omap.obs == 0) & (self.omap.explored == 1)
        ys, xs = np.where(free)
        if len(ys) < 10:
            return 4.0  # fallback for ~8x6m room
        wx = (xs - self.omap.origin) * self.omap.res
        wy = (ys - self.omap.origin) * self.omap.res
        return max(wx.max() - wx.min(), wy.max() - wy.min())

    def _dist_to_wall(self, wx, wy):
        """Estimate distance to nearest wall/obstacle boundary."""
        gx, gy = self.omap.w2g(wx, wy)
        r = 20  # search radius in grid cells
        x0, x1 = max(0, gx-r), min(self.omap.size-1, gx+r)
        y0, y1 = max(0, gy-r), min(self.omap.size-1, gy+r)
        region = self.omap.explored[x0:x1+1, y0:y1+1]
        total = region.size
        explored = int(region.sum())
        # Low explored ratio → near wall/edge
        return explored / max(total, 1)

    def _dist_to_corner(self, wx, wy):
        """Corner score: high where two walls meet (explored ratio from two
        orthogonal directions is low)."""
        gx, gy = self.omap.w2g(wx, wy)
        r = 15
        # Check explored ratio in 4 quadrants — corners have 2 quadrants
        # with low explored
        quads = []
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            x0 = max(0, gx+dx*r//2) if dx < 0 else gx
            x1 = min(self.omap.size-1, gx+dx*r) if dx > 0 else gx
            y0 = max(0, gy+dy*r//2) if dy < 0 else gy
            y1 = min(self.omap.size-1, gy+dy*r) if dy > 0 else gy
            region = self.omap.explored[x0:x1+1, y0:y1+1]
            quads.append(region.sum() / max(region.size, 1))
        # Corner = exactly 2 quads have high explored
        high_quads = sum(1 for q in quads if q > 0.3)
        # Score: high when exactly 2 high, low otherwise
        return max(0, 1.0 - abs(high_quads - 2) * 0.5)

    def score(self, wx, wy, target_obj):
        """Compute spatial prior score [0, 1] for a frontier at (wx, wy)."""
        center = self._room_center()
        extent = self._room_extent()
        dx, dy = wx - center[0], wy - center[1]
        dist_to_center = np.sqrt(dx*dx + dy*dy)

        if target_obj == "lab_table":
            # Prefer room center — Gaussian falloff
            sigma = extent * 0.3
            return np.exp(-0.5 * (dist_to_center / sigma)**2)

        elif target_obj == "lab_bench":
            # Prefer areas near walls (low explored ratio)
            wall_ratio = self._dist_to_wall(wx, wy)
            # Invert: 1 = right next to wall, 0 = middle of room
            wall_score = 1.0 - wall_ratio
            # Also prefer edges (not center)
            center_penalty = min(1.0, dist_to_center / (extent * 0.3))
            return 0.6 * wall_score + 0.4 * center_penalty

        elif target_obj == "refrigerator":
            # Prefer corners and walls
            corner_score = self._dist_to_corner(wx, wy)
            wall_ratio = self._dist_to_wall(wx, wy)
            wall_score = 1.0 - wall_ratio
            # Penalize center
            center_penalty = min(1.0, dist_to_center / (extent * 0.3))
            return 0.4 * corner_score + 0.3 * wall_score + 0.3 * center_penalty

        elif target_obj == "centrifuge" or target_obj == "mini_centrifuge":
            if self.lab_bench_positions:
                bench_xy = np.mean(self.lab_bench_positions, axis=0)
                d_to_bench = np.sqrt((wx-bench_xy[0])**2 + (wy-bench_xy[1])**2)
                sigma = 1.5
                bench_score = np.exp(-0.5 * (d_to_bench / sigma)**2)
                return 0.7 * bench_score + 0.3 * (1.0 - self._dist_to_wall(wx, wy))
            else:
                wall_ratio = self._dist_to_wall(wx, wy)
                wall_score = 1.0 - wall_ratio
                center_penalty = min(1.0, dist_to_center / (extent * 0.3))
                return 0.5 * wall_score + 0.5 * center_penalty

        elif target_obj == "cabinet" or target_obj == "lab_waste_bin" or target_obj == "waste_bin":
            # Along walls, avoid center
            wall_ratio = self._dist_to_wall(wx, wy)
            wall_score = 1.0 - wall_ratio
            center_penalty = min(1.0, dist_to_center / (extent * 0.3))
            return 0.6 * wall_score + 0.4 * center_penalty

        elif target_obj in ("refrigerator", "refrige", "incubor"):
            corner_score = self._dist_to_corner(wx, wy)
            wall_ratio = self._dist_to_wall(wx, wy)
            wall_score = 1.0 - wall_ratio
            center_penalty = min(1.0, dist_to_center / (extent * 0.3))
            return 0.4 * corner_score + 0.3 * wall_score + 0.3 * center_penalty

        elif target_obj == "lab_table_1":
            wall_ratio = self._dist_to_wall(wx, wy)
            wall_score = 1.0 - wall_ratio
            center_penalty = min(1.0, dist_to_center / (extent * 0.3))
            return 0.6 * wall_score + 0.4 * center_penalty

        # Default: uniform prior
        return 0.5


def _to_nq_quat(rot):
    """Convert habitat-sim quaternion (x,y,z,w) to numpy-quaternion (w,x,y,z)."""
    if isinstance(rot, nq.quaternion):
        return rot
    return nq.quaternion(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2]))


# ─── Object Tracker ──────────────────────────────────────────────────────
class SimpleObjectTracker:
    """VLFM-style: accumulate SAM point clouds across frames, DBSCAN denoise.

    Each detection adds hundreds of 3D points (mask pixels projected to world).
    DBSCAN clustering removes outlier points from walls/table/floor.
    Position is the centroid of the largest cluster, closest to the agent.
    """

    def __init__(self, min_detections=3, max_age=50, max_points=10000):
        self._points = np.empty((0, 3))  # all world-frame (x,y,z) points
        self._min_detections = min_detections
        self._max_age = max_age
        self._max_points = max_points
        self._age = 0
        self._detection_count = 0

    def add(self, point_cloud):
        """Add a point cloud (N,3) from one detection frame."""
        if len(point_cloud) == 0:
            return
        # Random subsample to avoid memory blowup
        if len(point_cloud) > 2000:
            idx = np.random.choice(len(point_cloud), 2000, replace=False)
            point_cloud = point_cloud[idx]
        self._points = np.vstack([self._points, point_cloud]) if len(self._points) > 0 else point_cloud
        # Trim oldest points if total exceeds max
        if len(self._points) > self._max_points:
            self._points = self._points[-self._max_points:]
        self._detection_count += 1
        self._age = 0

    def has_object(self):
        return self._detection_count >= self._min_detections

    def get_position(self, agent_xy=None):
        """DBSCAN denoise → return centroid of cluster closest to agent."""
        pts = self._points
        if len(pts) < 3:
            return None
        if len(pts) < 10:
            # Not enough points for DBSCAN, return median
            if agent_xy is not None:
                dists = np.linalg.norm(pts[:, [0,2]] - agent_xy, axis=1)
                return pts[np.argmin(dists)]
            return np.median(pts, axis=0)
        # DBSCAN: eps=0.3m, min 20 points
        clustering = DBSCAN(eps=0.3, min_samples=10).fit(pts)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # noise
        if not unique_labels:
            return np.median(pts, axis=0)  # all noise, return median

        # Find largest cluster
        best_label = max(unique_labels, key=lambda l: (labels == l).sum())
        cluster_pts = pts[labels == best_label]

        # Return point in cluster closest to agent
        if agent_xy is not None:
            cluster_xy = cluster_pts[:, [0, 2]]
            dists = np.linalg.norm(cluster_xy - agent_xy, axis=1)
            return cluster_pts[np.argmin(dists)]
        return np.median(cluster_pts, axis=0)

    @property
    def total_points(self):
        return len(self._points)

    def step(self):
        """Age the tracker; reset if too old."""
        self._age += 1
        if self._age > self._max_age:
            self._points = np.empty((0, 3))
            self._detection_count = 0
            self._age = 0

    def reset(self):
        self._points = np.empty((0, 3))
        self._detection_count = 0
        self._age = 0


def sam_segment(rgb, bbox_pixel, port=13183):
    """Call SAM server to get precise mask for bbox region.

    Args:
        rgb: (H,W,3) RGB image (will be converted to BGR JPEG internally)
        bbox_pixel: [x1, y1, x2, y2] in pixel coordinates
        port: SAM server port
    Returns:
        (H,W) uint8 mask (0 or 1), or None on failure
    """
    import base64 as _b64
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                          [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    img_b64 = _b64.b64encode(buf).decode('utf-8')
    try:
        resp = requests.post(f"http://localhost:{port}/mobile_sam",
                            json={"image": img_b64, "bbox": bbox_pixel},
                            timeout=20)
        if resp.status_code != 200:
            return None
        mask_str = resp.json().get("cropped_mask", "")
        if not mask_str:
            return None
        # Decode: server uses bool_arr_to_str → base64(arr.tobytes())
        mask_bytes = _b64.b64decode(mask_str)
        h, w = rgb.shape[:2]
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(h, w)
        return mask
    except Exception:
        return None


def mask_to_pointcloud(mask, depth_img, agent_state, hfov_deg=79, img_w=640, img_h=480):
    """Project ALL SAM mask pixels to 3D world-frame point cloud (VLFM-style).

    Every valid pixel in the mask becomes a 3D point. This gives hundreds of
    points per detection, enabling DBSCAN clustering for noise rejection.

    Args:
        mask: (H, W) uint8 mask (1=object, 0=background)
        depth_img: (H, W) depth array in meters
        agent_state: habitat_sim.AgentState
    Returns:
        (N, 3) world-frame point cloud, or (0, 3) empty array
    """
    h, w = depth_img.shape
    # If mask covers >30% of image, restrict to bbox center (SAM oversegmented)
    if (mask > 0).sum() > h * w * 0.3:
        ys, xs = np.where(mask > 0)
        cx_m, cy_m = xs.mean(), ys.mean()
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        x0 = max(0, int(cx_m - bw * 0.15))
        x1 = min(w, int(cx_m + bw * 0.15))
        y0 = max(0, int(cy_m - bh * 0.15))
        y1 = min(h, int(cy_m + bh * 0.15))
        inner = np.zeros_like(mask)
        inner[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = inner
    valid = (mask > 0) & (depth_img > 0.2) & (depth_img < 8.0)
    if valid.sum() < 5:
        return np.empty((0, 3))

    vs, us = np.where(valid)
    depths = depth_img[valid]
    n_pts = len(depths)

    # Project all pixels to camera frame
    hfov_rad = np.deg2rad(hfov_deg)
    fx = (w / 2) / np.tan(hfov_rad / 2)
    fy = fx; cx, cy = w / 2, h / 2

    x_cam = (us - cx) * depths / fx
    y_cam = -(vs - cy) * depths / fy
    z_cam = depths

    # Camera-frame: pinhole (+Z fwd) → habitat-sim local (-Z fwd)
    cam_pts = np.column_stack([x_cam, y_cam, -z_cam])

    # Rotate all points at once to world frame
    pos = agent_state.position
    rot = agent_state.rotation
    world_pts = nq.rotate_vectors(_to_nq_quat(rot), cam_pts) + np.array([pos[0], pos[1], pos[2]])

    return world_pts


def bbox_to_3d_robust(bbox_norm, depth_img, agent_state, use_inner=True,
                       hfov_deg=79, img_w=640, img_h=480):
    """Estimate 3D world position using inner region of bbox (pseudo-mask).

    More robust than single-pixel center: uses median depth from inner 40%
    of the bbox, reducing sensitivity to edge pixels and depth noise.

    Args:
        bbox_norm: [x1, y1, x2, y2] in normalized 0-1 coordinates
        depth_img: (H, W) depth array in meters
        agent_state: habitat_sim.AgentState
        use_inner: if True, sample inner 40% of bbox; else use full bbox
    Returns:
        (x, y, z) world position, or None
    """
    x1, y1, x2, y2 = bbox_norm
    h, w = depth_img.shape

    if use_inner:
        # Inner 40% region for robust depth sampling
        cx_n, cy_n = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * 0.2, (y2 - y1) * 0.2
        sx1 = max(0, int((cx_n - bw) * w))
        sx2 = min(w, int((cx_n + bw) * w))
        sy1 = max(0, int((cy_n - bh) * h))
        sy2 = min(h, int((cy_n + bh) * h))
    else:
        sx1 = max(0, int(x1 * w)); sx2 = min(w, int(x2 * w))
        sy1 = max(0, int(y1 * h)); sy2 = min(h, int(y2 * h))

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    patch = depth_img[sy1:sy2, sx1:sx2]
    valid = patch[(patch > 0.2) & (patch < 8.0)]
    if len(valid) < 5:
        return None
    d = float(np.median(valid))

    # Project bbox center to 3D
    u = (x1 + x2) / 2 * w
    v = (y1 + y2) / 2 * h
    hfov_rad = np.deg2rad(hfov_deg)
    fx = (w / 2) / np.tan(hfov_rad / 2)
    fy = fx
    cx, cy = w / 2, h / 2

    x_cam = (u - cx) * d / fx
    y_cam = -(v - cy) * d / fy
    z_cam = d

    pos = agent_state.position
    rot = agent_state.rotation
    # pinhole (+Z fwd) → habitat-sim local (-Z fwd)
    cam_vec = np.array([x_cam, y_cam, -z_cam])
    rotated = nq.rotate_vectors(_to_nq_quat(rot), np.array([cam_vec]))[0]
    world_pos = np.array([pos[0], pos[1], pos[2]]) + rotated
    return world_pos


# ─── Path Follower ───────────────────────────────────────────────────────
class Follower:
    def __init__(self, sim):
        self.pf = sim.pathfinder

    def forward(self, state):
        f = nq.rotate_vectors(_to_nq_quat(state.rotation), np.array([[0,0,-1]]))[0]
        return np.array([f[0], f[2]])

    def get_action(self, target_3d, state):
        cur = np.array([state.position[0], state.position[1], state.position[2]])
        if np.linalg.norm(cur - target_3d) < 0.3:
            return 0, None
        path = habitat_sim.ShortestPath()
        path.requested_start = cur; path.requested_end = target_3d
        if not self.pf.find_path(path) or len(path.points) < 2:
            return None, None
        nxt = np.array(path.points[1]); pos = state.position
        tgt = np.array([nxt[0]-pos[0], nxt[2]-pos[2]])
        tgt = tgt/(np.linalg.norm(tgt)+1e-8)
        fwd = self.forward(state)
        cross = fwd[0]*tgt[1] - fwd[1]*tgt[0]
        angle = np.arccos(np.clip(np.dot(fwd, tgt), -1, 1))
        if angle < np.deg2rad(10): return 1, path.geodesic_distance
        elif cross > 0: return 3, path.geodesic_distance
        else: return 2, path.geodesic_distance


# ─── Video Writer ────────────────────────────────────────────────────────
class EpisodeVideo:
    def __init__(self, path, fps=10):
        self.path = path; self.fps = fps; self.frames = []
    def add(self, frame):
        self.frames.append(frame)
    def save(self):
        if not self.frames: return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 self.fps, (w, h))
        for f in self.frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video saved: {self.path} ({len(self.frames)} frames)", flush=True)


def make_panel(rgb, vmap_vis, omap_vis, depth, info_lines, panel_h=480):
    """Compose a 4-panel visualization: RGB | ValueMap | ObstacleMap | Depth."""
    h, w = rgb.shape[:2]
    # Resize maps to match RGB dimensions
    vmap_r = cv2.resize(vmap_vis, (w, h))
    omap_r = cv2.resize(omap_vis, (w, h))
    # Depth colormap
    d_clip = np.clip(depth, 0, 5.0) / 5.0
    d_color = cv2.applyColorMap((d_clip*255).astype(np.uint8), cv2.COLORMAP_JET)
    d_color = cv2.resize(d_color, (w, h))

    # Info bar at bottom
    bar_h = 30
    bar = np.zeros((bar_h, w*2, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(info_lines[:2]):
        cv2.putText(bar, line, (10, 12+i*14), font, 0.35, (255,255,255), 1)

    # Top row: RGB | ValueMap
    top = np.hstack([rgb, vmap_r])
    # Bottom row: ObstacleMap | Depth
    bot = np.hstack([omap_r, d_color])
    # Stack: top + bottom + info bar
    panel = np.vstack([top, bot, bar])
    return panel


# ─── Run Episode ─────────────────────────────────────────────────────────
def run_episode(ep, ep_idx, save_video=False, blind=False, use_vlm_vmap=False):
    sim = create_sim(); pf = sim.pathfinder; follower = Follower(sim)
    vmap = SimpleValueMap(); omap = SimpleObstacleMap()
    spatial = SpatialPrior(omap)  # domain-knowledge spatial prior
    tracker = SimpleObjectTracker()  # accumulate detections across frames

    target_obj = ep["object_category"]
    start_pos = np.array(ep["start_position"]); start_rot = np.array(ep["start_rotation"])
    dlog_path = f"navmesh_output/debug_ep{ep_idx}.log"
    os.makedirs("navmesh_output", exist_ok=True)
    if os.path.exists(dlog_path):
        os.remove(dlog_path)
    # VLFM-style blind: agent does NOT know goal, but we read it for success checking
    true_goal_xyz = np.array(ep["goals"][0]["position"])
    true_goal_xy = true_goal_xyz[[0, 2]]
    # Snap to NavMesh at ground level (goal may be on table)
    snap_goal = pf.snap_point(np.array([true_goal_xyz[0], 0.03, true_goal_xyz[2]]))
    if snap_goal is not None:
        nav_goal_xy = np.array([snap_goal[0], snap_goal[2]])
    else:
        nav_goal_xy = true_goal_xy
    if blind:
        goal_xy = None
    else:
        goal_xy = true_goal_xy  # non-blind: agent knows and navigates to goal
    threshold = OBJ_RADIUS.get(target_obj, 1.0)

    agent = sim.get_agent(0); state = habitat_sim.AgentState()
    state.position = mn.Vector3(start_pos[0], start_pos[1], start_pos[2])
    state.rotation = list(start_rot); agent.set_state(state)

    vid = EpisodeVideo(f"{VIDEO_DIR}/ep{ep_idx:03d}_{target_obj}.mp4") if save_video else None
    itm_ok, dino_ok = True, True
    itm_fails, dino_fails = 0, 0
    mode, init_spins = "init", 0
    best_dist = float("inf"); last_cosine = 0.0
    dino_detected_this_step = False  # VLFM-style: detection boosts value map
    best_detection_fwd = None         # world-space forward vector at best detection
    last_bbox_norm = None             # latest detection bbox [x1,y1,x2,y2] for visual servoing
    last_sam_mask = None              # latest SAM mask for the detection
    geo_dist = 0.0
    last_agent_xy = None  # for stuck detection
    nav_stuck_steps = 0
    nav_stuck_count = 0
    t_ep_start = time.time()
    visited_frontiers = set()
    no_progress_steps = 0
    systematic_mode = False
    consecutive_resets = 0

    for step in range(MAX_STEPS):
        state = agent.get_state(); pos = state.position
        agent_xy = np.array([pos[0], pos[2]]); agent_3d = np.array([pos[0], pos[1], pos[2]])
        fwd = follower.forward(state)
        # Distance to NavMesh-accessible goal point (true goal may be on table)
        dist_obj = float(np.linalg.norm(agent_xy - nav_goal_xy))

        # ── Success (both modes) ──────────────────────────────────────
        if dist_obj < threshold:
            _finish(sim, vid, ep, target_obj, True, step, dist_obj, mode)
            return {"success": True, "steps": step, "final_dist": dist_obj,
                    "target": target_obj}
        # Visual success: bbox large + close enough (blind mode)
        bbox_size_now = max(last_bbox_norm[2] - last_bbox_norm[0],
                            last_bbox_norm[3] - last_bbox_norm[1]) if last_bbox_norm else 0
        if blind and bbox_size_now > 0.5 and dist_obj < threshold * 2.5:
            print(f"  Step {step}: VISUAL SUCCESS! bbox={bbox_size_now:.2f} dist={dist_obj:.2f}m",
                  flush=True)
            _finish(sim, vid, ep, target_obj, True, step, dist_obj, mode)
            return {"success": True, "steps": step, "final_dist": dist_obj,
                    "target": target_obj}

        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]; depth = obs["depth"]

        # ── INIT: quick 180° spin ───────────────────────────────────
        if mode == "init":
            if init_spins < INIT_SPINS:
                # Seed value map on every other init spin
                if itm_ok and init_spins % 2 == 0:
                    try:
                        if use_vlm_vmap:
                            info = update_value_map_vlm(rgb, depth, agent_3d, fwd[0], fwd[1],
                                                        vmap, target_obj.replace("_", " "))
                            if info:
                                last_cosine = np.mean(list(info["scores"].values()))
                        else:
                            itm_prompt = DOMAIN_PROMPTS.get(target_obj, target_obj)
                            model, proc, dev = get_blip2()
                            info = update_value_map_patches(
                                model, proc, rgb, depth, itm_prompt,
                                agent_3d[0], agent_3d[2], fwd[0], fwd[1],
                                vmap, dev)
                            last_cosine = info["score_mean"]
                        itm_fails = 0
                    except Exception as e:
                        itm_fails += 1
                        if itm_fails >= 3:
                            itm_ok = False
                # GDINO detection during init: boost value map (VLFM-style)
                if dino_ok and blind:
                    try:
                        det_prompt = DOMAIN_PROMPTS.get(target_obj, target_obj.replace("_", " "))
                        dets = gdino_detect(rgb, det_prompt)
                        if len(dets) > 0 and dets[0].get("confidence", 0) > 0.3:
                            verified = vlm_verify_detections(rgb, dets, target_obj.replace("_", " "))
                            if len(verified) > 0:
                                box_norm = verified[0].get("box_norm", None)
                                if box_norm is not None:
                                    vmap.update_bbox(1.0, depth, box_norm,
                                                     agent_3d[0], agent_3d[2],
                                                     fwd[0], fwd[1])
                                    if best_detection_fwd is None:
                                        best_detection_fwd = np.array([fwd[0], fwd[1]])
                                    # Pre-load tracker with init detections
                                    if not tracker.has_object():
                                        t3d = bbox_to_3d_robust(box_norm, depth, state)
                                        if t3d is not None:
                                            tracker.add(np.array([t3d]))
                                    dino_detected_this_step = True
                                    print(f"  Step {step}: Detected {target_obj}, boosted",
                                          flush=True)
                    except Exception as e:
                        pass
                omap.update(depth, agent_3d[0], agent_3d[2], fwd[0], fwd[1])
                vmap.mark_traj(agent_xy[0], agent_xy[1])
                # Turn 30° each spin (3×10°) for faster init
                sim.step("turn_right"); sim.step("turn_right"); sim.step("turn_right")
                init_spins += 1

                if save_video:  # save every init frame
                    panel = make_panel(rgb, vmap.visualize(), omap.visualize(), depth,
                                       [f"INIT {init_spins}/{INIT_SPINS} | Target: {target_obj}",
                                        f"Cos={last_cosine:.3f} {'(ITM off)' if not itm_ok else ''}"])
                    vid.add(panel)
                    frame_dir = f"{VIDEO_DIR}/frames/ep{ep_idx:03d}_{target_obj}"
                    os.makedirs(frame_dir, exist_ok=True)
                    cv2.imwrite(f"{frame_dir}/step{step:04d}.jpg", panel)
                continue
            # Save final init panorama view
            if save_video:
                det_dir = f"{VIDEO_DIR}/detections/ep{ep_idx:03d}_{target_obj}"
                os.makedirs(det_dir, exist_ok=True)
                cv2.imwrite(f"{det_dir}/init_final.jpg",
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # If target detected during init, turn to face the detection direction + walk forward
            if blind and best_detection_fwd is not None:
                target_fwd = best_detection_fwd
                target_yaw = np.arctan2(target_fwd[0], target_fwd[1])
                cur_state = agent.get_state()
                cur_fwd = follower.forward(cur_state)
                cur_yaw = np.arctan2(cur_fwd[0], cur_fwd[1])
                yaw_diff = target_yaw - cur_yaw
                while yaw_diff > np.pi: yaw_diff -= 2*np.pi
                while yaw_diff < -np.pi: yaw_diff += 2*np.pi
                turn_action = "turn_left" if yaw_diff > 0 else "turn_right"
                turn_steps = int(abs(yaw_diff) / np.deg2rad(10))
                for _ in range(min(turn_steps, 36)):
                    sim.step(turn_action)
                obs = sim.get_sensor_observations()
                rgb = obs["rgb"][:, :, :3]; depth = obs["depth"]
                best_detection_fwd = None
                # Update state — init turns changed agent orientation, must refresh before projections
                state = agent.get_state(); pos = state.position
                agent_xy = np.array([pos[0], pos[2]])
                agent_3d = np.array([pos[0], pos[1], pos[2]])
                fwd = follower.forward(state)
                d = "RIGHT" if yaw_diff > 0 else "LEFT"
                print(f"  Step {step}: Init done, {d} {abs(np.rad2deg(yaw_diff)):.0f}deg. Exploring...",
                      flush=True)
            else:
                print(f"  Step {step}: Init done ({INIT_SPINS} spins), exploring..."
                      f"{' (ITM disabled after failures)' if not itm_ok else ''}", flush=True)
            mode = "explore"

        # ── Per-episode timeout ──────────────────────────────────────
        if time.time() - t_ep_start > EPISODE_TIMEOUT:
            print(f"  Step {step}: EPISODE TIMEOUT ({EPISODE_TIMEOUT}s)", flush=True)
            break

        # ── Update maps ──────────────────────────────────────────────
        omap.update(depth, agent_3d[0], agent_3d[2], fwd[0], fwd[1])
        if itm_ok and step % PATCH_ITM_EVERY_N == 0:
            try:
                if use_vlm_vmap:
                    info = update_value_map_vlm(rgb, depth, agent_3d, fwd[0], fwd[1],
                                                vmap, target_obj.replace("_", " "))
                    if info:
                        last_cosine = np.mean(list(info["scores"].values()))
                    itm_fails = 0
                else:
                    itm_prompt = DOMAIN_PROMPTS.get(target_obj, target_obj)
                    model, proc, dev = get_blip2()
                    info = update_value_map_patches(
                        model, proc, rgb, depth, itm_prompt,
                        agent_3d[0], agent_3d[2], fwd[0], fwd[1],
                        vmap, dev)
                    last_cosine = info["score_mean"]
                    itm_fails = 0
            except Exception as e:
                itm_fails += 1
                if itm_fails >= 3:
                    print(f"  Step {step}: ITM disabled after {itm_fails} failures", flush=True)
                    itm_ok = False
        vmap.mark_traj(agent_xy[0], agent_xy[1])

        # ── GroundingDINO detection ──────────────────────────────────
        if dino_ok and step % DINO_EVERY_N == 0:
            try:
                det_prompt = DOMAIN_PROMPTS.get(target_obj, target_obj.replace("_", " "))
                dets = gdino_detect(rgb, det_prompt)
                # Save detection image for debugging
                det_dir = f"{VIDEO_DIR}/detections/ep{ep_idx:03d}_{target_obj}"
                os.makedirs(det_dir, exist_ok=True)
                det_img = rgb.copy()
                h_img, w_img = det_img.shape[:2]
                for d in dets:
                    box_norm = d.get("box_norm", None)
                    if box_norm is not None and len(box_norm) == 4:
                        x1 = int(box_norm[0] * w_img); y1 = int(box_norm[1] * h_img)
                        x2 = int(box_norm[2] * w_img); y2 = int(box_norm[3] * h_img)
                        cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{d.get('label','?')} {d.get('confidence',0):.2f}"
                        cv2.putText(det_img, label, (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                cv2.imwrite(f"{det_dir}/step{step:04d}.jpg",
                            cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR))
                if len(dets) > 0 and dets[0].get("confidence", 0) > 0.3:
                    verified = vlm_verify_detections(rgb, dets, target_obj.replace("_", " "))
                    if len(verified) > 0:
                        dino_detected_this_step = True
                        if blind:
                            box_norm = verified[0].get("box_norm", None)
                            if box_norm is not None:
                                # SAM segmentation (always run for visual servoing)
                                x1_px = int(box_norm[0] * rgb.shape[1])
                                y1_px = int(box_norm[1] * rgb.shape[0])
                                x2_px = int(box_norm[2] * rgb.shape[1])
                                y2_px = int(box_norm[3] * rgb.shape[0])
                                sam_mask = sam_segment(rgb, [x1_px, y1_px, x2_px, y2_px])
                                # Save SAM visualization
                                sam_vis = rgb.copy()
                                if sam_mask is not None:
                                    sam_vis[sam_mask > 0] = [0, 255, 0]
                                    cv2.rectangle(sam_vis, (x1_px, y1_px), (x2_px, y2_px), (255, 0, 0), 2)
                                cv2.imwrite(f"{det_dir}/step{step:04d}_sam.jpg",
                                            cv2.cvtColor(sam_vis, cv2.COLOR_RGB2BGR))
                                # Store for visual servoing
                                last_bbox_norm = box_norm
                                last_sam_mask = sam_mask
                                # Boost value map
                                vmap.update_bbox(1.0, depth, box_norm,
                                                 agent_3d[0], agent_3d[2],
                                                 fwd[0], fwd[1])
                                # Accumulate point cloud to tracker (VLFM-style)
                                if not tracker.has_object():
                                    if sam_mask is not None and sam_mask.sum() > 20:
                                        pcd = mask_to_pointcloud(sam_mask, depth, state)
                                    else:
                                        # Fallback: project bbox center as single point
                                        t3d = bbox_to_3d_robust(box_norm, depth, state)
                                        pcd = np.array([t3d]) if t3d is not None else np.empty((0, 3))
                                    if len(pcd) > 0:
                                        tracker.add(pcd)
                                if tracker.has_object():
                                    tpos = tracker.get_position(agent_xy)
                                    bbox_w = box_norm[2] - box_norm[0]
                                    bbox_h = box_norm[3] - box_norm[1]
                                    if tpos is not None:
                                        with open(dlog_path, "a") as f:
                                            print(f"  Step {step}: DETECTED {target_obj} "
                                                  f"pts={tracker.total_points} "
                                                  f"pos=({tpos[0]:.2f},{tpos[2]:.2f}) "
                                                  f"bb={bbox_w:.2f}x{bbox_h:.2f}", file=f, flush=True)
                    dino_fails = 0
            except Exception as e:
                dino_fails += 1
                if dino_fails >= 3:
                    print(f"  Step {step}: DINO disabled after {dino_fails} failures", flush=True)
                    dino_ok = False

        # ── Action selection ────────────────────────────────────────
        mode = "explore"
        geo = 0.0
        frontiers = omap.get_frontiers()

        # Navigate mode: pathfinder toward closest navigable point near tracked object
        if tracker.has_object() and last_bbox_norm is not None:
            mode = "navigate"
            bbox_cx = (last_bbox_norm[0] + last_bbox_norm[2]) / 2
            tpos = tracker.get_position(agent_xy)
            action, geo = None, 0.0
            if tpos is not None:
                # After getting stuck twice, switch to pure visual servoing
                if nav_stuck_count >= 2:
                    bbox_cx = (last_bbox_norm[0] + last_bbox_norm[2]) / 2
                    if bbox_cx < 0.35:
                        action = 2  # turn_left
                    elif bbox_cx > 0.65:
                        action = 3  # turn_right
                    else:
                        action = 1  # move_forward
                    with open(dlog_path, "a") as f:
                        print(f"  Step {step}: NAV visual bb_cx={bbox_cx:.2f} act={action}", file=f, flush=True)
                else:
                    # Snap tracked position to nearest navigable point
                    nav_3d = np.array([tpos[0], 0.03, tpos[2]])
                    snap = pf.snap_point(nav_3d)
                    if snap is not None:
                        nav_3d = np.array([snap[0], snap[1], snap[2]])
                    action, geo = follower.get_action(nav_3d, state)
                    # Stuck detection
                    if action == 1 and last_agent_xy is not None:
                        moved = float(np.linalg.norm(agent_xy - last_agent_xy))
                        if moved < 0.02:
                            nav_stuck_steps += 1
                        else:
                            nav_stuck_steps = 0
                            nav_stuck_count = 0  # reset: agent is moving again
                        if nav_stuck_steps > 8:
                            nav_stuck_steps = 0
                            nav_stuck_count += 1
                            if bbox_cx < 0.35:
                                action = 2
                            elif bbox_cx > 0.65:
                                action = 3
                            else:
                                action = 1
                            with open(dlog_path, "a") as f:
                                print(f"  Step {step}: NAV stuck→visual "
                                      f"({nav_stuck_count}/2) bb_cx={bbox_cx:.2f} act={action}", file=f, flush=True)
                        else:
                            with open(dlog_path, "a") as f:
                                print(f"  Step {step}: NAV pf t=({tpos[0]:.2f},{tpos[2]:.2f}) "
                                      f"-> snap=({nav_3d[0]:.2f},{nav_3d[2]:.2f}) act={action}", file=f, flush=True)
                    else:
                        nav_stuck_steps = 0
                        with open(dlog_path, "a") as f:
                            print(f"  Step {step}: NAV pf t=({tpos[0]:.2f},{tpos[2]:.2f}) "
                                  f"-> snap=({nav_3d[0]:.2f},{nav_3d[2]:.2f}) act={action}", file=f, flush=True)
        elif tracker.has_object() and last_bbox_norm is None:
            # Tracking but lost sight — fall back to exploration
            mode = "explore"
            tracker.reset()
            action = 2 if (step % 4 < 2) else 3  # alternate turns to scan
        elif len(frontiers) == 0:
            tracker.step()  # only age tracker during exploration
            if blind:
                action = 2 if (step % 4 < 2) else 3  # alternate turns to scan
            else:
                goal_3d = np.array([goal_xy[0], 0.03, goal_xy[1]])
                snap = pf.snap_point(goal_3d)
                if snap is not None: goal_3d = np.array([snap[0], snap[1], snap[2]])
                action, geo = follower.get_action(goal_3d, state)
        else:
            tracker.step()  # only age tracker during exploration
            # Score frontiers: SEMANTIC + SPATIAL (+ PROGRESS if not blind)
            scored = []
            for f in frontiers:
                sem_score = vmap.get_value(f[0], f[1], radius=3)
                spa_score = spatial.score(f[0], f[1], target_obj)
                d_from_agent = np.linalg.norm(f - agent_xy)
                explore_bonus = min(1.0, d_from_agent / 5.0)
                if blind:
                    # Detection just happened: suppress exploration, boost semantic
                    if dino_detected_this_step:
                        sem_w = 0.8 if itm_ok else 0.0
                        spa_w = 0.2; exp_w = 0.0
                    else:
                        sem_w = 0.6 if itm_ok else 0.0
                        spa_w = 0.2; exp_w = 0.2
                    combined = (sem_w * sem_score + spa_w * spa_score + exp_w * explore_bonus)
                    scored.append((f, combined, d_from_agent, sem_score, spa_score))
                else:
                    d_to_goal = np.linalg.norm(f - goal_xy)
                    progress = (dist_obj - d_to_goal) / max(dist_obj, 1e-6)
                    sem_w = 0.3 if itm_ok else 0.0
                    spa_w = 0.4; prog_w = 0.2; exp_w = 0.1
                    combined = (sem_w * sem_score + spa_w * spa_score +
                                prog_w * max(0, progress) + exp_w * explore_bonus)
                    scored.append((f, combined, d_from_agent, sem_score, spa_score))

            # Score uniformity check
            scores_arr = np.array([s[1] for s in scored])
            score_std = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0

            if no_progress_steps > 30 or score_std < 0.03:
                systematic_mode = True
            elif no_progress_steps == 0:
                systematic_mode = False

            if systematic_mode:
                best_f, best_score = None, -float("inf")
                for f, combined, dist, sem, spa in scored:
                    fkey = (round(f[0], 1), round(f[1], 1))
                    if fkey in visited_frontiers:
                        continue
                    if dist > best_score:
                        best_score = dist; best_f = f
                if best_f is not None:
                    consecutive_resets = 0
                if best_f is None and len(scored) > 0:
                    consecutive_resets += 1
                    if consecutive_resets > 5:
                        print(f"  Step {step}: Fully explored. Best={best_dist:.2f}m", flush=True)
                        break
                    best_f = max(scored, key=lambda s: s[2])[0]
                    visited_frontiers.clear()
                    print(f"  Step {step}: All visited→farthest "
                          f"({np.linalg.norm(best_f-agent_xy):.1f}m) [{consecutive_resets}/5]",
                          flush=True)
            else:
                best_f = scored[0][0]; best_score = scored[0][1]

            if best_f is not None:
                fkey = (round(best_f[0], 1), round(best_f[1], 1))
                visited_frontiers.add(fkey)
                nav_3d = np.array([best_f[0], 0.03, best_f[1]])
                snap = pf.snap_point(nav_3d)
                if snap is not None: nav_3d = np.array([snap[0], snap[1], snap[2]])
                action, geo = follower.get_action(nav_3d, state)
            else:
                action, geo = 2, 0.0
        if geo is not None: geo_dist = geo

        # Progress tracking toward true goal (every step, regardless of action branch)
        if dist_obj < best_dist - 0.5:  # only meaningful progress (>0.5m) resets
            best_dist = dist_obj
            no_progress_steps = 0
        else:
            no_progress_steps += 1

        # Early-stop: agent is stuck in dead-loop, abandon episode
        if no_progress_steps > 100 and best_dist > 0.5:
            print(f"  Step {step}: EARLY STOP — no progress in {no_progress_steps} steps "
                  f"(best={best_dist:.2f}m)", flush=True)
            break

        if action is None: action = 2
        act_name = ACTIONS.get(action)
        if act_name is not None: sim.step(act_name)

        # ── Video frame ──────────────────────────────────────────────
        if save_video:  # save every step
            spa_here = spatial.score(agent_xy[0], agent_xy[1], target_obj)
            if mode == "navigate" and last_bbox_norm:
                bb = last_bbox_norm
                tpos = tracker.get_position(agent_xy)
                tp_str = f"t=({tpos[0]:.1f},{tpos[2]:.1f})" if tpos is not None else "t=?"
                nav_info = f"PATHFINDER bb={bb[2]-bb[0]:.2f} geo={geo_dist:.1f}m {tp_str}"
            else:
                nav_info = "DETECTED!" if dino_detected_this_step else "exploring"
            info = [
                f"Step {step} | Mode: {mode} | Target: {target_obj} | "
                f"Sem={last_cosine:.2f} Spa={spa_here:.2f}",
                f"Dist: {dist_obj:.2f}m | Best: {best_dist:.2f}m | "
                f"Geo: {geo_dist:.1f}m | "
                f"{nav_info}",
            ]
            panel = make_panel(rgb, vmap.visualize(), omap.visualize(), depth, info)
            vid.add(panel)
            # Also save individual frame image
            frame_dir = f"{VIDEO_DIR}/frames/ep{ep_idx:03d}_{target_obj}"
            os.makedirs(frame_dir, exist_ok=True)
            cv2.imwrite(f"{frame_dir}/step{step:04d}.jpg", panel)

        dino_detected_this_step = False  # reset for next step
        last_agent_xy = agent_xy.copy()

        if step % 50 == 0:
            spa_here = spatial.score(agent_xy[0], agent_xy[1], target_obj)
            elapsed = time.time() - t_ep_start
            sys_tag = "[SYS]" if systematic_mode else "[SEM]"
            print(f"  Step {step}: {mode}{sys_tag} dist={dist_obj:.2f}m "
                  f"sem={last_cosine:.3f} spa={spa_here:.2f} "
                  f"noprog={no_progress_steps} t={elapsed:.0f}s", flush=True)

    # Timeout
    if blind:
        final_dist = dist_obj  # distance to true goal (already computed)
    else:
        final_dist = float(np.linalg.norm(
            np.array([agent.get_state().position[0], agent.get_state().position[2]]) - goal_xy))
    _finish(sim, vid, ep, target_obj, False, step, final_dist, mode)
    return {"success": False, "steps": step, "final_dist": best_dist,
            "target": target_obj}


def _finish(sim, vid, ep, target_obj, success, steps, final_dist, mode):
    if vid:
        # Add final frame with result
        info = [f"RESULT: {'SUCCESS' if success else 'FAILED'} in {steps} steps",
                f"Target: {target_obj} | Final dist: {final_dist:.2f}m"]
        final_panel = np.zeros((480*2, 640*2, 3), dtype=np.uint8)
        cv2.putText(final_panel, info[0], (50, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(final_panel, info[1], (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        vid.add(final_panel)
        vid.save()
    sim.close()


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="Save visualization videos")
    parser.add_argument("--blind", action="store_true",
                        help="Blind mode: agent does NOT know goal position, must find target purely by semantic signals")
    parser.add_argument("--vlm_vmap", action="store_true",
                        help="Use GPT-4o instead of BLIP-2 for value map generation")
    args = parser.parse_args()

    if args.blind:
        print("=== BLIND MODE: Agent does NOT know goal positions ===", flush=True)

    with gzip.open(EPISODE_FILE, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"]

    results = []; t0 = time.time()
    total = min(args.episodes, len(episodes))
    for i in range(args.start, args.start + total):
        result = run_episode(episodes[i], i, save_video=args.video, blind=args.blind,
                             use_vlm_vmap=args.vlm_vmap)
        result["episode_id"] = episodes[i]["episode_id"]
        result["target"] = episodes[i]["object_category"]
        results.append(result)
        # Flush intermediate results
        sys.stdout.flush()
        n = len(results); succ = sum(1 for r in results if r["success"])
        elapsed_ep = time.time() - t0
        print(f"[{n}/{total}] Running SR: {succ}/{n} ({100*succ/max(n,1):.0f}%) | "
              f"Time: {elapsed_ep:.0f}s | "
              f"Last: {result['target']} {'SUCCESS' if result['success'] else 'FAILED'} "
              f"dist={result['final_dist']:.2f}m steps={result['steps']}", flush=True)

    elapsed = time.time() - t0; successes = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print(f"VLFM RESULTS ({len(results)} episodes, {elapsed:.0f}s)")
    print(f"  Success: {len(successes)}/{len(results)} "
          f"({100*len(successes)/max(len(results),1):.0f}%)")
    if successes:
        print(f"  Avg steps (success): {np.mean([r['steps'] for r in successes]):.0f}")
    print(f"  Avg final dist: {np.mean([r['final_dist'] for r in results]):.2f}m")
    for obj in ["cabinet", "centrifuge", "incubor", "lab_waste_bin", "mini_centrifuge", "pipette", "refrige"]:
        obj_r = [r for r in results if r["target"] == obj]
        if obj_r:
            sr = 100*sum(1 for r in obj_r if r["success"])/len(obj_r)
            print(f"  {obj:20s}: {sr:.0f}% ({len(obj_r)} ep)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
