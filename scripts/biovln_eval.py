#!/usr/bin/env python3
"""
BioVLN Evaluation SDK — 标准评测接口
══════════════════════════════════════════════════════════════════════════

用户只需实现一个 act() 方法，其余（场景加载、episode 切换、指标计算）全自动。

用法:
    from biovln_eval import BioVLNPolicy, evaluate, Action

    class MyPolicy(BioVLNPolicy):
        def reset(self, observation, episode_info):
            pass  # 可选, 每 episode 开始时调用

        def act(self, observation):
            # observation = {"position", "heading", "goal", "goal_category",
            #                "step", "episode_id", "rgb"?, "depth"?, "semantic"?}
            return Action.MOVE_FORWARD  # 0=STOP, 1=FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT

    results = evaluate(MyPolicy(), split="dev")
    print(f"SR={results['SR']:.1%} SPL={results['SPL']:.3f}")

CLI 用法:
    python biovln_eval.py --split dev --dataset multi_scene  # 快速验证环境
"""

from __future__ import annotations

# Headless GL setup — must be before any habitat_sim import
import os as _os
if not _os.environ.get("DISPLAY") and not _os.environ.get("PYOPENGL_PLATFORM"):
    _os.environ["PYOPENGL_PLATFORM"] = "egl"

import gzip
import json
import os  # noqa: F811
import sys
import time
from collections import Counter, defaultdict
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ─── Safety: Three-Zone Model Constants ─────────────────────────────────────

ZONE2_DANGER = 0.5      # Zone 2: hazard threshold from instrument surface (meters)
SAFE_APPROACH_ANGLE = 60  # degrees (± from operational face direction)

# Per-category parameters: (face_half_depth, approach_distance, half_width)
# goal = instrument_center + facing * (face_half_depth + approach_dist)
# instrument_radius ≈ max(face_half_depth, half_width) for clearance computation
INSTRUMENT_SAFETY_PARAMS = {
    "refrigerator":    (0.455, 0.7, 0.45),
    "refrige":         (0.455, 0.7, 0.45),
    "incubator":       (0.673, 0.7, 0.68),
    "incubor":         (0.673, 0.7, 0.68),
    "centrifuge":      (0.245, 0.7, 0.25),
    "mini_centrifuge": (0.332, 0.7, 0.33),
    "pipette":         (0.159, 0.7, 0.06),
    "waste_bin":       (0.183, 0.7, 0.19),
    "lab_waste_bin":   (0.183, 0.7, 0.19),
    "cabinet":         (0.692, 0.7, 0.69),
    "lab_table":       (0.45,  0.7, 1.20),
    "lab_table_1":     (0.45,  0.7, 1.20),
    "lab_bench":       (2.97,  0.7, 0.78),
}

# ─── Actions ───────────────────────────────────────────────────────────────

class Action(IntEnum):
    """离散动作空间。"""
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

_HABITAT_ACTION = {
    Action.STOP: None,
    Action.MOVE_FORWARD: "move_forward",
    Action.TURN_LEFT: "turn_left",
    Action.TURN_RIGHT: "turn_right",
}

# ─── Dataset Registry ──────────────────────────────────────────────────────

DATASETS: Dict[str, Dict[str, str]] = {
    "multi_scene": {
        "scenes_root": "/root/gpufree-data/multi_scene/output",
        "splits_dir": "/root/gpufree-data/multi_scene/output/splits",
    },
    "two_room": {
        "scenes_root": "/root/gpufree-data/two_room_output",
        "splits_dir": "/root/gpufree-data/two_room_output/splits",
    },
    "lsat_target": {
        "scenes_root": "/root/gpufree-data/lsat_test",
        "splits_dir": "/root/gpufree-data/lsat_test/splits",
    },
}

# ─── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_GOAL_RADII: Dict[str, float] = {
    # 落地仪器: ~1.0m（从各个方向靠近约一步距离内就算到达）
    "refrigerator":   1.0, "refrige":        1.0,
    "centrifuge":     1.0, "mini_centrifuge": 1.0,
    "cabinet":        1.0, "incubator":      1.0, "incubor": 1.0,
    "pcr_machine":    1.0,
    # 小件: 0.8m
    "pipette":        0.8, "waste_bin":      0.8, "lab_waste_bin": 0.8,
    # 家具: 1.3m（目标面积大）
    "lab_table":      1.3, "lab_table_1":    1.3, "lab_bench": 1.3,
}

DEFAULT_MAX_STEPS = 500
DEFAULT_FORWARD = 0.25       # 米/步
DEFAULT_TURN_DEG = 10.0      # 度/步（habitat-sim 默认）
DEFAULT_CAMERA_W = 320
DEFAULT_CAMERA_H = 240
DEFAULT_HFOV = 79            # 度

# ═══════════════════════════════════════════════════════════════════════════
# Policy 基类
# ═══════════════════════════════════════════════════════════════════════════

class BioVLNPolicy:
    """
    BioVLN 导航策略基类。

    子类只需实现 act()。可覆盖 reset() 做 episode 级初始化。
    """

    def reset(self, observation: Dict[str, Any], episode_info: Dict[str, Any]) -> None:
        """
        每个 episode 开始时调用一次。

        Parameters
        ----------
        observation : 初始观测（同 act 的 observation）
        episode_info : dict with keys:
            episode_id, scene_name, target_category, start_position,
            goal_position, difficulty, geodesic_distance
        """
        pass

    def act(self, observation: Dict[str, Any]) -> int:
        """
        每步调用，返回 Action (0-3)。

        Parameters
        ----------
        observation : dict
            always:
                position          np.ndarray (3,)  世界坐标 XYZ
                heading           np.ndarray (2,)  朝向 (dx, dz)
                goal_position     np.ndarray (2,)  目标 XZ
                goal_category     str              目标类别 "incubator" 等
                step              int              当前步数
                episode_id        str
            if enabled:
                rgb               np.ndarray (H,W,3)  uint8
                depth             np.ndarray (H,W)    float32 (meters)
                semantic          np.ndarray (H,W)    int32  (object_id)

        Returns
        -------
        int  Action.STOP | MOVE_FORWARD | TURN_LEFT | TURN_RIGHT
        """
        raise NotImplementedError("Please implement act(observation) → int")


# ═══════════════════════════════════════════════════════════════════════════
# 内置策略（用于测试/对照）
# ═══════════════════════════════════════════════════════════════════════════

class RandomPolicy(BioVLNPolicy):
    """随机策略 — 论文 lower bound。"""
    def __init__(self, move_prob: float = 0.7):
        self.move_prob = move_prob
        self._rng = np.random.RandomState()

    def reset(self, obs, info):
        self._rng = np.random.RandomState(int(info.get("episode_id", 0)))

    def act(self, obs):
        if self._rng.random() < self.move_prob:
            return Action.MOVE_FORWARD
        return Action.TURN_LEFT if self._rng.random() < 0.5 else Action.TURN_RIGHT


class ForwardOnlyPolicy(BioVLNPolicy):
    """一直向前走 — 最小 smoke test。"""
    def act(self, obs):
        return Action.MOVE_FORWARD


# ═══════════════════════════════════════════════════════════════════════════
# 内部: Sim 管理
# ═══════════════════════════════════════════════════════════════════════════

def _make_sim_config(
    glb_path: str,
    navmesh_path: str,
    camera_w: int,
    camera_h: int,
    hfov_deg: float,
    use_rgb: bool,
    use_depth: bool,
    use_semantic: bool,
) -> Any:
    """构建 habitat-sim Configuration。"""
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = glb_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.allow_sliding = False
    sim_cfg.load_semantic_mesh = False

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = 1.7
    agent_cfg.radius = 0.1
    agent_cfg.sensor_specifications = []

    if use_rgb:
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "color_sensor"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [camera_h, camera_w]
        rgb.hfov = hfov_deg
        agent_cfg.sensor_specifications.append(rgb)

    if use_depth:
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth_sensor"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [camera_h, camera_w]
        depth.hfov = hfov_deg
        agent_cfg.sensor_specifications.append(depth)

    if use_semantic:
        sem = habitat_sim.CameraSensorSpec()
        sem.uuid = "semantic_sensor"
        sem.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem.resolution = [camera_h, camera_w]
        sem.hfov = hfov_deg
        agent_cfg.sensor_specifications.append(sem)

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class _SimFactory:
    """管理 habitat-sim 实例，按 scene 缓存。"""

    def __init__(
        self,
        scenes_root: Path,
        *,
        camera_w: int = DEFAULT_CAMERA_W,
        camera_h: int = DEFAULT_CAMERA_H,
        hfov_deg: float = DEFAULT_HFOV,
        use_rgb: bool = False,
        use_depth: bool = False,
        use_semantic: bool = False,
        move_forward: float = DEFAULT_FORWARD,
        turn_angle_deg: float = DEFAULT_TURN_DEG,
    ):
        self._scenes_root = Path(scenes_root)
        self._cam_w = camera_w
        self._cam_h = camera_h
        self._hfov = hfov_deg
        self._use_rgb = use_rgb
        self._use_depth = use_depth
        self._use_semantic = use_semantic
        self._move_forward = move_forward
        self._turn_angle = turn_angle_deg
        self._sim = None
        self._current_scene: Optional[str] = None

    def get(self, scene_name: str):
        """返回 (sim, pathfinder) 对。"""
        import habitat_sim

        if self._current_scene == scene_name and self._sim is not None:
            # 重用已有实例
            self._sim.get_agent(0)  # 确保 agent 激活
            return self._sim, self._sim.pathfinder

        # 关闭旧实例
        self.close()

        scene_dir = self._scenes_root / scene_name
        glb_path = str(scene_dir / f"{scene_name}.glb")
        navmesh_path = str(scene_dir / f"{scene_name}.navmesh")

        if not os.path.exists(glb_path):
            raise FileNotFoundError(f"Scene GLB not found: {glb_path}")

        cfg = _make_sim_config(
            glb_path, navmesh_path,
            self._cam_w, self._cam_h, self._hfov,
            self._use_rgb, self._use_depth, self._use_semantic,
        )

        self._sim = habitat_sim.Simulator(cfg)
        self._current_scene = scene_name

        # 加载 navmesh
        if os.path.exists(navmesh_path):
            self._sim.pathfinder.load_nav_mesh(navmesh_path)
        else:
            print(f"  ⚠  Navmesh not found: {navmesh_path} — pathfinder unavailable",
                  file=sys.stderr)

        # 覆盖默认动作参数
        agent = self._sim.get_agent(0)
        agent_cfg = agent.agent_config
        if "move_forward" in agent_cfg.action_space:
            agent_cfg.action_space["move_forward"].actuation.amount = self._move_forward
        if "turn_left" in agent_cfg.action_space:
            agent_cfg.action_space["turn_left"].actuation.amount = self._turn_angle
        if "turn_right" in agent_cfg.action_space:
            agent_cfg.action_space["turn_right"].actuation.amount = self._turn_angle

        seed = int(time.time() * 1000) % (2**31)
        self._sim.seed(seed)

        return self._sim, self._sim.pathfinder

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None
            self._current_scene = None


# ═══════════════════════════════════════════════════════════════════════════
# 安全指标: Three-Zone Safety Tracker
# ═══════════════════════════════════════════════════════════════════════════

class _SafetyTracker:
    """记录 agent 轨迹，计算基于三区模型的安全指标。

    测量 agent 到场景中**所有仪器**物理表面的最近距离，不只是目标仪器。
    """

    def __init__(self, goal_xy: np.ndarray, instrument_facing: np.ndarray,
                 target_category: str, all_instruments: List[Tuple[np.ndarray, float]]):
        """Args:
            goal_xy: target goal position (for approach angle)
            instrument_facing: operational face direction of target (for approach angle)
            target_category: target instrument category
            all_instruments: list of (center_xy, radius_m) for ALL instruments in the scene
        """
        self._goal = goal_xy.copy()
        self._facing = instrument_facing / (np.linalg.norm(instrument_facing) or 1.0)
        self._all_insts = all_instruments  # [(center, radius), ...] for ALL scene instruments

        self._trajectory: List[np.ndarray] = []
        self._clearances: List[float] = []   # agent ↔ nearest instrument surface
        self._zone2_steps = 0
        self._total_steps = 0

    def _clearance(self, agent_xy: np.ndarray) -> float:
        """agent 到场景中任意仪器表面的最近距离 (负数 = 穿模)。"""
        min_d = float('inf')
        for center, radius in self._all_insts:
            d = float(np.linalg.norm(agent_xy - center)) - radius
            if d < min_d:
                min_d = d
        return max(0.0, min_d)

    def record(self, agent_xy: np.ndarray):
        self._trajectory.append(agent_xy.copy())
        c = self._clearance(agent_xy)
        self._clearances.append(c)
        self._total_steps += 1
        if c < ZONE2_DANGER:
            self._zone2_steps += 1

    def compute(self, final_xy: np.ndarray) -> dict:
        if not self._clearances:
            return self._empty_result()

        # ① 最小仪器间距 (Min Clearance to instrument surface)
        min_clear = round(float(np.min(self._clearances)), 3)

        # ② 最终接近角 (goal-relative, for operational face analysis)
        approach_vec = final_xy - self._goal
        approach_dist = float(np.linalg.norm(approach_vec))
        if approach_dist < 0.01:
            angle = 0.0
        else:
            approach_vec /= approach_dist
            dot = float(np.clip(np.dot(approach_vec, self._facing), -1, 1))
            angle = float(np.degrees(np.arccos(dot)))
        safe_approach = angle < SAFE_APPROACH_ANGLE

        # ③ Zone 2 闯入率 = 离仪器表面 <0.5m 的步数占比
        zone2_rate = round(self._zone2_steps / max(self._total_steps, 1), 4)

        # ④ 轨迹采样
        step = max(1, len(self._trajectory) // 50)
        traj_sample = [t.tolist() for t in self._trajectory[::step]]
        if len(self._trajectory) > 0:
            traj_sample.append(self._trajectory[-1].tolist())

        # ⑤ 危险检查
        dangerously_close = min_clear < ZONE2_DANGER

        return {
            "min_clearance_inst": min_clear,
            "approach_angle_deg": round(angle, 1),
            "safe_approach": safe_approach,
            "zone2_violation_rate": zone2_rate,
            "dangerously_close": dangerously_close,
            "trajectory_sample": traj_sample,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "min_clearance_inst": -1.0,
            "approach_angle_deg": 180.0,
            "safe_approach": False,
            "zone2_violation_rate": 0.0,
            "dangerously_close": True,
            "trajectory_sample": [],
        }


def _extract_facing_from_episode(ep: dict) -> np.ndarray:
    """从 episode 的 view_point 提取操作面朝向 (goal→instrument)。"""
    try:
        vp = ep["goals"][0].get("view_points")
        if vp and len(vp) > 0:
            rot = vp[0].get("agent_state", {}).get("rotation")
            if rot and len(rot) == 4:
                import magnum as mn
                q = mn.Quaternion(
                    mn.Vector3(float(rot[0]), float(rot[1]), float(rot[2])),
                    float(rot[3]))
                fwd = q.transform_vector(mn.Vector3(0, 0, -1))
                return np.array([fwd[0], fwd[2]], dtype=np.float64)
    except Exception:
        pass
    return np.array([0.0, -1.0], dtype=np.float64)


def _build_all_instruments(episodes: list) -> List[Tuple[np.ndarray, float, str]]:
    """从所有 episode 的 goal 反推场景中全部仪器的 (center_xy, radius, category)。
    用于安全指标：测 agent 到任意仪器表面的距离。
    """
    inst_map = {}  # category → (center_xy, radius, facing)
    for ep in episodes:
        goal = ep["goals"][0]
        cat = goal.get("object_category", ep.get("object_category", "?"))
        if cat in inst_map:
            continue
        goal_xy = np.array([goal["position"][0], goal["position"][2]], dtype=np.float64)
        facing = _extract_facing_from_episode(ep)
        fh, ap, hw = INSTRUMENT_SAFETY_PARAMS.get(cat, (0.5, 0.7, 0.5))
        fn = facing / (np.linalg.norm(facing) or 1.0)
        center = goal_xy - fn * (fh + ap)
        radius = max(fh, hw)
        inst_map[cat] = (center, radius, cat)
    return [(c, r) for c, r, _ in inst_map.values()]


# ═══════════════════════════════════════════════════════════════════════════
# 观测组装
# ═══════════════════════════════════════════════════════════════════════════

def _get_heading(rotation) -> np.ndarray:
    """从 quaternion 提取 x-z 平面朝向。"""
    import magnum as mn
    # habitat-sim 的 rotation 是 numpy-quaternion 类型 (有 .x .y .z .w 属性)
    try:
        rot = mn.Quaternion(
            mn.Vector3(float(rotation.x), float(rotation.y), float(rotation.z)),
            float(rotation.w))
    except Exception:
        rot = mn.Quaternion(list(rotation))
    forward = rot.transform_vector(mn.Vector3(0, 0, -1))
    return np.array([forward[0], forward[2]], dtype=np.float32)


def _build_observation(
    sim,
    sensor_data: dict,
    goal_position: np.ndarray,
    goal_category: str,
    step: int,
    episode_id: str,
    use_rgb: bool,
    use_depth: bool,
    use_semantic: bool,
) -> Dict[str, Any]:
    """从 sim state + sensor data 构建 observation dict。"""
    agent = sim.get_agent(0)
    state = agent.get_state()
    pos = state.position
    heading = _get_heading(state.rotation)

    obs: Dict[str, Any] = {
        "position":      np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32),
        "heading":       heading,
        "goal_position": goal_position.copy(),
        "goal_category": goal_category,
        "step":          step,
        "episode_id":    episode_id,
    }

    if use_rgb and "color_sensor" in sensor_data:
        rgba = sensor_data["color_sensor"]
        obs["rgb"] = rgba[..., :3].copy()  # RGBA→RGB, uint8

    if use_depth and "depth_sensor" in sensor_data:
        obs["depth"] = sensor_data["depth_sensor"].copy()

    if use_semantic and "semantic_sensor" in sensor_data:
        obs["semantic"] = sensor_data["semantic_sensor"].copy()

    return obs


def _get_initial_observation(
    sim, goal_position, goal_category, episode_id,
    use_rgb, use_depth, use_semantic,
) -> Dict[str, Any]:
    """获取 episode 初始观测（step=0 时的 sensor 数据）。"""
    try:
        sensor_data = sim.get_sensor_observations()
    except Exception:
        sensor_data = {}
    return _build_observation(
        sim, sensor_data, goal_position, goal_category,
        step=0, episode_id=episode_id,
        use_rgb=use_rgb, use_depth=use_depth, use_semantic=use_semantic,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 核心 API: evaluate()
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    policy: BioVLNPolicy,
    *,
    split: str = "dev",
    dataset: Optional[str] = None,
    scenes_root: Optional[str] = None,
    splits_dir: Optional[str] = None,
    max_episodes: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    move_forward: float = DEFAULT_FORWARD,
    turn_angle_deg: float = DEFAULT_TURN_DEG,
    camera_w: int = DEFAULT_CAMERA_W,
    camera_h: int = DEFAULT_CAMERA_H,
    hfov_deg: float = DEFAULT_HFOV,
    use_rgb: bool = False,
    use_depth: bool = False,
    use_semantic: bool = False,
    goal_radius: Optional[float] = None,
    verbose: bool = True,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    在指定 split 上评测一个 policy。

    Parameters
    ----------
    policy : BioVLNPolicy
        实现了 act() 的策略实例。
    split : str
        默认 "dev"。也支持 "train", "val", "test"。
    dataset : str or None
        "multi_scene" | "two_room" | "lsat_target"（从 registry 取路径）。
        与 scenes_root+splits_dir 互斥。
    scenes_root, splits_dir : str or None
        手动指定场景目录和 episodes 目录（不用 registry 时）。
    max_episodes : int or None
        限制评测 episode 数（None=全部）。
    max_steps : int
        每 episode 最大步数（默认 500）。
    move_forward : float
        前进步长 (m)。
    turn_angle_deg : float
        转弯角度 (度)。
    camera_w, camera_h, hfov_deg : int
        传感器参数（use_rgb/depth/semantic 为 True 时生效）。
    use_rgb, use_depth, use_semantic : bool
        是否启用对应传感器（会增加计算开销）。
    goal_radius : float or None
        统一成功半径 (m)。None=使用默认 per-category 半径。
    verbose : bool
        是否打印实时进度。
    output_file : str or None
        可选，将结果写入此 JSON 文件。

    Returns
    -------
    dict with keys:
        split, total, success, SR, SPL, avg_steps, avg_dtg_failed, elapsed,
        per_category, per_difficulty, per_episode
    """
    import habitat_sim

    # ── 解析路径 ───────────────────────────────────────────────────────
    if dataset is not None:
        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}")
        cfg = DATASETS[dataset]
        _scenes_root = Path(cfg["scenes_root"])
        _splits_dir = Path(cfg["splits_dir"])
    elif scenes_root and splits_dir:
        _scenes_root = Path(scenes_root)
        _splits_dir = Path(splits_dir)
    else:
        raise ValueError(
            "Either `dataset` or both `scenes_root`+`splits_dir` must be provided")

    # ── 加载 episodes ──────────────────────────────────────────────────
    ep_file = _splits_dir / f"episodes_{split}.json.gz"
    if not ep_file.exists():
        # 尝试 plain JSON
        ep_file = _splits_dir / f"episodes_{split}.json"
    if not ep_file.exists():
        raise FileNotFoundError(
            f"Episode file not found: {_splits_dir}/episodes_{split}.json.gz\n"
            f"Available files: {list(_splits_dir.glob('episodes_*'))}")

    with (gzip.open(ep_file, "rt", encoding="utf-8") if ep_file.suffix == ".gz"
          else open(ep_file, "r")) as f:
        data = json.load(f)

    episodes = data.get("episodes", data)  # 兼容顶层 list 或 keyed dict
    if not isinstance(episodes, list):
        raise ValueError(f"Unexpected episode file format (keys: {list(data.keys())})")

    if max_episodes:
        episodes = episodes[:max_episodes]

    total = len(episodes)
    if total == 0:
        print("No episodes loaded.", file=sys.stderr)
        return {"split": split, "total": 0, "SR": 0.0, "per_episode": []}

    # ── 预建全场景仪器列表 (for safety: clearance to ALL instruments) ──
    all_instruments = _build_all_instruments(episodes)

    # ── 按 scene 分组（减少 sim 切换） ──────────────────────────────────
    by_scene: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
    for i, ep in enumerate(episodes):
        scene = ep.get("scene_name", "unknown")
        by_scene[scene].append((i, ep))

    if verbose:
        ds_label = dataset or f"{_scenes_root}"
        print(f"  BioVLN evaluate — [{ds_label}] [{split.upper()}] "
              f"{total} episodes, {len(by_scene)} scenes")
        print(f"  Policy: {type(policy).__name__}")

    # ── 初始化 sim factory ─────────────────────────────────────────────
    sim_factory = _SimFactory(
        _scenes_root,
        camera_w=camera_w, camera_h=camera_h, hfov_deg=hfov_deg,
        use_rgb=use_rgb, use_depth=use_depth, use_semantic=use_semantic,
        move_forward=move_forward, turn_angle_deg=turn_angle_deg,
    )

    # ── 逐 scene 评测 ──────────────────────────────────────────────────
    all_results: List[dict] = []
    t_start = time.time()

    for scene_name, eps_in_scene in sorted(by_scene.items()):
        if verbose:
            print(f"  Scene: {scene_name} ({len(eps_in_scene)} episodes)", end="")
            sys.stdout.flush()

        try:
            sim, pf = sim_factory.get(scene_name)
        except Exception as e:
            # 场景载入失败 — 所有 episode 标记为失败
            if verbose:
                print(f"  ❌ failed to load: {e}")
            for _, ep in eps_in_scene:
                all_results.append(_make_error_result(ep, f"scene_load_failed: {e}"))
            continue

        for _, ep in eps_in_scene:
            result = _run_single_episode(
                ep, sim, pf, policy,
                max_steps=max_steps,
                goal_radius=goal_radius,
                use_rgb=use_rgb, use_depth=use_depth, use_semantic=use_semantic,
                all_instruments=all_instruments,
            )
            all_results.append(result)

        if verbose:
            successes = sum(1 for r in all_results[-len(eps_in_scene):] if r["success"])
            print(f"  ✓ {successes}/{len(eps_in_scene)}")

    sim_factory.close()
    elapsed = time.time() - t_start

    # ── 汇总指标 ───────────────────────────────────────────────────────
    metrics = _compute_metrics(all_results, split, elapsed)
    ds_label = dataset or f"{_scenes_root.name}"
    metrics["dataset"] = ds_label

    if verbose:
        _print_summary(metrics)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  Results saved → {out_path}")

    return metrics


# ─── Episode runner ─────────────────────────────────────────────────────────

def _run_single_episode(
    ep: dict,
    sim,
    pf,
    policy: BioVLNPolicy,
    max_steps: int,
    goal_radius: Optional[float],
    use_rgb: bool,
    use_depth: bool,
    use_semantic: bool,
    all_instruments: Optional[List[Tuple[np.ndarray, float]]] = None,
) -> dict:
    """运行单个 episode，返回结果 dict。"""
    import magnum as mn

    episode_id = str(ep.get("episode_id", "?"))
    scene_name = str(ep.get("scene_name", "?"))
    target_cat = str(ep.get("object_category", "?"))
    difficulty = ep.get("difficulty", "unknown")
    geo_dist = float(ep.get("info", {}).get("geodesic_distance", -1.0))

    start_pos = np.array(ep["start_position"], dtype=np.float64)
    start_rot = list(ep["start_rotation"])

    goal = ep["goals"][0]
    goal_pos = np.array(goal["position"], dtype=np.float64)
    goal_xy = np.array([goal_pos[0], goal_pos[2]], dtype=np.float64)

    # Safety: extract operational face direction from view_points
    instrument_facing = _extract_facing_from_episode(ep)

    threshold = goal_radius if goal_radius is not None \
        else DEFAULT_GOAL_RADII.get(target_cat, 1.0)

    # Optimal geodesic (for SPL)
    if geo_dist > 0:
        optimal = geo_dist
    else:
        optimal = float(np.linalg.norm(start_pos[[0, 2]] - goal_xy))
        if optimal < 1e-6:
            optimal = 1.0

    # Set agent state
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = mn.Vector3(float(start_pos[0]), float(start_pos[1]), float(start_pos[2]))
    state.rotation = list(start_rot)
    agent.set_state(state)

    # Initial observation + reset
    episode_info = {
        "episode_id": episode_id,
        "scene_name": scene_name,
        "target_category": target_cat,
        "start_position": start_pos.tolist(),
        "goal_position": goal_pos.tolist(),
        "difficulty": difficulty,
        "geodesic_distance": geo_dist,
    }

    init_obs = _get_initial_observation(
        sim, goal_xy, target_cat, episode_id,
        use_rgb, use_depth, use_semantic,
    )
    try:
        policy.reset(init_obs, episode_info)
    except Exception:
        pass  # reset 不应中断评测

    # ── 主循环 ──────────────────────────────────────────────────────────
    last_pos = np.array([start_pos[0], start_pos[2]], dtype=np.float64)
    path_length = 0.0
    insts = all_instruments if all_instruments else [(goal_xy, 10.0)]  # fallback: single huge instrument
    safety = _SafetyTracker(goal_xy, instrument_facing, target_cat, insts)
    safety.record(last_pos)  # record initial position

    for step in range(max_steps):
        st = agent.get_state()
        agent_xy = np.array([st.position[0], st.position[2]], dtype=np.float64)
        dist = float(np.linalg.norm(agent_xy - goal_xy))

        # Update path
        path_length += float(np.linalg.norm(agent_xy - last_pos))
        last_pos = agent_xy.copy()

        # Track safety
        safety.record(agent_xy)

        # ── 成功检查 ──────────────────────────────────────────────────
        if dist < threshold:
            spl = optimal / max(path_length, optimal, 1e-6)
            sm = safety.compute(agent_xy)
            return {
                "episode_id": episode_id,
                "scene_name": scene_name,
                "target": target_cat,
                "difficulty": difficulty,
                "success": True,
                "steps": step,
                "final_dist": round(dist, 2),
                "path_length": round(path_length, 2),
                "optimal_geo": round(optimal, 2),
                "SPL": round(spl, 4),
                **{f"safety_{k}": v for k, v in sm.items()},
            }

        # ── 获取观测 ──────────────────────────────────────────────────
        try:
            sensor_data = sim.get_sensor_observations()
        except Exception:
            sensor_data = {}
        observation = _build_observation(
            sim, sensor_data, goal_xy, target_cat,
            step, episode_id,
            use_rgb, use_depth, use_semantic,
        )

        # ── 策略决策 ──────────────────────────────────────────────────
        try:
            action = int(policy.act(observation))
        except Exception:
            action = Action.MOVE_FORWARD  # 出错就往前走

        if not (0 <= action <= 3):
            action = Action.MOVE_FORWARD

        # ── STOP ──────────────────────────────────────────────────────
        if action == Action.STOP:
            # Policy 主动停止 — 检查是否在目标范围内
            if dist < threshold:
                spl = optimal / max(path_length, optimal, 1e-6)
                sm = safety.compute(agent_xy)
                return {
                    "episode_id": episode_id,
                    "scene_name": scene_name,
                    "target": target_cat,
                    "difficulty": difficulty,
                    "success": True,
                    "steps": step,
                    "final_dist": round(dist, 2),
                    "path_length": round(path_length, 2),
                    "optimal_geo": round(optimal, 2),
                    "SPL": round(spl, 4),
                    **{f"safety_{k}": v for k, v in sm.items()},
                }
            else:
                # 过早停止
                sm = safety.compute(agent_xy)
                return {
                    "episode_id": episode_id, "SPL": 0.0,
                    "scene_name": scene_name, "target": target_cat,
                    "difficulty": difficulty,
                    "success": False, "steps": step,
                    "final_dist": round(dist, 2),
                    "path_length": round(path_length, 2),
                    "optimal_geo": round(optimal, 2),
                    "reason": "early_stop",
                    **{f"safety_{k}": v for k, v in sm.items()},
                }

        # ── 执行动作 ──────────────────────────────────────────────────
        act_name = _HABITAT_ACTION.get(action)
        if act_name:
            try:
                sim.step(act_name)
            except Exception:
                # 动作执行失败（如撞墙），继续下一步
                pass
        # action=STOP 但不在目标内，继续走

    # ── 超时 ───────────────────────────────────────────────────────────
    st = agent.get_state()
    final_dist = float(np.linalg.norm(
        np.array([st.position[0], st.position[2]]) - goal_xy))
    final_xy = np.array([st.position[0], st.position[2]], dtype=np.float64)
    sm = safety.compute(final_xy)
    return {
        "episode_id": episode_id, "SPL": 0.0,
        "scene_name": scene_name, "target": target_cat,
        "difficulty": difficulty,
        "success": False, "steps": max_steps,
        "final_dist": round(final_dist, 2),
        "path_length": round(path_length, 2),
        "optimal_geo": round(optimal, 2),
        "reason": "timeout",
        **{f"safety_{k}": v for k, v in sm.items()},
    }


def _make_error_result(ep: dict, reason: str) -> dict:
    return {
        "episode_id": str(ep.get("episode_id", "?")), "SPL": 0.0,
        "scene_name": str(ep.get("scene_name", "?")),
        "target": str(ep.get("object_category", "?")),
        "difficulty": ep.get("difficulty", "unknown"),
        "success": False, "steps": 0, "final_dist": -1.0,
        "path_length": 0.0, "optimal_geo": -1.0, "reason": reason,
    }


# ─── Metrics ────────────────────────────────────────────────────────────────

def _compute_metrics(results: List[dict], split: str, elapsed: float) -> dict:
    n = len(results)
    successes = [r for r in results if r["success"]]
    n_success = len(successes)

    # Per-category
    by_cat: Dict[str, dict] = {}
    for r in results:
        cat = r["target"]
        by_cat.setdefault(cat, {"total": 0, "success": 0})
        by_cat[cat]["total"] += 1
        if r["success"]:
            by_cat[cat]["success"] += 1
    per_category = {
        cat: {
            "total": v["total"],
            "success": v["success"],
            "SR": round(v["success"] / v["total"], 4) if v["total"] > 0 else 0.0,
        }
        for cat, v in sorted(by_cat.items())
    }

    # Per-difficulty
    by_diff: Dict[str, dict] = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, {"total": 0, "success": 0})
        by_diff[d]["total"] += 1
        if r["success"]:
            by_diff[d]["success"] += 1
    per_difficulty = {
        d: {
            "total": v["total"],
            "success": v["success"],
            "SR": round(v["success"] / v["total"], 4) if v["total"] > 0 else 0.0,
        }
        for d, v in sorted(by_diff.items())
    }

    # SPL
    spl_vals = [r.get("SPL", 0.0) for r in results]

    # DTG on failures
    fail_dtgs = [r.get("final_dist", 0.0) for r in results if not r["success"]]

    # Steps
    step_vals = [r.get("steps", 0) for r in results]

    # ── Safety aggregation ──────────────────────────────────────────────
    safety_keys = ["min_clearance_inst", "approach_angle_deg", "safe_approach",
                   "zone2_violation_rate", "dangerously_close"]

    def _avg(key): return np.mean([r.get(f"safety_{key}", 0) for r in results
                                    if r.get(f"safety_{key}", -1) >= 0])

    n_dangerous = sum(1 for r in results if r.get("safety_dangerously_close", False))

    safety_summary = {
        "avg_min_clearance": round(_avg("min_clearance_inst"), 3),
        "zone2_violation_rate": round(_avg("zone2_violation_rate"), 4),
        "dangerous_episodes": n_dangerous,
        "dangerous_rate": round(n_dangerous / n, 4) if n > 0 else 0.0,
    }

    # Safety per-category
    safety_per_cat = {}
    for r in results:
        cat = r["target"]
        if cat not in safety_per_cat:
            safety_per_cat[cat] = {
                "clearances": [], "zone2_violations": [], "dangerous": 0, "total": 0,
            }
        sc = safety_per_cat[cat]
        sc["total"] += 1
        c = r.get("safety_min_clearance_inst", -1)
        if c >= 0:
            sc["clearances"].append(c)
        v = r.get("safety_zone2_violation_rate", 0)
        if v >= 0:
            sc["zone2_violations"].append(v)
        if r.get("safety_dangerously_close", False):
            sc["dangerous"] += 1

    safety_per_cat_summary = {}
    for cat, v in sorted(safety_per_cat.items()):
        safety_per_cat_summary[cat] = {
            "avg_min_clearance": round(np.mean(v["clearances"]), 3) if v["clearances"] else 0,
            "zone2_violation_rate": round(np.mean(v["zone2_violations"]), 4) if v["zone2_violations"] else 0,
            "dangerous_rate": round(v["dangerous"] / v["total"], 4) if v["total"] else 0,
        }

    return {
        "split": split,
        "total": n,
        "success": n_success,
        "SR": round(n_success / n, 4) if n > 0 else 0.0,
        "SPL": round(float(np.mean(spl_vals)), 4),
        "avg_steps": round(float(np.mean(step_vals)), 1),
        "avg_dtg_failed": round(float(np.mean(fail_dtgs)), 2) if fail_dtgs else 0.0,
        "elapsed": round(elapsed, 1),
        "per_category": per_category,
        "per_difficulty": per_difficulty,
        "per_episode": results,
        "safety_summary": safety_summary,
        "safety_per_category": safety_per_cat_summary,
    }


def _print_summary(metrics: dict):
    print(f"\n{'='*60}")
    print(f"  [{metrics['split'].upper()}]  {metrics.get('dataset', '')}")
    print(f"  SR:  {metrics['SR']:.1%}  ({metrics['success']}/{metrics['total']})")
    print(f"  SPL: {metrics['SPL']:.4f}  |  avg steps: {metrics['avg_steps']}")
    print(f"  avg DTG (failed): {metrics['avg_dtg_failed']}m  |  elapsed: {metrics['elapsed']:.1f}s")

    # Safety summary
    ss = metrics.get("safety_summary", {})
    if ss:
        print(f"\n  ═══ Safety (Instrument Clearance, Zone 2 = {ZONE2_DANGER}m) ═══")
        print(f"  MCR (Min Clearance Radius): {ss['avg_min_clearance']:.3f}m  "
              f"(dangerous <{ZONE2_DANGER}m: {ss['dangerous_episodes']}/{metrics['total']} = {ss['dangerous_rate']:.1%})")
        print(f"  VRT (Violation Rate, Threshold={ZONE2_DANGER}m): {ss['zone2_violation_rate']:.1%}")

        spc = metrics.get("safety_per_category", {})
        if spc:
            print(f"\n  Per-category safety:")
            print(f"  {'Category':<20s} {'Clearance':>8s} {'Zone2%':>8s} {'Danger%':>8s}")
            for cat, v in spc.items():
                print(f"  {cat:<20s} {v['avg_min_clearance']:>7.3f}m "
                      f"{v['zone2_violation_rate']:>7.1%} "
                      f"{v['dangerous_rate']:>7.1%}")

    if metrics.get("per_category"):
        print(f"\n  Per-category SR:")
        for cat, v in metrics["per_category"].items():
            bar = "█" * int(v["SR"] * 20)
            print(f"    {cat:<20} {v['SR']:.1%}  {bar}  ({v['success']}/{v['total']})")

    if metrics.get("per_difficulty"):
        ordered = ["easy", "medium", "hard", "unknown"]
        present = [d for d in ordered if d in metrics["per_difficulty"]]
        if present:
            print(f"\n  Per-difficulty SR:")
            for d in present:
                v = metrics["per_difficulty"][d]
                print(f"    {d:<10} {v['SR']:.1%}  ({v['success']}/{v['total']})")

    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _cli():
    import argparse

    p = argparse.ArgumentParser(
        description="BioVLN Evaluation — run a policy on a split")
    p.add_argument("--split", default="dev",
                   choices=["dev", "val", "train", "test"])
    p.add_argument("--dataset", default="multi_scene",
                   choices=["multi_scene", "two_room", "lsat_target"])
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Limit number of episodes (default: all)")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--output", type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument("--policy", type=str, default="forward",
                   choices=["forward", "random"],
                   help="Built-in policy to run")
    p.add_argument("--rgb", action="store_true", help="Enable RGB sensor")
    p.add_argument("--depth", action="store_true", help="Enable depth sensor")
    p.add_argument("--semantic", action="store_true", help="Enable semantic sensor")
    p.add_argument("--quiet", action="store_true", help="Suppress per-episode output")

    args = p.parse_args()

    policy_map = {
        "forward": ForwardOnlyPolicy(),
        "random": RandomPolicy(),
    }
    policy = policy_map[args.policy]

    evaluate(
        policy,
        split=args.split,
        dataset=args.dataset,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        use_rgb=args.rgb,
        use_depth=args.depth,
        use_semantic=args.semantic,
        verbose=not args.quiet,
        output_file=args.output,
    )


if __name__ == "__main__":
    _cli()
