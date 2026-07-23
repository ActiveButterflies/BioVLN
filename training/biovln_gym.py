#!/usr/bin/env python3
"""
BioVLN Gym Environment — 标准 RL 训练接口
══════════════════════════════════════════════════════

符合 OpenAI Gym / Gymnasium 规范，可直接对接 Stable-Baselines3、
Ray RLlib、CleanRL 等训练框架。

用法:
    from biovln_gym import BioVLNGym

    env = BioVLNGym(dataset="multi_scene", split="train", use_rgb=True)
    obs, info = env.reset()
    for _ in range(500):
        action = model.predict(obs)         # 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()

设计原则:
  - 复用 biovln_eval 内部的 _SimFactory、_build_observation、_SafetyTracker
  - 不修改 biovln_eval.py
  - 连续训练时按 episode 循环 reset() → step() × N
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 导入内部实现（不暴露给用户）
from biovln_eval import (
    Action, DATASETS, DEFAULT_GOAL_RADII,
    _SimFactory, _build_observation, _get_initial_observation,
    _SafetyTracker, _extract_facing_from_episode,
    _build_all_instruments, ZONE2_DANGER,
)


# ═══════════════════════════════════════════════════════════════════════════
# BioVLN Gym Environment
# ═══════════════════════════════════════════════════════════════════════════

class BioVLNGym:
    """
    BioVLN 标准强化学习环境。

    观测空间 (observation):
        rgb         np.ndarray (3, H, W) float32 [0,1]  — 仅当 use_rgb=True
        depth       np.ndarray (1, H, W) float32         — 仅当 use_depth=True
        goal_vec    np.ndarray (4,)      float32         — agent-centric 目标向量

    动作空间:
        0 = STOP    (episode 结束)
        1 = FORWARD (0.25m)
        2 = TURN_LEFT  (10°)
        3 = TURN_RIGHT (10°)

    Parameters
    ----------
    dataset : str
        "multi_scene" | "two_room" | "lsat_target"
    split : str
        Episode 划分: "dev", "val", "test", "train"
    use_rgb, use_depth : bool
        是否返回图像观测
    max_steps : int
        Episode 最大步数 (默认 500)
    reward_scale : float
        奖励缩放系数 (默认 1.0)
    seed : int or None
        环境随机种子
    """

    # ─── Gym 标准属性 ───
    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: str = "multi_scene",
        split: str = "dev",
        use_rgb: bool = False,
        use_depth: bool = False,
        max_steps: int = 500,
        reward_scale: float = 1.0,
        seed: Optional[int] = None,
    ):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}")

        cfg = DATASETS[dataset]
        self._dataset = dataset
        self._split = split
        self._scenes_root = Path(cfg["scenes_root"])
        self._splits_dir = Path(cfg["splits_dir"])
        self._use_rgb = use_rgb
        self._use_depth = use_depth
        self._max_steps = max_steps
        self._reward_scale = reward_scale

        # Episode 随机种子
        self._rng = np.random.RandomState(seed)

        # 加载 episodes
        self._episodes = self._load_episodes()

        # 预建全场景仪器列表 (safety)
        self._all_instruments = _build_all_instruments(self._episodes)

        # Sim 工厂
        self._sim_factory = _SimFactory(
            self._scenes_root,
            use_rgb=use_rgb,
            use_depth=use_depth,
            use_semantic=False,
            move_forward=0.25,
            turn_angle_deg=10.0,
        )

        # 当前 episode 状态
        self._sim = None
        self._pf = None
        self._current_scene: Optional[str] = None
        self._ep_idx: int = 0
        self._step_count: int = 0
        self._episode_info: dict = {}

        # 轨迹累积
        self._path_length: float = 0.0
        self._last_xy: np.ndarray = np.zeros(2)
        self._goal_xy: np.ndarray = np.zeros(2)
        self._goal_category: str = ""
        self._goal_radius: float = 1.0
        self._optimal: float = 1.0
        self._safety: Any = None

        # 观测缓存
        self._last_rgb: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None

    # ─── Episode 加载 ────────────────────────────────────────────────────

    def _load_episodes(self) -> list:
        ep_file = self._splits_dir / f"episodes_{self._split}.json.gz"
        if not ep_file.exists():
            ep_file = self._splits_dir / f"episodes_{self._split}.json"

        if not ep_file.exists():
            raise FileNotFoundError(
                f"Episode file not found: {self._splits_dir}/episodes_{self._split}.json.gz\n"
                f"Available: {list(self._splits_dir.glob('episodes_*'))}")

        with (gzip.open(ep_file, "rt", encoding="utf-8") if ep_file.suffix == ".gz"
              else open(ep_file, "r")) as f:
            data = json.load(f)

        episodes = data.get("episodes", data)
        if not isinstance(episodes, list):
            raise ValueError(f"Unexpected episode file format")
        self._rng.shuffle(episodes)  # 初始洗牌
        return episodes

    # ─── 观测空间 (动态) ────────────────────────────────────────────────

    @property
    def observation_space(self):
        """返回当前环境的观测空间描述 (不依赖 gymnasium)。"""
        obs_dict = {
            "goal_vec": "Box(4,) float32",  # [dx_local, dz_local, dist_norm, progress]
            "position": "Box(2,) float32",  # agent XZ (归一化)
            "heading": "Box(2,) float32",    # heading (dx, dz)
        }
        if self._use_rgb:
            obs_dict["rgb"] = "Box(3, 240, 320) float32 [0,1]"
        if self._use_depth:
            obs_dict["depth"] = "Box(1, 240, 320) float32"
        return obs_dict

    @property
    def action_space(self):
        return "Discrete(4)"  # STOP, FORWARD, TURN_LEFT, TURN_RIGHT

    # ══════════════════════════════════════════════════════════════════════
    # 核心 API: reset / step
    # ══════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """开始一个新的 episode。

        Returns:
            observation: dict with keys goal_vec, position, heading, (+ rgb, depth)
            info: dict with episode metadata
        """
        import magnum as mn

        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # 循环取下一个 episode
        ep = self._episodes[self._ep_idx % len(self._episodes)]
        self._ep_idx += 1

        self._episode_info = ep
        episode_id = str(ep.get("episode_id", self._ep_idx))
        scene_name = str(ep.get("scene_name", "?"))
        target_cat = str(ep.get("object_category", "?"))
        geo_dist = float(ep.get("info", {}).get("geodesic_distance", -1.0))

        start_pos = np.array(ep["start_position"], dtype=np.float64)
        start_rot = list(ep["start_rotation"])

        goal = ep["goals"][0]
        goal_pos = np.array(goal["position"], dtype=np.float64)
        self._goal_xy = np.array([goal_pos[0], goal_pos[2]], dtype=np.float64)
        self._goal_category = target_cat
        self._goal_radius = DEFAULT_GOAL_RADII.get(target_cat, 1.0)

        # 最优路径长度 (for 奖励计算)
        if geo_dist > 0:
            self._optimal = geo_dist
        else:
            self._optimal = float(np.linalg.norm(start_pos[[0, 2]] - self._goal_xy))
            if self._optimal < 1e-6:
                self._optimal = 1.0

        # 加载场景
        try:
            self._sim, self._pf = self._sim_factory.get(scene_name)
        except Exception as e:
            # 场景加载失败 → 返回全零观测
            return self._empty_obs(), {"error": str(e), "episode_id": episode_id}

        self._current_scene = scene_name

        # 设置 agent 初始状态
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        state.position = mn.Vector3(float(start_pos[0]), float(start_pos[1]), float(start_pos[2]))
        state.rotation = list(start_rot)
        agent.set_state(state)

        # 初始观测
        init_obs = _get_initial_observation(
            self._sim, self._goal_xy, target_cat, episode_id,
            use_rgb=self._use_rgb, use_depth=self._use_depth, use_semantic=False,
        )

        # Safety tracker
        instrument_facing = _extract_facing_from_episode(ep)
        self._safety = _SafetyTracker(self._goal_xy, instrument_facing, target_cat,
                                      self._all_instruments)

        # 轨迹累积
        self._step_count = 0
        self._path_length = 0.0
        self._last_xy = np.array([start_pos[0], start_pos[2]], dtype=np.float64)
        self._safety.record(self._last_xy)

        # 缓存图像
        self._last_rgb = init_obs.get("rgb")
        self._last_depth = init_obs.get("depth")

        info = {
            "episode_id": episode_id,
            "scene_name": scene_name,
            "target_category": target_cat,
            "difficulty": ep.get("difficulty", "unknown"),
            "start_position": start_pos.tolist(),
            "goal_position": goal_pos.tolist(),
        }

        return self._format_obs(init_obs), info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """执行一个动作。

        Returns:
            observation, reward, terminated, truncated, info
        """
        import magnum as mn

        # ── 动作验证 ──
        if not (0 <= action <= 3):
            action = Action.MOVE_FORWARD
        self._step_count += 1

        # ── 执行 Habitat 动作 ──
        act_name = {0: None, 1: "move_forward", 2: "turn_left", 3: "turn_right"}.get(action)
        if act_name:
            try:
                self._sim.step(act_name)
            except Exception:
                pass  # 碰撞等 → 停留在原位

        # ── 获取新状态 ──
        agent = self._sim.get_agent(0)
        st = agent.get_state()
        agent_xy = np.array([st.position[0], st.position[2]], dtype=np.float64)

        # 更新路径
        self._path_length += float(np.linalg.norm(agent_xy - self._last_xy))
        self._last_xy = agent_xy.copy()

        # Safety tracking
        self._safety.record(agent_xy)

        # 到目标的距离
        dist = float(np.linalg.norm(agent_xy - self._goal_xy))

        # 获取传感器数据
        try:
            sensor_data = self._sim.get_sensor_observations()
        except Exception:
            sensor_data = {}

        obs = _build_observation(
            self._sim, sensor_data, self._goal_xy, self._goal_category,
            step=self._step_count, episode_id=str(self._ep_idx),
            use_rgb=self._use_rgb, use_depth=self._use_depth, use_semantic=False,
        )

        # ── 奖励 ──
        reward = self._compute_reward(action, dist, agent_xy)

        # ── 终止判断 ──
        terminated = (action == Action.STOP and dist < self._goal_radius)
        truncated = (self._step_count >= self._max_steps)

        # ── 信息 ──
        info = {
            "dist_to_goal": round(dist, 2),
            "path_length": round(self._path_length, 2),
            "step": self._step_count,
        }

        if terminated:
            info["success"] = True
            info["reason"] = "goal_reached"
            info["spl"] = round(self._optimal / max(self._path_length, self._optimal, 1e-6), 4)
            sm = self._safety.compute(agent_xy)
            for k, v in sm.items():
                info[f"safety_{k}"] = v
            # 成功奖励 (减掉最后一步的距离误差作为 bonus)
            reward += self._reward_scale * 10.0

        if truncated and not terminated:
            info["success"] = False
            info["reason"] = "timeout"
            sm = self._safety.compute(agent_xy)
            for k, v in sm.items():
                info[f"safety_{k}"] = v

        return self._format_obs(obs), float(reward), terminated, truncated, info

    # ─── 奖励函数 ────────────────────────────────────────────────────────

    def _compute_reward(self, action: int, dist: float, agent_xy: np.ndarray) -> float:
        """计算增量奖励。

        奖励设计 (dense + sparse):
          + 接近目标:  delta_dist < 0 → 正奖励
          + 安全惩罚:  进入 Zone 2 → 负奖励
          - 步数惩罚:  -0.01 per step (鼓励快速到达)
          - 撞墙惩罚:  agent_xy ≈ last_xy 且前进动作 → 负奖励
        """
        scale = self._reward_scale
        reward = 0.0

        # 步数惩罚
        reward -= 0.01 * scale

        # 距离减少奖励 (dense reward)
        prev_dist = float(np.linalg.norm(self._last_xy - self._goal_xy))
        delta = prev_dist - dist
        reward += 2.0 * scale * delta

        # 安全惩罚 (Zone 2)
        clearance = self._safety._clearance(agent_xy)
        if clearance < ZONE2_DANGER and ZONE2_DANGER > 0:
            reward -= 1.0 * scale * (1.0 - clearance / ZONE2_DANGER)

        # 撞墙惩罚
        if action == Action.MOVE_FORWARD and prev_dist - dist < 0.01:
            reward -= 0.5 * scale

        return reward

    # ─── 观测格式化 ──────────────────────────────────────────────────────

    def _format_obs(self, raw_obs: dict) -> Dict[str, np.ndarray]:
        """将 biovln_eval 的内部 observation 转为 RL 训练格式。"""
        agent_pos = raw_obs.get("position", np.zeros(3))[:2]
        heading = raw_obs.get("heading", np.array([1.0, 0.0]))
        goal_pos = raw_obs.get("goal_position", np.zeros(2))

        # Agent-centric 目标向量
        gx, gz = float(goal_pos[0]), float(goal_pos[1])
        ax, az = float(agent_pos[0]), float(agent_pos[1])

        dx_global = gx - ax
        dz_global = gz - az
        dist_global = max(float(np.sqrt(dx_global**2 + dz_global**2)), 1e-6)
        dx_global /= dist_global
        dz_global /= dist_global

        # 旋转到 agent 局部坐标系 ← heading = 前方
        hx, hz = float(heading[0]), float(heading[1])
        dx_local = hx * dx_global + hz * dz_global
        dz_local = -hz * dx_global + hx * dz_global

        obs = {
            "goal_vec": np.array([dx_local, dz_local,
                                  dist_global,
                                  float(np.clip(dist_global / 5.0, 0.0, 1.0))
                                  ], dtype=np.float32),
            "position": agent_pos.astype(np.float32) / 4.0,  # 归一化到 [-1,1]
            "heading":  heading.astype(np.float32),
        }

        if self._use_rgb:
            rgb = raw_obs.get("rgb")
            if rgb is not None:
                obs["rgb"] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # (C,H,W)
            elif self._last_rgb is not None:
                obs["rgb"] = self._last_rgb  # 首帧缓存

        if self._use_depth:
            depth = raw_obs.get("depth")
            if depth is not None:
                obs["depth"] = depth[np.newaxis, ...].astype(np.float32)  # (1,H,W)
            elif self._last_depth is not None:
                obs["depth"] = self._last_depth

        return obs

    def _empty_obs(self) -> Dict[str, np.ndarray]:
        """场景加载失败时的零观测。"""
        obs = {
            "goal_vec": np.zeros(4, dtype=np.float32),
            "position": np.zeros(2, dtype=np.float32),
            "heading":  np.array([1.0, 0.0], dtype=np.float32),
        }
        if self._use_rgb:
            obs["rgb"] = np.zeros((3, 240, 320), dtype=np.float32)
        if self._use_depth:
            obs["depth"] = np.zeros((1, 240, 320), dtype=np.float32)
        return obs

    # ─── 生命周期 ────────────────────────────────────────────────────────

    def close(self):
        if self._sim_factory is not None:
            self._sim_factory.close()

    def seed(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        return [seed]

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Stable-Baselines3 兼容包装函数
# ═══════════════════════════════════════════════════════════════════════════

def make_biovln_env(
    dataset: str = "multi_scene",
    split: str = "dev",
    use_rgb: bool = False,
    use_depth: bool = False,
    max_steps: int = 500,
    reward_scale: float = 1.0,
    seed: int = 0,
) -> BioVLNGym:
    """创建 BioVLN Gym 环境的工厂函数, 兼容 SB3 的 make_vec_env 接口。

    Usage:
        from stable_baselines3 import PPO
        from biovln_gym import make_biovln_env

        env = make_biovln_env(dataset="multi_scene", split="dev", use_rgb=True)
        model = PPO("CnnPolicy", env, verbose=1)
        model.learn(total_timesteps=100_000)
    """
    return BioVLNGym(
        dataset=dataset, split=split,
        use_rgb=use_rgb, use_depth=use_depth,
        max_steps=max_steps, reward_scale=reward_scale,
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Demo: 快速验证环境
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Smoke test: 跑 3 个 episode 检查接口。
    Usage:
        python biovln_gym.py
    """
    print("BioVLN Gym Environment — Smoke Test")
    print("=" * 50)

    env = BioVLNGym(dataset="lsat_target", split="dev", use_rgb=False)

    for ep in range(3):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        for s in range(500):
            # 随机策略
            action = np.random.randint(0, 4)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print(f"  Ep {ep}: steps={steps} reward={total_reward:.2f} "
              f"success={info.get('success', False)} dist={info.get('dist_to_goal', -1):.2f}m")

    env.close()
    print("  ✅ All episodes completed")
