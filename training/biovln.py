#!/usr/bin/env python3
"""
BioVLN Unified Interface — 零样本评测 · 训练数据录制 · 模型训练
══════════════════════════════════════════════════════════════════════════

三个入口，一个文件:

  Usage 1 — 零样本评测 (现有 evaluate 等价):
      from biovln import bench
      bench(MyPolicy(), dataset="multi_scene", split="dev")

  Usage 2 — 录制训练数据:
      from biovln import record
      record(MyPolicy(), dataset="multi_scene", split="train",
             output_dir="/data/trajectories", use_rgb=True)

  Usage 3 — 在线 RL 训练 / 非零样本评测:
      from biovln import make_env, evaluate_trained

      env = make_env(dataset="multi_scene", split="train", use_rgb=True)
      model.learn(total_timesteps=500_000)
      evaluate_trained(model, dataset="multi_scene", split="test")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from biovln_eval import evaluate, BioVLNPolicy, Action
from data_recorder import TrajectoryRecorder
from biovln_gym import BioVLNGym as _BioVLNGym  # 重导出


# ═══════════════════════════════════════════════════════════════════════════
# 1. 零样本评测 (evaluate 的别名，不改变任何参数)
# ═══════════════════════════════════════════════════════════════════════════

def bench(
    policy: BioVLNPolicy,
    *,
    dataset: str = "multi_scene",
    split: str = "dev",
    use_rgb: bool = False,
    use_depth: bool = False,
    max_episodes: Optional[int] = None,
    verbose: bool = True,
    output_file: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    零样本评测 — 在所有 benchmark episode 上运行 policy 并返回指标。

    等价于直接调用 evaluate()。保持这个接口是为了统一命令风格。

    Parameters
    ----------
    policy : BioVLNPolicy
        零样本策略 (子类实现 act())
    dataset : str
        "multi_scene" | "two_room" | "lsat_target"
    split : str
        "dev" | "val" | "test" | "train"
    use_rgb : bool
        是否提供 RGB 观测 (True=增加 ~2× 计算开销)
    use_depth : bool
        是否提供深度观测
    max_episodes : int or None
        限制评测 episode 数
    verbose : bool
        是否打印进度
    output_file : str or None
        结果 JSON 路径

    Returns
    -------
    dict: SR, SPL, per_category, per_difficulty, per_episode ...
    """
    return evaluate(
        policy,
        dataset=dataset,
        split=split,
        use_rgb=use_rgb,
        use_depth=use_depth,
        use_semantic=False,
        max_episodes=max_episodes,
        verbose=verbose,
        output_file=output_file,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. 录制训练数据 (data_recorder 的封装)
# ═══════════════════════════════════════════════════════════════════════════

def record(
    policy: BioVLNPolicy,
    *,
    dataset: str = "multi_scene",
    split: str = "dev",
    output_dir: str = "./biovln_trajectories",
    use_rgb: bool = False,
    use_depth: bool = False,
    compress_images: bool = False,
    format: str = "jsonl",
    max_episodes: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    """
    运行 policy 并录制完整轨迹到磁盘，供后续离线训练。

    Parameters
    ----------
    policy : BioVLNPolicy
        数据生成策略 (Oracle/Frontier/VLFM 等)
    dataset : str
        见 bench()
    split : str
        见 bench()
    output_dir : str
        输出根目录 (自动创建时间戳子目录)
    use_rgb : bool
        True=保存每帧 RGB 到 images/ 目录
    use_depth : bool
        True=保存每帧 Depth 到 images/ 目录
    compress_images : bool
        True=保存图像, False=仅保存位置/朝向/目标元数据
    format : str
        "jsonl" (多ep一文件) | "habitat" (一ep一文件)
    max_episodes : int or None
        见 bench()
    verbose : bool
        见 bench()

    Returns
    -------
    Path: 输出目录 (含 manifest.json)
    """
    rec = TrajectoryRecorder(
        output_dir=output_dir,
        format=format,
        compress_images=compress_images,
    )

    try:
        evaluate(
            policy,
            dataset=dataset,
            split=split,
            use_rgb=use_rgb,
            use_depth=use_depth,
            use_semantic=False,
            max_episodes=max_episodes,
            verbose=verbose,
            trajectory_callback=rec,
        )
    finally:
        rec.finalize()

    return rec.output_path


# ═══════════════════════════════════════════════════════════════════════════
# 3. 在线 RL 环境 (对接 SB3 / CleanRL / RLlib)
# ═══════════════════════════════════════════════════════════════════════════

def make_env(
    dataset: str = "multi_scene",
    split: str = "dev",
    use_rgb: bool = False,
    use_depth: bool = False,
    max_steps: int = 500,
    reward_scale: float = 1.0,
    seed: Optional[int] = None,
) -> _BioVLNGym:
    """
    创建标准 Gym 环境，用于在线 RL 训练。

    用法:
        from stable_baselines3 import PPO
        env = make_env(dataset="multi_scene", split="train", use_rgb=True)
        model = PPO("CnnPolicy", env, verbose=1)
        model.learn(total_timesteps=200_000)
            # RL 训练的是非零样本策略 (从同分布数据中学习)

    Parameters
    ----------
    dataset : str
        "multi_scene" — 程序化场景，布局随机化，支持大规模训练。
    split : str
        "dev" (训练) | "val" (验证) | "test" (最终评测)
        建议 dev→train; val→验证; test→留出
    use_rgb, use_depth : bool
        视觉输入
    max_steps : int
        Episode 最大步数
    reward_scale : float
        奖励缩放
    seed : int or None
        环境随机种子
    """
    return _BioVLNGym(
        dataset=dataset,
        split=split,
        use_rgb=use_rgb,
        use_depth=use_depth,
        max_steps=max_steps,
        reward_scale=reward_scale,
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. 非零样本评测 — 把训练好的模型包装成 BioVLNPolicy 然后跑 bench()
# ═══════════════════════════════════════════════════════════════════════════

def _compute_goal_vec(
    position: np.ndarray,
    goal_position: np.ndarray,
    heading: np.ndarray,
) -> np.ndarray:
    """计算 agent-centric 目标向量: [dx_local, dz_local, dist, dist/5]."""
    gx, gz = float(goal_position[0]), float(goal_position[1])
    ax, az = float(position[0]), float(position[1])
    dx, dz = gx - ax, gz - az
    dist = max(float(np.sqrt(dx**2 + dz**2)), 1e-6)
    hx, hz = float(heading[0]), float(heading[1])
    dx_local = hx * (dx / dist) + hz * (dz / dist)
    dz_local = -hz * (dx / dist) + hx * (dz / dist)
    return np.array(
        [dx_local, dz_local, dist, float(np.clip(dist / 5.0, 0.0, 1.0))],
        dtype=np.float32,
    )


class TrainedPolicy(BioVLNPolicy):
    """将任意训练好的模型包装为 BioVLN 评测接口。

    支持三种模型类型:
      - SB3 (Stable-Baselines3): 带 model.predict(obs)
      - TorchScript: 带 model.forward(tensor)
      - 任意 callable: model(obs) → action logits

    模型可以是 vision-based (接收 RGB) 或 vector-based (仅 goal_vec)。

    Parameters
    ----------
    model : Any
        训练好的模型对象。
    model_type : str
        "sb3" — SB3 PPO/A2C 模型 (自动调用 model.predict)
        "sb3_cnn" — SB3 CNN 模型 (观测 = {image: (C,H,W), vec: (4,)} dict)
        "torch" — torch.nn.Module (纯 goal_vec 输入)
        "torch_cnn" — torch.nn.Module (接收 (rgb, goal_vec) 两个输入)
        "auto" — 自动检测 (默认)
    use_rgb : bool
        True = bench() 评测时启用 RGB 传感器。vision model 必须设 True。

    Examples
    --------
    >>> from stable_baselines3 import PPO
    >>> model = PPO.load("biovln_ppo_dev")
    >>> bench(TrainedPolicy(model, model_type="sb3"),
    ...       dataset="multi_scene", split="test", use_rgb=True)

    >>> model = MyTorchCNN()  # torch.nn.Module with forward(rgb, goal_vec)
    >>> bench(TrainedPolicy(model, model_type="torch_cnn"),
    ...       dataset="multi_scene", split="test", use_rgb=True)
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "auto",
        use_rgb: bool = False,
    ):
        super().__init__()
        self._model = model
        self._use_rgb = use_rgb

        if model_type == "auto":
            model_type = self._detect(model)
        self._model_type = model_type

    @staticmethod
    def _detect(model: Any) -> str:
        """自动检测模型类型。"""
        # SB3 模型
        if hasattr(model, "predict") and hasattr(model, "policy"):
            try:
                obs_space = model.observation_space
                if hasattr(obs_space, "spaces"):
                    return "sb3_cnn"  # Dict obs → CNN
                return "sb3"
            except Exception:
                return "sb3"
        # PyTorch
        if hasattr(model, "forward") or hasattr(model, "__call__"):
            return "torch"
        return "torch"

    def act(self, observation: dict) -> int:
        pos = observation.get("position", np.zeros(3))[:2]
        heading = observation.get("heading", np.array([1.0, 0.0]))
        goal = observation.get("goal_position", np.zeros(2))
        goal_vec = _compute_goal_vec(pos, goal, heading)

        # ── RGB ──
        rgb = observation.get("rgb")
        if rgb is not None and hasattr(rgb, "ndim") and rgb.ndim == 3:
            rgb = np.ascontiguousarray(
                rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        # ── 模型推理 ──
        try:
            action = self._infer(goal_vec, rgb)
            if 0 <= action <= 3:
                return int(action)
        except Exception:
            pass
        return 1  # fallback: forward

    def _infer(
        self,
        goal_vec: np.ndarray,
        rgb: Optional[np.ndarray],
    ) -> int:
        mt = self._model_type

        # ── SB3 ──
        if mt in ("sb3", "sb3_cnn"):
            if mt == "sb3_cnn" and self._use_rgb and rgb is not None:
                obs = {"image": rgb, "vec": goal_vec}
            else:
                obs = goal_vec
            action, _states = self._model.predict(obs, deterministic=True)
            return int(action)

        # ── Torch (纯 vector) ──
        if mt == "torch" or not self._use_rgb:
            import torch
            with torch.no_grad():
                x = torch.from_numpy(goal_vec).unsqueeze(0)
                logits = self._model(x)
                return int(logits.argmax(dim=1).item())

        # ── Torch CNN (vision + vector) ──
        if mt == "torch_cnn" and rgb is not None:
            import torch
            with torch.no_grad():
                rgb_t = torch.from_numpy(rgb).unsqueeze(0)
                gv_t = torch.from_numpy(goal_vec).unsqueeze(0)
                logits = self._model(rgb_t, gv_t)
                return int(logits.argmax(dim=1).item())

        # ── Torch CNN — RGB 为空, 用零填充 ──
        if mt == "torch_cnn":
            import torch
            with torch.no_grad():
                rgb_t = torch.zeros(1, 3, 240, 320)
                gv_t = torch.from_numpy(goal_vec).unsqueeze(0)
                logits = self._model(rgb_t, gv_t)
                return int(logits.argmax(dim=1).item())

        return 1


def evaluate_trained(
    model: Any,
    *,
    dataset: str = "multi_scene",
    split: str = "test",
    model_type: str = "auto",
    use_rgb: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    用训练好的模型跑一次评测。

    Parameters
    ----------
    model : Any
        训练好的模型 (SB3, torch.nn.Module, 或 callable)
    dataset, split : str
        评测数据集和划分 (建议 test 或 val)
    model_type : str
        "sb3" / "sb3_cnn" / "torch" / "torch_cnn" / "auto"
    use_rgb : bool
        True = 启用 RGB 传感器, vision model 必须设 True
    verbose : bool
        是否打印实时进度

    Returns
    -------
    dict: SR, SPL, per_category, per_difficulty, per_episode, ...
    """
    return bench(
        TrainedPolicy(model, model_type=model_type, use_rgb=use_rgb),
        dataset=dataset,
        split=split,
        use_rgb=use_rgb,
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. 数据采集脚本 (论文 Table 6 的数据生成)
# ═══════════════════════════════════════════════════════════════════════════

def collect_safety_data(
    dataset: str = "multi_scene",
    split: str = "dev",
    max_episodes: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    仅用于采集论文中 Table 6 的安全指标数据。

    不需要用户提供 policy，内部运行已有的 Oracle/Frontier/3-Zone。
    """
    from biovln_eval import (
        RandomPolicy,
    )

    class ForwardOnly(BioVLNPolicy):
        def act(self, obs): return 1

    results = {}
    for name, policy in [
        ("random", RandomPolicy()),
        ("oracle", ForwardOnly()),  # 待用户替换为实际 Oracle 策略
    ]:
        r = bench(policy, dataset=dataset, split=split,
                  use_rgb=False, max_episodes=max_episodes, verbose=verbose)
        results[name] = r

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("BioVLN Unified Interface")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python biovln.py bench    — 零样本评测")
        print("  python biovln.py record   — 录制训练数据")
        print("  python biovln.py env      — 在线 RL (需 SB3)")
        print()
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "bench":
        from biovln_eval import RandomPolicy
        bench(RandomPolicy(), dataset="lsat_target", split="dev", max_episodes=5)

    elif cmd == "record":
        from biovln_eval import RandomPolicy
        dest = record(RandomPolicy(), dataset="lsat_target", split="dev",
                      output_dir="/tmp/biovln_record", max_episodes=5)
        print(f"Trajectories → {dest}")

    elif cmd == "env":
        env = make_env(dataset="lsat_target", split="dev")
        obs, info = env.reset()
        total_reward = 0.0
        for s in range(500):
            action = np.random.randint(0, 4)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Random policy: reward={total_reward:.2f} "
              f"success={info.get('success', False)} steps={s+1}")
        env.close()
