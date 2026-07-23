#!/usr/bin/env python3
"""
BioVLN Trajectory Dataset — 将 recorded 轨迹转为 PyTorch Dataset
══════════════════════════════════════════════════════════════

用于行为克隆 (Behavioral Cloning) 训练:
  - 输入: 每步的 observation (RGB/Depth + goal vector)
  - 标签: action (0=STOP, 1=FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT)

用法:
    from trajectory_dataset import TrajectoryDataset

    ds = TrajectoryDataset("/data/biovln_trajectories/run_xxx", split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    for batch in loader:
        rgb = batch["rgb"]           # (B, 3, H, W) float32 [0,1]
        goal_vec = batch["goal_vec"] # (B, 4)  (dx, dz, sin_heading, cos_heading)
        actions = batch["action"]    # (B,)  long [0-3]
        # 训练...
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # type: ignore


# ─── 坐标归一化常量 ──────────────────────────────────────────────────────

# 房间尺寸: 8m × 6m (single_room), 中心在原点 → position range ≈ [-4,4] × [-3,3]
POS_SCALE = 4.0  # 归一化到 [-1, 1]


# ═══════════════════════════════════════════════════════════════════════════
# 轻量级数据加载器 (不依赖 torch，可在 CLI 直接使用)
# ═══════════════════════════════════════════════════════════════════════════

def _load_episodes(data_dir: str) -> List[dict]:
    """加载目录下所有 jsonl.gz 文件，返回所有 episode 的 step 列表。"""
    root = Path(data_dir)
    steps: List[dict] = []
    for gz_file in sorted(root.glob("trajectories_*.jsonl.gz")):
        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            for line in f:
                ep = json.loads(line)
                traj = ep.get("trajectory", [])
                ep_info = ep.get("episode_info", {})
                for step in traj:
                    # 跳过 STOP 动作（无意义）
                    if step.get("action") == 0:
                        continue
                    steps.append({
                        **step,
                        "episode_id": ep.get("episode_id", ""),
                        "scene_name": ep_info.get("scene_name", ""),
                        "target_category": ep_info.get("target_category", ""),
                    })
    return steps


def _normalize_position(pos: List[float]) -> np.ndarray:
    """将全局坐标归一化到 [-1, 1]。"""
    return np.array(pos[:2], dtype=np.float32) / POS_SCALE


def _compute_goal_vector(goal_pos: List[float], agent_pos: List[float],
                         agent_heading: List[float]) -> np.ndarray:
    """计算 agent-centric 目标向量。

    Returns:
        np.ndarray shape (4,): [dx, dz, sin(goal_heading), cos(goal_heading)]
        其中 dx, dz 是目标在 agent 局部坐标系中的方向。
    """
    gx, gz = goal_pos[0], goal_pos[1]
    ax, az = agent_pos[0], agent_pos[1]
    hx, hz = agent_heading[0], agent_heading[1]

    # 全局偏移
    dx_global, dz_global = gx - ax, gz - az
    dist = max(float(np.sqrt(dx_global**2 + dz_global**2)), 1e-6)
    dx_global /= dist
    dz_global /= dist

    # 旋转到 agent 局部坐标系
    dx_local = hx * dx_global + hz * dz_global
    dz_local = -hz * dx_global + hx * dz_global

    return np.array([dx_local, dz_local, dist, float(np.clip(dist / 5.0, 0.0, 1.0))],
                    dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class TrajectoryDataset(Dataset):
        """
        从 BioVLN 录制数据中读取 (observation, action) 对，供行为克隆训练。

        Parameters
        ----------
        data_dir : str
            包含 trajectories_*.jsonl.gz 和 images/ 的目录。
        use_rgb : bool
            True 时从 images/ 目录读取 PNG 并转为 tensor。
            False 时只返回 goal_vector（纯特征训练，适合几何方法）。
        pos_noise : float
            位置噪声标准差 (m)，数据增强用。默认为 0（无噪声）。
        success_only : bool
            True 时仅加载成功 episode 的步（高质量教学数据）。
        """

        def __init__(
            self,
            data_dir: str,
            use_rgb: bool = False,
            pos_noise: float = 0.0,
            success_only: bool = False,
        ):
            self._root = Path(data_dir)
            self._use_rgb = use_rgb
            self._pos_noise = pos_noise
            self._steps = _load_episodes(data_dir)

            if success_only:
                self._steps = [s for s in self._steps
                               if s.get("success", False)]

            if len(self._steps) == 0:
                raise ValueError(f"No training steps found in {data_dir}")

            n_eps = len(set(s.get("episode_id", "") for s in self._steps))
            print(f"  [Dataset] {len(self._steps)} steps from {n_eps} episodes")

        def __len__(self):
            return len(self._steps)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            step = self._steps[idx]
            obs = step.get("observation", {})

            # ── 位置归一化 ──
            pos = _normalize_position(obs.get("position", [0, 0, 0]))
            if self._pos_noise > 0:
                pos += np.random.normal(0, self._pos_noise / POS_SCALE, size=2)

            # ── 目标向量 ──
            goal_vec = _compute_goal_vector(
                obs.get("goal_position", [0, 0]),
                obs.get("position", [0, 0, 0]),
                obs.get("heading", [1, 0]),
            )

            # ── 局部朝向 ──
            heading = np.array(obs.get("heading", [1, 0]), dtype=np.float32)

            sample: Dict[str, torch.Tensor] = {
                "position":   torch.from_numpy(pos),
                "heading":    torch.from_numpy(heading),
                "goal_vec":   torch.from_numpy(goal_vec),
                "action":     torch.tensor(step.get("action", 1), dtype=torch.long),
                "dist":       torch.tensor(step.get("dist_to_goal", 0.0)),
                "category":   step.get("target_category", ""),
                "episode_id": step.get("episode_id", ""),
            }

            # ── RGB ──
            if self._use_rgb:
                rgb_path = obs.get("rgb_path")
                if rgb_path:
                    from PIL import Image as PILImage
                    img = PILImage.open(str(self._root / rgb_path)).convert("RGB")
                    rgb_tensor = torch.from_numpy(
                        np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                    )
                    sample["rgb"] = rgb_tensor

            return sample


else:

    class TrajectoryDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TrajectoryDataset requires PyTorch. "
                "Install with: pip install torch"
            )


# ═══════════════════════════════════════════════════════════════════════════
# CLI: 快速预览数据集
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python trajectory_dataset.py /data/biovln_trajectories/run_xxx")
        print("\nPrints summary of the recorded dataset.\n")
        sys.exit(1)

    data_dir = sys.argv[1]
    steps = _load_episodes(data_dir)

    # 统计
    eps = {}
    for s in steps:
        eid = s.get("episode_id", "?")
        eps.setdefault(eid, []).append(s)

    actions = [s["action"] for s in steps]
    dists = [s.get("dist_to_goal", 0) for s in steps]
    obs_presents = [1 if s.get("observation", {}).get("position") else 0 for s in steps]

    print(f"Data directory: {data_dir}")
    print(f"  Total episodes:  {len(eps)}")
    print(f"  Total steps:     {len(steps)}")
    print(f"  Avg steps/ep:    {len(steps)/max(len(eps),1):.1f}")
    print(f"  Action dist:     FORWARD={actions.count(1)} L={actions.count(2)} R={actions.count(3)} STOP={actions.count(0)}")
    print(f"  Dist to goal:    min={min(dists):.2f} max={max(dists):.2f} mean={np.mean(dists):.2f}")
    print(f"  Has observation: {sum(obs_presents)}/{len(steps)}")
    print(f"  Image dir:       {'present' if (Path(data_dir)/'images').exists() else 'NOT FOUND'}")
