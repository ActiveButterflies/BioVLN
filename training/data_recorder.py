#!/usr/bin/env python3
"""
BioVLN Trajectory Data Recorder — 将评测过程保存为训练数据
══════════════════════════════════════════════════════════════

非侵入式数据采集：通过回调机制在 evaluate() 中插入，不修改现有代码。

用法:
    from biovln_eval import evaluate
    from data_recorder import TrajectoryRecorder

    rec = TrajectoryRecorder(output_dir="/path/to/data", format="habitat_rl")
    results = evaluate(MyPolicy(), split="dev", trajectory_callback=rec)
    rec.finalize()  # 冲刷缓冲区、写入 manifest

设计原则：
  - 不修改 biovln_eval.py 中任何一行
  - 支持多种输出格式（Habitat RL / RLDS-like / 简洁 JSONL）
  - 增量写入，避免内存溢出
  - 支持 RGB/Depth/Semantic 图像（可选压缩）
"""

from __future__ import annotations

import gzip
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ─── Array / JSON 编码工具 ────────────────────────────────────────────────

def _ndarray_to_list(obj):
    """递归将 ndarray 转为 list，确保 JSON 可序列化。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(v) for v in obj]
    return obj


def _encode_image(arr: np.ndarray, fmt: str = "png") -> bytes:
    """将 (H,W,3) uint8 或 (H,W) float32 numpy 数组编码为指定格式的二进制。"""
    import io
    if fmt == "npy":
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        buf.seek(0)
        return buf.read()
    # PNG encoding
    from PIL import Image as PILImage
    if arr.ndim == 2:
        # Depth: normalize to 0-255
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin < 1e-6:
            arr_norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr_norm = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        img = PILImage.fromarray(arr_norm, mode="L")
    else:
        img = PILImage.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════
# 回调协议（在 evaluate 侧调用）
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryCallback(ABC):
    """数据记录回调基类。

    在与 biovln_eval.evaluate() 集成时，按以下顺序调用：
      on_episode_start → (on_step ...) × N → on_episode_end → on_episode_start ...
    """

    @abstractmethod
    def on_episode_start(self, episode_id: str, episode_info: dict) -> None:
        ...

    @abstractmethod
    def on_step(self, step: int, observation: dict, action: int, agent_xy: np.ndarray,
                dist_to_goal: float, safety: Optional[dict] = None) -> None:
        ...

    @abstractmethod
    def on_episode_end(self, episode_id: str, success: bool, result: dict) -> None:
        ...

    def finalize(self) -> None:
        """评测全部结束后调用，用于冲刷缓冲区和写入元数据。"""
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 具体实现：保存训练数据到磁盘
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryRecorder(TrajectoryCallback):
    """
    将每条 episode 的完整轨迹保存到磁盘。

    Parameters
    ----------
    output_dir : str
        数据输出目录。自动按时间戳创建子目录。
    format : str
        "jsonl"    — 每条 episode 一个 .jsonl.gz 文件，每行一个 {obs, action} 对。
        "habitat"  — 每条 episode 一个 .json.gz 文件，兼容 Habitat RL 训练格式。
    compress_images : bool
        True 时将 RGB/Depth 图像编码为 PNG 并 Base64 内嵌（体积大但完整）。
        False 时仅保存图像尺寸元数据，不存储像素（减少数据体积，适合几何方法）。
    max_episodes_per_file : int
        JSONL 模式下，每 N 个 episode 写入一个新文件。
    """

    def __init__(
        self,
        output_dir: str = "./biovln_trajectories",
        format: str = "jsonl",
        compress_images: bool = False,
        max_episodes_per_file: int = 100,
    ):
        assert format in ("jsonl", "habitat"), f"Unknown format: {format}"
        self._format = format
        self._compress_images = compress_images
        self._max_eps_per_file = max_episodes_per_file

        # 按时间戳创建输出子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._root = Path(output_dir) / f"run_{timestamp}"
        self._root.mkdir(parents=True, exist_ok=True)

        # 元数据
        self._manifest: List[dict] = []
        self._split: Optional[str] = None
        self._dataset_name: Optional[str] = None

        # 当前 episode 缓冲区
        self._ep_id: Optional[str] = None
        self._steps: List[dict] = []
        self._ep_info: dict = {}

        # JSONL 聚合缓冲区
        self._ep_buffer: List[Tuple[str, dict, List[dict]]] = []
        self._file_counter = 0

    # ── 元数据设置（在 evaluate 中调用） ──

    def set_metadata(self, split: str, dataset_name: str):
        self._split = split
        self._dataset_name = dataset_name

    # ── 回调方法 ──

    def on_episode_start(self, episode_id: str, episode_info: dict) -> None:
        self._ep_id = episode_id
        self._steps = []
        self._ep_info = _ndarray_to_list(episode_info)

    def on_step(self, step: int, observation: dict, action: int,
                agent_xy: np.ndarray, dist_to_goal: float,
                safety: Optional[dict] = None) -> None:

        step_data: Dict[str, Any] = {
            "step": step,
            "action": int(action),
            "position": agent_xy.tolist(),
            "dist_to_goal": round(float(dist_to_goal), 4),
        }

        # 观测元数据（始终记录）
        obs_serialized: Dict[str, Any] = {
            "position": observation.get("position", []).tolist()
            if isinstance(observation.get("position"), np.ndarray) else [],
            "heading": observation.get("heading", []).tolist()
            if isinstance(observation.get("heading"), np.ndarray) else [],
            "goal_position": observation.get("goal_position", []).tolist()
            if isinstance(observation.get("goal_position"), np.ndarray) else [],
            "goal_category": observation.get("goal_category", ""),
        }

        # 图像数据（仅在 compress_images=True 时保存）
        if self._compress_images:
            for key in ("rgb", "depth", "semantic"):
                arr = observation.get(key)
                if arr is not None and isinstance(arr, np.ndarray):
                    fmt = "png"
                    raw = _encode_image(arr, fmt=fmt)
                    img_dir = self._root / "images" / (self._ep_id or "unknown")
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img_path = img_dir / f"step_{step:04d}_{key}.{fmt}"
                    with open(img_path, "wb") as f:
                        f.write(raw)
                    obs_serialized[f"{key}_path"] = str(img_path.relative_to(self._root))
                    obs_serialized[f"{key}_shape"] = list(arr.shape)
                elif arr is not None and isinstance(arr, np.ndarray):
                    obs_serialized[f"{key}_shape"] = list(arr.shape)

        step_data["observation"] = obs_serialized

        if safety:
            step_data["safety"] = safety

        self._steps.append(step_data)

    def on_episode_end(self, episode_id: str, success: bool, result: dict) -> None:
        if not self._steps:
            return

        # 打包 episode 数据
        ep_data = {
            "episode_id": episode_id,
            "episode_info": self._ep_info,
            "success": success,
            "result": {k: v for k, v in _ndarray_to_list(result).items()
                       if not k.startswith("safety_")},
            "num_steps": len(self._steps),
            "trajectory": self._steps,
        }

        if self._format == "jsonl":
            self._ep_buffer.append((episode_id, ep_data, self._steps))
            # 达到阈值写入文件
            if len(self._ep_buffer) >= self._max_eps_per_file:
                self._flush_buffer()
            # 记录 manifest
            self._manifest.append({
                "episode_id": episode_id,
                "scene_name": self._ep_info.get("scene_name", ""),
                "target_category": self._ep_info.get("target_category", ""),
                "difficulty": self._ep_info.get("difficulty", ""),
                "success": success,
                "num_steps": len(self._steps),
            })
        else:
            # habitat 格式：单文件
            if self._compress_images:
                # Habitat RL 格式：含图像的完整 episode
                out_path = self._root / "habitat" / f"{episode_id}.json.gz"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(out_path, "wt", encoding="utf-8") as f:
                    json.dump(ep_data, f, ensure_ascii=False)
            else:
                out_path = self._root / "habitat" / f"{episode_id}.json.gz"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(out_path, "wt", encoding="utf-8") as f:
                    # 不含像素数据
                    json.dump(_strip_images(ep_data), f, ensure_ascii=False)

            self._manifest.append({
                "episode_id": episode_id,
                "file": str(out_path.relative_to(self._root)),
                "scene_name": self._ep_info.get("scene_name", ""),
                "target_category": self._ep_info.get("target_category", ""),
                "difficulty": self._ep_info.get("difficulty", ""),
                "success": success,
                "num_steps": len(self._steps),
            })

        self._ep_id = None
        self._steps = []
        self._ep_info = {}

    # ── JSONL 缓冲区管理 ──

    def _flush_buffer(self):
        if not self._ep_buffer:
            return

        fname = f"trajectories_{self._split or 'unlabeled'}_{self._file_counter:04d}.jsonl.gz"
        out_path = self._root / fname
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            for _eid, ep_data, _ in self._ep_buffer:
                if not self._compress_images:
                    ep_data = _strip_images(ep_data)
                f.write(json.dumps(ep_data, ensure_ascii=False) + "\n")

        self._file_counter += 1
        self._ep_buffer = []

    def finalize(self) -> None:
        """写入剩余缓冲区和 manifest。"""
        if self._format == "jsonl" and self._ep_buffer:
            self._flush_buffer()

        # 写入 manifest
        manifest_path = self._root / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "split": self._split,
                "dataset": self._dataset_name,
                "num_episodes": len(self._manifest),
                "format": self._format,
                "compress_images": self._compress_images,
                "created_at": datetime.now().isoformat(),
                "episodes": self._manifest,
            }, f, indent=2, ensure_ascii=False)

        print(f"  [Recorder] Saved {len(self._manifest)} episode trajectories")
        print(f"  [Recorder] Output: {self._root.resolve()}")

    @property
    def output_path(self) -> Path:
        return self._root


# ─── 辅助函数 ───────────────────────────────────────────────────────────

def _strip_images(ep_data: dict) -> dict:
    """仅移除 trajectory 中的图像路径/尺寸字段，保留 position/heading/goal 等元数据。"""
    d = ep_data.copy()
    for step in d.get("trajectory", []):
        obs = step.get("observation")
        if obs:
            # 只删图像相关字段，保留 position/heading/goal_position/goal_category
            for key in list(obs.keys()):
                if key.endswith("_path") or key.endswith("_shape"):
                    del obs[key]
    return d


# ═══════════════════════════════════════════════════════════════════════════
# 便捷工厂函数
# ═══════════════════════════════════════════════════════════════════════════

def make_recorder(
    output_dir: str = "./biovln_trajectories",
    format: str = "jsonl",
    compress_images: bool = False,
) -> TrajectoryRecorder:
    """创建 TrajectoryRecorder 的快捷方法。

    Examples
    --------
    >>> rec = make_recorder("/data/biovln_train")
    >>> results = evaluate(policy, split="train",
    ...                    trajectory_callback=rec, use_rgb=True)
    >>> rec.finalize()
    """
    return TrajectoryRecorder(
        output_dir=output_dir,
        format=format,
        compress_images=compress_images,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Demo: 将任意 evaluate() 调用变成训练数据生成
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    快速演示：在 LSAT Target 上跑 Random 策略并记录轨迹。

    Usage:
        python data_recorder.py
    """
    from biovln_eval import BioVLNPolicy, evaluate

    class DemoPolicy(BioVLNPolicy):
        def act(self, obs):
            return 1  # 一直向前走

    rec = TrajectoryRecorder(output_dir="/tmp/biovln_demo", format="jsonl")
    try:
        results = evaluate(
            DemoPolicy(), split="dev", dataset="lsat_target",
            trajectory_callback=rec,
            max_episodes=5, use_rgb=False, use_depth=True,
        )
        print(f"\nSR={results['SR']:.1%}, SPL={results['SPL']:.3f}")
    finally:
        rec.finalize()
