#!/usr/bin/env python3
"""
Patch-level BLIP-2 scoring → spatial value map projection.
Uses HuggingFace transformers (not LAVIS) — supports HF_ENDPOINT mirror.

Drop-in replacement for biovln/patch_value_map.py.
Exports: update_value_map_patches(), extract_patches(), scores_to_heatmap(),
         project_heatmap_to_grid(), get_blip2_model()
"""
import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

MODEL_ID = "Salesforce/blip2-itm-vit-g"

# ─── Global lazy-loaded model ────────────────────────────────────────────
_model = None
_processor = None
_device = None


def get_blip2_model():
    """Lazy-load BLIP-2 HF model. Returns (model, processor, device)."""
    global _model, _processor, _device
    if _model is None:
        from transformers import Blip2Processor, Blip2ForImageTextRetrieval
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BLIP2-HF] Loading {MODEL_ID} on {_device}...", flush=True)
        _processor = Blip2Processor.from_pretrained(MODEL_ID)
        _model = Blip2ForImageTextRetrieval.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        ).to(_device)
        _model.eval()
        print(f"[BLIP2-HF] Ready", flush=True)
    return _model, _processor, _device


# ─── Patch extraction (unchanged from original) ──────────────────────────
def extract_patches(
    rgb: np.ndarray, window: int = 128, stride: int = 64
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    H, W = rgb.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    patch_list = []
    for y in range(0, H - window + 1, stride):
        for x in range(0, W - window + 1, stride):
            x2, y2 = x + window, y + window
            boxes.append((x, y, x2, y2))
            patch_list.append(rgb[y:y2, x:x2])
    patches = np.stack(patch_list, axis=0)
    return boxes, patches


# ─── Batched HF BLIP-2 scoring ───────────────────────────────────────────
def compute_patch_scores(
    model,
    processor,
    patches: np.ndarray,
    prompt: str,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Batched ITM scoring. model=Blip2ForImageTextRetrieval, processor=Blip2Processor."""
    N = patches.shape[0]
    all_scores = []
    for i in range(0, N, batch_size):
        batch = patches[i: i + batch_size]
        imgs = [Image.fromarray(p) for p in batch]
        inputs = processor(images=imgs, text=[prompt] * len(imgs),
                          return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, use_itm_head=True)
        scores = torch.sigmoid(out.itm_score).cpu().numpy().ravel()
        all_scores.append(scores)
    return np.concatenate(all_scores).astype(np.float32)


# ─── Score → heatmap ─────────────────────────────────────────────────────
def scores_to_heatmap(
    boxes: List[Tuple[int, int, int, int]],
    scores: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    H, W = image_shape
    heatmap = np.full((H, W), -np.inf, dtype=np.float32)
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        np.maximum(heatmap[y1:y2, x1:x2], s, out=heatmap[y1:y2, x1:x2])
    heatmap = np.where(np.isinf(heatmap), 0.0, heatmap)
    return heatmap


# ─── Heatmap → 2D value map grid ─────────────────────────────────────────
def project_heatmap_to_grid(
    heatmap: np.ndarray,
    depth: np.ndarray,
    value_map,
    agent_x: float,
    agent_y: float,
    fwd_x: float,
    fwd_y: float,
    hfov_deg: float = 79.0,
    depth_min: float = 0.5,
    depth_max: float = 5.0,
) -> int:
    H, W = depth.shape
    fx = fy = W / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    cx, cy = W // 2, H // 2

    right_x, right_y = -fwd_y, fwd_x
    vv, uu = np.mgrid[0:H, 0:W]
    valid = (depth > depth_min) & (depth < depth_max)

    Z = depth[valid]
    X = (uu[valid] - cx) * Z / fx
    Y = -(vv[valid] - cy) * Z / fy

    wx_arr = agent_x + X * right_x + Z * fwd_x
    wy_arr = agent_y + X * right_y + Z * fwd_y
    scores = heatmap[valid]

    gx_arr = np.clip((wx_arr / value_map.res).astype(np.int32) + value_map.origin,
                      0, value_map.size - 1)
    gy_arr = np.clip((wy_arr / value_map.res).astype(np.int32) + value_map.origin,
                      0, value_map.size - 1)

    flat_idx = gx_arr * value_map.size + gy_arr
    np.maximum.at(value_map.data.ravel(), flat_idx, scores)
    return len(scores)


# ─── High-level convenience ──────────────────────────────────────────────
def update_value_map_patches(
    model,
    processor,
    rgb: np.ndarray,
    depth: np.ndarray,
    target_prompt: str,
    agent_x: float,
    agent_y: float,
    fwd_x: float,
    fwd_y: float,
    value_map,
    device: torch.device,
    window: int = 128,
    stride: int = 64,
) -> dict:
    t0 = time.time()
    boxes, patches = extract_patches(rgb, window=window, stride=stride)
    t1 = time.time()
    scores = compute_patch_scores(model, processor, patches, target_prompt, device)
    t2 = time.time()
    heatmap = scores_to_heatmap(boxes, scores, rgb.shape[:2])
    t3 = time.time()
    n = project_heatmap_to_grid(heatmap, depth, value_map, agent_x, agent_y, fwd_x, fwd_y)
    t4 = time.time()
    return {
        "n_patches": len(boxes),
        "n_projected": n,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "t_extract": t1 - t0,
        "t_infer": t2 - t1,
        "t_heatmap": t3 - t2,
        "t_project": t4 - t3,
        "t_total": t4 - t0,
    }
