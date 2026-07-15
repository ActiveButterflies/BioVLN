#!/usr/bin/env python3
"""BLIP-2 ITM HTTP server (port 12182) — LAVIS + downloaded weights.
Compatible with run_vlfm.py blip2_cosine().

Requires LAVIS (salesforce-lavis) and the two downloaded files:
  ~/.cache/torch/hub/checkpoints/eva_vit_g.pth
  ~/.cache/torch/hub/checkpoints/blip2_pretrained.pth
Usage: python blip_server.py [--port 12182]
"""
import json, traceback, base64, io
import numpy as np
import cv2
import torch
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[BLIP2] Loading LAVIS BLIP-2 ITM on {DEVICE}...", flush=True)
from lavis.models import load_model_and_preprocess
model, vis_processor, _ = load_model_and_preprocess(
    "blip2_image_text_matching", "pretrain", is_eval=True, device=DEVICE)
print(f"[BLIP2] Ready on port 12182", flush=True)


@app.route("/blip2itm", methods=["POST"])
def blip2itm():
    """Full-image ITM score. Returns {"response": float} for run_vlfm.py compat."""
    try:
        data = request.get_json() or {}
        img_b64 = data.get("image", "")
        text_prompt = data.get("txt", "object")

        if not img_b64:
            return jsonify({"response": 0.0})

        img_bytes = base64.b64decode(img_b64)
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        # LAVIS expects a preprocessed image tensor
        img_tensor = vis_processor["eval"](image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model({"image": img_tensor, "text_input": [text_prompt]},
                       match_head="itm")
        score = float(out.cpu().item())

        return jsonify({"response": score})

    except Exception:
        traceback.print_exc()
        return jsonify({"response": 0.0})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, threaded=False)
