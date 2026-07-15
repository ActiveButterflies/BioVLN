#!/usr/bin/env python3
"""MobileSAM HTTP server (port 13183) — uses uploaded checkpoint.
Compatible with run_vlfm.py sam_segment().
Usage: python sam_server.py [--port 13183]
"""
import io, json, traceback, base64
import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify

from mobile_sam import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "/root/gpufree-data/data/mobile_sam.pt"

app = Flask(__name__)

print(f"[SAM] Loading MobileSAM from {CHECKPOINT} on {DEVICE}...", flush=True)
sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
sam.eval()
predictor = SamPredictor(sam)
print(f"[SAM] Ready on port 13183", flush=True)


@app.route("/mobile_sam", methods=["POST"])
def mobile_sam():
    """Segment an object in image at bbox.
    Expects JSON: {"image": base64_jpg, "bbox": [x1, y1, x2, y2]}
    Returns: {"cropped_mask": base64_mask_bytes}
    """
    try:
        data = request.get_json() or {}
        img_b64 = data.get("image", "")
        bbox = data.get("bbox", [0, 0, 100, 100])

        if not img_b64:
            return jsonify({"cropped_mask": ""}), 400

        img_bytes = base64.b64decode(img_b64)
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return jsonify({"cropped_mask": ""})

        predictor.set_image(img_rgb)
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=input_box[None, :], multimask_output=False)

        if masks is None or masks.shape[0] == 0:
            return jsonify({"cropped_mask": ""})

        mask = masks[0].astype(np.uint8)
        # Encode mask as base64 of raw bytes (matches run_vlfm.py decode)
        mask_b64 = base64.b64encode(mask.tobytes()).decode("utf-8")
        return jsonify({"cropped_mask": mask_b64})

    except Exception:
        traceback.print_exc()
        return jsonify({"cropped_mask": ""})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13183)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, threaded=False)
