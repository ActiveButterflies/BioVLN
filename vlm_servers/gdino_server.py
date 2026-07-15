#!/usr/bin/env python3
"""GroundingDINO HTTP server (port 12181) — HuggingFace transformers.
Compatible with run_vlfm.py gdino_detect().
Usage: python gdino_server.py [--port 12181]
"""
# Use isolated newer transformers (LAVIS needs older version in main env)
import sys
sys.path.insert(0, "/tmp/hf_new")

import io, json, traceback, base64
import numpy as np
import cv2
import torch
from PIL import Image
from flask import Flask, request, jsonify

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

print(f"[GDINO] Loading {MODEL_ID} on {DEVICE}...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print(f"[GDINO] Ready on port 12181", flush=True)


@app.route("/gdino", methods=["POST"])
def detect():
    try:
        # Support both JSON (base64) and multipart form uploads
        if request.is_json:
            data = request.get_json()
            img_b64 = data.get("image", "")
            # Remove data:image prefix if present
            if "," in str(img_b64):
                img_b64 = str(img_b64).split(",", 1)[1]
            img_bytes = base64.b64decode(img_b64)
            text_prompt = data.get("caption", data.get("text", data.get("prompt", "object")))
        elif "image" in request.files:
            img_bytes = request.files["image"].read()
            text_prompt = request.form.get("text", request.form.get("prompt", "object"))
        else:
            return jsonify({"error": "no image"}), 400

        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        h, w = img_np.shape[:2]

        inputs = processor(images=Image.fromarray(img_rgb), text=text_prompt,
                          return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            box_threshold=0.3, text_threshold=0.25,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu()   # [N, 4] in [x1,y1,x2,y2] (corner format)
        scores = results.get("scores", torch.ones(len(boxes)))
        labels = results.get("labels", ["object"] * len(boxes))

        # HF returns corner format [x1,y1,x2,y2]. Convert to normalized [x1,y1,x2,y2]
        # (VLFM client expects corner format, not center+size)
        boxes_norm = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            boxes_norm.append([x1 / w, y1 / h, x2 / w, y2 / h])

        result = {
            "boxes": json.dumps(boxes_norm),
            "phrases": json.dumps(labels),
            "logits": json.dumps(scores.tolist() if hasattr(scores, 'tolist') else list(scores)),
        }
        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, threaded=False)
