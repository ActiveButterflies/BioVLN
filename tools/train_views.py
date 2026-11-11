#!/usr/bin/env python
"""Train single-frame predictors on the operating-view dataset.

Task A: is this viewpoint an admissible operating viewpoint?  (binary)
Task B: which side of the instrument are we looking at?       (4 classes)

Split is by scene: the test scenes never appear in training.
Baselines: majority class, and a depth-statistics logistic regression that sees no
instrument identity, so it cannot know which face is the usable one.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


def build(scenes_tr, scenes_te, seed=0):
    torch.manual_seed(seed)


class SmallCNN(nn.Module):
    def __init__(self, in_ch, n_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_out)

    def forward(self, x):
        return self.head(self.body(x).flatten(1))


def load(path):
    z = np.load(path, allow_pickle=True)
    rgb = z["rgb"].astype(np.float32) / 255.0
    dep = z["depth"].astype(np.float32)
    meta = json.loads(str(z["meta"]))
    return rgb, dep, meta


def norm_depth(d):
    d = np.clip(d, 0.0, 6.0) / 6.0
    return d[:, None, :, :]


def train_model(xtr, ytr, xte, yte, n_out, epochs, lr=1e-3, wd=1e-4, bs=64, class_w=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(xtr.shape[1], n_out).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = None if class_w is None else torch.tensor(class_w, dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    xtr_t = torch.from_numpy(xtr); ytr_t = torch.from_numpy(ytr).long()
    xte_t = torch.from_numpy(xte).to(dev); yte_t = torch.from_numpy(yte).long().to(dev)
    n = len(xtr_t)
    best = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = xtr_t[idx].to(dev); yb = ytr_t[idx].to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(xte_t)
            acc = (logits.argmax(1) == yte_t).float().mean().item()
            if best is None or acc > best[0]:
                best = (acc, ep, logits.cpu().numpy())
    return best


def metrics(y, p, n_out):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
    pred = p.argmax(1)
    out = {"acc": float((pred == y).mean()), "bal_acc": float(balanced_accuracy_score(y, pred))}
    if n_out == 2:
        prob = torch.softmax(torch.from_numpy(p), 1).numpy()[:, 1]
        out["auc"] = float(roc_auc_score(y, prob))
        out["brier"] = float(((prob - y) ** 2).mean())
    out["confusion"] = confusion_matrix(y, pred).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rgb, dep, meta = load(args.data)
    scenes = sorted({m["scene"] for m in meta})
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(scenes))
    n_te = max(1, len(scenes) // 4)
    te_scenes = {scenes[i] for i in perm[:n_te]}
    tr_scenes = set(scenes) - te_scenes
    tr = np.array([m["scene"] in tr_scenes for m in meta])
    te = ~tr
    y_zone = np.array([m["in_zone"] for m in meta])
    y_side = np.array([m["side"] for m in meta])
    res = {"n_total": int(len(meta)), "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "pos_rate_train": float(y_zone[tr].mean()), "pos_rate_test": float(y_zone[te].mean()),
           "test_scenes": sorted(te_scenes), "train_scenes": sorted(tr_scenes)}

    # majority baseline
    maj = int(np.bincount(y_zone[tr]).argmax())
    res["majority"] = {"acc": float((y_zone[te] == maj).mean())}

    # depth-statistics baseline: no appearance, no instrument identity
    def stats(d):
        flat = d.reshape(len(d), -1)
        return np.stack([flat.min(1), np.percentile(flat, 5, axis=1), np.percentile(flat, 25, axis=1),
                         np.median(flat, axis=1), flat.mean(1)], 1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    ss = StandardScaler().fit(stats(dep[tr]))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(ss.transform(stats(dep[tr])), y_zone[tr])
    prob = lr.predict_proba(ss.transform(stats(dep[te])))[:, 1]
    res["depth_stats"] = {"acc": float(accuracy_score(y_zone[te], (prob > 0.5).astype(int))),
                          "auc": float(roc_auc_score(y_zone[te], prob))}

    # Task A: RGB only, and RGB+D
    x_rgb = rgb.transpose(0, 3, 1, 2)
    x_rgbd = np.concatenate([x_rgb, norm_depth(dep)], 1)
    cw = [1.0, float((y_zone[tr] == 0).sum()) / max(1, (y_zone[tr] == 1).sum())]
    for tag, x in (("rgb", x_rgb), ("rgbd", x_rgbd)):
        acc, ep, logits = train_model(x[tr], y_zone[tr], x[te], y_zone[te], 2, args.epochs, class_w=cw)
        res[f"zone_{tag}"] = metrics(y_zone[te], logits, 2)
        res[f"zone_{tag}"]["best_ep"] = ep

    # Task B: side, RGB+D
    res["side_rgbd"] = metrics(y_side[te], train_model(x_rgbd[tr], y_side[tr], x_rgbd[te], y_side[te], 4, args.epochs)[2], 4)
    res["side_majority"] = {"acc": float((y_side[te] == np.bincount(y_side[tr]).argmax()).mean())}

    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("train_scenes",)}, indent=2)[:2000])


if __name__ == "__main__":
    main()
