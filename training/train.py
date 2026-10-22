"""Train the on-device Thai Sign Language classifier.

Input:  one or more JSON files in the app's export format (fang-handframes-v1),
        i.e. what "Export JSON" in the sign tab produces, or what an adapter converts a dataset into.
Output: models/tsl-model.json (loaded by js/sign/model.js) and a metrics report.

Only numpy is required, so it runs anywhere, including Colab.

    python training/train.py data/*.json --out models/tsl-model.json
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import sequence_features  # noqa: E402

MODEL_FORMAT = "fang-mlp-v1"


def load_samples(paths):
    samples = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if data.get("format") != "fang-handframes-v1":
            raise SystemExit(f"{p}: expected format fang-handframes-v1, got {data.get('format')}")
        for s in data["samples"]:
            if any(f["hands"] for f in s["frames"]):
                samples.append(s)
    return samples


def crops(frames, window, count, rng):
    """Random windows of `window` frames, the same length the app feeds the model at run time."""
    if len(frames) <= window:
        return [frames]
    starts = [rng.randrange(0, len(frames) - window + 1) for _ in range(count)]
    return [frames[s : s + window] for s in starts]


def split(samples, test_frac, rng):
    """Hold out whole signers when the data names them, so the score reflects unseen people."""
    signers = sorted({s.get("signer") for s in samples if s.get("signer")})
    if len(signers) >= 2:
        n_test = max(1, round(len(signers) * test_frac))
        test_signers = set(rng.sample(signers, n_test))
        train = [s for s in samples if s.get("signer") not in test_signers]
        test = [s for s in samples if s.get("signer") in test_signers]
        return train, test, f"signer-held-out (test signers: {sorted(test_signers)})"
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)
    train, test = [], []
    for group in by_label.values():
        rng.shuffle(group)
        k = int(len(group) * test_frac) if len(group) >= 3 else 0
        test += group[:k]
        train += group[k:]
    return train, test, "random per-class split (no signer ids: optimistic, same people in train and test)"


def featurize(samples, labels, window, crops_per_sample, rng):
    xs, ys = [], []
    index = {l: i for i, l in enumerate(labels)}
    for s in samples:
        for frames in crops(s["frames"], window, crops_per_sample, rng):
            xs.append(sequence_features(frames))
            ys.append(index[s["label"]])
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.int64)


def train_mlp(x, y, n_classes, hidden, epochs, lr, weight_decay, seed):
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    w1 = rng.normal(0, np.sqrt(2 / d), (hidden, d))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(2 / hidden), (n_classes, hidden))
    b2 = np.zeros(n_classes)
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    onehot = np.eye(n_classes)[y]
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 64):
            idx = order[start : start + 64]
            xb, tb = x[idx], onehot[idx]
            h_pre = xb @ w1.T + b1
            h = np.maximum(h_pre, 0)
            logits = h @ w2.T + b2
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            g_logits = (p - tb) / len(idx)
            g_w2 = g_logits.T @ h + weight_decay * w2
            g_b2 = g_logits.sum(0)
            g_h = g_logits @ w2 * (h_pre > 0)
            g_w1 = g_h.T @ xb + weight_decay * w1
            g_b1 = g_h.sum(0)
            step += 1
            for i, g in enumerate([g_w1, g_b1, g_w2, g_b2]):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                mh = m[i] / (1 - 0.9**step)
                vh = v[i] / (1 - 0.999**step)
                params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return params


def predict(params, x):
    w1, b1, w2, b2 = params
    logits = np.maximum(x @ w1.T + b1, 0) @ w2.T + b2
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def model_probs(m, frames):
    """Reference for MlpModel.predict in js/sign/model.js (minus the no-hands check): class probabilities."""
    if m.get("fps"):
        span = 1000 * m["window"] / m["fps"]
        end, i, picked = frames[-1]["t"], 0, []
        for k in range(m["window"]):
            target = end - span + (k + 0.5) * span / m["window"]
            while i < len(frames) - 1 and frames[i]["t"] < target:
                i += 1
            picked.append(frames[i])
        frames = picked
    if m.get("skipEmptyFrames"):
        frames = [f for f in frames if f["hands"]]
    x = (np.array(sequence_features(frames, m.get("features", 1), bool(m.get("canonicalHand")))) - m["mean"]) / m["std"]
    layers = m.get("layers") or [{"W": m["W1"], "b": m["b1"]}, {"W": m["W2"], "b": m["b2"]}]
    for i, layer in enumerate(layers):
        x = np.array(layer["W"]) @ x + layer["b"]
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    p = np.exp(x - x.max())
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="models/tsl-model.json")
    ap.add_argument("--window", type=int, default=30, help="frames per prediction; must match the app")
    ap.add_argument("--crops", type=int, default=8, help="random windows per training sample")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples = load_samples(args.inputs)
    counts = Counter(s["label"] for s in samples)
    dropped = sorted(l for l, c in counts.items() if c < args.min_per_class)
    samples = [s for s in samples if s["label"] not in dropped]
    labels = sorted({s["label"] for s in samples})
    if len(labels) < 2:
        raise SystemExit(f"need at least 2 labels with >= {args.min_per_class} samples; have {dict(counts)}")

    train, test, split_kind = split(samples, args.test_frac, rng)
    x_tr, y_tr = featurize(train, labels, args.window, args.crops, rng)
    mean = x_tr.mean(0)
    std = x_tr.std(0)
    std[std < 1e-4] = 1.0  # constant features (e.g. the wrist, always 0 after normalizing); avoids 0/0 in the app
    params = train_mlp((x_tr - mean) / std, y_tr, len(labels), args.hidden, args.epochs, args.lr, args.weight_decay, args.seed)

    report = {
        "samples": len(samples),
        "labels": len(labels),
        "dropped_labels_too_few_samples": dropped,
        "split": split_kind,
        "train_samples": len(train),
        "test_samples": len(test),
    }
    if test:
        # One centered window per test sample, like a single prediction in the app.
        x_te, y_te = featurize(test, labels, args.window, 1, random.Random(args.seed + 1))
        pred = predict(params, (x_te - mean) / std).argmax(1)
        report["test_accuracy"] = round(float((pred == y_te).mean()), 4)
        per_class = {}
        for c, label in enumerate(labels):
            mask = y_te == c
            if mask.any():
                per_class[label] = {"n": int(mask.sum()), "accuracy": round(float((pred[mask] == c).mean()), 4)}
        report["per_class"] = per_class
        confusions = Counter((labels[a], labels[b]) for a, b in zip(y_te, pred) if a != b)
        report["top_confusions"] = [{"true": a, "predicted": b, "count": n} for (a, b), n in confusions.most_common(10)]

    w1, b1, w2, b2 = params
    model = {
        "format": MODEL_FORMAT,
        "labels": labels,
        "window": args.window,
        "mean": mean.round(6).tolist(),
        "std": std.round(6).tolist(),
        "W1": w1.round(6).tolist(),
        "b1": b1.round(6).tolist(),
        "W2": w2.round(6).tolist(),
        "b2": b2.round(6).tolist(),
        "report": report,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
