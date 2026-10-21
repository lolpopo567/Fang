"""Train the app's sign classifier on TSL-51 and score it on a signer it has never seen.

    python training/tsl51.py prepare DATA_DIR --out data/tsl51.npz
    python training/train_tsl51.py data/tsl51.npz --out models/tsl-model.json

By default the two interpreters (expert_01, expert_02, with the dataset's augmented copies) are the
training set and the researcher's own webcam clips (user_01) are the test set, so the reported
accuracy is for a person, camera and room the model did not train on.

License: TSL-51 is CC BY-NC-SA 4.0, so the trained model is for non-commercial use only.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import MODEL_FORMAT, predict, train_mlp  # noqa: E402
from tsl51 import frame_vectors, load_cache, mirror, window_features  # noqa: E402

NULL = "null_act"


def motion_center(vecs, hands, aspect):
    """Frame index at the centre of the hands' movement, used to place windows on the sign itself."""
    wrists = hands[:, :, 0, :2].astype(np.float64) * [aspect, 1.0]
    step = np.nan_to_num(np.linalg.norm(np.diff(wrists, axis=0), axis=-1)).sum(1)
    if step.sum() == 0:
        return len(vecs) // 2
    return int(round(float((np.arange(len(step)) + 0.5) @ step / step.sum())))


def windows(n, window, center, count, rng, anywhere):
    if n <= window:
        return [0]
    hi = n - window
    if anywhere:
        return list(rng.integers(0, hi + 1, count))
    lo_s, hi_s = center - int(0.75 * window), center - int(0.25 * window)
    return list(np.clip(rng.integers(lo_s, hi_s + 1, count), 0, hi))


def featurize(idx, clips, meta, labels, args, rng, train):
    index = {l: i for i, l in enumerate(labels)}
    xs, ys = [], []
    for i in idx:
        vecs = frame_vectors(clips[i], meta["aspect"][i])
        label = meta["label"][i]
        center = motion_center(vecs, clips[i], meta["aspect"][i])
        if train:
            starts = windows(len(vecs), args.window, center, args.crops, rng, label == NULL)
        else:
            starts = [int(np.clip(center - args.window // 2, 0, max(0, len(vecs) - args.window)))]
        variants = [vecs, mirror(vecs)] if train and args.mirror else [vecs]
        for v in variants:
            for s in starts:
                xs.append(window_features(v[s : s + args.window]))
                ys.append(index[label])
    return np.array(xs), np.array(ys)


def evaluate(params, mean, std, x, y, labels):
    prob = predict(params, (x - mean) / std)
    pred = prob.argmax(1)
    top3 = (np.argsort(-prob, 1)[:, :3] == y[:, None]).any(1)
    per_class = {
        labels[c]: {"n": int((y == c).sum()), "accuracy": round(float((pred[y == c] == c).mean()), 4)}
        for c in range(len(labels))
        if (y == c).any()
    }
    confusions = Counter((labels[a], labels[b]) for a, b in zip(y, pred) if a != b)
    return {
        "test_accuracy": round(float((pred == y).mean()), 4),
        "test_top3_accuracy": round(float(top3.mean()), 4),
        "per_class": per_class,
        "top_confusions": [{"true": a, "predicted": b, "count": n} for (a, b), n in confusions.most_common(15)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("--out", default="models/tsl-model.json")
    ap.add_argument("--test-signers", default="user_01")
    ap.add_argument("--train-signers", default="", help="default: every signer not in --test-signers")
    ap.add_argument("--window", type=int, default=30, help="frames per prediction; the app uses the model's value")
    ap.add_argument("--crops", type=int, default=3, help="windows per training clip")
    ap.add_argument("--mirror", action="store_true", help="also train on mirrored clips (left-handed signers)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clips, meta = load_cache(args.cache)
    test_signers = set(args.test_signers.split(",")) if args.test_signers else set()
    train_signers = set(args.train_signers.split(",")) if args.train_signers else {str(s) for s in meta["signer"]} - test_signers
    train_idx = [i for i, s in enumerate(meta["signer"]) if s in train_signers]
    test_idx = [i for i, s in enumerate(meta["signer"]) if s in test_signers and meta["augmentation"][i] == "original"]
    labels = sorted({meta["label"][i] for i in train_idx})
    test_idx = [i for i in test_idx if meta["label"][i] in labels]
    display = {meta["label"][i]: meta["display"][i] for i in train_idx}

    rng = np.random.default_rng(args.seed)
    x_tr, y_tr = featurize(train_idx, clips, meta, labels, args, rng, train=True)
    mean = x_tr.mean(0)
    std = x_tr.std(0)
    std[std < 1e-4] = 1.0  # constant features (e.g. the wrist, always 0 after normalizing); avoids 0/0 in the app
    params = train_mlp((x_tr - mean) / std, y_tr, len(labels), args.hidden, args.epochs, args.lr, args.weight_decay, args.seed)

    report = {
        "dataset": "TSL-51 (CC BY-NC-SA 4.0, non-commercial)",
        "labels": len(labels),
        "split": f"signer-held-out: train {sorted(train_signers)}, test {sorted(test_signers)}" if test_signers else f"no test split, trained on all signers {sorted(train_signers)}",
        "train_clips": len(train_idx),
        "train_windows": len(x_tr),
        "test_clips": len(test_idx),
        "window": args.window,
        "mirror": args.mirror,
    }
    if test_idx:
        x_te, y_te = featurize(test_idx, clips, meta, labels, args, rng, train=False)
        report.update(evaluate(params, mean, std, x_te, y_te, labels))

    w1, b1, w2, b2 = params
    model = {
        "format": MODEL_FORMAT,
        "labels": labels,
        "display": display,
        "window": args.window,
        "license": "Trained on TSL-51 (huggingface.co/datasets/Namonpas/thai-sign-language-tsl51), CC BY-NC-SA 4.0: non-commercial use only.",
        "mean": mean.round(5).tolist(),
        "std": std.round(5).tolist(),
        "W1": w1.round(5).tolist(),
        "b1": b1.round(5).tolist(),
        "W2": w2.round(5).tolist(),
        "b2": b2.round(5).tolist(),
        "report": report,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "per_class"}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
