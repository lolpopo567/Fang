"""Train the app's sign classifier on TSL-51 and score it on a signer it has never seen.

    python training/tsl51.py prepare DATA_DIR --out data/tsl51.npz
    python training/tsl51.py track DATA_DIR --out data/tsl51-user-app.npz      # optional, see below
    python training/train_tsl51.py data/tsl51.npz --app-test data/tsl51-user-app.npz --out models/tsl-model.json

By default the two interpreters (expert_01, expert_02, with the dataset's augmented copies) are the
training set and the researcher's own webcam clips (user_01) are the test set, so the reported
accuracy is for a person, camera and room the model did not train on. With --app-test, user_01 is also
scored on landmarks from the app's own hand tracker, which is closest to what the app sees on a phone.

Each training example is what the app would hold at one moment: the last window/fps seconds of a clip
as seen by a camera running at a random 5-30 fps, resampled to `window` frames, with random rotation,
stretch, jitter and dropped hands. Needs PyTorch (CPU is fine, about 10 minutes).

License: TSL-51 is CC BY-NC-SA 4.0, so the trained model is for non-commercial use only.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import MODEL_FORMAT  # noqa: E402
from tsl51 import frame_vectors, label_hands, load_cache, load_tracked, main_hand_is_left, mirror, window_features  # noqa: E402

NULL = "null_act"
DATA_FPS = 30.0  # every TSL-51 clip


def motion_center(hands, aspect):
    """Time in frames at the centre of the hands' movement, used to place windows on the sign itself."""
    wrists = hands[:, :, 0, :2].astype(np.float64) * [aspect, 1.0]
    step = np.nan_to_num(np.linalg.norm(np.diff(wrists, axis=0), axis=-1)).sum(1)
    if step.sum() == 0:
        return len(hands) / 2
    return float((np.arange(len(step)) + 1.0) @ step / step.sum())


class Windows:
    """Turns clips into the app's input: frames a phone would keep at prediction time, then features."""

    def __init__(self, args, rng):
        self.args = args
        self.rng = rng
        self.span = args.window / args.fps  # seconds

    def times(self, n_frames, center, train):
        """Clip times (s) of the `window` frames the app would pass to the model, nearest-earlier device frame."""
        a, rng, span = self.args, self.rng, self.span
        if train:
            device_fps = rng.uniform(a.min_device_fps, a.max_device_fps)
            end = center + span * rng.uniform(0.25, 0.75)
            end -= rng.uniform(0, 1 / device_fps)
        else:
            device_fps, end = DATA_FPS, center + span / 2
        end = min(max(end, span), max(n_frames / DATA_FPS, span))
        device = np.arange(end, end - span - 1 / device_fps, -1 / device_fps)[::-1]
        targets = end - span + (np.arange(a.window) + 0.5) * span / a.window
        return device[np.clip(np.searchsorted(device, targets), 0, len(device) - 1)]

    def augment(self, h):
        rng = self.rng
        h = h.copy()
        angle = rng.normal(0, 0.15)
        c, s = np.cos(angle), np.sin(angle)
        x, y = h[..., 0] - 0.5, h[..., 1] - 0.5
        sx, sy = rng.normal(1, 0.08, 2)
        h[..., 0] = (c * x - s * y) * sx + 0.5
        h[..., 1] = (s * x + c * y) * sy + 0.5
        if self.args.hand_position:  # camera framing: where the signer sits in the frame and how near
            zoom, (dx, dy) = rng.normal(1, 0.05), rng.normal(0, 0.025, 2)
            h[..., 0] = (h[..., 0] - 0.5) * zoom + 0.5 + dx
            h[..., 1] = (h[..., 1] - 0.5) * zoom + 0.5 + dy
        h[..., 2] *= rng.normal(1, 0.2)
        h += rng.normal(0, 0.003, h.shape).astype(np.float32)
        if rng.random() < 0.5:  # the tracker loses a hand now and then
            h[rng.random(h.shape[:2]) < rng.uniform(0, 0.25)] = np.nan
        return h

    def features(self, hands, aspect, label, train):
        center = motion_center(hands, aspect) / DATA_FPS
        if train and label == NULL:
            center = self.rng.uniform(0, len(hands) / DATA_FPS)
        t = self.times(len(hands), center, train)
        frame = np.round(t * DATA_FPS).astype(int)
        h = hands[np.clip(frame, 0, len(hands) - 1)].copy()
        h[(frame < 0) | (frame >= len(hands))] = np.nan  # before or after the clip: hands at rest, out of view
        if train:
            h = self.augment(h)
        seen = ~np.isnan(h).all(axis=(1, 2, 3))
        if seen.sum() >= 2:
            h = h[seen]  # skipEmptyFrames
        vecs = frame_vectors(h, aspect, self.args.hand_position)
        if main_hand_is_left(h, aspect):
            vecs = mirror(vecs)  # canonicalHand
        return window_features(vecs, 2)

    def batch(self, items, train):
        """items: (hands, aspect, label) triples."""
        return np.array([self.features(h, a, l, train) for h, a, l in items], np.float32)


def train_net(x_fn, y, n_classes, args):
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    x = x_fn()
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-4] = 1.0  # constant features (e.g. the wrist, always 0 after normalizing); avoids 0/0 in the app
    sizes = [x.shape[1], *map(int, args.hidden.split(","))]
    mods = []
    for i, o in zip(sizes, sizes[1:]):
        mods += [nn.Linear(i, o), nn.ReLU(), nn.Dropout(args.dropout)]
    net = nn.Sequential(*mods, nn.Linear(sizes[-1], n_classes))
    opt = torch.optim.AdamW(net.parameters(), args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(y) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    yt = torch.tensor(y)
    for epoch in range(args.epochs):
        if epoch:
            x = x_fn()  # fresh windows and augmentation every epoch
        xt = torch.tensor((x - mean) / std)
        order = torch.randperm(len(xt))
        for b in range(0, len(xt), 128):
            k = order[b : b + 128]
            loss = nn.functional.cross_entropy(net(xt[k]), yt[k], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {epoch + 1}/{args.epochs} loss {loss.item():.3f}", flush=True)
    net.eval()
    layers = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in net if isinstance(m, nn.Linear)]
    return layers, mean.astype(np.float64), std.astype(np.float64)


def predict(layers, x):
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    p = np.exp(x - x.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def evaluate(layers, mean, std, x, y, labels):
    prob = predict(layers, (x - mean) / std)
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
    ap.add_argument("--app-test", default="", help="output of `tsl51.py track`: also score user_01 on the app tracker's landmarks")
    ap.add_argument("--window", type=int, default=24, help="frames per prediction")
    ap.add_argument("--fps", type=float, default=15, help="the window covers window/fps seconds, at any camera frame rate")
    ap.add_argument("--min-device-fps", type=float, default=5)
    ap.add_argument("--max-device-fps", type=float, default=30)
    ap.add_argument("--no-hand-position", dest="hand_position", action="store_false", help="hand shapes only, not where the hands are")
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=4)
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
    index = {l: i for i, l in enumerate(labels)}

    win = Windows(args, np.random.default_rng(args.seed))
    train_items = [(clips[i], meta["aspect"][i], meta["label"][i]) for i in train_idx]
    y_tr = np.array([index[meta["label"][i]] for i in train_idx])
    layers, mean, std = train_net(lambda: win.batch(train_items, True), y_tr, len(labels), args)

    report = {
        "dataset": "TSL-51 (CC BY-NC-SA 4.0, non-commercial)",
        "labels": len(labels),
        "split": f"signer-held-out: train {sorted(train_signers)}, test {sorted(test_signers)}" if test_signers else f"no test split, trained on all signers {sorted(train_signers)}",
        "train_clips": len(train_idx),
        "test_clips": len(test_idx),
        "window": args.window,
        "fps": args.fps,
        "hidden": args.hidden,
        "hand_position": args.hand_position,
    }
    if test_idx:
        y_te = np.array([index[meta["label"][i]] for i in test_idx])
        x_te = win.batch([(clips[i], meta["aspect"][i], meta["label"][i]) for i in test_idx], False)
        report.update(evaluate(layers, mean, std, x_te, y_te, labels))
        tracked = load_tracked(args.app_test) if args.app_test else {}
        idx = [i for i in test_idx if str(meta["original"][i]) in tracked]
        if idx:
            hands = [label_hands(*tracked[str(meta["original"][i])], float(meta["aspect"][i])) for i in idx]
            x_app = win.batch([(h, meta["aspect"][i], meta["label"][i]) for h, i in zip(hands, idx)], False)
            app = evaluate(layers, mean, std, x_app, np.array([index[meta["label"][i]] for i in idx]), labels)
            report["app_tracker"] = {"test_clips": len(idx), **{k: app[k] for k in ("test_accuracy", "test_top3_accuracy", "top_confusions")}}

    model = {
        "format": MODEL_FORMAT,
        "language": "tsl",
        "labels": labels,
        "display": display,
        "window": args.window,
        "fps": args.fps,
        "features": 2,
        "canonicalHand": True,
        "handPosition": args.hand_position,
        "skipEmptyFrames": True,
        "license": "Trained on TSL-51 (huggingface.co/datasets/Namonpas/thai-sign-language-tsl51), CC BY-NC-SA 4.0: non-commercial use only.",
        "mean": mean.round(5).tolist(),
        "std": std.round(5).tolist(),
        "layers": [{"W": w.round(5).tolist(), "b": b.round(5).tolist()} for w, b in layers],
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
