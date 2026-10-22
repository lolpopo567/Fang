"""Train the app's American Sign Language classifier on PopSign ASL and score it on signers it has never seen.

    python training/popsign.py prepare --subset non-game --out data/popsign.npz
    python training/popsign.py prepare --subset game --out data/popsign-game.npz
    python training/train_popsign.py data/popsign.npz data/popsign-game.npz --final --out models/asl-model.json

PopSign's own splits are signer-disjoint, so the test split (8 Deaf signers) is a score for people the model
did not train on. Without --final the model trains on the train split and reports the validation split
(use this to compare settings); --final also trains on validation for the shipped model and reports test.

The clips are about 5 frames per second and hold the signing part only (about 1.2 s). The model gets
`fps: 5`, so the app resamples its last window / 5 seconds of camera frames to the same rate. Frames where
the tracker found no hand are dropped before the features are computed, here and in the app
(the model's "skipEmptyFrames"), because at 256x256 the tracker misses about half the frames.

Features are version 2 (mean, spread, half difference and quarter means of the frame vectors) with
handPosition (where each wrist is in the frame and how big the hand looks), see js/sign/model.js.
Training uses PyTorch (pip install torch) for an MLP with two hidden layers and dropout, and augments
every batch afresh: mirrored (left-handed) copies, small rotations, zoom and shifts of the whole frame,
landmark jitter, shorter spans and dropped frames or hands, like a phone tracker that loses a hand now and then.
The features are computed in torch for speed; the script checks them against npfeatures.py before training.

License: PopSign ASL v1.0 is CC BY 4.0, so the model may be used commercially with credit.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from npfeatures import app_frame_vectors, mirror, window_features  # noqa: E402
from popsign import load_cache  # noqa: E402
from train import MODEL_FORMAT  # noqa: E402
from train_tsl51 import evaluate  # noqa: E402

LICENSE = (
    "Trained on PopSign ASL v1.0 (Starner et al., NeurIPS 2023; huggingface.co/datasets/sign/popsign-images), "
    "CC BY 4.0."
)
FEATURES = 2
SEGMENTS = 4
MAX_FRAMES = 24  # longer clips (under 1%) are cut

# PopSign glosses that run words together; everything else is shown as is.
DISPLAY = {
    "callonphone": "call (on phone)", "frenchfries": "french fries", "glasswindow": "glass / window",
    "haveto": "have to", "hesheit": "he / she / it", "icecream": "ice cream", "minemy": "mine / my",
    "thankyou": "thank you", "weus": "we / us", "shhh": "shh", "owie": "ouch (owie)",
}


def pack(clips, idx):
    """Frames with at least one hand, padded: (N, MAX_FRAMES, 2, 21, 3) NaN-filled, and lengths."""
    x = np.full((len(idx), MAX_FRAMES, 2, 21, 3), np.nan, np.float32)
    lengths = np.zeros(len(idx), np.int64)
    for j, i in enumerate(idx):
        h = clips[i][~np.isnan(clips[i][:, :, 0, 0]).all(1)][:MAX_FRAMES]
        x[j, : len(h)] = h
        lengths[j] = len(h)
    return torch.from_numpy(x), torch.from_numpy(lengths)


def frame_vectors(h):
    """torch version of npfeatures.app_frame_vectors(h, 1, hand_position=True) for a batch: (N, T, 134)."""
    present = ~torch.isnan(h[..., 0, 0])
    h = torch.nan_to_num(h)
    rel = h - h[..., :1, :]
    scale = rel[..., 9, :].norm(dim=-1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    norm = rel / scale[..., None, None] * present[..., None, None]
    pos = torch.stack([h[..., 0, 0] - 0.5, h[..., 0, 1] - 0.5, torch.log(scale / 0.1)], -1) * present[..., None]
    return torch.cat([norm.flatten(2), present.float(), pos.flatten(2)], -1)


def window_feats(v, lengths):
    """torch version of npfeatures.window_features(v[:n], 2) for each row: (N, 7 * 134)."""
    t = torch.arange(v.shape[1])[None]
    n = lengths[:, None]

    def span_mean(lo, hi):
        m = ((t >= lo[:, None]) & (t < hi[:, None])).float()[..., None]
        return (v * m).sum(1) / m.sum(1).clamp_min(1)

    mean = span_mean(torch.zeros_like(lengths), lengths)
    valid = (t < n).float()[..., None]
    std = (((v - mean[:, None]) ** 2 * valid).sum(1) / n).sqrt()
    half = (lengths // 2).clamp_min(1)
    delta = (span_mean(half, lengths) - span_mean(torch.zeros_like(lengths), half)) * (lengths > half)[:, None]
    out = [mean, std, delta]
    for k in range(SEGMENTS):
        lo = k * lengths // SEGMENTS
        out.append(span_mean(lo, torch.maximum((k + 1) * lengths // SEGMENTS, lo + 1)))
    return torch.cat(out, 1)


def features(h, lengths):
    return window_feats(frame_vectors(h), lengths)


def augment(h, lengths, rng):
    """A fresh variation of each clip; returns new (h, lengths) with frames re-packed."""
    n = len(h)
    h = h.clone()
    flip = torch.from_numpy(rng.random(n) < 0.5)
    hf = h[flip].flip(2)  # swap the two hand slots
    hf[..., 0] = 1 - hf[..., 0]
    h[flip] = hf
    angle = torch.from_numpy(rng.normal(0, math.radians(8), n)).float().view(n, 1, 1, 1)
    zoom = torch.from_numpy(np.exp(rng.normal(0, 0.1, n))).float().view(n, 1, 1, 1)
    shift = torch.from_numpy(rng.normal(0, 0.04, (n, 2))).float()
    x, y = h[..., 0] - 0.5, h[..., 1] - 0.5
    h[..., 0] = (torch.cos(angle) * x - torch.sin(angle) * y) * zoom + 0.5 + shift[:, 0].view(n, 1, 1, 1)
    h[..., 1] = (torch.sin(angle) * x + torch.cos(angle) * y) * zoom + 0.5 + shift[:, 1].view(n, 1, 1, 1)
    h[..., 2] *= zoom
    h += torch.from_numpy(rng.normal(0, 0.004, h.shape)).float()
    out = torch.full_like(h, float("nan"))
    new_len = torch.zeros_like(lengths)
    for i in range(n):
        k = int(lengths[i])
        a = rng.integers(0, k // 3 + 1)
        b = max(k - rng.integers(0, k // 3 + 1), a + 1)
        keep = [j for j in range(a, b) if rng.random() < 0.85] or [a]
        clip = h[i, keep]
        clip[torch.from_numpy(rng.random((len(keep), 2)) < 0.1)] = float("nan")
        clip = clip[~torch.isnan(clip[:, :, 0, 0]).all(1)]
        if not len(clip):
            clip = h[i, keep[:1]]
        out[i, : len(clip)] = clip
        new_len[i] = len(clip)
    return out, new_len


def check_parity(clips, idx):
    """The torch features must equal npfeatures.py (and so the app) on real clips, mirrored ones included."""
    h, lengths = pack(clips, idx)
    got = features(h, lengths).double().numpy()
    for j, i in enumerate(idx):
        v = app_frame_vectors(h[j, : lengths[j]].numpy(), 1.0, hand_position=True)
        err = np.abs(got[j] - window_features(v, FEATURES)).max()
        assert err < 1e-4, f"torch features differ from npfeatures.py on clip {i}: {err}"
    hf = h.flip(2)
    hf[..., 0] = 1 - hf[..., 0]
    want = np.array([window_features(mirror(app_frame_vectors(h[j, : lengths[j]].numpy(), 1.0, True)), FEATURES) for j in range(len(idx))])
    assert np.abs(features(hf, lengths).double().numpy() - want).max() < 1e-4, "mirrored features differ"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--out", default="models/asl-model.json")
    ap.add_argument("--final", action="store_true", help="train on train + validation, report test")
    ap.add_argument("--window", type=int, default=8, help="frames per prediction, at --fps")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--hidden", default="512,256")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--repeat-first", type=int, default=1, help="use the first cache's training clips this many times")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    clips, meta = [], {k: [] for k in ("label", "signer", "split", "source")}
    subsets = []
    for path in args.caches:
        c, m = load_cache(path)
        clips += c
        for k in ("label", "signer", "split"):
            meta[k] += [str(v) for v in m[k]]
        meta["source"] += [len(subsets)] * len(c)
        subsets.append(Path(path).stem)
    train_splits = {"train", "validation"} if args.final else {"train"}
    test_split = "test" if args.final else "validation"
    train_idx = [i for i, s in enumerate(meta["split"]) if s in train_splits]
    labels = sorted({meta["label"][i] for i in train_idx})
    index = {l: i for i, l in enumerate(labels)}
    test_idx = [i for i, s in enumerate(meta["split"]) if s == test_split and meta["label"][i] in index]
    hand_frames = lambda i: (~np.isnan(clips[i][:, :, 0, 0]).all(1)).sum()  # noqa: E731
    train_idx = [i for i in train_idx if hand_frames(i) >= 2]
    test_with_hands = [i for i in test_idx if hand_frames(i) >= 2]

    check_parity(clips, test_with_hands[:200])
    x_tr, len_tr = pack(clips, train_idx)
    y_tr = torch.tensor([index[meta["label"][i]] for i in train_idx])
    # Rows of x_tr per epoch; the first cache's clips (careful, one sign per video) can count more than once.
    rows = torch.cat([torch.arange(len(train_idx))] + [torch.tensor([j for j, i in enumerate(train_idx) if meta["source"][i] == 0])] * (args.repeat_first - 1))
    x_te, len_te = pack(clips, test_with_hands)
    y_te = np.array([index[meta["label"][i]] for i in test_with_hands])
    print(f"{len(train_idx)} training clips ({len(rows)} per epoch), {len(test_with_hands)} {test_split} clips, {len(labels)} signs", flush=True)

    with torch.no_grad():
        f0 = torch.cat([features(x_tr[b], len_tr[b]) for b in torch.arange(len(x_tr)).split(8192)])
    mean, std = f0.mean(0), f0.std(0, unbiased=False)
    del f0
    std[std < 1e-4] = 1.0
    hidden = [int(h) for h in args.hidden.split(",")]
    dims = [len(mean), *hidden]
    mods = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(args.dropout)]
    net = nn.Sequential(*mods, nn.Linear(dims[-1], len(labels)))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(len(rows) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    start = time.time()
    for epoch in range(args.epochs):
        for b in rows[torch.randperm(len(rows))].split(args.batch):
            h, lengths = augment(x_tr[b], len_tr[b], rng)
            loss = F.cross_entropy(net((features(h, lengths) - mean) / std), y_tr[b], label_smoothing=args.label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {epoch + 1}: loss {loss.item():.3f} ({time.time() - start:.0f} s)", file=sys.stderr, flush=True)
    net.eval()

    layers = [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy()) for m in net if isinstance(m, nn.Linear)]
    mean, std = mean.double().numpy(), std.double().numpy()
    score = lambda h, lengths, y: evaluate(layers, mean, std, features(h, lengths).double().numpy(), y, labels)  # noqa: E731

    signers = lambda idx: sorted({meta["signer"][i] for i in idx})  # noqa: E731
    report = {
        "dataset": f"PopSign ASL v1.0 ({', '.join(subsets)}; CC BY 4.0)",
        "labels": len(labels),
        "split": f"signer-held-out: PopSign's {'+'.join(sorted(train_splits))} split(s) ({len(signers(train_idx))} signers) vs its {test_split} split ({len(signers(test_idx))} signers)",
        "train_clips": len(train_idx),
        "test_clips": len(test_idx),
        "test_clips_with_hands": len(test_with_hands),
        "hidden": hidden,
    }
    report.update(score(x_te, len_te, y_te))
    hf = x_te.flip(2)
    hf[..., 0] = 1 - hf[..., 0]
    report["test_accuracy_mirrored"] = score(hf, len_te, y_te)["test_accuracy"]

    r = lambda a: a.round(5).tolist()  # noqa: E731
    model = {
        "format": MODEL_FORMAT,
        "language": "ase",
        "labels": labels,
        "display": {l: DISPLAY.get(l, l) for l in labels},
        "window": args.window,
        "fps": args.fps,
        "features": FEATURES,
        "handPosition": True,
        "skipEmptyFrames": True,
        "license": LICENSE,
        "mean": r(mean),
        "std": r(std),
        "layers": [{"W": r(w), "b": r(b)} for w, b in layers],
        "report": report,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, ensure_ascii=False, indent=1))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
