"""Train the app's American Sign Language classifier on PopSign ASL and score it on signers it has never seen.

    python training/popsign.py prepare --subset non-game --out data/popsign.npz
    python training/train_popsign.py data/popsign.npz --out models/asl-model.json

PopSign's own splits are signer-disjoint, so the test split (8 Deaf signers) is a score for people the model
did not train on. By default the model trains on the train split, picks nothing from validation, and reports
the test split; --final also trains on validation for the shipped model, still reporting test.

The clips are about 5 frames per second, so one clip (the signing part only, about 1.2 s) is one training
window. The app feeds about 1.5 s of camera frames at 15-30 fps; the features are means, spreads and the
first-half/second-half change of the hand shapes, which do not depend much on the frame rate. Frames where
the tracker found no hand are dropped before the features are computed, here and in the app
(the model's "skipEmptyFrames"), because at 256x256 the tracker misses about half the frames.

License: PopSign ASL v1.0 is CC BY 4.0, so the model may be used commercially with credit.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from npfeatures import app_frame_vectors, mirror, window_features  # noqa: E402
from popsign import load_cache  # noqa: E402
from train import MODEL_FORMAT, train_mlp  # noqa: E402
from train_tsl51 import evaluate  # noqa: E402

LICENSE = (
    "Trained on PopSign ASL v1.0 (Starner et al., NeurIPS 2023; huggingface.co/datasets/sign/popsign-images), "
    "CC BY 4.0."
)


# PopSign glosses that run words together; everything else is shown as is.
DISPLAY = {
    "callonphone": "call (on phone)", "frenchfries": "french fries", "glasswindow": "glass / window",
    "haveto": "have to", "hesheit": "he / she / it", "icecream": "ice cream", "minemy": "mine / my",
    "thankyou": "thank you", "weus": "we / us", "shhh": "shh", "owie": "ouch (owie)",
}


def hand_frames(hands, aspect):
    vecs = app_frame_vectors(hands, aspect)
    return vecs[vecs[:, -2:].any(1)]


def featurize(idx, clips, meta, labels, args, rng, train, flip=False):
    index = {l: i for i, l in enumerate(labels)}
    xs, ys = [], []
    for i in idx:
        vecs = hand_frames(clips[i], meta["aspect"][i])
        if len(vecs) < args.min_hand_frames:
            continue
        if flip:
            vecs = mirror(vecs)
        variants = [vecs, mirror(vecs)] if args.mirror and train else [vecs]
        for v in variants:
            xs.append(window_features(v))
            ys.append(index[meta["label"][i]])
            if train:
                # Random sub-spans, like an app window that starts late or ends early.
                for _ in range(args.crops):
                    if len(v) < 3:
                        break
                    a = int(rng.integers(0, len(v) // 3 + 1))
                    b = len(v) - int(rng.integers(0, len(v) // 3 + 1))
                    xs.append(window_features(v[a:b]))
                    ys.append(index[meta["label"][i]])
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("--out", default="models/asl-model.json")
    ap.add_argument("--final", action="store_true", help="train on train + validation splits")
    ap.add_argument("--test-split", default="test", help="split to score on (validation: 8 other held-out signers)")
    ap.add_argument("--window", type=int, default=45, help="frames the app collects per prediction")
    ap.add_argument("--min-hand-frames", type=int, default=2)
    ap.add_argument("--crops", type=int, default=2)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", help="skip mirrored copies (left-handed signers)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clips, meta = load_cache(args.cache)
    train_splits = {"train", "validation"} if args.final else {"train"}
    train_idx = [i for i, s in enumerate(meta["split"]) if s in train_splits]
    test_idx = [i for i, s in enumerate(meta["split"]) if s == args.test_split and s not in train_splits]
    labels = sorted({str(meta["label"][i]) for i in train_idx})
    test_idx = [i for i in test_idx if meta["label"][i] in labels]

    rng = np.random.default_rng(args.seed)
    x_tr, y_tr = featurize(train_idx, clips, meta, labels, args, rng, train=True)
    mean = x_tr.mean(0)
    std = x_tr.std(0)
    std[std < 1e-4] = 1.0
    params = train_mlp((x_tr - mean) / std, y_tr, len(labels), args.hidden, args.epochs, args.lr, args.weight_decay, args.seed)

    x_te, y_te = featurize(test_idx, clips, meta, labels, args, rng, train=False)
    signers = lambda idx: sorted({str(meta["signer"][i]) for i in idx})  # noqa: E731
    report = {
        "dataset": "PopSign ASL v1.0, non-game subset (CC BY 4.0)",
        "labels": len(labels),
        "split": f"signer-held-out: PopSign's {'+'.join(sorted(train_splits))} split(s) ({len(signers(train_idx))} signers) vs its {args.test_split} split ({len(signers(test_idx))} signers)",
        "train_clips": len(train_idx),
        "train_windows": len(x_tr),
        "test_clips": len(test_idx),
        "test_clips_with_hands": len(x_te),
        "mirror": args.mirror,
    }
    report.update(evaluate(params, mean, std, x_te, y_te, labels))
    # The same test clips mirrored: if the score holds up, it does not matter whether the source videos were
    # mirrored selfies, and left-handed signers are covered.
    x_mi, y_mi = featurize(test_idx, clips, meta, labels, args, rng, train=False, flip=True)
    report["test_accuracy_mirrored"] = evaluate(params, mean, std, x_mi, y_mi, labels)["test_accuracy"]

    w1, b1, w2, b2 = params
    model = {
        "format": MODEL_FORMAT,
        "language": "ase",
        "labels": labels,
        "display": {l: DISPLAY.get(l, l) for l in labels},
        "window": args.window,
        "skipEmptyFrames": True,
        "license": LICENSE,
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
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, ensure_ascii=False, indent=1))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
