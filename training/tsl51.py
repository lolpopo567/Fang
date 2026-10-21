"""Adapter for the TSL-51 Thai Sign Language dataset (huggingface.co/datasets/Namonpas/thai-sign-language-tsl51).

License: CC BY-NC-SA 4.0, non-commercial only. A model trained on it inherits that restriction.

TSL-51 ships MediaPipe Holistic landmarks as CSV, one row per frame. Only the 2 x 21 hand points
are used, because the app tracks hands only. The clips are too many to convert to the app's JSON
export format, so this script caches them as arrays and computes the same features as features.py
with numpy (checked against features.py by `selftest`).

    python training/tsl51.py prepare DATA_DIR --out data/tsl51.npz
    python training/tsl51.py selftest data/tsl51.npz

DATA_DIR holds metadata/*.csv plus the unzipped landmark CSVs anywhere below it
(landmarks/expert_primary_02.zip, landmarks/expert_primary_03.zip and landmarks/user_sign/).
"""

import argparse
import csv
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import FRAME_DIM, MIDDLE_MCP, NUM_LANDMARKS, WRIST  # noqa: E402

HAND_COLS = [f"{side}_{axis}{i}" for side in ("lh", "rh") for i in range(NUM_LANDMARKS) for axis in "xyz"]


def decode_name(name):
    # The second expert zip escapes Thai characters in file names as #Uxxxx.
    return re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), name)


def read_hands(path):
    """(T, 2, 21, 3) float32 in dataset order [left hand, right hand]; NaN where a hand is missing."""
    import pandas as pd

    df = pd.read_csv(path, usecols=lambda c: c in HAND_COLS)
    return df[HAND_COLS].to_numpy(np.float32).reshape(-1, 2, NUM_LANDMARKS, 3)


def load_rows(root):
    rows = []
    meta = root / "metadata"
    with open(meta / "expert_metadata.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["source_group"] != "expert-primary":
                continue  # web-scraped clips: 1-5 per sign and unclear rights, see docs/datasets.md
            rows.append({
                "file": os.path.basename(r["landmark_path"]),
                "label": r["sign_id"],
                "display": r["sign_clean"],
                "signer": r["signer_name"],
                "original": r["original_video_id"],
                "augmentation": r["augmentation_type"],
                "aspect": float(r["width_extracted"]) / float(r["height_extracted"]),
            })
    with open(meta / "user_sign_metadata.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "file": os.path.basename(r["landmark_path"]),
                "label": r["sign_id"],
                "display": r["sign_clean"],
                "signer": "user_01",  # the researcher, recorded on a 640x480 webcam
                "original": r["video_id"],
                "augmentation": "original",
                "aspect": float(r["width_extracted"]) / float(r["height_extracted"]),
            })
    return rows


def prepare(args):
    root = Path(args.data_dir)
    index = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(".csv") and "metadata" not in dirpath:
                index[decode_name(name)] = os.path.join(dirpath, name)
    rows = load_rows(root)
    missing = [r["file"] for r in rows if r["file"] not in index]
    if missing:
        raise SystemExit(f"{len(missing)} landmark files not found, e.g. {missing[:3]}")
    with Pool() as pool:
        seqs = pool.map(read_hands, [index[r["file"]] for r in rows], chunksize=64)
    lengths = np.array([len(s) for s in seqs])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        hands=np.concatenate(seqs),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        **{k: np.array([r[k] for r in rows]) for k in rows[0] if k != "file"},
    )
    print(f"wrote {out}: {len(rows)} clips, {lengths.sum()} frames")


def load_cache(path):
    d = np.load(path)
    hands, offsets = d["hands"], d["offsets"]
    clips = [hands[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    meta = {k: d[k] for k in ("label", "display", "signer", "original", "augmentation", "aspect")}
    return clips, meta


def frame_vectors(hands, aspect):
    """numpy version of features.frame_vector for a whole clip; returns (T, 128).

    Holistic names hands from the signer's point of view and the videos are not mirrored. The app runs
    HandLandmarker on the raw camera image, which labels hands as if the image were mirrored, so the
    signer's right hand comes out as 'Left' (slot 0) and the signer's left hand as 'Right' (slot 1).
    """
    h = hands[:, ::-1].astype(np.float64)  # slot 0 = dataset right hand, slot 1 = dataset left hand
    h[..., 0] *= aspect
    h[..., 2] *= aspect
    present = ~np.isnan(h).any(axis=(2, 3))
    rel = h - h[:, :, WRIST : WRIST + 1]
    scale = np.linalg.norm(rel[:, :, MIDDLE_MCP], axis=-1)
    scale[~(scale > 0)] = 1.0
    norm = np.where(present[:, :, None, None], rel / scale[:, :, None, None], 0.0)
    out = np.zeros((len(h), FRAME_DIM))
    out[:, : 2 * NUM_LANDMARKS * 3] = norm.reshape(len(h), -1)
    out[:, 2 * NUM_LANDMARKS * 3 :] = present
    return out


def mirror(vecs):
    """The same clip signed by a left-handed signer: flip x and swap the two hand slots."""
    n = NUM_LANDMARKS * 3
    hands = vecs[:, : 2 * n].reshape(len(vecs), 2, NUM_LANDMARKS, 3)[:, ::-1].copy()
    hands[..., 0] *= -1
    return np.concatenate([hands.reshape(len(vecs), -1), vecs[:, 2 * n :][:, ::-1]], axis=1)


def window_features(vecs):
    """numpy version of features.sequence_features over one window of frame vectors; returns (384,)."""
    n = len(vecs)
    half = n // 2 or 1
    delta = vecs[half:].mean(0) - vecs[:half].mean(0) if n - half > 0 else np.zeros(vecs.shape[1])
    return np.concatenate([vecs.mean(0), vecs.std(0), delta])


def to_frames(hands, aspect):
    """Rebuild app-style HandFrames from a cached clip, for checking against features.py."""
    frames = []
    for f in hands:
        out = []
        for idx, name in ((1, "Left"), (0, "Right")):
            if not np.isnan(f[idx]).any():
                out.append({"handedness": name, "landmarks": f[idx].astype(float).tolist()})
        frames.append({"aspect": float(aspect), "hands": out})
    return frames


def selftest(args):
    from features import sequence_features

    clips, meta = load_cache(args.cache)
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in rng.choice(len(clips), 50, replace=False):
        hands = clips[i][:30]
        a = window_features(frame_vectors(hands, meta["aspect"][i]))
        b = np.array(sequence_features(to_frames(hands, meta["aspect"][i])))
        worst = max(worst, float(np.abs(a - b).max()))
    print(f"max difference vs features.py over 50 clips: {worst:.2e}")
    if worst > 1e-6:
        raise SystemExit("numpy features drifted from features.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("data_dir")
    p.add_argument("--out", default="data/tsl51.npz")
    p.set_defaults(fn=prepare)
    p = sub.add_parser("selftest")
    p.add_argument("cache")
    p.set_defaults(fn=selftest)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
