"""Adapter for the TSL-51 Thai Sign Language dataset (huggingface.co/datasets/Namonpas/thai-sign-language-tsl51).

License: CC BY-NC-SA 4.0, non-commercial only. A model trained on it inherits that restriction.

TSL-51 ships MediaPipe Holistic landmarks as CSV, one row per frame. Only the 2 x 21 hand points
are used, because the app tracks hands only. The clips are too many to convert to the app's JSON
export format, so this script caches them as arrays and computes the same features as features.py
with numpy (checked against features.py by `selftest`).

    python training/tsl51.py prepare DATA_DIR --out data/tsl51.npz
    python training/tsl51.py selftest data/tsl51.npz
    python training/tsl51.py track DATA_DIR --out data/tsl51-user-app.npz   # optional, needs mediapipe + opencv

DATA_DIR holds metadata/*.csv plus the unzipped landmark CSVs anywhere below it
(landmarks/expert_primary_02.zip, landmarks/expert_primary_03.zip and landmarks/user_sign/).

`track` runs the app's own hand tracker over the researcher's videos (videos/user_sign/, the only videos
TSL-51 publishes), which gives a test set with the landmarks the app really sees, not Holistic's.
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
from features import NUM_LANDMARKS, WRIST  # noqa: E402
from npfeatures import app_frame_vectors, main_hand_is_left, mirror, window_features  # noqa: E402,F401

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


def frame_vectors(hands, aspect, hand_position=False):
    """numpy version of features.frame_vector for a whole clip; returns (T, 128), or (T, 134) with hand_position.

    The dataset's left hand goes in the app's 'Left' slot. Checked 2026-10-22 by running the app's
    hand_landmarker.task on 40 raw TSL-51 videos: it called Holistic's left hand 'Left' 97% of the time.
    (The first model swapped them, which put the signing hand in the wrong slot in the app.)
    """
    return app_frame_vectors(hands, aspect, hand_position)


def to_frames(hands, aspect):
    """Rebuild app-style HandFrames from a cached clip, for checking against features.py."""
    frames = []
    for f in hands:
        out = []
        for idx, name in ((0, "Left"), (1, "Right")):
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


def track_video(path):
    """The app's HandLandmarker on one video: up to 2 hands per frame in the tracker's order, as
    (T, 2, 21, 3) landmarks (NaN when absent), (T, 2) labels (0 'Left', 1 'Right', -1 absent) and (T, 2) scores."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    hand_model = str(Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task")
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=hand_model), running_mode=vision.RunningMode.VIDEO, num_hands=2)
    hands, labels, scores = [], [], []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(image, int(len(hands) * 1000 / fps))
            h = np.full((2, NUM_LANDMARKS, 3), np.nan, np.float32)
            lb = np.full(2, -1, np.int8)
            sc = np.zeros(2, np.float32)
            for k, (hand, handedness) in enumerate(zip(res.hand_landmarks[:2], res.handedness[:2])):
                h[k] = [[p.x, p.y, p.z] for p in hand]
                lb[k] = 0 if handedness[0].category_name == "Left" else 1
                sc[k] = handedness[0].score
            hands.append(h)
            labels.append(lb)
            scores.append(sc)
    return np.array(hands).reshape(-1, 2, NUM_LANDMARKS, 3), np.array(labels).reshape(-1, 2), np.array(scores).reshape(-1, 2)


def track(args):
    root = Path(args.data_dir)
    with open(root / "metadata" / "user_sign_metadata.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with Pool() as pool:
        res = pool.map(track_video, [str(root / r["video_path"]) for r in rows], chunksize=8)
    np.savez(
        args.out,
        original=np.array([r["video_id"] for r in rows]),
        hands=np.concatenate([r[0] for r in res]),
        labels=np.concatenate([r[1] for r in res]),
        scores=np.concatenate([r[2] for r in res]),
        offsets=np.concatenate([[0], np.cumsum([len(r[0]) for r in res])]),
    )
    print(f"wrote {args.out}: {len(rows)} videos")


def label_hands(hands, labels, scores, aspect, step=1):
    """Replays the app's HandLabeler (js/sign/features.js) over a tracked clip, keeping every `step`-th
    frame (a slower camera); returns (T', 2, 21, 3) in the app's slot order."""
    hands, labels, scores = hands[::step], labels[::step], scores[::step]
    out = np.full(hands.shape, np.nan, np.float32)
    last = [None, None]

    def dist(w, s):
        return np.hypot((w[0] - last[s][0]) * aspect, w[1] - last[s][1]) if last[s] is not None else np.inf

    for t in range(len(hands)):
        ks = [k for k in (0, 1) if labels[t, k] >= 0]
        wr = [hands[t, k, WRIST] for k in ks]
        if len(ks) == 1:
            slots = [int(labels[t, ks[0]])]
        elif len(ks) == 2:
            a, b = (int(labels[t, k]) for k in ks)
            if a != b:
                slots = [a, b]
            elif last[0] is not None and last[1] is not None:
                slots = [0, 1] if dist(wr[0], 0) + dist(wr[1], 1) <= dist(wr[0], 1) + dist(wr[1], 0) else [1, 0]
            else:
                keep = 0 if scores[t, ks[0]] >= scores[t, ks[1]] else 1
                s = (a, b)[keep]
                slots = [s, 1 - s] if keep == 0 else [1 - s, s]
        else:
            slots = []
        for k, s, w in zip(ks, slots, wr):
            last[s] = w
            out[t, s] = hands[t, k]
    return out


def load_tracked(path):
    """{original video id: (hands, labels, scores)} from `track`."""
    d = np.load(path)
    hands, labels, scores, offsets = d["hands"], d["labels"], d["scores"], d["offsets"]
    return {str(o): (hands[a:b], labels[a:b], scores[a:b]) for o, a, b in zip(d["original"], offsets[:-1], offsets[1:])}


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
    p = sub.add_parser("track")
    p.add_argument("data_dir")
    p.add_argument("--out", default="data/tsl51-user-app.npz")
    p.set_defaults(fn=track)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
