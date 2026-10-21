"""Adapter for PopSign ASL v1.0 (American Sign Language), via huggingface.co/datasets/sign/popsign-images.

License: CC BY 4.0. Credit: Starner et al., "PopSign ASL v1.0: An Isolated American Sign Language Dataset
Collected via Smartphones", NeurIPS 2023 Datasets and Benchmarks. 250 signs by 47 Deaf adults on phone
selfie cameras, with signer-disjoint train / validation / test splits.

The Hugging Face copy holds the signing part of each clip as 256x256 frames at about 5 fps, not videos
or landmarks. This script runs the app's own hand tracker (models/hand_landmarker.task, the same
MediaPipe HandLandmarker the app runs) on those frames and caches the hands in the app's slot order
('Left' = slot 0, 'Right' = slot 1), so no left/right remapping is needed later.

    python training/popsign.py prepare --subset non-game --out data/popsign.npz
    python training/popsign.py prepare --subset non-game --limit-files 2 --out data/popsign-small.npz   # quick try

Needs: pip install mediapipe huggingface_hub pyarrow pillow numpy. Streams the parquet files
(the non-game subset is about 19 GB) without storing them.
"""

import argparse
import io
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = "datasets/sign/popsign-images"
HAND_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task")
NUM_LANDMARKS = 21
FPS = 5.0  # frame rate of the Hugging Face copy


def signer_of(file):
    # e.g. test/animal/gtsignstudy4a.8031-animal-2023_01_24_20_29_03.260-0.mp4 -> gtsignstudy4a.8031
    return os.path.basename(file).split("-")[0]


def list_files(subset, splits):
    from huggingface_hub import HfFileSystem

    names = HfFileSystem().ls(f"{REPO}/data/{subset}", detail=False)
    # Each split is sharded as <split>-NNNNN-of-MMMMM; the single <split>-00000-of-00001 files are a
    # separate small upload and are skipped.
    return sorted(n for n in names if re.search(rf"/({'|'.join(splits)})-\d{{5}}-of-(?!00001)\d{{5}}\.parquet$", n))


def detect_file(path):
    """Returns a list of (meta dict, hands array (T, 2, 21, 3) with NaN for missing hands)."""
    import mediapipe as mp
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image

    # IMAGE mode: frames are 200 ms apart, so the app's frame-to-frame tracking would not help here.
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=HAND_MODEL), num_hands=2)
    lm = vision.HandLandmarker.create_from_options(opts)
    split = os.path.basename(path).split("-")[0]
    out = []
    pf = pq.ParquetFile(HfFileSystem().open(path, block_size=64 << 20))
    for g in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(g).to_pylist():
            hands = np.full((len(r["images"]), 2, NUM_LANDMARKS, 3), np.nan, np.float32)
            for k, im in enumerate(r["images"]):
                img = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
                for cats, pts in zip(res.handedness, res.hand_landmarks):
                    slot = {"Left": 0, "Right": 1}.get(cats[0].category_name)
                    if slot is not None and np.isnan(hands[k, slot, 0, 0]):
                        hands[k, slot] = [[p.x, p.y, p.z] for p in pts]
            out.append(({"label": r["text"], "signer": signer_of(r["file"]), "split": split, "clip": r["file"]}, hands))
    lm.close()
    print(f"{os.path.basename(path)}: {len(out)} clips", file=sys.stderr, flush=True)
    return out


def prepare(args):
    files = list_files(args.subset, args.splits.split(","))
    if args.limit_files:
        files = files[: args.limit_files]
    print(f"{len(files)} parquet files", file=sys.stderr)
    with Pool(args.workers) as pool:
        results = [c for chunk in pool.imap_unordered(detect_file, files) for c in chunk]
    results.sort(key=lambda c: c[0]["clip"])
    seqs = [h for _, h in results]
    lengths = np.array([len(s) for s in seqs])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        hands=np.concatenate(seqs),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        aspect=np.ones(len(results)),  # the frames are square crops
        **{k: np.array([m[k] for m, _ in results]) for k in ("label", "signer", "split", "clip")},
    )
    print(f"wrote {out}: {len(results)} clips, {lengths.sum()} frames")


def load_cache(path):
    d = np.load(path)
    hands, offsets = d["hands"], d["offsets"]
    clips = [hands[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    meta = {k: d[k] for k in ("label", "signer", "split", "clip", "aspect")}
    return clips, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--subset", default="non-game", choices=["non-game", "game"])
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--limit-files", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="data/popsign.npz")
    p.set_defaults(fn=prepare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
