"""numpy versions of features.py for whole clips, shared by the dataset adapters.

Hands come in as (T, 2, 21, 3) in the app's slot order: slot 0 is the hand MediaPipe HandLandmarker labels
'Left', slot 1 'Right'; NaN where a hand is missing. tsl51.py selftest checks these against features.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import FRAME_DIM, MIDDLE_MCP, NUM_LANDMARKS, SEGMENTS, WRIST  # noqa: E402


def app_frame_vectors(hands, aspect, hand_position=False):
    """numpy version of features.frame_vector for a whole clip; returns (T, 128), or (T, 134) with hand_position."""
    h = hands.astype(np.float64)
    h[..., 0] *= aspect
    h[..., 2] *= aspect
    present = ~np.isnan(h).any(axis=(2, 3))
    rel = h - h[:, :, WRIST : WRIST + 1]
    scale = np.linalg.norm(rel[:, :, MIDDLE_MCP], axis=-1)
    scale[~(scale > 0)] = 1.0
    norm = np.where(present[:, :, None, None], rel / scale[:, :, None, None], 0.0)
    out = np.zeros((len(h), FRAME_DIM + (6 if hand_position else 0)))
    out[:, : 2 * NUM_LANDMARKS * 3] = norm.reshape(len(h), -1)
    out[:, 2 * NUM_LANDMARKS * 3 : FRAME_DIM] = present
    if hand_position:
        pos = np.stack([h[:, :, WRIST, 0] - 0.5 * aspect, h[:, :, WRIST, 1] - 0.5, np.log(scale / 0.1)], -1)
        out[:, FRAME_DIM:] = np.where(present[:, :, None], pos, 0.0).reshape(len(h), 6)
    return out


def mirror(vecs):
    """The same clip signed by a left-handed signer: flip x and swap the two hand slots."""
    n = NUM_LANDMARKS * 3
    hands = vecs[:, : 2 * n].reshape(len(vecs), 2, NUM_LANDMARKS, 3)[:, ::-1].copy()
    hands[..., 0] *= -1
    parts = [hands.reshape(len(vecs), -1), vecs[:, 2 * n : FRAME_DIM][:, ::-1]]
    if vecs.shape[1] > FRAME_DIM:  # hand_position values, [x, y, size] per slot
        pos = vecs[:, FRAME_DIM:].reshape(len(vecs), 2, 3)[:, ::-1].copy()
        pos[..., 0] *= -1
        parts.append(pos.reshape(len(vecs), 6))
    return np.concatenate(parts, axis=1)


def main_hand_is_left(hands, aspect):
    """numpy version of features.main_hand_is_left for one window of (T, 2, 21, 3) hands."""
    w = hands[:, :, WRIST, :2].astype(np.float64) * [aspect, 1.0]
    present = (~np.isnan(w).any(-1)).sum(0)
    if present[0] != present[1] and present.min() == 0:
        return bool(present[0] > present[1])
    moved = np.nan_to_num(np.linalg.norm(np.diff(w, axis=0), axis=-1)).sum(0)
    return bool(moved[0] > moved[1])


def window_features(vecs, version=1):
    """numpy version of features.sequence_features over one window of frame vectors; returns (384,) or (896,)."""
    n = len(vecs)
    half = n // 2 or 1
    delta = vecs[half:].mean(0) - vecs[:half].mean(0) if n - half > 0 else np.zeros(vecs.shape[1])
    out = [vecs.mean(0), vecs.std(0), delta]
    if version >= 2:
        for j in range(SEGMENTS):
            lo = j * n // SEGMENTS
            out.append(vecs[lo : max((j + 1) * n // SEGMENTS, lo + 1)].mean(0))
    return np.concatenate(out)
