"""numpy versions of features.py for whole clips, shared by the dataset adapters.

Hands come in as (T, 2, 21, 3) in the app's slot order: slot 0 is the hand MediaPipe HandLandmarker labels
'Left', slot 1 'Right'; NaN where a hand is missing. tsl51.py selftest checks these against features.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import FRAME_DIM, MIDDLE_MCP, NUM_LANDMARKS, WRIST  # noqa: E402


def app_frame_vectors(hands, aspect):
    """numpy version of features.frame_vector for a whole clip; returns (T, 128)."""
    h = hands.astype(np.float64)
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
