"""Feature extraction, kept identical to js/sign/features.js and js/sign/model.js.

tests/parity.mjs runs both on the same input and fails if they drift apart.
"""

import math

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
FRAME_DIM = 2 * NUM_LANDMARKS * 3 + 2  # 128


def normalize_hand(landmarks, aspect=1.0):
    """aspect = frame width / height. MediaPipe x and z are in units of frame width, y of height,
    so x and z are rescaled to height units first; a portrait phone and a landscape webcam then agree."""
    if not landmarks or len(landmarks) != NUM_LANDMARKS:
        return None
    landmarks = [(x * aspect, y, z * aspect) for x, y, z in landmarks]
    wx, wy, wz = landmarks[WRIST]
    mx, my, mz = landmarks[MIDDLE_MCP]
    scale = math.sqrt((mx - wx) ** 2 + (my - wy) ** 2 + (mz - wz) ** 2) or 1.0
    return [((x - wx) / scale, (y - wy) / scale, (z - wz) / scale) for x, y, z in landmarks]


def frame_vector(frame):
    out = [0.0] * FRAME_DIM
    for hand in frame["hands"]:
        slot = {"Left": 0, "Right": 1}.get(hand.get("handedness"))
        if slot is None:
            continue
        norm = normalize_hand(hand["landmarks"], frame.get("aspect") or 1.0)
        if norm is None:
            continue
        base = slot * NUM_LANDMARKS * 3
        for i, (x, y, z) in enumerate(norm):
            out[base + i * 3] = x
            out[base + i * 3 + 1] = y
            out[base + i * 3 + 2] = z
        out[2 * NUM_LANDMARKS * 3 + slot] = 1.0
    return out


def sequence_features(frames):
    """[mean, std, mean(second half) - mean(first half)] over frame vectors; 384 values."""
    vecs = [frame_vector(f) for f in frames]
    n = len(vecs)
    if n == 0:
        return [0.0] * (3 * FRAME_DIM)
    half = n // 2 or 1
    mean, std, delta = [], [], []
    for j in range(FRAME_DIM):
        col = [v[j] for v in vecs]
        m = sum(col) / n
        mean.append(m)
        std.append(math.sqrt(sum((c - m) ** 2 for c in col) / n))
        a = sum(col[:half]) / half
        delta.append(sum(col[half:]) / (n - half) - a if n - half > 0 else 0.0)
    return mean + std + delta
