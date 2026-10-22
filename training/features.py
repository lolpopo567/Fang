"""Feature extraction, kept identical to js/sign/features.js and js/sign/model.js.

tests/parity.mjs runs both on the same input and fails if they drift apart.
"""

import math

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
FRAME_DIM = 2 * NUM_LANDMARKS * 3 + 2  # 128
SEGMENTS = 4  # features version 2: mean of each quarter of the window


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


def mirror_vector(v):
    """The same frame signed with the other hand leading: swap the two hand slots and flip x."""
    n = NUM_LANDMARKS * 3
    out = v[n : 2 * n] + v[:n] + [v[2 * n + 1], v[2 * n]]
    for i in range(0, 2 * n, 3):
        out[i] = -out[i]
    return out


def main_hand_is_left(frames):
    """True when the 'Left' slot hand leads the window: it is the only hand seen, or its wrist moves more.
    Models with canonicalHand mirror such windows, so left- and right-handed signers look the same."""
    present = [0, 0]
    moved = [0.0, 0.0]
    prev = [None, None]
    for f in frames:
        aspect = f.get("aspect") or 1.0
        wrist = [None, None]
        for hand in f["hands"]:
            slot = {"Left": 0, "Right": 1}.get(hand.get("handedness"))
            if slot is not None and len(hand["landmarks"]) == NUM_LANDMARKS:
                x, y, _ = hand["landmarks"][WRIST]
                wrist[slot] = (x * aspect, y)
        for s in (0, 1):
            if wrist[s] is not None:
                present[s] += 1
                if prev[s] is not None:
                    moved[s] += math.hypot(wrist[s][0] - prev[s][0], wrist[s][1] - prev[s][1])
            prev[s] = wrist[s]
    if present[0] != present[1] and min(present) == 0:
        return present[0] > present[1]
    return moved[0] > moved[1]


def sequence_features(frames, version=1, canonical_hand=False):
    """[mean, std, mean(second half) - mean(first half)] over frame vectors; 384 values.
    Version 2 adds the mean of each of SEGMENTS equal parts of the window (896 values), which keeps
    the order of movements; canonical_hand mirrors windows led by the 'Left' slot hand."""
    vecs = [frame_vector(f) for f in frames]
    if canonical_hand and main_hand_is_left(frames):
        vecs = [mirror_vector(v) for v in vecs]
    n = len(vecs)
    if n == 0:
        return [0.0] * ((3 + (SEGMENTS if version >= 2 else 0)) * FRAME_DIM)
    half = n // 2 or 1
    mean, std, delta = [], [], []
    for j in range(FRAME_DIM):
        col = [v[j] for v in vecs]
        m = sum(col) / n
        mean.append(m)
        std.append(math.sqrt(sum((c - m) ** 2 for c in col) / n))
        a = sum(col[:half]) / half
        delta.append(sum(col[half:]) / (n - half) - a if n - half > 0 else 0.0)
    out = mean + std + delta
    if version >= 2:
        for j in range(SEGMENTS):
            lo = j * n // SEGMENTS
            seg = vecs[lo : max((j + 1) * n // SEGMENTS, lo + 1)]
            out += [sum(v[k] for v in seg) / len(seg) for k in range(FRAME_DIM)]
    return out
