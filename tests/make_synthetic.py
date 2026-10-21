"""Synthetic fang-handframes-v1 data for testing the training pipeline only (not real signs)."""
import json, math, random, sys

def hand(spread, angle, dx, dy, rng):
    pts = []
    for i in range(21):
        finger, joint = divmod(max(i - 1, 0), 4)
        a = angle + (finger - 2) * spread
        r = 0.0 if i == 0 else 0.03 * (joint + 1) + (0.02 if i == 9 else 0)
        pts.append([0.5 + dx + r * math.cos(a) + rng.gauss(0, 0.002), 0.5 + dy + r * math.sin(a) + rng.gauss(0, 0.002), rng.gauss(0, 0.002)])
    return pts

def sample(label, signer, rng, n=45):
    frames = []
    for k in range(n):
        p = k / n
        if label == 'A':   hs = hand(0.1, -1.5, 0, 0, rng)
        elif label == 'B': hs = hand(0.4, -1.5, 0.2 * p, 0, rng)
        else:              hs = hand(0.25, -0.5 + p, 0, -0.1 * p, rng)
        frames.append({'t': k * 33, 'hands': [{'handedness': 'Right', 'score': 0.9, 'landmarks': hs}]})
    return {'label': label, 'signer': signer, 'frames': frames}

rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
samples = [sample(l, f's{s}', rng) for s in range(4) for l in 'ABC' for _ in range(6)]
json.dump({'format': 'fang-handframes-v1', 'samples': samples}, open(sys.argv[1], 'w'))
