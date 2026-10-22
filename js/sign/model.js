// On-device sign classifier: a small MLP whose weights come from training/train.py as JSON.
// sequenceFeatures() must stay identical to training/features.py (checked by tests/parity).

import { frameVector, NUM_LANDMARKS } from './features.js';

export const MODEL_FORMAT = 'fang-mlp-v1';
export const NO_HANDS = '_none';
const MIN_HAND_FRAMES = 0.3;
const SEGMENTS = 4;
const SLOTS = { Left: 0, Right: 1 };

/**
 * Picks `n` frames evenly spread over the last n / fps seconds, so a model sees the same time span
 * whether the phone tracks hands at 8 or 30 frames a second. For target time t it takes the first frame
 * at or after t (the last frame if none), as training/train_tsl51.py does.
 */
export function resampleByTime(frames, n, fps) {
  if (!frames.length) return frames;
  const span = (1000 * n) / fps;
  const end = frames.at(-1).t;
  const out = [];
  let i = 0;
  for (let k = 0; k < n; k++) {
    const target = end - span + ((k + 0.5) * span) / n;
    while (i < frames.length - 1 && frames[i].t < target) i++;
    out.push(frames[i]);
  }
  return out;
}

/** The same frame signed with the other hand leading: swap the two hand slots and flip x. */
export function mirrorVector(v) {
  const n = NUM_LANDMARKS * 3;
  const out = [...v.slice(n, 2 * n), ...v.slice(0, n), v[2 * n + 1], v[2 * n]];
  for (let i = 0; i < 2 * n; i += 3) out[i] = -out[i];
  return out;
}

/**
 * True when the 'Left' slot hand leads the window: it is the only hand seen, or its wrist moves more.
 * Models with canonicalHand mirror such windows, so left- and right-handed signers look the same.
 */
export function mainHandIsLeft(frames) {
  const present = [0, 0];
  const moved = [0, 0];
  let prev = [null, null];
  for (const f of frames) {
    const aspect = f.aspect || 1;
    const wrist = [null, null];
    for (const hand of f.hands) {
      const slot = SLOTS[hand.handedness];
      if (slot == null || hand.landmarks?.length !== NUM_LANDMARKS) continue;
      const [x, y] = hand.landmarks[0];
      wrist[slot] = [x * aspect, y];
    }
    for (const s of [0, 1]) {
      if (!wrist[s]) continue;
      present[s]++;
      if (prev[s]) moved[s] += Math.hypot(wrist[s][0] - prev[s][0], wrist[s][1] - prev[s][1]);
    }
    prev = wrist;
  }
  if (present[0] !== present[1] && Math.min(...present) === 0) return present[0] > present[1];
  return moved[0] > moved[1];
}

/**
 * Summarizes a window of HandFrames into one vector:
 * [mean of frame vectors, std of frame vectors, mean(second half) - mean(first half)], 3 x 128 values.
 * Mean/std capture the hand shape; the half difference captures the direction of movement.
 * Version 2 adds the mean of each quarter of the window (7 x 128 values), which keeps the order of movements.
 */
export function sequenceFeatures(frames, { version = 1, canonicalHand = false } = {}) {
  let vecs = frames.map(frameVector);
  if (canonicalHand && mainHandIsLeft(frames)) vecs = vecs.map(mirrorVector);
  const n = vecs.length;
  const d = vecs[0]?.length ?? 128;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  const delta = new Array(d).fill(0);
  const segments = version >= 2 ? Array.from({ length: SEGMENTS }, () => new Array(d).fill(0)) : [];
  if (!n) return [...mean, ...std, ...delta, ...segments.flat()];
  const half = Math.floor(n / 2) || 1;
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += vecs[i][j];
    const m = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (vecs[i][j] - m) ** 2;
    let a = 0;
    for (let i = 0; i < half; i++) a += vecs[i][j];
    let b = 0;
    for (let i = half; i < n; i++) b += vecs[i][j];
    mean[j] = m;
    std[j] = Math.sqrt(v / n);
    delta[j] = n - half > 0 ? b / (n - half) - a / half : 0;
    segments.forEach((seg, k) => {
      const lo = Math.floor((k * n) / SEGMENTS);
      const hi = Math.max(Math.floor(((k + 1) * n) / SEGMENTS), lo + 1);
      let c = 0;
      for (let i = lo; i < hi; i++) c += vecs[i][j];
      seg[j] = c / (hi - lo);
    });
  }
  return [...mean, ...std, ...delta, ...segments.flat()];
}

export class MlpModel {
  constructor(json) {
    if (json.format !== MODEL_FORMAT) throw new Error(`unsupported model format ${json.format}`);
    Object.assign(this, json);
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) return null;
    return new MlpModel(await res.json());
  }

  /** Returns { label, text, confidence, alternatives } with probabilities from softmax; text is the word to show. */
  async predict(frames) {
    if (this.fps) frames = resampleByTime(frames, this.window, this.fps);
    // The model never saw empty windows, so without hands in view it would still name a sign.
    const withHands = frames.filter((f) => f.hands.length);
    if (withHands.length < MIN_HAND_FRAMES * frames.length) {
      return { label: NO_HANDS, text: NO_HANDS, confidence: 1, alternatives: [] };
    }
    // Models trained on frames where the tracker often missed the hands (PopSign) skip hand-less frames.
    if (this.skipEmptyFrames) frames = withHands;
    const opts = { version: this.features ?? 1, canonicalHand: Boolean(this.canonicalHand) };
    let h = sequenceFeatures(frames, opts).map((v, i) => (v - this.mean[i]) / this.std[i]);
    // `layers` [{W, b}, ...] with ReLU between them; older models have one hidden layer as W1/b1/W2/b2.
    const layers = this.layers ?? [{ W: this.W1, b: this.b1 }, { W: this.W2, b: this.b2 }];
    layers.forEach(({ W, b }, l) => {
      h = b.map((bias, k) => bias + dot(W[k], h));
      if (l < layers.length - 1) h = h.map((v) => Math.max(0, v));
    });
    const logits = h;
    const max = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const ranked = exps
      .map((e, c) => ({ label: this.labels[c], text: this.display?.[this.labels[c]] ?? this.labels[c], confidence: e / sum }))
      .sort((a, b) => b.confidence - a.confidence);
    return { ...ranked[0], alternatives: ranked.slice(1, 4) };
  }
}

function dot(w, x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += w[i] * x[i];
  return s;
}
