// On-device sign classifier: a small MLP whose weights come from training/train.py as JSON.
// sequenceFeatures() must stay identical to training/features.py (checked by tests/parity).

import { frameVector } from './features.js';

export const MODEL_FORMAT = 'fang-mlp-v1';
export const NO_HANDS = '_none';
const MIN_HAND_FRAMES = 0.3;

/**
 * Summarizes a window of HandFrames into one vector:
 * [mean of frame vectors, std of frame vectors, mean(second half) - mean(first half)], 3 x 128 values.
 * Mean/std capture the hand shape; the half difference captures the direction of movement.
 */
export function sequenceFeatures(frames) {
  const vecs = frames.map(frameVector);
  const n = vecs.length;
  const d = vecs[0]?.length ?? 128;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  const delta = new Array(d).fill(0);
  if (!n) return [...mean, ...std, ...delta];
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
  }
  return [...mean, ...std, ...delta];
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
    // The model never saw empty windows, so without hands in view it would still name a sign.
    const withHands = frames.filter((f) => f.hands.length);
    if (withHands.length < MIN_HAND_FRAMES * frames.length) {
      return { label: NO_HANDS, text: NO_HANDS, confidence: 1, alternatives: [] };
    }
    // Models trained on frames where the tracker often missed the hands (PopSign) skip hand-less frames.
    if (this.skipEmptyFrames) frames = withHands;
    const x = sequenceFeatures(frames).map((v, i) => (v - this.mean[i]) / this.std[i]);
    const h = this.b1.map((b, k) => Math.max(0, b + dot(this.W1[k], x)));
    const logits = this.b2.map((b, c) => b + dot(this.W2[c], h));
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
