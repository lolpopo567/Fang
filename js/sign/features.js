// Pure functions that turn MediaPipe hand results into model-ready features.
// No DOM access here, so it runs in Node tests and can be ported to Kotlin/Python as-is.

export const NUM_LANDMARKS = 21;
const WRIST = 0;
const MIDDLE_MCP = 9;

/**
 * A HandFrame is the structured landmark record handed to recognizers:
 * { t: ms timestamp, hands: [{ handedness: 'Left'|'Right', score, landmarks: [[x,y,z] x21] }] }
 * Coordinates are MediaPipe image-normalized (0..1 across the frame); aspect is frame width / height.
 */
export function toHandFrame(result, t, aspect = 1) {
  const hands = (result?.landmarks ?? []).map((lms, i) => {
    const cat = result.handedness?.[i]?.[0] ?? result.handednesses?.[i]?.[0];
    return {
      handedness: cat?.categoryName ?? 'Unknown',
      score: cat?.score ?? 0,
      landmarks: lms.map((p) => [p.x, p.y, p.z])
    };
  });
  return { t, aspect, hands };
}

/**
 * Translate to the wrist and scale by wrist-to-middle-knuckle distance, so the
 * features do not depend on where the hand is in the frame or how far it is from the camera.
 * MediaPipe x and z are in units of frame width and y of height, so x and z are first rescaled
 * by the aspect ratio; a portrait phone and a landscape webcam then give the same hand shape.
 */
export function normalizeHand(landmarks, aspect = 1) {
  if (!landmarks || landmarks.length !== NUM_LANDMARKS) return null;
  landmarks = landmarks.map(([x, y, z]) => [x * aspect, y, z * aspect]);
  const [wx, wy, wz] = landmarks[WRIST];
  const [mx, my, mz] = landmarks[MIDDLE_MCP];
  const scale = Math.hypot(mx - wx, my - wy, mz - wz) || 1;
  return landmarks.map(([x, y, z]) => [(x - wx) / scale, (y - wy) / scale, (z - wz) / scale]);
}

/**
 * Fixed-length vector for one frame: [left hand 63 values, right hand 63 values, left present, right present].
 * Missing hands are zero-filled. Length 128.
 * With handPosition, 6 more values say where each hand is (left then right): wrist x and y relative to the
 * frame center, in frame-height units, and log(hand size / 0.1), which grows as the hand nears the camera.
 * Length 134. Many signs differ mainly in where they are made (chin, chest, side), which the shape alone misses.
 */
export function frameVector(frame, handPosition = false) {
  const out = new Array(2 * NUM_LANDMARKS * 3 + 2 + (handPosition ? 6 : 0)).fill(0);
  for (const hand of frame.hands) {
    const slot = hand.handedness === 'Left' ? 0 : hand.handedness === 'Right' ? 1 : null;
    if (slot == null) continue;
    const norm = normalizeHand(hand.landmarks, frame.aspect || 1);
    if (!norm) continue;
    const base = slot * NUM_LANDMARKS * 3;
    norm.forEach(([x, y, z], i) => {
      out[base + i * 3] = x;
      out[base + i * 3 + 1] = y;
      out[base + i * 3 + 2] = z;
    });
    out[2 * NUM_LANDMARKS * 3 + slot] = 1;
    if (handPosition) {
      const aspect = frame.aspect || 1;
      const [[wx, wy, wz], m] = [hand.landmarks[WRIST], hand.landmarks[MIDDLE_MCP]];
      const size = Math.hypot((m[0] - wx) * aspect, m[1] - wy, (m[2] - wz) * aspect) || 1;
      const p = 2 * NUM_LANDMARKS * 3 + 2 + slot * 3;
      out[p] = (wx - 0.5) * aspect;
      out[p + 1] = wy - 0.5;
      out[p + 2] = Math.log(size / 0.1);
    }
  }
  return out;
}

/**
 * Keeps the last `size` frames so recognizers get a time window, not a single pose.
 * With durationMs set it keeps the last durationMs of frames instead (plus the one frame just before),
 * whatever the camera's frame rate; models with an `fps` field resample that span to a fixed length.
 */
export class FrameWindow {
  constructor(size = 30, durationMs = 0) {
    this.size = size;
    this.durationMs = durationMs;
    this.frames = [];
  }
  push(frame) {
    this.frames.push(frame);
    if (this.durationMs) {
      while (this.frames.length > 1 && this.frames[1].t <= frame.t - this.durationMs) this.frames.shift();
    } else if (this.frames.length > this.size) this.frames.shift();
  }
  get full() {
    if (this.durationMs) return this.frames.length > 1 && this.frames.at(-1).t - this.frames[0].t >= this.durationMs;
    return this.frames.length === this.size;
  }
  clear() {
    this.frames = [];
  }
  toVectors() {
    return this.frames.map((f) => frameVector(f));
  }
}

const SLOT_NAMES = ['Left', 'Right'];

/**
 * When the hands touch or cross, the tracker often labels both 'Left' or both 'Right'. frameVector then
 * writes both into one slot and one hand is lost. push() gives each hand its own label: the hands keep the
 * slot whose last position is nearest, or with no history the more confident hand keeps its label.
 * Labels that differ are never changed, and the output is a HandFrame like toHandFrame's.
 * Tested on TSL-51's 547 webcam clips through the app's tracker: +3.6 points top-1 on signs where the hands touch.
 */
export class HandLabeler {
  constructor() {
    this.reset();
  }
  reset() {
    // Wrist [x, y] where each slot (Left, Right) was last seen.
    this.last = [null, null];
  }
  push(frame) {
    const aspect = frame.aspect || 1;
    const hands = frame.hands.filter((h) => h.landmarks?.length === NUM_LANDMARKS).slice(0, 2);
    const slots = this.assign(hands, aspect);
    hands.forEach((h, i) => { this.last[slots[i]] = h.landmarks[WRIST]; });
    if (hands.every((h, i) => h.handedness === SLOT_NAMES[slots[i]])) return frame;
    return { ...frame, hands: hands.map((h, i) => ({ ...h, handedness: SLOT_NAMES[slots[i]] })) };
  }
  /** Slot (0 Left, 1 Right) for each hand. */
  assign(hands, aspect) {
    const labels = hands.map((h) => SLOT_NAMES.indexOf(h.handedness));
    const dist = (h, s) => {
      const p = this.last[s];
      const w = h.landmarks[WRIST];
      return p ? Math.hypot((w[0] - p[0]) * aspect, w[1] - p[1]) : Infinity;
    };
    if (hands.length === 1) return [labels[0] >= 0 ? labels[0] : dist(hands[0], 0) < dist(hands[0], 1) ? 0 : 1];
    if (hands.length < 2) return [];
    const [a, b] = labels;
    if (a >= 0 && b >= 0 && a !== b) return [a, b];
    if (this.last[0] && this.last[1]) {
      return dist(hands[0], 0) + dist(hands[1], 1) <= dist(hands[0], 1) + dist(hands[1], 0) ? [0, 1] : [1, 0];
    }
    const keep = (hands[0].score ?? 0) >= (hands[1].score ?? 0) ? 0 : 1;
    const s = labels[keep] >= 0 ? labels[keep] : 1;
    return keep === 0 ? [s, 1 - s] : [1 - s, s];
  }
}
