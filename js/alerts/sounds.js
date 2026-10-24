// Which sounds raise an alert, and when. Pure logic (no audio, no DOM) so it can be unit-tested.
// Each alert groups a few of YAMNet's 521 AudioSet classes; the label strings must match
// models/yamnet.tflite exactly.
export const ALERT_SOUNDS = [
  { id: 'door', icon: 'i-door', on: true, labels: ['Doorbell', 'Ding-dong', 'Knock'] },
  { id: 'baby', icon: 'i-baby', on: true, labels: ['Baby cry, infant cry', 'Crying, sobbing'] },
  { id: 'alarm', icon: 'i-alarm-smoke', on: true, labels: ['Smoke detector, smoke alarm', 'Fire alarm', 'Alarm', 'Alarm clock', 'Buzzer', 'Car alarm'] },
  { id: 'siren', icon: 'i-siren', on: true, labels: ['Siren', 'Civil defense siren', 'Police car (siren)', 'Ambulance (siren)', 'Fire engine, fire truck (siren)', 'Emergency vehicle'] },
  { id: 'horn', icon: 'i-car', on: true, labels: ['Vehicle horn, car horn, honking', 'Air horn, truck horn'] },
  { id: 'phone', icon: 'i-phone', on: true, labels: ['Telephone bell ringing', 'Ringtone', 'Telephone'] },
  { id: 'dog', icon: 'i-dog', on: true, labels: ['Dog', 'Bark'] },
  { id: 'shout', icon: 'i-megaphone', on: true, labels: ['Shout', 'Yell', 'Screaming', 'Children shouting'] },
  { id: 'glass', icon: 'i-glass', on: true, labels: ['Shatter', 'Smash, crash'] },
  { id: 'water', icon: 'i-droplets', on: false, labels: ['Water tap, faucet', 'Sink (filling or washing)'] },
  { id: 'speech', icon: 'i-message', on: false, labels: ['Speech'] }
];

export const ALL_LABELS = ALERT_SOUNDS.flatMap((s) => s.labels);

// Score needed (0..1) for each sensitivity setting; see docs/sound-alerts.md for how these were chosen.
export const SENSITIVITY = { low: 0.5, normal: 0.3, high: 0.15 };

// A group's score is the highest score among its classes.
export function groupScores(categories) {
  const byLabel = new Map(categories.map((c) => [c.categoryName, c.score]));
  const out = {};
  for (const s of ALERT_SOUNDS) out[s.id] = Math.max(0, ...s.labels.map((l) => byLabel.get(l) ?? 0));
  return out;
}

// Turns a stream of per-window group scores into alerts. A sound must score above the threshold
// on average over the last `frames` windows (so one window alone must be twice as loud), and each sound
// then stays quiet for `cooldownMs` so a ringing doorbell gives one alert, not ten.
export class AlertDetector {
  constructor({ threshold = SENSITIVITY.normal, frames = 2, cooldownMs = 10000, enabled = null } = {}) {
    this.threshold = threshold;
    this.frames = frames;
    this.cooldownMs = cooldownMs;
    this.enabled = enabled ?? new Set(ALERT_SOUNDS.filter((s) => s.on).map((s) => s.id));
    this.history = [];
    this.lastFired = new Map();
  }

  reset() {
    this.history = [];
    this.lastFired.clear();
  }

  // scores: { id: 0..1 }. Returns the alerts to raise now, strongest first.
  push(scores, t) {
    this.history.push(scores);
    if (this.history.length > this.frames) this.history.shift();
    const fired = [];
    for (const id of this.enabled) {
      const avg = this.history.reduce((sum, h) => sum + (h[id] ?? 0), 0) / this.frames;
      if (avg < this.threshold) continue;
      if (t - (this.lastFired.get(id) ?? -Infinity) < this.cooldownMs) continue;
      this.lastFired.set(id, t);
      fired.push({ id, score: avg, t });
    }
    return fired.sort((a, b) => b.score - a.score);
  }
}

// Keeps the last `seconds` of microphone audio so each classification sees a full window.
export class RingBuffer {
  constructor(size) {
    this.data = new Float32Array(size);
    this.filled = 0;
  }

  push(chunk) {
    const n = this.data.length;
    if (chunk.length >= n) {
      this.data.set(chunk.subarray(chunk.length - n));
    } else {
      this.data.copyWithin(0, chunk.length);
      this.data.set(chunk, n - chunk.length);
    }
    this.filled = Math.min(n, this.filled + chunk.length);
  }

  get full() { return this.filled === this.data.length; }

  // Root-mean-square level of the buffer, for the input meter.
  rms() {
    let sum = 0;
    for (const v of this.data) sum += v * v;
    return Math.sqrt(sum / this.data.length);
  }
}
