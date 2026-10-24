import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toHandFrame, normalizeHand, frameVector, FrameWindow, HandLabeler } from '../js/sign/features.js';
import { OfflineRecognizer, OnlineRecognizer, RecognizerHub, isUncertain } from '../js/sign/recognizer.js';

const hand = (dx = 0, s = 1) => Array.from({ length: 21 }, (_, i) => [0.5 + dx + (i === 9 ? 0.1 * s : i * 0.01 * s), 0.5, 0]);

test('toHandFrame maps MediaPipe result to plain arrays', () => {
  const f = toHandFrame({ landmarks: [hand().map(([x, y, z]) => ({ x, y, z }))], handedness: [[{ categoryName: 'Right', score: 0.9 }]] }, 42);
  assert.equal(f.t, 42);
  assert.equal(f.hands[0].handedness, 'Right');
  assert.equal(f.hands[0].landmarks.length, 21);
  assert.deepEqual(toHandFrame({ landmarks: [] }, 1).hands, []);
});

test('normalizeHand is invariant to position and scale', () => {
  const a = normalizeHand(hand(0, 1));
  const b = normalizeHand(hand(0.2, 1));
  const c = normalizeHand(hand(0, 2));
  a.forEach((p, i) => p.forEach((v, j) => {
    assert.ok(Math.abs(v - b[i][j]) < 1e-9);
    assert.ok(Math.abs(v - c[i][j]) < 1e-9);
  }));
  assert.deepEqual(a[0], [0, 0, 0]);
  assert.equal(normalizeHand([[0, 0, 0]]), null);
});

test('frameVector has fixed length and presence flags', () => {
  const v = frameVector({ t: 0, hands: [{ handedness: 'Left', landmarks: hand() }] });
  assert.equal(v.length, 128);
  assert.equal(v[126], 1);
  assert.equal(v[127], 0);
  assert.equal(frameVector({ t: 0, hands: [] }).every((x) => x === 0), true);
});

test('FrameWindow keeps the last N frames', () => {
  const w = new FrameWindow(3);
  for (let i = 0; i < 5; i++) w.push({ t: i, hands: [] });
  assert.ok(w.full);
  assert.deepEqual(w.frames.map((f) => f.t), [2, 3, 4]);
  assert.equal(w.toVectors().length, 3);
});

test('HandLabeler keeps both hands when the tracker gives them the same label', () => {
  const at = (handedness, dx, score = 0.9) => ({ handedness, score, landmarks: hand(dx) });
  const lab = new HandLabeler();
  // No history: the more confident hand keeps its label, the other gets the free one.
  let f = lab.push({ t: 0, aspect: 1, hands: [at('Right', -0.2, 0.6), at('Right', 0.2, 0.9)] });
  assert.deepEqual(f.hands.map((h) => h.handedness), ['Left', 'Right']);
  assert.equal(frameVector(f).slice(126).join(), '1,1');
  // With history each hand keeps the slot it was last seen nearest, whatever the scores say.
  f = lab.push({ t: 1, aspect: 1, hands: [at('Left', 0.19, 0.95), at('Left', -0.19, 0.5)] });
  assert.deepEqual(f.hands.map((h) => h.handedness), ['Right', 'Left']);
  // Labels that already differ are left alone.
  const ok = { t: 2, aspect: 1, hands: [at('Right', -0.2), at('Left', 0.2)] };
  assert.equal(lab.push(ok), ok);
});

test('offline recognizer reports no_model until a model is plugged in', async () => {
  assert.equal((await new OfflineRecognizer().predict([])).status, 'no_model');
  const r = await new OfflineRecognizer({ predict: async () => ({ label: 'สวัสดี', confidence: 0.4 }) }).predict([]);
  assert.equal(r.label, 'สวัสดี');
  assert.ok(isUncertain(r));
});

test('online recognizer sends landmarks and handles failures', async () => {
  let sent;
  const ok = new OnlineRecognizer('https://x/api', { fetchImpl: async (_u, o) => { sent = JSON.parse(o.body); return { ok: true, json: async () => ({ label: 'ขอบคุณ', confidence: 0.9 }) }; } });
  const r = await ok.predict([{ t: 0, hands: [] }]);
  assert.equal(sent.version, 1);
  assert.equal(r.status, 'ok');
  assert.equal(isUncertain(r), false);
  assert.equal((await new OnlineRecognizer('').predict([])).status, 'not_configured');
  const bad = new OnlineRecognizer('https://x', { fetchImpl: async () => { throw new Error('offline'); } });
  assert.equal((await bad.predict([])).status, 'error');
  const slow = new OnlineRecognizer('https://x', { timeoutMs: 20, fetchImpl: (_u, o) => new Promise((_, rej) => o.signal.addEventListener('abort', () => rej(Object.assign(new Error(), { name: 'AbortError' })))) });
  assert.equal((await slow.predict([])).error, 'timeout');
});

test('hybrid mode falls back from offline to online', async () => {
  const online = new OnlineRecognizer('https://x', { fetchImpl: async () => ({ ok: true, json: async () => ({ label: 'a', confidence: 1 }) }) });
  const hub = new RecognizerHub(new OfflineRecognizer(), online);
  hub.mode = 'hybrid';
  assert.equal((await hub.predict([])).source, 'online');
  online.endpoint = '';
  assert.equal((await hub.predict([])).status, 'no_model');
});

test('sound alert groups take the best score among their classes', async () => {
  const { groupScores } = await import('../js/alerts/sounds.js');
  const s = groupScores([{ categoryName: 'Knock', score: 0.4 }, { categoryName: 'Doorbell', score: 0.7 }, { categoryName: 'Bark', score: 0.2 }]);
  assert.equal(s.door, 0.7);
  assert.equal(s.dog, 0.2);
  assert.equal(s.baby, 0);
});

test('every alert label is a real YAMNet class', async () => {
  const { ALL_LABELS } = await import('../js/alerts/sounds.js');
  // The model file ends with a zip holding yamnet_label_list.txt (stored uncompressed).
  const fs = await import('node:fs');
  const model = fs.readFileSync(new URL('../models/yamnet.tflite', import.meta.url)).toString('latin1');
  const labels = new Set(model.slice(model.lastIndexOf('Speech\n')).split('\n'));
  for (const l of ALL_LABELS) assert.ok(labels.has(l), l);
});

test('alert detector needs two loud windows and then waits out the cooldown', async () => {
  const { AlertDetector } = await import('../js/alerts/sounds.js');
  const d = new AlertDetector({ threshold: 0.3, frames: 2, cooldownMs: 10000, enabled: new Set(['door', 'dog']) });
  assert.deepEqual(d.push({ door: 0.5 }, 0), []);          // (0.5 + 0) / 2 is under 0.3
  assert.deepEqual(d.push({ door: 0.2 }, 500).map((a) => a.id), ['door']); // (0.5 + 0.2) / 2 = 0.35
  assert.deepEqual(d.push({ door: 0.9 }, 1000), []);       // cooling down
  assert.deepEqual(d.push({ door: 0.9, dog: 0.8 }, 11000).map((a) => a.id), ['door', 'dog']);
  assert.deepEqual(d.push({ baby: 1 }, 12000), []);        // baby is not enabled; door and dog still cooling down
  const quiet = new AlertDetector({ threshold: 0.3 });
  assert.deepEqual([0.5, 0.05, 0.5, 0.05].flatMap((v, i) => quiet.push({ door: v }, i * 500)), []);
});

test('ring buffer keeps the newest samples', async () => {
  const { RingBuffer } = await import('../js/alerts/sounds.js');
  const b = new RingBuffer(4);
  b.push(new Float32Array([1, 2, 3]));
  assert.equal(b.full, false);
  b.push(new Float32Array([4, 5]));
  assert.deepEqual([...b.data], [2, 3, 4, 5]);
  b.push(new Float32Array([6, 7, 8, 9, 10]));
  assert.deepEqual([...b.data], [7, 8, 9, 10]);
  assert.ok(b.full);
});
