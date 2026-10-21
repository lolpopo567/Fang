import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toHandFrame, normalizeHand, frameVector, FrameWindow } from '../js/sign/features.js';
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
