import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { sequenceFeatures, MlpModel, mainHandIsLeft, resampleByTime } from '../js/sign/model.js';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fang-'));
const py = (code, ...args) => execFileSync('python3', ['-c', code, ...args], { cwd: root, encoding: 'utf8' });

test('JS and Python features match', () => {
  const data = path.join(tmp, 'parity.json');
  execFileSync('python3', ['tests/make_synthetic.py', data, '7'], { cwd: root });
  const sample = JSON.parse(fs.readFileSync(data, 'utf8')).samples[4];
  // Add a left hand and a missing frame so every branch is compared.
  sample.frames[3].hands.push({ ...sample.frames[3].hands[0], handedness: 'Left' });
  sample.frames[5].hands = [];
  // Portrait and landscape frames, plus one frame without the field (old recordings).
  sample.frames.forEach((f, i) => { if (i % 3 === 0) f.aspect = 0.5625; else if (i % 3 === 1) f.aspect = 1.333; });
  fs.writeFileSync(data, JSON.stringify(sample));
  for (const [version, canonical, dim] of [[1, false, 384], [2, false, 896], [2, true, 896]]) {
    const pyOut = JSON.parse(py(`import json,sys; sys.path.insert(0,"training"); from features import sequence_features; print(json.dumps(sequence_features(json.load(open(sys.argv[1]))["frames"], ${version}, ${canonical ? 'True' : 'False'})))`, data));
    const jsOut = sequenceFeatures(sample.frames, { version, canonicalHand: canonical });
    assert.equal(jsOut.length, dim);
    assert.equal(pyOut.length, dim);
    jsOut.forEach((v, i) => assert.ok(Math.abs(v - pyOut[i]) < 1e-9, `v${version} feature ${i}: ${v} vs ${pyOut[i]}`));
  }
});

test('numpy clip features (dataset adapters) match features.py', () => {
  const code = [
    'import sys; sys.path.insert(0,"training")',
    'import numpy as np; from features import sequence_features; from npfeatures import app_frame_vectors, window_features',
    'rng=np.random.default_rng(1); h=rng.random((9,2,21,3)).astype(np.float32); h[2,0]=np.nan; h[5]=np.nan; h[7,1]=np.nan',
    'frames=[{"aspect":0.75,"hands":[{"handedness":n,"landmarks":f[s].astype(float).tolist()} for s,n in ((0,"Left"),(1,"Right")) if not np.isnan(f[s]).any()]} for f in h]',
    'print(float(np.abs(window_features(app_frame_vectors(h,0.75))-np.array(sequence_features(frames))).max()))',
    'print(float(np.abs(window_features(app_frame_vectors(h,0.75),2)-np.array(sequence_features(frames,2))).max()))'
  ].join('\n');
  for (const d of py(code).trim().split('\n')) assert.ok(Number(d) < 1e-6);
});

test('canonicalHand mirrors windows led by the left-slot hand', () => {
  const hand = (name, k, dx) => ({ handedness: name, landmarks: Array.from({ length: 21 }, (_, i) => [0.5 + dx + i * 0.01 + k * (name === 'Left' ? 0.03 : 0.002), 0.4 + (i % 5) * 0.02, 0.01 * i]) });
  // Two hands, the 'Left' one moving: a left-handed signer. Mirroring it must give the right-handed version.
  const lefty = Array.from({ length: 8 }, (_, k) => ({ aspect: 1, hands: [hand('Left', k, 0.1), hand('Right', k, -0.1)] }));
  assert.ok(mainHandIsLeft(lefty));
  const flipped = sequenceFeatures(lefty, { version: 2, canonicalHand: true });
  assert.notDeepEqual(flipped, sequenceFeatures(lefty, { version: 2 }));
  // The 'Left' slot's presence flag (mean over frames) moves to the 'Right' slot, and x flips sign.
  assert.equal(flipped[127], 1);
  assert.equal(flipped[63 + 3], -sequenceFeatures(lefty)[3]);
  // A right-led window is left alone.
  const rightLed = lefty.map((f) => ({ ...f, hands: [f.hands[1], { ...f.hands[0], handedness: 'Right' }].map((h, i) => ({ ...h, handedness: i ? 'Right' : 'Left' })) }));
  assert.ok(!mainHandIsLeft(rightLed));
  assert.deepEqual(sequenceFeatures(rightLed, { version: 2, canonicalHand: true }), sequenceFeatures(rightLed, { version: 2 }));
});

test('resampleByTime covers a fixed time span at any frame rate', () => {
  const at = (fps, seconds) => Array.from({ length: Math.round(fps * seconds) }, (_, i) => ({ t: (i * 1000) / fps, hands: [] }));
  for (const fps of [8, 15, 30]) {
    const out = resampleByTime(at(fps, 3), 24, 15);
    assert.equal(out.length, 24);
    const span = out.at(-1).t - out[0].t;
    assert.ok(span > 1300 && span <= 1600, `${fps} fps: span ${span}`);
  }
});

test('skipEmptyFrames models ignore frames without hands', async () => {
  const hand = (k) => ({ handedness: 'Left', landmarks: Array.from({ length: 21 }, (_, i) => [0.3 + i * 0.01 + k * 0.02, 0.5 + (i % 4) * 0.02, 0]) });
  const frames = Array.from({ length: 10 }, (_, k) => ({ aspect: 1, hands: k % 2 ? [] : [hand(k)] }));
  const d = 384;
  const json = { format: 'fang-mlp-v1', labels: ['a', 'b'], window: 10, mean: new Array(d).fill(0), std: new Array(d).fill(1),
    W1: [Array.from({ length: d }, (_, i) => (i % 7) / 7)], b1: [0], W2: [[1], [-1]], b2: [0, 0] };
  const full = await new MlpModel(json).predict(frames);
  const skip = await new MlpModel({ ...json, skipEmptyFrames: true }).predict(frames);
  const only = await new MlpModel(json).predict(frames.filter((f) => f.hands.length));
  assert.notEqual(full.confidence, skip.confidence);
  assert.equal(skip.confidence, only.confidence);
});

test('trained model runs in JS and recognizes held-out synthetic data', async () => {
  const trainData = path.join(tmp, 'train.json');
  const testData = path.join(tmp, 'test.json');
  const out = path.join(tmp, 'model.json');
  execFileSync('python3', ['tests/make_synthetic.py', trainData, '1'], { cwd: root });
  execFileSync('python3', ['tests/make_synthetic.py', testData, '2'], { cwd: root });
  execFileSync('python3', ['training/train.py', trainData, '--out', out, '--epochs', '30', '--min-per-class', '3'], { cwd: root });
  const json = JSON.parse(fs.readFileSync(out, 'utf8'));
  assert.match(json.report.split, /signer-held-out/);
  assert.ok(json.report.test_accuracy > 0.9, `python test accuracy ${json.report.test_accuracy}`);

  const model = new MlpModel(json);
  const samples = JSON.parse(fs.readFileSync(testData, 'utf8')).samples;
  let correct = 0;
  for (const s of samples) {
    const res = await model.predict(s.frames.slice(5, 5 + json.window));
    if (res.label === s.label) correct++;
    assert.ok(res.confidence > 0 && res.confidence <= 1);
  }
  assert.ok(correct / samples.length > 0.9, `js accuracy ${correct}/${samples.length}`);
});

for (const name of ['tsl-model.json', 'asl-model.json']) test(`bundled ${name} gives the same probabilities in JS and Python`, async (t) => {
  const modelPath = path.join(root, 'models', name);
  if (!fs.existsSync(modelPath)) return t.skip('no bundled model');
  const json = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
  assert.equal(json.labels.length, (json.layers?.at(-1).W ?? json.W2).length);
  assert.ok(json.labels.every((l) => typeof json.display[l] === 'string'));
  const data = path.join(tmp, 'bundled.json');
  execFileSync('python3', ['tests/make_synthetic.py', data, '3'], { cwd: root });
  const frames = JSON.parse(fs.readFileSync(data, 'utf8')).samples[7].frames.slice(0, json.window);
  // Uneven timestamps, as from a phone whose frame rate wobbles, exercise resampling for models with fps.
  frames.forEach((f, i) => { f.aspect = 1.333; f.t = i * 70 + (i % 3) * 25; });
  fs.writeFileSync(data, JSON.stringify(frames));
  const code = [
    'import json,sys; sys.path.insert(0,"training")',
    'from train import model_probs',
    'm=json.load(open(sys.argv[1])); p=model_probs(m, json.load(open(sys.argv[2])))',
    'print(json.dumps({"label": m["labels"][int(p.argmax())], "p": float(p.max())}))'
  ].join('\n');
  const pyOut = JSON.parse(py(code, modelPath, data));
  const res = await new MlpModel(json).predict(frames);
  assert.equal(res.label, pyOut.label);
  assert.ok(Math.abs(res.confidence - pyOut.p) < 1e-9);
  assert.equal(res.text, json.display[res.label]);
});
