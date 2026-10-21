import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { sequenceFeatures, MlpModel } from '../js/sign/model.js';

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
  fs.writeFileSync(data, JSON.stringify(sample));
  const pyOut = JSON.parse(py('import json,sys; sys.path.insert(0,"training"); from features import sequence_features; print(json.dumps(sequence_features(json.load(open(sys.argv[1]))["frames"])))', data));
  const jsOut = sequenceFeatures(sample.frames);
  assert.equal(jsOut.length, 384);
  assert.equal(pyOut.length, 384);
  jsOut.forEach((v, i) => assert.ok(Math.abs(v - pyOut[i]) < 1e-9, `feature ${i}: ${v} vs ${pyOut[i]}`));
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
