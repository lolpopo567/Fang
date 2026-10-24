// Writes app/src/test/resources/parity.json: inputs and the web app's outputs for the sign pipeline, so
// ParityTest can check the Kotlin port gives the same numbers. Run from android/: node tools/make-parity-fixture.mjs
import fs from 'node:fs';
import { frameVector, HandLabeler } from '../../js/sign/features.js';
import { sequenceFeatures, resampleByTime, MlpModel } from '../../js/sign/model.js';

// Small seeded PRNG so the fixture is the same every run.
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);

function hand(label, cx, cy, angle, spread) {
  const landmarks = [];
  for (let i = 0; i < 21; i++) {
    const finger = Math.floor(Math.max(i - 1, 0) / 4);
    const joint = Math.max(i - 1, 0) % 4;
    const a = angle + (finger - 2) * spread;
    const r = i === 0 ? 0 : 0.03 * (joint + 1) + (i === 9 ? 0.02 : 0);
    landmarks.push([cx + r * Math.cos(a) + (rand() - 0.5) * 0.004, cy + r * Math.sin(a) + (rand() - 0.5) * 0.004, (rand() - 0.5) * 0.02]);
  }
  return { handedness: label, score: 0.8 + rand() * 0.2, landmarks };
}

// A clip: hands move over ~2 s at an uneven frame rate, with dropped hands and a both-'Right' mix-up.
function clip(kind) {
  const frames = [];
  let t = 1000;
  for (let k = 0; k < 40; k++) {
    const p = k / 40;
    t += 40 + Math.floor(rand() * 50);
    const hands = [];
    if (kind !== 'left') hands.push(hand('Right', 0.35 + 0.2 * p, 0.55 - 0.1 * p, -1.5 + p, 0.25));
    if (kind !== 'right') hands.push(hand(kind === 'touch' && k % 5 === 0 ? 'Right' : 'Left', 0.65 - 0.15 * p, 0.6, -1.2 - p, 0.3));
    frames.push({ t, aspect: k % 3 === 0 ? 0.75 : 0.5625, hands: k % 9 === 4 ? [] : hands });
  }
  return frames;
}

const models = Object.fromEntries(['tsl', 'asl'].map((m) => [m, new MlpModel(JSON.parse(fs.readFileSync(new URL(`../../models/${m}-model.json`, import.meta.url))))]));
const opts = [[1, false, false, false], [2, false, false, false], [2, true, false, false], [2, false, true, false], [2, true, true, false], [2, false, true, true]];
const cases = [];
for (const kind of ['right', 'left', 'both', 'touch']) {
  const raw = clip(kind);
  const labeler = new HandLabeler();
  const frames = raw.map((f) => labeler.push(f));
  const c = {
    kind,
    raw,
    labeled: frames.map((f) => f.hands.map((h) => h.handedness)),
    frameVectors: [0, 7, 20].map((i) => frameVector(frames[i], true)),
    resampledT: resampleByTime(frames, 24, 15).map((f) => f.t),
    sequence: opts.map(([version, canonicalHand, handPosition, mirrored]) => ({
      version, canonicalHand, handPosition, mirrored,
      out: sequenceFeatures(frames, { version, canonicalHand, handPosition, mirrored })
    })),
    predictions: {}
  };
  for (const [name, model] of Object.entries(models)) {
    const p = await model.predict(frames);
    c.predictions[name] = { label: p.label, confidence: p.confidence, alternatives: p.alternatives.map((a) => ({ label: a.label, confidence: a.confidence })) };
  }
  cases.push(c);
}
fs.writeFileSync(new URL('../app/src/test/resources/parity.json', import.meta.url), JSON.stringify({ cases }));
console.log(cases.map((c) => `${c.kind}: tsl ${c.predictions.tsl.label} ${c.predictions.tsl.confidence.toFixed(3)}, asl ${c.predictions.asl.label} ${c.predictions.asl.confidence.toFixed(3)}`).join('\n'));
