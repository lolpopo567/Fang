// Unit tests for offline captions and transcript helpers. Set FANG_SPEECH_DIR to a folder with the
// speech pack files (encoder/decoder/joiner .int8.onnx, tokens.txt, 0.wav... and trans.txt from the
// Hugging Face repo in js/offline/pack.js) to also run the real model.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fbank, numFrames, NUM_BINS } from '../js/offline/fbank.js';
import { Segmenter, Resampler, FRAME, cleanText } from '../js/offline/segmenter.js';
import { parseTokens, tokensToText } from '../js/offline/transducer.js';
import { chooseEngine } from '../js/captions.js';
import { transcriptText, matches, clock } from '../js/transcripts.js';

test('fbank gives 80 log-mel bins every 10 ms', () => {
  const tone = Float32Array.from({ length: 16000 }, (_, i) => 0.3 * Math.sin((2 * Math.PI * 1000 * i) / 16000));
  const { data, frames } = fbank(tone);
  assert.equal(frames, 100);
  assert.equal(numFrames(16080), 101);
  assert.equal(data.length, frames * NUM_BINS);
  assert.ok(data.every(Number.isFinite));
  // A 1 kHz tone puts its energy in the matching mel bin, not at the edges.
  const row = Array.from(data.subarray(50 * NUM_BINS, 51 * NUM_BINS));
  const peak = row.indexOf(Math.max(...row));
  assert.ok(peak > 15 && peak < 30, `peak bin ${peak}`);
  const silent = fbank(new Float32Array(1600)).data;
  assert.ok(silent.every((v) => v < -15));
});

test('Segmenter emits one final per utterance, with interims while speaking', () => {
  const seg = new Segmenter({ interimMs: 500 });
  const frame = new Float32Array(FRAME);
  const events = [];
  const feed = (n, p) => { for (let i = 0; i < n; i++) events.push(...seg.push(frame, p)); };
  feed(30, 0.05); // silence
  feed(60, 0.9); // ~1.9 s of speech
  feed(30, 0.05); // pause ends it
  const finals = events.filter((e) => e.type === 'final');
  assert.equal(finals.length, 1);
  assert.ok(events.filter((e) => e.type === 'interim').length >= 2);
  // pre-roll + speech + a short tail, but not the whole pause
  const frames = finals[0].audio.length / FRAME;
  assert.ok(frames > 60 && frames < 85, `${frames} frames`);
  // A blip shorter than the minimum speech length is ignored.
  events.length = 0;
  feed(3, 0.9);
  feed(30, 0.05);
  assert.equal(events.filter((e) => e.type === 'final').length, 0);
});

test('Segmenter cuts very long speech and flushes on stop', () => {
  const seg = new Segmenter({ maxMs: 2000, interimMs: 100000 });
  const frame = new Float32Array(FRAME);
  const events = [];
  for (let i = 0; i < 100; i++) events.push(...seg.push(frame, 0.9));
  assert.ok(events.filter((e) => e.type === 'final').length >= 1);
  assert.equal(seg.flush().length, 1);
  assert.equal(seg.flush().length, 0);
});

test('Resampler converts 48 kHz to 16 kHz across chunk boundaries', () => {
  const r = new Resampler(48000);
  let n = 0;
  for (let i = 0; i < 10; i++) n += r.process(new Float32Array(4800).fill(0.5)).length;
  assert.ok(Math.abs(n - 16000) <= 2, `${n}`);
  assert.equal(new Resampler(16000).process(new Float32Array(10)).length, 10);
});

test('cleanText and token joining', () => {
  assert.equal(cleanText('สําหรับ '), 'สำหรับ');
  assert.equal(cleanText('หนึ่งใน๑๐ประเทศ'), 'หนึ่งใน10ประเทศ');
  const tokens = parseTokens('<blk> 0\n▁ 3\nสวัสดี 4\n▁ครับ 5\n');
  assert.equal(tokensToText(tokens, [4, 5]), 'สวัสดี ครับ');
});

test('chooseEngine picks the offline model only when it helps', () => {
  const base = { setting: 'auto', lang: 'th-TH', online: true, browserSupported: true, packInstalled: true };
  assert.equal(chooseEngine(base), 'browser');
  assert.equal(chooseEngine({ ...base, online: false }), 'device');
  assert.equal(chooseEngine({ ...base, browserSupported: false }), 'device');
  assert.equal(chooseEngine({ ...base, online: false, packInstalled: false }), 'browser');
  assert.equal(chooseEngine({ ...base, browserSupported: false, packInstalled: false }), 'none');
  assert.equal(chooseEngine({ ...base, browserSupported: false, lang: 'en-US' }), 'needs-thai');
  assert.equal(chooseEngine({ ...base, setting: 'device' }), 'device');
  assert.equal(chooseEngine({ ...base, setting: 'device', packInstalled: false }), 'needs-pack');
  assert.equal(chooseEngine({ ...base, setting: 'device', lang: 'en-US' }), 'browser');
  assert.equal(chooseEngine({ ...base, setting: 'browser', online: false }), 'browser');
});

test('transcript text, search and times', () => {
  const at = new Date(2026, 9, 24, 9, 5, 7).getTime();
  const doc = { lines: [{ at, text: 'สวัสดีครับ' }, { at: at + 1000, text: 'Hello there' }] };
  assert.equal(clock(at), '09:05:07');
  assert.equal(transcriptText(doc), '[09:05:07] สวัสดีครับ\n[09:05:08] Hello there');
  assert.equal(transcriptText(doc, { times: false }), 'สวัสดีครับ\nHello there');
  assert.ok(matches(doc, 'hello'));
  assert.ok(matches(doc, 'สวัสดี'));
  assert.ok(matches(doc, ' '));
  assert.ok(!matches(doc, 'ลาก่อน'));
});

const dir = process.env.FANG_SPEECH_DIR;
test('offline Thai model transcribes the reference clips', { skip: !dir && 'FANG_SPEECH_DIR not set' }, async () => {
  // Same runtime version as vendor/onnxruntime; the npm build also works under Node.
  const ort = await import('onnxruntime-web/wasm');
  const { Transducer } = await import('../js/offline/transducer.js');
  ort.env.wasm.numThreads = 1;
  const rd = (f) => fs.readFileSync(path.join(dir, f));
  const asr = await Transducer.create(ort, {
    encoder: rd('encoder-epoch-12-avg-5.int8.onnx'), decoder: rd('decoder-epoch-12-avg-5.int8.onnx'),
    joiner: rd('joiner-epoch-12-avg-5.int8.onnx'), tokens: rd('tokens.txt').toString()
  });
  const refs = Object.fromEntries(rd('trans.txt').toString().trim().split('\n').map((l) => l.split('\t')));
  let errors = 0, total = 0;
  for (const [id, ref] of Object.entries(refs)) {
    const hyp = cleanText(await asr.recognize(readWav(rd(`${id}.wav`))));
    const r = [...cleanText(ref)], h = [...hyp];
    errors += editDistance(r, h);
    total += r.length;
    console.log(id, hyp);
  }
  const cer = errors / total;
  console.log('character error rate', cer.toFixed(3));
  assert.ok(cer < 0.1, `CER ${cer}`);
});

function readWav(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let p = 12; p < buf.length;) {
    const id = buf.toString('ascii', p, p + 4), size = dv.getUint32(p + 4, true);
    if (id === 'data') return Float32Array.from({ length: size / 2 }, (_, i) => dv.getInt16(p + 8 + 2 * i, true) / 32768);
    p += 8 + size;
  }
  throw new Error('no data chunk');
}

function editDistance(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[b.length];
}
