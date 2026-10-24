// Runs offline Thai speech recognition off the main thread.
// In:  { type: 'load' } | { type: 'audio', samples: Float32Array, rate } | { type: 'flush' } | { type: 'reset' }
// Out: { type: 'ready', loadMs } | { type: 'speech', active } | { type: 'interim', text } |
//      { type: 'final', text, ms, seconds } | { type: 'error', message }
import * as ort from '../../vendor/onnxruntime/ort.wasm.bundle.min.mjs';
import { FILES, getFile } from './pack.js';
import { Transducer } from './transducer.js';
import { Segmenter, Resampler, FRAME, cleanText } from './segmenter.js';

let asr = null;
let vad = null;
let segmenter = new Segmenter();
let resampler = null;
let pending = new Float32Array(0);
const finals = [];
let interim = null;
let busy = false;
let audioChain = Promise.resolve();

const bytes = async (file) => {
  const res = await getFile(file.url);
  if (!res.ok) throw new Error(`missing ${file.url.split('/').pop()}`);
  return new Uint8Array(await res.arrayBuffer());
};

class Vad {
  constructor(session) {
    this.session = session;
    this.sr = new ort.Tensor('int64', BigInt64Array.of(16000n), []);
    this.reset();
  }
  reset() {
    this.state = new ort.Tensor('float32', new Float32Array(256), [2, 1, 128]);
    this.context = new Float32Array(64);
  }
  async prob(frame) {
    const input = new Float32Array(64 + frame.length);
    input.set(this.context);
    input.set(frame, 64);
    this.context = frame.slice(-64);
    const out = await this.session.run({ input: new ort.Tensor('float32', input, [1, input.length]), state: this.state, sr: this.sr });
    this.state = out.stateN;
    return out.output.data[0];
  }
}

async function load() {
  const t0 = performance.now();
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot enable cross-origin isolation, so no threads.
  ort.env.wasm.wasmBinary = await bytes(FILES.ortWasm);
  const [encoder, decoder, joiner, vadModel, tokens] = await Promise.all([
    bytes(FILES.encoder), bytes(FILES.decoder), bytes(FILES.joiner), bytes(FILES.vad),
    getFile(FILES.tokens.url).then((r) => r.text())
  ]);
  asr = await Transducer.create(ort, { encoder, decoder, joiner, tokens });
  vad = new Vad(await ort.InferenceSession.create(vadModel, { executionProviders: ['wasm'] }));
  // Warm up so the first real utterance is not slowed by one-time setup.
  await asr.recognize(new Float32Array(16000));
  postMessage({ type: 'ready', loadMs: Math.round(performance.now() - t0) });
}

async function onAudio(samples, rate) {
  if (!resampler || resampler.from !== rate) { resampler = new Resampler(rate); resampler.from = rate; }
  const s = resampler.process(samples);
  const merged = new Float32Array(pending.length + s.length);
  merged.set(pending);
  merged.set(s, pending.length);
  let o = 0;
  for (; o + FRAME <= merged.length; o += FRAME) {
    const frame = merged.slice(o, o + FRAME);
    const p = await vad.prob(frame);
    const wasSpeaking = segmenter.speaking;
    handle(segmenter.push(frame, p));
    if (segmenter.speaking !== wasSpeaking) postMessage({ type: 'speech', active: segmenter.speaking });
  }
  pending = merged.slice(o);
}

function handle(events) {
  for (const e of events) {
    if (e.type === 'final') { finals.push(e.audio); interim = null; }
    else interim = e.audio; // only the newest interim matters
  }
  pump();
}

async function pump() {
  if (busy) return;
  busy = true;
  try {
    while (finals.length || interim) {
      if (finals.length) {
        const audio = finals.shift();
        const t = performance.now();
        const text = cleanText(await asr.recognize(audio));
        postMessage({ type: 'final', text, ms: Math.round(performance.now() - t), seconds: audio.length / 16000 });
      } else {
        const audio = interim;
        interim = null;
        const text = cleanText(await asr.recognize(audio));
        // A final for this utterance may have been queued meanwhile; then this interim is stale.
        if (!finals.length) postMessage({ type: 'interim', text });
      }
    }
  } catch (err) {
    postMessage({ type: 'error', message: String(err?.message || err) });
  } finally {
    busy = false;
  }
}

function reset() {
  segmenter = new Segmenter();
  vad?.reset();
  pending = new Float32Array(0);
  interim = null;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'load') {
    load().catch((err) => postMessage({ type: 'error', message: String(err?.message || err), fatal: true }));
  } else if (msg.type === 'audio') {
    if (!vad) return;
    audioChain = audioChain.then(() => onAudio(msg.samples, msg.rate))
      .catch((err) => postMessage({ type: 'error', message: String(err?.message || err) }));
  } else if (msg.type === 'flush') {
    audioChain = audioChain.then(() => { handle(segmenter.flush()); reset(); });
  } else if (msg.type === 'reset') {
    audioChain = audioChain.then(reset);
  }
};
