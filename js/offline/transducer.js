// Greedy decoding for a k2/icefall Zipformer transducer exported to ONNX (encoder, decoder,
// joiner). `ort` is the onnxruntime-web module, passed in so this file also runs under Node tests.
import { fbank, NUM_BINS } from './fbank.js';

const BLANK = 0;
const CONTEXT = 2;

export function parseTokens(text) {
  const tokens = [];
  for (const line of text.split('\n')) {
    const m = line.match(/^(.*) (\d+)$/);
    if (m) tokens[Number(m[2])] = m[1];
  }
  return tokens;
}

export function tokensToText(tokens, ids) {
  return ids.map((id) => tokens[id] ?? '').join('').replace(/▁/g, ' ').replace(/\s+/g, ' ').trim();
}

export class Transducer {
  constructor(ort, { encoder, decoder, joiner, tokens }) {
    this.ort = ort;
    this.encoder = encoder;
    this.decoder = decoder;
    this.joiner = joiner;
    this.tokens = tokens;
  }

  static async create(ort, { encoder, decoder, joiner, tokens }, options = {}) {
    const make = (bytes) => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], ...options });
    const [enc, dec, join] = await Promise.all([make(encoder), make(decoder), make(joiner)]);
    return new Transducer(ort, { encoder: enc, decoder: dec, joiner: join, tokens: parseTokens(tokens) });
  }

  async runDecoder(context) {
    const y = new this.ort.Tensor('int64', BigInt64Array.from(context.map(BigInt)), [1, CONTEXT]);
    const out = await this.decoder.run({ y });
    return out.decoder_out;
  }

  // samples: Float32Array, 16 kHz mono in [-1, 1]. Returns the recognised text.
  async recognize(samples) {
    const { data, frames } = fbank(samples);
    if (frames < 10) return '';
    const { Tensor } = this.ort;
    const enc = await this.encoder.run({
      x: new Tensor('float32', data, [1, frames, NUM_BINS]),
      x_lens: new Tensor('int64', BigInt64Array.of(BigInt(frames)), [1])
    });
    const encOut = enc.encoder_out;
    const [, T, D] = encOut.dims;
    const len = Math.min(T, Number(enc.encoder_out_lens.data[0]));
    const context = new Array(CONTEXT).fill(BLANK);
    const hyp = [];
    let decOut = await this.runDecoder(context);
    for (let t = 0; t < len; t++) {
      const frame = new Tensor('float32', encOut.data.subarray(t * D, (t + 1) * D), [1, D]);
      const { logit } = await this.joiner.run({ encoder_out: frame, decoder_out: decOut });
      const l = logit.data;
      let best = 0;
      for (let k = 1; k < l.length; k++) if (l[k] > l[best]) best = k;
      // One symbol per frame (the "max symbols per frame = 1" greedy search used by sherpa-onnx).
      if (best !== BLANK && best !== 2) {
        hyp.push(best);
        context.shift();
        context.push(best);
        decOut = await this.runDecoder(context);
      }
    }
    return tokensToText(this.tokens, hyp);
  }
}
