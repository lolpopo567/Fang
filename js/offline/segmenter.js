// Cuts a stream of 16 kHz audio into utterances using per-frame speech probabilities from the
// voice detector. The recogniser is not a streaming model, so it runs on each finished utterance
// ("final"), and also every so often on the utterance so far ("interim") so text appears while
// the person is still talking.
export const FRAME = 512; // samples per voice-detector frame (32 ms at 16 kHz)
const FRAME_MS = (FRAME / 16000) * 1000;

export class Segmenter {
  constructor({
    startProb = 0.5, endProb = 0.35, silenceMs = 600, preRollMs = 300, tailMs = 200,
    minSpeechMs = 250, maxMs = 12000, interimMs = 1500
  } = {}) {
    Object.assign(this, { startProb, endProb });
    this.silenceFrames = Math.round(silenceMs / FRAME_MS);
    this.preRollFrames = Math.round(preRollMs / FRAME_MS);
    this.tailFrames = Math.round(tailMs / FRAME_MS);
    this.minSpeechFrames = Math.round(minSpeechMs / FRAME_MS);
    this.maxFrames = Math.round(maxMs / FRAME_MS);
    this.interimFrames = Math.round(interimMs / FRAME_MS);
    this.reset();
  }

  reset() {
    this.speaking = false;
    this.preRoll = [];
    this.frames = [];
    this.speech = 0;
    this.silence = 0;
    this.sinceInterim = 0;
  }

  // Returns a list of { type: 'interim' | 'final', audio: Float32Array }.
  push(frame, prob) {
    const out = [];
    if (!this.speaking) {
      this.preRoll.push(frame);
      if (this.preRoll.length > this.preRollFrames) this.preRoll.shift();
      if (prob >= this.startProb) {
        this.speaking = true;
        this.frames = this.preRoll;
        this.preRoll = [];
        this.speech = 1;
        this.silence = 0;
        this.sinceInterim = 0;
      }
      return out;
    }
    this.frames.push(frame);
    this.sinceInterim++;
    if (prob < this.endProb) this.silence++;
    else { this.silence = 0; this.speech++; }

    if (this.silence >= this.silenceFrames) {
      const keep = this.frames.length - this.silence + this.tailFrames;
      if (this.speech >= this.minSpeechFrames) out.push({ type: 'final', audio: concat(this.frames.slice(0, keep)) });
      this.reset();
    } else if (this.frames.length >= this.maxFrames) {
      // Very long speech without a pause: cut here and carry on in a new utterance.
      out.push({ type: 'final', audio: concat(this.frames) });
      this.frames = [];
      this.speech = 0;
      this.silence = 0;
      this.sinceInterim = 0;
    } else if (this.sinceInterim >= this.interimFrames) {
      this.sinceInterim = 0;
      out.push({ type: 'interim', audio: concat(this.frames) });
    }
    return out;
  }

  // Ends the current utterance, e.g. when the user presses stop.
  flush() {
    const out = [];
    if (this.speaking && this.speech >= this.minSpeechFrames) out.push({ type: 'final', audio: concat(this.frames) });
    this.reset();
    return out;
  }
}

export function concat(frames) {
  let n = 0;
  for (const f of frames) n += f.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const f of frames) { out.set(f, o); o += f.length; }
  return out;
}

// Streaming linear-interpolation resampler, for browsers that will not open the microphone at
// 16 kHz directly. Speech energy sits well below 8 kHz, so this is good enough for recognition.
export class Resampler {
  constructor(from, to = 16000) {
    this.ratio = from / to;
    this.pos = 0;
    this.last = 0;
  }

  process(input) {
    if (this.ratio === 1) return input;
    const out = [];
    // pos is measured in input samples, relative to the start of `input`; index -1 is `last`.
    while (this.pos < input.length - 1) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = i < 0 ? this.last : input[i];
      out.push(a + (input[i + 1] - a) * frac);
      this.pos += this.ratio;
    }
    this.pos -= input.length;
    this.last = input[input.length - 1];
    return Float32Array.from(out);
  }
}

// Fixes spellings the model learned from its training text (nikhahit + sara aa instead of sara am),
// and writes Thai digits as 0-9, which is how most everyday Thai text writes numbers.
export function cleanText(text) {
  return text
    .replace(/\u0E4D\u0E32/g, '\u0E33')
    .replace(/[\u0E50-\u0E59]/g, (d) => String(d.charCodeAt(0) - 0x0E50))
    .trim();
}
