// Kaldi-style log-mel filterbank features, matching what the icefall/k2 Thai Zipformer was trained
// on: 16 kHz input in [-1, 1], 25 ms frames every 10 ms, Povey window, pre-emphasis 0.97, DC
// removal, 512-point FFT, 80 mel bins from 20 Hz to 7600 Hz, no dither, snip_edges=false.
export const SAMPLE_RATE = 16000;
export const NUM_BINS = 80;
const FRAME = 400;
const SHIFT = 160;
const FFT = 512;
const PREEMPH = 0.97;
const EPS = 1.1920928955078125e-7;

const mel = (f) => 1127 * Math.log(1 + f / 700);

const window = new Float32Array(FRAME);
for (let i = 0; i < FRAME; i++) window[i] = Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FRAME - 1)), 0.85);

// Sparse mel filters: for each bin, the first FFT index it covers and its weights.
const filters = (() => {
  const lo = mel(20), hi = mel(SAMPLE_RATE / 2 - 400);
  const delta = (hi - lo) / (NUM_BINS + 1);
  const width = SAMPLE_RATE / FFT;
  const out = [];
  for (let b = 0; b < NUM_BINS; b++) {
    const left = lo + b * delta, center = left + delta, right = center + delta;
    let start = -1;
    const w = [];
    for (let i = 0; i < FFT / 2; i++) {
      const m = mel(i * width);
      if (m > left && m < right) {
        if (start < 0) start = i;
        w.push(m <= center ? (m - left) / (center - left) : (right - m) / (right - center));
      }
    }
    out.push({ start, weights: Float32Array.from(w) });
  }
  return out;
})();

// Radix-2 FFT tables for 512 points.
const bitrev = new Uint16Array(FFT);
for (let i = 0, bits = Math.log2(FFT); i < FFT; i++) {
  let r = 0;
  for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
  bitrev[i] = r;
}
const cosT = new Float64Array(FFT / 2), sinT = new Float64Array(FFT / 2);
for (let i = 0; i < FFT / 2; i++) { cosT[i] = Math.cos((2 * Math.PI * i) / FFT); sinT[i] = -Math.sin((2 * Math.PI * i) / FFT); }

function fftPower(re, im, power) {
  for (let i = 0; i < FFT; i++) {
    const j = bitrev[i];
    if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let size = 2; size <= FFT; size <<= 1) {
    const half = size >> 1, step = FFT / size;
    for (let s = 0; s < FFT; s += size) {
      for (let k = 0; k < half; k++) {
        const c = cosT[k * step], n = sinT[k * step];
        const a = s + k, b = a + half;
        const tr = re[b] * c - im[b] * n, ti = re[b] * n + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
  for (let i = 0; i <= FFT / 2; i++) power[i] = re[i] * re[i] + im[i] * im[i];
}

export function numFrames(numSamples) {
  return Math.floor((numSamples + SHIFT / 2) / SHIFT);
}

// samples: Float32Array at 16 kHz. Returns { data: Float32Array(frames * 80), frames }.
export function fbank(samples) {
  const n = samples.length;
  const frames = numFrames(n);
  const data = new Float32Array(frames * NUM_BINS);
  const re = new Float64Array(FFT), im = new Float64Array(FFT), power = new Float64Array(FFT / 2 + 1);
  const buf = new Float64Array(FRAME);
  for (let f = 0; f < frames; f++) {
    const start = f * SHIFT + SHIFT / 2 - FRAME / 2;
    let mean = 0;
    for (let i = 0; i < FRAME; i++) {
      let s = start + i;
      if (s < 0) s = -s - 1;
      else if (s >= n) s = 2 * n - 1 - s;
      buf[i] = samples[Math.min(Math.max(s, 0), n - 1)];
      mean += buf[i];
    }
    mean /= FRAME;
    for (let i = 0; i < FRAME; i++) buf[i] -= mean;
    for (let i = FRAME - 1; i > 0; i--) buf[i] -= PREEMPH * buf[i - 1];
    buf[0] -= PREEMPH * buf[0];
    re.fill(0); im.fill(0);
    for (let i = 0; i < FRAME; i++) re[i] = buf[i] * window[i];
    fftPower(re, im, power);
    const o = f * NUM_BINS;
    for (let b = 0; b < NUM_BINS; b++) {
      const { start: s0, weights } = filters[b];
      let e = 0;
      for (let k = 0; k < weights.length; k++) e += weights[k] * power[s0 + k];
      data[o + b] = Math.log(Math.max(e, EPS));
    }
  }
  return { data, frames };
}
