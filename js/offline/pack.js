// The offline Thai speech pack: a Zipformer transducer trained on GigaSpeech2 Thai by the k2/icefall
// team (Apache-2.0), plus the Silero voice detector and the ONNX runtime. The model is too large
// for GitHub Pages (its encoder alone is 155 MB), so it is downloaded once from Hugging Face,
// pinned to a fixed revision, and kept in its own cache that app updates do not clear.
export const CACHE = 'fang-speech-th-v1';

const HF = 'https://huggingface.co/yfyeung/icefall-asr-gigaspeech2-th-zipformer-2024-06-20/resolve/e33c75c3e75fd0830acec75bc41f4e5d711bbec9/';
const here = (path) => new URL(`../../${path}`, import.meta.url).href;

export const FILES = {
  encoder: { url: `${HF}exp/encoder-epoch-12-avg-5.int8.onnx`, size: 154671320 },
  decoder: { url: `${HF}exp/decoder-epoch-12-avg-5.int8.onnx`, size: 1308690 },
  joiner: { url: `${HF}exp/joiner-epoch-12-avg-5.int8.onnx`, size: 1033417 },
  tokens: { url: `${HF}data/lang_bpe_2000/tokens.txt`, size: 39142 },
  vad: { url: here('models/silero_vad_v5.onnx'), size: 2327524 },
  ortWasm: { url: here('vendor/onnxruntime/ort-wasm-simd-threaded.wasm'), size: 14239897 },
  ort: { url: here('vendor/onnxruntime/ort.wasm.bundle.min.mjs'), size: 73054 },
  // The app's own code for offline captions, so it still loads if the app cache is replaced.
  code: [
    'asr-worker.js', 'fbank.js', 'transducer.js', 'segmenter.js', 'pack.js', 'engine.js', 'capture-worklet.js'
  ].map((f) => here(`js/offline/${f}`))
};

const ALL = () => [
  ...Object.values(FILES).filter((f) => f.url).map((f) => f.url),
  ...FILES.code
];

export const TOTAL_BYTES = Object.values(FILES).reduce((sum, f) => sum + (f.size || 0), 0);

export function supported() {
  return typeof caches !== 'undefined' && typeof Worker !== 'undefined' && typeof WebAssembly !== 'undefined' &&
    !!(navigator.mediaDevices?.getUserMedia) && typeof AudioWorkletNode !== 'undefined';
}

// Returns the cached Response, or fetches it (when online) as a fallback.
export async function getFile(url) {
  const hit = typeof caches !== 'undefined' ? await caches.match(url, { cacheName: CACHE }) : null;
  return hit || fetch(url);
}

export async function isInstalled() {
  if (typeof caches === 'undefined') return false;
  try {
    if (!(await caches.has(CACHE))) return false;
    const cache = await caches.open(CACHE);
    const hits = await Promise.all(ALL().map((u) => cache.match(u)));
    return hits.every(Boolean);
  } catch {
    return false;
  }
}

// Downloads every file into the cache. onProgress(fraction 0..1). Large files report progress
// as they stream. Throws on failure, leaving any files already fetched in the cache so a retry
// only fetches what is missing.
export async function install(onProgress = () => {}, signal) {
  const cache = await caches.open(CACHE);
  // Ask the browser not to evict a 170 MB download when space runs low.
  navigator.storage?.persist?.().catch(() => {});
  const urls = ALL();
  const sizes = new Map(Object.values(FILES).filter((f) => f.url).map((f) => [f.url, f.size || 50000]));
  const total = urls.reduce((s, u) => s + (sizes.get(u) || 50000), 0);
  let done = 0;
  for (const url of urls) {
    const expected = sizes.get(url) || 50000;
    if (await cache.match(url)) {
      done += expected;
      onProgress(done / total);
      continue;
    }
    const res = await fetch(url, { signal, cache: 'no-store' });
    if (!res.ok) throw new Error(`${res.status} ${url.split('/').pop()}`);
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done: end, value } = await reader.read();
      if (end) break;
      chunks.push(value);
      got += value.length;
      onProgress((done + Math.min(got, expected)) / total);
    }
    const type = res.headers.get('content-type') || 'application/octet-stream';
    await cache.put(url, new Response(new Blob(chunks, { type }), { headers: { 'content-type': type } }));
    done += expected;
    onProgress(done / total);
  }
}

export async function remove() {
  if (typeof caches !== 'undefined') await caches.delete(CACHE);
}
