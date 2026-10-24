// Main-thread side of offline captions: opens the microphone, streams audio to the recognition
// worker, and reports text back. The worker (and its 160 MB of model) stays loaded between
// sessions so stopping and starting again is instant.
let worker = null;
let ready = null;

function getWorker(onMessage) {
  if (!worker) {
    worker = new Worker(new URL('./asr-worker.js', import.meta.url), { type: 'module' });
    ready = new Promise((resolve, reject) => {
      worker.addEventListener('message', function first(e) {
        if (e.data.type === 'ready') { worker.removeEventListener('message', first); resolve(e.data); }
        else if (e.data.type === 'error' && e.data.fatal) {
          worker.removeEventListener('message', first);
          worker.terminate();
          worker = null;
          reject(new Error(e.data.message));
        }
      });
    });
    worker.postMessage({ type: 'load' });
  }
  worker.onmessage = (e) => onMessage(e.data);
  return ready;
}

export class OfflineEngine {
  constructor(onEvent) {
    this.onEvent = onEvent;
    this.stream = null;
    this.ctx = null;
  }

  get loaded() {
    return !!worker;
  }

  // Resolves once the microphone is open and the model is loaded.
  async start() {
    const loading = getWorker((msg) => this.onEvent(msg));
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    try {
      // Chrome and Safari resample the microphone to 16 kHz for us; others fall back to the
      // worker's resampler at the device rate.
      this.ctx = new AudioContext({ sampleRate: 16000 });
      await this.connect();
    } catch {
      await this.ctx?.close().catch(() => {});
      this.ctx = new AudioContext();
      await this.connect();
    }
    await loading;
  }

  async connect() {
    await this.ctx.audioWorklet.addModule(new URL('./capture-worklet.js', import.meta.url));
    const source = this.ctx.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(this.ctx, 'fang-capture');
    const rate = this.ctx.sampleRate;
    node.port.onmessage = (e) => worker?.postMessage({ type: 'audio', samples: e.data, rate }, [e.data.buffer]);
    source.connect(node);
    // Some browsers only pull audio through nodes that reach the destination; output is silent.
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    node.connect(mute).connect(this.ctx.destination);
    if (this.ctx.state === 'suspended') await this.ctx.resume();
  }

  stop() {
    worker?.postMessage({ type: 'flush' });
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close().catch(() => {});
    this.stream = null;
    this.ctx = null;
  }
}
