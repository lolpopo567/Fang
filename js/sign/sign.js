import { t } from '../i18n.js';
import { load, save, download } from '../store.js';
import { toHandFrame, FrameWindow } from './features.js';
import { OfflineRecognizer, OnlineRecognizer, RecognizerHub, isUncertain } from './recognizer.js';

const WASM_PATH = new URL('../../vendor/mediapipe/wasm', import.meta.url).href;
const MODEL_PATH = new URL('../../models/hand_landmarker.task', import.meta.url).href;
const PREDICT_EVERY_MS = 500;
const RECORD_MS = 2000;

let landmarkerPromise = null;

const round4 = (n) => Math.round(n * 1e4) / 1e4;

// MediaPipe (~13 MB of WASM) loads only when the camera is first opened.
async function getLandmarker() {
  landmarkerPromise ??= (async () => {
    const { FilesetResolver, HandLandmarker } = await import('../../vendor/mediapipe/vision_bundle.mjs');
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_PATH, delegate },
      runningMode: 'VIDEO',
      numHands: 2
    });
    let lm;
    try {
      lm = await HandLandmarker.createFromOptions(fileset, opts('GPU'));
    } catch {
      lm = await HandLandmarker.createFromOptions(fileset, opts('CPU'));
    }
    return { lm, connections: HandLandmarker.HAND_CONNECTIONS };
  })();
  try {
    return await landmarkerPromise;
  } catch (err) {
    landmarkerPromise = null;
    throw err;
  }
}

export function initSign() {
  const video = document.getElementById('sign-video');
  const canvas = document.getElementById('sign-overlay');
  const ctx = canvas.getContext('2d');
  const empty = document.getElementById('stage-empty');
  const toggle = document.getElementById('sign-toggle');
  const modeSel = document.getElementById('sign-mode');
  const endpointRow = document.getElementById('sign-endpoint-row');
  const endpoint = document.getElementById('sign-endpoint');
  const stat = {
    pipeline: document.getElementById('stat-pipeline'),
    hands: document.getElementById('stat-hands'),
    fps: document.getElementById('stat-fps'),
    latency: document.getElementById('stat-latency')
  };
  const resultLabel = document.getElementById('result-label');
  const resultMeta = document.getElementById('result-meta');
  const recStatus = document.getElementById('rec-status');

  const online = new OnlineRecognizer(load('fang.signEndpoint', ''));
  const hub = new RecognizerHub(new OfflineRecognizer(), online);
  const window_ = new FrameWindow(30);

  let stream = null;
  let facing = 'user';
  let running = false;
  let rafId = 0;
  let lastVideoTime = -1;
  let fpsCount = 0;
  let fpsStart = 0;
  let lastPredict = 0;
  let predicting = false;
  let pipelineKey = 'sign.stopped';
  let lastPrediction = null;
  let recording = null;
  let samples = load('fang.samples', []);

  modeSel.value = load('fang.signMode', 'offline');
  hub.mode = modeSel.value;
  endpoint.value = online.endpoint;
  endpointRow.hidden = hub.mode === 'offline';

  modeSel.addEventListener('change', () => {
    hub.mode = modeSel.value;
    endpointRow.hidden = hub.mode === 'offline';
    save('fang.signMode', hub.mode);
  });
  endpoint.addEventListener('change', () => {
    online.endpoint = endpoint.value.trim();
    save('fang.signEndpoint', online.endpoint);
  });

  function setPipeline(key, extra = '') {
    pipelineKey = key;
    stat.pipeline.textContent = t(key) + extra;
  }

  function setToggle() {
    toggle.textContent = t(running ? 'sign.stop' : 'sign.start');
  }

  function showPrediction(p) {
    lastPrediction = p;
    if (!p) {
      resultLabel.textContent = '—';
      resultMeta.textContent = '';
      return;
    }
    const msg = { no_model: 'sign.noModel', not_configured: 'sign.noEndpoint', error: 'sign.onlineError' }[p.status];
    if (msg) {
      resultLabel.textContent = '—';
      resultMeta.textContent = t(msg) + (p.error ? ` (${p.error})` : '');
      resultLabel.classList.remove('uncertain');
      return;
    }
    resultLabel.textContent = p.label;
    const pct = Math.round((p.confidence ?? 0) * 100);
    resultMeta.textContent = `${t('sign.confidence')} ${pct}%` + (isUncertain(p) ? ` · ${t('sign.uncertain')}` : '');
    resultLabel.classList.toggle('uncertain', isUncertain(p));
  }

  function draw(frame, connections) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.width;
    const h = canvas.height;
    for (const hand of frame.hands) {
      const color = hand.handedness === 'Left' ? '#ffb000' : '#00d1ff';
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, w / 240);
      for (const { start, end } of connections) {
        const a = hand.landmarks[start];
        const b = hand.landmarks[end];
        ctx.beginPath();
        ctx.moveTo(a[0] * w, a[1] * h);
        ctx.lineTo(b[0] * w, b[1] * h);
        ctx.stroke();
      }
      ctx.fillStyle = '#fff';
      for (const [x, y] of hand.landmarks) {
        ctx.beginPath();
        ctx.arc(x * w, y * h, Math.max(3, w / 160), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  async function loop(lm, connections) {
    if (!running) return;
    const now = performance.now();
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const t0 = performance.now();
      const result = lm.detectForVideo(video, now);
      const latency = performance.now() - t0;
      const frame = toHandFrame(result, now);
      draw(frame, connections);
      window_.push(frame);
      recording?.frames.push(frame);

      stat.hands.textContent = String(frame.hands.length);
      stat.latency.textContent = `${latency.toFixed(0)} ms`;
      fpsCount++;
      if (now - fpsStart >= 1000) {
        stat.fps.textContent = String(Math.round((fpsCount * 1000) / (now - fpsStart)));
        fpsCount = 0;
        fpsStart = now;
      }

      if (!predicting && now - lastPredict > PREDICT_EVERY_MS && window_.full) {
        lastPredict = now;
        predicting = true;
        hub.predict(window_.frames.slice()).then(showPrediction).finally(() => { predicting = false; });
      }
    }
    rafId = requestAnimationFrame(() => loop(lm, connections));
  }

  async function start() {
    running = true;
    setToggle();
    setPipeline('sign.loading');
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
      video.srcObject = stream;
      video.classList.toggle('mirror', facing === 'user');
      canvas.classList.toggle('mirror', facing === 'user');
      await video.play();
      empty.hidden = true;
      const { lm, connections } = await getLandmarker();
      if (!running) return;
      setPipeline('sign.running');
      fpsStart = performance.now();
      fpsCount = 0;
      loop(lm, connections);
    } catch (err) {
      stopCamera();
      setPipeline('sign.cameraError', err.message || String(err));
    }
  }

  function stopCamera() {
    running = false;
    cancelAnimationFrame(rafId);
    stream?.getTracks().forEach((tr) => tr.stop());
    stream = null;
    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    empty.hidden = false;
    window_.clear();
    lastVideoTime = -1;
    for (const el of [stat.hands, stat.fps, stat.latency]) el.textContent = '—';
    setToggle();
  }

  toggle.addEventListener('click', () => {
    if (running) {
      stopCamera();
      setPipeline('sign.stopped');
    } else {
      start();
    }
  });

  document.getElementById('sign-flip').addEventListener('click', () => {
    facing = facing === 'user' ? 'environment' : 'user';
    if (running) {
      stopCamera();
      start();
    }
  });

  // Stop the camera when the user leaves the tab so it is not left running in the background.
  document.addEventListener('tabchange', (e) => {
    if (e.detail !== 'tab-sign' && running) {
      stopCamera();
      setPipeline('sign.stopped');
    }
  });

  // Dataset sample recorder: landmarks only, no video.
  const recLabel = document.getElementById('rec-label');
  const showCount = () => { recStatus.textContent = t('rec.count') + samples.length; };

  document.getElementById('rec-start').addEventListener('click', () => {
    if (!running) return void (recStatus.textContent = t('rec.needCamera'));
    const label = recLabel.value.trim();
    if (!label) return void (recStatus.textContent = t('rec.needLabel'));
    if (recording) return;
    recording = { label, frames: [] };
    recStatus.textContent = t('rec.recording');
    setTimeout(() => {
      const r = recording;
      recording = null;
      if (!r.frames.some((f) => f.hands.length)) {
        recStatus.textContent = t('rec.noHands');
        return;
      }
      const t0 = r.frames[0].t;
      samples.push({
        label: r.label,
        recordedAt: new Date().toISOString(),
        frames: r.frames.map((f) => ({
          t: Math.round(f.t - t0),
          // 4 decimals keeps sub-pixel precision and makes samples ~3x smaller in storage.
          hands: f.hands.map((h) => ({ ...h, score: round4(h.score), landmarks: h.landmarks.map((p) => p.map(round4)) }))
        }))
      });
      save('fang.samples', samples);
      showCount();
    }, RECORD_MS);
  });

  document.getElementById('rec-export').addEventListener('click', () => {
    const stamp = new Date().toISOString().slice(0, 10);
    download(`fang-sign-samples-${stamp}.json`, JSON.stringify({ format: 'fang-handframes-v1', samples }), 'application/json');
  });

  document.getElementById('rec-clear').addEventListener('click', () => {
    samples = [];
    save('fang.samples', samples);
    showCount();
  });

  document.addEventListener('langchange', () => {
    setToggle();
    setPipeline(pipelineKey);
    showPrediction(lastPrediction);
    if (recStatus.textContent) showCount();
  });

  setToggle();
  setPipeline('sign.stopped');
  if (samples.length) showCount();
}
