import { t } from '../i18n.js';
import { load, save } from '../store.js';
import { ALERT_SOUNDS, ALL_LABELS, SENSITIVITY, AlertDetector, RingBuffer, groupScores } from './sounds.js';

const WASM_PATH = new URL('../../vendor/mediapipe/wasm', import.meta.url).href;
const MODEL_PATH = new URL('../../models/yamnet.tflite', import.meta.url).href;
const WINDOW_S = 1;      // YAMNet looks at 0.975 s of audio
const HOP_MS = 500;      // classify twice a second
const LOG_MAX = 30;
// Strong-weak-strong buzz; long enough to notice in a pocket.
const VIBRATE = [600, 200, 600, 200, 900];

let classifierPromise = null;

// MediaPipe audio (~7 MB of WASM plus the 4 MB YAMNet model) loads only when listening starts.
async function getClassifier() {
  classifierPromise ??= (async () => {
    const { FilesetResolver, AudioClassifier } = await import('../../vendor/mediapipe/audio_bundle.mjs');
    const fileset = await FilesetResolver.forAudioTasks(WASM_PATH);
    return AudioClassifier.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      maxResults: -1,
      categoryAllowlist: ALL_LABELS
    });
  })();
  try {
    return await classifierPromise;
  } catch (err) {
    classifierPromise = null;
    throw err;
  }
}

const soundName = (id) => t(`alerts.sound.${id}`);
const timeOf = (ms) => new Date(ms).toLocaleTimeString(document.documentElement.lang, { hour: '2-digit', minute: '2-digit', second: '2-digit' });

export function initAlerts() {
  const toggle = document.getElementById('alerts-toggle');
  const status = document.getElementById('alerts-status');
  const orb = document.getElementById('alerts-orb');
  const meterList = document.getElementById('alerts-meters');
  const logList = document.getElementById('alerts-log');
  const logEmpty = document.getElementById('alerts-log-empty');
  const flash = document.getElementById('alert-flash');
  const flashIcon = document.getElementById('alert-flash-icon');
  const flashName = document.getElementById('alert-flash-name');
  const flashTime = document.getElementById('alert-flash-time');
  const sensSel = document.getElementById('alerts-sensitivity');
  const flashChk = document.getElementById('alerts-flash');
  const vibrateChk = document.getElementById('alerts-vibrate');
  const notifyChk = document.getElementById('alerts-notify');
  const soundList = document.getElementById('alerts-sounds');
  const navTab = document.getElementById('nav-alerts');

  const enabled = new Set(load('fang.alertSounds', ALERT_SOUNDS.filter((s) => s.on).map((s) => s.id)));
  const detector = new AlertDetector({ threshold: SENSITIVITY[load('fang.alertSensitivity', 'normal')] ?? SENSITIVITY.normal, enabled });
  let log = load('fang.alertLog', []);

  sensSel.value = load('fang.alertSensitivity', 'normal');
  flashChk.checked = load('fang.alertFlash', true);
  vibrateChk.checked = load('fang.alertVibrate', true);
  notifyChk.checked = load('fang.alertNotify', false);
  sensSel.addEventListener('change', () => {
    detector.threshold = SENSITIVITY[sensSel.value];
    save('fang.alertSensitivity', sensSel.value);
  });
  flashChk.addEventListener('change', () => save('fang.alertFlash', flashChk.checked));
  vibrateChk.addEventListener('change', () => save('fang.alertVibrate', vibrateChk.checked));
  notifyChk.addEventListener('change', async () => {
    if (notifyChk.checked && 'Notification' in window && Notification.permission === 'default') {
      notifyChk.checked = (await Notification.requestPermission()) === 'granted';
    }
    save('fang.alertNotify', notifyChk.checked);
  });
  if (!('Notification' in window)) notifyChk.closest('label').hidden = true;
  if (!('vibrate' in navigator)) vibrateChk.closest('label').hidden = true;

  // One checkbox per sound in settings, and one live meter per sound on the alerts screen.
  const meters = new Map();
  function buildLists() {
    soundList.replaceChildren();
    meterList.replaceChildren();
    meters.clear();
    for (const s of ALERT_SOUNDS) {
      const row = document.createElement('label');
      row.className = 'check-row';
      row.innerHTML = `<span class="icon ${s.icon}" aria-hidden="true"></span><span></span><input type="checkbox">`;
      row.children[1].textContent = soundName(s.id);
      const box = row.querySelector('input');
      box.checked = enabled.has(s.id);
      box.addEventListener('change', () => {
        if (box.checked) enabled.add(s.id); else enabled.delete(s.id);
        save('fang.alertSounds', [...enabled]);
        buildLists();
      });
      soundList.append(row);

      if (!enabled.has(s.id)) continue;
      const li = document.createElement('li');
      li.innerHTML = `<span class="icon ${s.icon}" aria-hidden="true"></span><span class="meter-name"></span><span class="meter"><i></i></span>`;
      li.querySelector('.meter-name').textContent = soundName(s.id);
      meterList.append(li);
      meters.set(s.id, li);
    }
  }

  function renderLog() {
    logList.replaceChildren(...log.map((e) => {
      const li = document.createElement('li');
      const s = ALERT_SOUNDS.find((x) => x.id === e.id);
      li.innerHTML = `<span class="tile" aria-hidden="true"><span class="icon ${s?.icon ?? 'i-bell'}"></span></span><span class="log-name"></span><time></time>`;
      li.querySelector('.log-name').textContent = soundName(e.id);
      li.querySelector('time').textContent = timeOf(e.t);
      return li;
    }));
    logEmpty.hidden = log.length > 0;
  }

  function showAlert(id, now = Date.now()) {
    const s = ALERT_SOUNDS.find((x) => x.id === id);
    log = [{ id, t: now }, ...log].slice(0, LOG_MAX);
    save('fang.alertLog', log);
    renderLog();
    flashIcon.className = `icon ${s.icon}`;
    flashName.textContent = soundName(id);
    flashTime.textContent = timeOf(now);
    flash.classList.toggle('no-flash', !flashChk.checked);
    flash.hidden = false;
    document.getElementById('alert-flash-close').focus();
    if (vibrateChk.checked) navigator.vibrate?.(VIBRATE);
    if (notifyChk.checked && document.hidden && Notification.permission === 'granted') {
      navigator.serviceWorker?.ready.then((reg) => reg.showNotification(`Fang: ${soundName(id)}`, {
        body: timeOf(now), tag: `fang-${id}`, renotify: true, vibrate: VIBRATE, icon: 'icons/icon-192.png'
      })).catch(() => {});
    }
  }

  document.getElementById('alert-flash-close').addEventListener('click', () => {
    flash.hidden = true;
    navigator.vibrate?.(0);
  });
  document.getElementById('alerts-test').addEventListener('click', () => showAlert('door'));
  document.getElementById('alerts-log-clear').addEventListener('click', () => {
    log = [];
    save('fang.alertLog', log);
    renderLog();
  });

  // Listening: microphone -> ring buffer -> YAMNet every HOP_MS -> detector -> alert.
  let session = null;

  function setStatus(key, extra = '') {
    status.textContent = t(key) + extra;
  }

  async function start() {
    toggle.disabled = true;
    setStatus('alerts.loading');
    try {
      const [classifier, stream] = await Promise.all([
        getClassifier(),
        // Phone voice processing would treat alarms and doorbells as noise and remove them.
        navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
      ]);
      const ctx = new AudioContext();
      await ctx.resume();
      const src = ctx.createMediaStreamSource(stream);
      const buffer = new RingBuffer(Math.round(ctx.sampleRate * WINDOW_S));
      // ScriptProcessorNode is deprecated but works in every browser that has a microphone,
      // and its 4096-sample chunks are plenty for twice-a-second classification.
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = (e) => buffer.push(e.inputBuffer.getChannelData(0));
      const mute = ctx.createGain();
      mute.gain.value = 0;
      src.connect(proc).connect(mute).connect(ctx.destination);
      detector.reset();
      session = { stream, ctx, timer: 0, wakeLock: null };
      session.timer = setInterval(() => {
        if (!buffer.full) return;
        const result = classifier.classify(buffer.data.slice(), ctx.sampleRate)[0];
        const scores = groupScores(result?.classifications[0]?.categories ?? []);
        for (const [id, li] of meters) li.style.setProperty('--level', Math.min(1, scores[id] / detector.threshold).toFixed(3));
        orb.style.setProperty('--level', Math.min(1, buffer.rms() * 8).toFixed(3));
        for (const a of detector.push(scores, performance.now())) showAlert(a.id);
      }, HOP_MS);
      // Keep the screen on while listening: browsers pause the microphone when a phone sleeps.
      session.wakeLock = await navigator.wakeLock?.request('screen').catch(() => null);
      toggle.classList.add('is-live');
      navTab.classList.add('is-live');
      orb.classList.add('is-live');
      toggle.textContent = t('alerts.stop');
      setStatus('alerts.listening');
    } catch (err) {
      stop();
      setStatus(err?.name === 'NotAllowedError' ? 'alerts.noMic' : 'alerts.error', err?.name === 'NotAllowedError' ? '' : String(err?.message ?? err));
    } finally {
      toggle.disabled = false;
    }
  }

  function stop() {
    if (session) {
      clearInterval(session.timer);
      session.stream.getTracks().forEach((tr) => tr.stop());
      session.ctx.close();
      session.wakeLock?.release().catch(() => {});
      session = null;
    }
    for (const li of meters.values()) li.style.setProperty('--level', 0);
    orb.style.setProperty('--level', 0);
    toggle.classList.remove('is-live');
    navTab.classList.remove('is-live');
    orb.classList.remove('is-live');
    toggle.textContent = t('alerts.start');
    setStatus('alerts.stopped');
  }

  toggle.addEventListener('click', () => (session ? stop() : start()));
  // The wake lock is dropped when the app is hidden; take it again on return.
  document.addEventListener('visibilitychange', async () => {
    if (session && document.visibilityState === 'visible') {
      session.wakeLock = await navigator.wakeLock?.request('screen').catch(() => null);
    }
  });
  document.addEventListener('langchange', () => {
    buildLists();
    renderLog();
    toggle.textContent = t(session ? 'alerts.stop' : 'alerts.start');
    setStatus(session ? 'alerts.listening' : 'alerts.stopped');
  });

  buildLists();
  renderLog();
  status.textContent = '';
  toggle.textContent = t('alerts.start');
}
