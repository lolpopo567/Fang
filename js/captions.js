import { t } from './i18n.js';
import { load, save, download } from './store.js';
import { newTranscript, putTranscript } from './transcripts.js';
import * as pack from './offline/pack.js';

const Recognition = globalThis.SpeechRecognition || globalThis.webkitSpeechRecognition;

// Which recogniser to use. 'browser' is the Web Speech API (usually sends audio to the browser
// vendor, needs internet). 'device' is Fang's own offline Thai model. 'auto' uses the browser
// when it can and the offline model when there is no connection or no browser support.
export function chooseEngine({ setting, lang, online, browserSupported, packInstalled }) {
  const deviceOk = packInstalled && lang.startsWith('th');
  if (setting === 'device') return deviceOk ? 'device' : (packInstalled ? 'browser' : 'needs-pack');
  if (setting === 'browser') return browserSupported ? 'browser' : 'none';
  if (deviceOk && (!online || !browserSupported)) return 'device';
  return browserSupported ? 'browser' : (packInstalled ? 'needs-thai' : 'none');
}

export function initCaptions() {
  const box = document.getElementById('caption-box');
  const toggle = document.getElementById('caption-toggle');
  const langSel = document.getElementById('caption-lang');
  const engineSel = document.getElementById('caption-engine');
  const size = document.getElementById('caption-size');
  const status = document.getElementById('caption-status');
  const engineBadge = document.getElementById('caption-engine-badge');
  const privacy = document.getElementById('caption-privacy');
  const autoSave = document.getElementById('caption-autosave');

  const lines = [];
  let interimEl = null;
  let rec = null;
  let offline = null;
  let engine = null; // engine of the running session
  let wanted = false; // user wants to keep listening; browsers end sessions on silence
  let packInstalled = false;
  let transcript = null;

  langSel.value = load('fang.captionLang', 'th-TH');
  engineSel.value = load('fang.captionEngine', 'auto');
  autoSave.checked = load('fang.captionAutoSave', true);
  size.value = load('fang.captionSize', 34);
  box.style.fontSize = `${size.value}px`;

  size.addEventListener('input', () => {
    box.style.fontSize = `${size.value}px`;
    save('fang.captionSize', Number(size.value));
  });

  langSel.addEventListener('change', () => {
    save('fang.captionLang', langSel.value);
    if (wanted) restart();
    showEngine();
  });
  engineSel.addEventListener('change', () => {
    save('fang.captionEngine', engineSel.value);
    if (wanted) restart();
    showEngine();
  });
  autoSave.addEventListener('change', () => save('fang.captionAutoSave', autoSave.checked));

  const pick = () => chooseEngine({
    setting: engineSel.value, lang: langSel.value, online: navigator.onLine,
    browserSupported: !!Recognition, packInstalled
  });

  // Shows which engine will be (or is being) used, and the matching privacy note.
  function showEngine() {
    const e = wanted ? engine : pick();
    const onDevice = e === 'device';
    engineBadge.hidden = !(e === 'device' || e === 'browser');
    engineBadge.classList.toggle('on-device', onDevice);
    engineBadge.querySelector('span:last-child').textContent = t(onDevice ? 'captions.engineDevice' : 'captions.engineBrowser');
    engineBadge.querySelector('.icon').className = `icon ${onDevice ? 'i-wifi-off' : 'i-cloud'}`;
    privacy.textContent = t(onDevice ? 'captions.privacyDevice' : 'captions.privacy');
    const unsupported = e === 'none';
    document.getElementById('captions-unsupported').hidden = !unsupported;
    toggle.disabled = unsupported;
  }

  async function refreshPack() {
    packInstalled = await pack.isInstalled();
    showEngine();
  }
  document.addEventListener('speechpackchange', refreshPack);
  window.addEventListener('online', showEngine);
  window.addEventListener('offline', showEngine);
  refreshPack();

  const setLabel = () => {
    toggle.textContent = t(wanted ? 'captions.stop' : 'captions.start');
    toggle.classList.toggle('is-live', wanted);
    box.classList.toggle('is-live', wanted);
  };
  document.addEventListener('langchange', () => { setLabel(); showEngine(); });

  function render() {
    box.replaceChildren();
    if (!lines.length && !interimEl) {
      const p = document.createElement('p');
      p.className = 'placeholder';
      p.textContent = t('captions.placeholder');
      box.append(p);
      return;
    }
    for (const line of lines.slice(-50)) {
      const p = document.createElement('p');
      p.textContent = line;
      box.append(p);
    }
    if (interimEl) box.append(interimEl);
    box.scrollTop = box.scrollHeight;
  }

  function setInterim(text) {
    if (text) {
      interimEl = document.createElement('p');
      interimEl.className = 'interim';
      interimEl.textContent = text;
    } else {
      interimEl = null;
    }
  }

  function addLine(text) {
    lines.push(text);
    if (navigator.vibrate) navigator.vibrate(30);
    if (!autoSave.checked) return;
    transcript ??= newTranscript({ lang: langSel.value, engine });
    transcript.lines.push({ at: Date.now(), text });
    transcript.endedAt = Date.now();
    putTranscript(transcript).catch(() => {});
  }

  function showError(code) {
    const key = `captions.err.${code}`;
    status.textContent = t(key) === key ? t('captions.err.generic') + code : t(key);
  }

  // Browser (Web Speech API) engine
  async function startBrowser() {
    rec = new Recognition();
    rec.lang = langSel.value;
    rec.continuous = true;
    rec.interimResults = true;
    // Newer Chrome can recognise on the device when it has a language pack installed.
    if (!navigator.onLine && 'processLocally' in rec && Recognition.available) {
      try {
        if (await Recognition.available({ langs: [rec.lang], processLocally: true }) === 'available') rec.processLocally = true;
      } catch {}
    }

    rec.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const text = e.results[i][0].transcript.trim();
        if (!text) continue;
        if (e.results[i].isFinal) addLine(text);
        else interim += text + ' ';
      }
      setInterim(interim.trim());
      render();
    };

    rec.onerror = (e) => {
      if (e.error === 'no-speech' || e.error === 'aborted') return;
      // Lost the connection: carry on with the offline model if we can.
      if (e.error === 'network' && engineSel.value === 'auto' && packInstalled && langSel.value.startsWith('th')) {
        status.textContent = t('captions.switchedOffline');
        engine = 'device';
        rec.onend = null;
        rec.abort();
        rec = null;
        showEngine();
        startDevice();
        return;
      }
      showError(e.error);
      if (e.error === 'not-allowed' || e.error === 'audio-capture') {
        wanted = false;
        setLabel();
      }
    };

    rec.onend = () => {
      setInterim('');
      render();
      if (wanted && engine === 'browser') {
        // Chrome and Safari stop after a pause; keep captions running until the user stops them.
        setTimeout(() => { if (wanted && engine === 'browser') startBrowser(); }, 250);
      } else if (!wanted) {
        status.textContent = t('captions.stopped');
      }
    };

    try {
      rec.start();
      status.textContent = t('captions.listening');
    } catch (err) {
      status.textContent = t('captions.err.generic') + err.message;
    }
  }

  // Offline (on-device model) engine
  async function startDevice() {
    const { OfflineEngine } = await import('./offline/engine.js');
    offline ??= new OfflineEngine((msg) => {
      if (msg.type === 'interim') { setInterim(msg.text); render(); }
      else if (msg.type === 'final') {
        setInterim('');
        if (msg.text) addLine(msg.text);
        render();
      } else if (msg.type === 'speech') {
        box.classList.toggle('hearing', msg.active);
      } else if (msg.type === 'error') {
        status.textContent = t('captions.err.generic') + msg.message;
      }
    });
    status.textContent = t(offline.loaded ? 'captions.listening' : 'captions.loadingModel');
    try {
      await offline.start();
      if (wanted) status.textContent = t('captions.listening');
      else offline.stop();
    } catch (err) {
      offline.stop();
      wanted = false;
      setLabel();
      showEngine();
      if (err?.name === 'NotAllowedError') showError('not-allowed');
      else if (err?.name === 'NotFoundError') showError('audio-capture');
      else status.textContent = t('captions.err.generic') + (err?.message || err);
    }
  }

  function start() {
    engine = pick();
    transcript = null; // each listening session is saved as its own transcript
    if (engine === 'needs-pack') {
      status.textContent = t('captions.needPack');
      return false;
    }
    if (engine === 'needs-thai') {
      status.textContent = t('captions.offlineThaiOnly');
      return false;
    }
    showEngine();
    if (engine === 'device') startDevice();
    else startBrowser();
    return true;
  }

  function stop() {
    wanted = false;
    rec?.stop();
    if (engine === 'device') {
      offline?.stop();
      status.textContent = t('captions.stopped');
    }
    box.classList.remove('hearing');
    setLabel();
    showEngine();
  }

  function restart() {
    if (engine === 'device') offline?.stop();
    else { if (rec) rec.onend = null; rec?.abort(); }
    rec = null;
    setInterim('');
    render();
    start();
  }

  toggle.addEventListener('click', () => {
    if (wanted) return stop();
    wanted = true;
    if (!start()) wanted = false;
    setLabel();
  });

  document.getElementById('caption-copy').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(lines.join('\n'));
      status.textContent = t('captions.copied');
    } catch (err) {
      status.textContent = t('captions.err.generic') + err.message;
    }
  });

  document.getElementById('caption-save').addEventListener('click', () => {
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
    download(`captions-${stamp}.txt`, lines.join('\n'));
  });

  document.getElementById('caption-clear').addEventListener('click', () => {
    lines.length = 0;
    interimEl = null;
    transcript = null; // the next line starts a new saved transcript
    render();
  });

  setLabel();
  document.addEventListener('langchange', render);
}
