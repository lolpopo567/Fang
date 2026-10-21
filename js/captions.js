import { t } from './i18n.js';
import { load, save, download } from './store.js';

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;

export function initCaptions() {
  const box = document.getElementById('caption-box');
  const toggle = document.getElementById('caption-toggle');
  const langSel = document.getElementById('caption-lang');
  const size = document.getElementById('caption-size');
  const status = document.getElementById('caption-status');

  const lines = [];
  let interimEl = null;
  let rec = null;
  let wanted = false; // user wants to keep listening; browsers end sessions on silence

  langSel.value = load('fang.captionLang', 'th-TH');
  size.value = load('fang.captionSize', 34);
  box.style.fontSize = `${size.value}px`;

  size.addEventListener('input', () => {
    box.style.fontSize = `${size.value}px`;
    save('fang.captionSize', Number(size.value));
  });

  langSel.addEventListener('change', () => {
    save('fang.captionLang', langSel.value);
    if (wanted) restart();
  });

  const setLabel = () => { toggle.textContent = t(wanted ? 'captions.stop' : 'captions.start'); };
  document.addEventListener('langchange', setLabel);

  if (!Recognition) {
    document.getElementById('captions-unsupported').hidden = false;
    toggle.disabled = true;
  }

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

  function start() {
    rec = new Recognition();
    rec.lang = langSel.value;
    rec.continuous = true;
    rec.interimResults = true;

    rec.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const text = e.results[i][0].transcript.trim();
        if (!text) continue;
        if (e.results[i].isFinal) lines.push(text);
        else interim += text + ' ';
      }
      if (interim) {
        interimEl = document.createElement('p');
        interimEl.className = 'interim';
        interimEl.textContent = interim.trim();
      } else {
        interimEl = null;
      }
      render();
      if (navigator.vibrate && !interim) navigator.vibrate(30);
    };

    rec.onerror = (e) => {
      if (e.error === 'no-speech' || e.error === 'aborted') return;
      const key = `captions.err.${e.error}`;
      status.textContent = t(key) === key ? t('captions.err.generic') + e.error : t(key);
      if (e.error === 'not-allowed' || e.error === 'audio-capture') {
        wanted = false;
        setLabel();
      }
    };

    rec.onend = () => {
      interimEl = null;
      render();
      if (wanted) {
        // Chrome and Safari stop after a pause; keep captions running until the user stops them.
        setTimeout(() => { if (wanted) start(); }, 250);
      } else {
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

  function stop() {
    wanted = false;
    rec?.stop();
    setLabel();
  }

  function restart() {
    rec?.abort();
  }

  toggle.addEventListener('click', () => {
    if (wanted) return stop();
    wanted = true;
    setLabel();
    start();
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
    render();
  });

  setLabel();
  document.addEventListener('langchange', render);
}
