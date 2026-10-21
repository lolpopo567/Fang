import { t, getLang } from './i18n.js';
import { load, save } from './store.js';

const DEFAULT_PHRASES = {
  th: [
    'ฉันหูหนวก/หูตึง กรุณาพิมพ์หรือเขียนให้อ่าน',
    'กรุณาพูดช้าๆ และหันหน้ามาทางฉัน',
    'ช่วยเขียนให้หน่อยได้ไหม',
    'ขอบคุณครับ/ค่ะ',
    'ห้องน้ำอยู่ที่ไหน',
    'ฉันต้องการความช่วยเหลือ',
    'ช่วยเรียกรถพยาบาล (1669)'
  ],
  en: [
    'I am deaf / hard of hearing. Please type or write it down.',
    'Please speak slowly and face me.',
    'Could you write that down?',
    'Thank you.',
    'Where is the restroom?',
    'I need help.',
    'Please call an ambulance.'
  ]
};

export function initSpeak() {
  const text = document.getElementById('speak-text');
  const status = document.getElementById('speak-status');
  const list = document.getElementById('phrase-list');
  const big = document.getElementById('bigtext');
  const bigContent = document.getElementById('bigtext-content');

  let custom = load('fang.phrases', []);

  function voiceFor(lang) {
    const voices = window.speechSynthesis?.getVoices() ?? [];
    return voices.find((v) => v.lang.replace('_', '-').toLowerCase().startsWith(lang));
  }

  function say(phrase) {
    if (!phrase.trim()) {
      status.textContent = t('speak.empty');
      return;
    }
    // Speak in the script of the text so Thai text gets a Thai voice regardless of UI language.
    const lang = /[฀-๿]/.test(phrase) ? 'th' : 'en';
    const voice = voiceFor(lang);
    if (!('speechSynthesis' in window) || !voice) {
      status.textContent = t('speak.noVoice');
      showBig(phrase);
      return;
    }
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(phrase);
    u.voice = voice;
    u.lang = voice.lang;
    u.rate = 0.95;
    speechSynthesis.speak(u);
    status.textContent = '';
  }

  function showBig(phrase) {
    if (!phrase.trim()) {
      status.textContent = t('speak.empty');
      return;
    }
    bigContent.textContent = phrase;
    big.hidden = false;
    document.getElementById('bigtext-close').focus();
  }

  function render() {
    list.replaceChildren();
    const all = [...custom.map((p) => ({ p, custom: true })), ...DEFAULT_PHRASES[getLang()].map((p) => ({ p }))];
    for (const { p, custom: isCustom } of all) {
      const li = document.createElement('li');
      const sayBtn = document.createElement('button');
      sayBtn.className = 'phrase';
      sayBtn.textContent = p;
      sayBtn.addEventListener('click', () => { text.value = p; say(p); });
      const showBtn = document.createElement('button');
      showBtn.className = 'btn small';
      showBtn.textContent = t('speak.show');
      showBtn.addEventListener('click', () => showBig(p));
      li.append(sayBtn, showBtn);
      if (isCustom) {
        const del = document.createElement('button');
        del.className = 'btn small';
        del.textContent = t('speak.delete');
        del.addEventListener('click', () => {
          custom = custom.filter((c) => c !== p);
          save('fang.phrases', custom);
          render();
        });
        li.append(del);
      }
      list.append(li);
    }
  }

  document.getElementById('speak-say').addEventListener('click', () => say(text.value));
  document.getElementById('speak-show').addEventListener('click', () => showBig(text.value));
  document.getElementById('speak-save').addEventListener('click', () => {
    const p = text.value.trim();
    if (!p) return void (status.textContent = t('speak.empty'));
    if (!custom.includes(p)) custom.unshift(p);
    save('fang.phrases', custom);
    status.textContent = t('speak.saved');
    render();
  });
  document.getElementById('bigtext-close').addEventListener('click', () => { big.hidden = true; });
  big.addEventListener('keydown', (e) => { if (e.key === 'Escape') big.hidden = true; });

  // Chrome fills the voice list asynchronously; asking once starts the load.
  window.speechSynthesis?.getVoices();
  document.addEventListener('langchange', render);
  render();
}
