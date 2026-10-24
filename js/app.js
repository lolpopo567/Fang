import { setLang } from './i18n.js';
import { load, save } from './store.js';
import { initCaptions } from './captions.js';
import { initSpeak } from './speak.js';
import { initSign } from './sign/sign.js';
import { initHistory } from './history.js';
import { initSpeechPack } from './offline/settings.js';

const langSel = document.getElementById('ui-lang');
langSel.value = load('fang.uiLang', 'th');
setLang(langSel.value);
langSel.addEventListener('change', () => {
  setLang(langSel.value);
  save('fang.uiLang', langSel.value);
  document.dispatchEvent(new Event('langchange'));
});

const themeSel = document.getElementById('theme');
themeSel.value = load('fang.theme', 'auto');
themeSel.addEventListener('change', () => {
  if (themeSel.value === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = themeSel.value;
  save('fang.theme', themeSel.value);
});

// Screens are addressed by the URL hash (#home, #captions, #speak, #sign, #settings, #history),
// so the phone's back button returns to the home screen and each mode can be bookmarked.
// #history/<id> opens one saved transcript.
const SCREENS = ['home', 'captions', 'speak', 'sign', 'settings', 'history'];
const TAB_OF = { history: 'captions' }; // screens without their own tab light up their parent's
const tabs = document.querySelectorAll('[role="tab"]');
function showScreen(name) {
  const id = `tab-${name}`;
  for (const s of SCREENS) document.getElementById(`tab-${s}`).hidden = s !== name;
  const tabId = `tab-${TAB_OF[name] || name}`;
  for (const tab of tabs) tab.setAttribute('aria-selected', String(tab.dataset.tab === tabId));
  document.getElementById('open-settings').classList.toggle('active', name === 'settings');
  window.scrollTo(0, 0);
  document.dispatchEvent(new CustomEvent('tabchange', { detail: id }));
}
function route() {
  const name = location.hash.slice(1).split('/')[0];
  showScreen(SCREENS.includes(name) ? name : 'home');
}
for (const tab of tabs) {
  tab.addEventListener('click', () => { location.hash = tab.dataset.tab.replace('tab-', ''); });
}
window.addEventListener('hashchange', route);

initCaptions();
initSpeak();
initSign();
initHistory();
initSpeechPack();

// Settings that also have a control inside a mode stay in step with it. The mode's own control
// keeps doing the work; the settings copy just forwards to it.
function mirror(copyId, realId, eventName) {
  const copy = document.getElementById(copyId);
  const real = document.getElementById(realId);
  copy.value = real.value;
  copy.addEventListener(eventName, () => {
    real.value = copy.value;
    real.dispatchEvent(new Event(eventName));
  });
  real.addEventListener(eventName, () => { copy.value = real.value; });
}
mirror('set-caption-size', 'caption-size', 'input');
mirror('set-sign-lang', 'sign-lang', 'change');

const preview = document.getElementById('caption-preview');
const sizeInput = document.getElementById('caption-size');
const showPreviewSize = () => { preview.style.fontSize = `${sizeInput.value}px`; };
sizeInput.addEventListener('input', showPreviewSize);
showPreviewSize();

route();

if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
