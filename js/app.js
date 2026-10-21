import { setLang } from './i18n.js';
import { load, save } from './store.js';
import { initCaptions } from './captions.js';
import { initSpeak } from './speak.js';
import { initSign } from './sign/sign.js';

const langSel = document.getElementById('ui-lang');
langSel.value = load('fang.uiLang', 'th');
setLang(langSel.value);
langSel.addEventListener('change', () => {
  setLang(langSel.value);
  save('fang.uiLang', langSel.value);
  document.dispatchEvent(new Event('langchange'));
});

const tabs = document.querySelectorAll('[role="tab"]');
function showTab(id) {
  for (const tab of tabs) {
    const on = tab.dataset.tab === id;
    tab.setAttribute('aria-selected', String(on));
    document.getElementById(tab.dataset.tab).hidden = !on;
  }
  save('fang.tab', id);
  document.dispatchEvent(new CustomEvent('tabchange', { detail: id }));
}
for (const tab of tabs) tab.addEventListener('click', () => showTab(tab.dataset.tab));

initCaptions();
initSpeak();
initSign();
showTab(load('fang.tab', 'tab-captions'));

if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
