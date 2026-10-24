import { t, getLang } from './i18n.js';
import { download } from './store.js';
import { listTranscripts, getTranscript, deleteTranscript, clearTranscripts, transcriptText, clock, matches } from './transcripts.js';

// The transcript history screen. #history lists saved transcripts; #history/<id> shows one, so
// the phone's back button goes from a transcript back to the list.
export function initHistory() {
  const listView = document.getElementById('history-list-view');
  const detail = document.getElementById('history-detail');
  const list = document.getElementById('history-list');
  const empty = document.getElementById('history-empty');
  const search = document.getElementById('history-search');
  const status = document.getElementById('history-status');
  let current = null;

  const fmtDate = (ms) => new Intl.DateTimeFormat(getLang() === 'th' ? 'th-TH' : 'en-GB', {
    weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'
  }).format(ms);
  const minutes = (doc) => Math.max(1, Math.round((doc.endedAt - doc.startedAt) / 60000));
  const meta = (doc) => `${minutes(doc)} ${t('history.minutes')} · ${doc.lines.length} ${t('history.lines')}`;

  async function showList() {
    detail.hidden = true;
    listView.hidden = false;
    let docs = [];
    try { docs = await listTranscripts(); } catch {}
    const shown = docs.filter((d) => matches(d, search.value));
    list.replaceChildren(...shown.map((doc) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = `#history/${doc.id}`;
      a.className = 'history-item';
      const head = document.createElement('span');
      head.className = 'history-when';
      head.textContent = fmtDate(doc.startedAt);
      if (doc.engine === 'device') {
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = t('history.offline');
        head.append(' ', badge);
      }
      const preview = document.createElement('span');
      preview.className = 'history-preview';
      preview.textContent = doc.lines.map((l) => l.text).join(' ');
      const info = document.createElement('span');
      info.className = 'history-meta';
      info.textContent = meta(doc);
      a.append(head, preview, info);
      li.append(a);
      return li;
    }));
    empty.hidden = shown.length > 0;
    empty.querySelector('span').textContent = t(docs.length ? 'history.noMatch' : 'history.empty');
  }

  async function showDetail(id) {
    current = await getTranscript(id).catch(() => null);
    if (!current) { location.hash = 'history'; return; }
    listView.hidden = true;
    detail.hidden = false;
    status.textContent = '';
    document.getElementById('history-title').textContent = fmtDate(current.startedAt);
    document.getElementById('history-meta').textContent = meta(current);
    const box = document.getElementById('history-lines');
    box.replaceChildren(...current.lines.map((l) => {
      const p = document.createElement('p');
      const time = document.createElement('time');
      time.textContent = clock(l.at);
      p.append(time, document.createTextNode(l.text));
      return p;
    }));
  }

  function route() {
    const [screen, id] = location.hash.slice(1).split('/');
    if (screen !== 'history') return;
    if (id) showDetail(id);
    else showList();
  }

  document.addEventListener('tabchange', route);
  document.addEventListener('langchange', route);
  search.addEventListener('input', showList);

  document.getElementById('history-back').addEventListener('click', () => { location.hash = 'history'; });
  document.getElementById('history-copy').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(transcriptText(current, { times: false }));
      status.textContent = t('captions.copied');
    } catch (err) {
      status.textContent = t('captions.err.generic') + err.message;
    }
  });
  document.getElementById('history-download').addEventListener('click', () => {
    const d = new Date(current.startedAt);
    const stamp = new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16).replace(/[:T]/g, '-');
    download(`captions-${stamp}.txt`, transcriptText(current));
  });
  document.getElementById('history-delete').addEventListener('click', async () => {
    if (!confirm(t('history.confirmDelete'))) return;
    await deleteTranscript(current.id).catch(() => {});
    location.hash = 'history';
  });
  document.getElementById('history-clear-all').addEventListener('click', async () => {
    if (!confirm(t('history.confirmClearAll'))) return;
    await clearTranscripts().catch(() => {});
    document.getElementById('history-clear-status').textContent = t('history.cleared');
  });
}
