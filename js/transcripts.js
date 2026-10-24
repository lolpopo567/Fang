// Saved caption transcripts, kept in IndexedDB on this device only.
// A transcript is { id, startedAt, endedAt, lang, engine, lines: [{ at, text }] }.
const DB = 'fang';
const STORE = 'transcripts';
let dbPromise = null;

function open() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB, 1);
      req.onupgradeneeded = () => {
        const store = req.result.createObjectStore(STORE, { keyPath: 'id' });
        store.createIndex('startedAt', 'startedAt');
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    dbPromise.catch(() => { dbPromise = null; });
  }
  return dbPromise;
}

function tx(mode, fn) {
  return open().then((db) => new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const result = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(result?.result ?? result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  }));
}

export const putTranscript = (doc) => tx('readwrite', (s) => s.put(doc));
export const deleteTranscript = (id) => tx('readwrite', (s) => s.delete(id));
export const clearTranscripts = () => tx('readwrite', (s) => s.clear());
export const getTranscript = (id) => tx('readonly', (s) => s.get(id));

// Newest first.
export async function listTranscripts() {
  const all = await tx('readonly', (s) => s.getAll());
  return all.sort((a, b) => b.startedAt - a.startedAt);
}

export function newTranscript({ lang, engine }) {
  const now = Date.now();
  return { id: `t${now.toString(36)}${Math.random().toString(36).slice(2, 6)}`, startedAt: now, endedAt: now, lang, engine, lines: [] };
}

export function transcriptText(doc, { times = true } = {}) {
  return doc.lines.map((l) => (times ? `[${clock(l.at)}] ${l.text}` : l.text)).join('\n');
}

export function clock(ms) {
  const d = new Date(ms);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, '0')).join(':');
}

export function matches(doc, query) {
  const q = query.trim().toLowerCase();
  return !q || doc.lines.some((l) => l.text.toLowerCase().includes(q));
}
