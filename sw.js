// Offline support: app shell is precached; the large MediaPipe files are cached the first time
// the camera or sound alerts are used, so later visits work without a connection.
const CACHE = 'fang-v12';
const SHELL = [
  './',
  'index.html',
  'css/icons.css',
  'css/app.css',
  'vendor/fonts/plex-thai-thai-400.woff2',
  'vendor/fonts/plex-thai-thai-600.woff2',
  'vendor/fonts/plex-thai-thai-700.woff2',
  'vendor/fonts/plex-thai-latin-400.woff2',
  'vendor/fonts/plex-thai-latin-600.woff2',
  'vendor/fonts/plex-thai-latin-700.woff2',
  'js/app.js',
  'js/i18n.js',
  'js/store.js',
  'js/captions.js',
  'js/speak.js',
  'js/sign/sign.js',
  'js/sign/features.js',
  'js/sign/recognizer.js',
  'js/sign/model.js',
  'js/alerts/alerts.js',
  'js/alerts/sounds.js',
  'manifest.webmanifest',
  'img/kitten.webp',
  'icons/favicon-32.png',
  'icons/icon-192.png',
  'icons/icon-512.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;
  // Network first so updates show up; cache as fallback when offline.
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
