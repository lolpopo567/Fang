// Browser test for offline captions and transcript history.
// Usage: node tests/e2e-captions.mjs <speech-pack-dir> <speech.wav>
// The pack dir holds the Hugging Face files named in js/offline/pack.js (served locally instead of
// downloading them); the wav (16 kHz mono) is played as the fake microphone, with pauses between
// sentences. Chrome loops the file.
import { chromium } from 'playwright';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import https from 'node:https';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { serve } from './serve.mjs';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const [packDir, wav] = process.argv.slice(2).map((p) => path.resolve(p));
const shots = process.env.SHOTS || path.join(root, 'tests');
const server = await serve(root);
const base = `http://localhost:${server.address().port}/`;
// Stand-in for huggingface.co: an HTTPS server with a throwaway certificate that serves the
// pack files by name, so the real download code runs without the network.
let hfRequests = 0;
const tls = makeCert();
const hf = https.createServer(tls, (req, res) => {
  hfRequests++;
  const file = path.join(packDir, path.basename(new URL(req.url, 'https://x').pathname));
  if (!fs.existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, { 'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/octet-stream', 'Content-Length': fs.statSync(file).size });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => hf.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: [
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${wav}`,
    // REAL_HF=1 downloads from the real Hugging Face instead (checks CORS and the pinned URLs).
    ...(process.env.REAL_HF ? [] : [
      `--host-resolver-rules=MAP huggingface.co 127.0.0.1:${hf.address().port}`, '--ignore-certificate-errors', '--no-proxy-server'
    ])
  ]
});
const errors = [];
const settle = (page) => page.waitForTimeout(400); // let the screen's enter animation finish

try {
  const ctx = await browser.newContext({ permissions: ['microphone'], viewport: { width: 390, height: 844 } });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('crash', () => console.log('page crashed'));
  page.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text()); });

  await page.goto(`${base}#settings`);
  await page.evaluate(() => navigator.serviceWorker.ready);
  await page.reload();
  await page.waitForFunction(() => !!navigator.serviceWorker.controller);

  // Settings: the pack starts missing, then downloads.
  await page.waitForFunction(() => document.getElementById('pack-state').textContent.includes('MB'));
  await page.click('#pack-download');
  await page.waitForSelector('#pack-delete:not([hidden])', { timeout: 120000 });
  if (!process.env.REAL_HF) assert.equal(hfRequests, 4);
  await page.locator('#caption-engine').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(shots, 'light-settings-captions.png') });

  // No internet: automatic mode switches to the on-device model.
  await ctx.setOffline(true);
  await page.click('#nav-captions');
  await page.waitForFunction(() => document.querySelector('#caption-engine-badge.on-device:not([hidden])'));
  await page.click('#caption-toggle');
  await page.waitForFunction(() => document.querySelectorAll('#caption-box p:not(.placeholder):not(.interim)').length >= 2, null, { timeout: 180000 });
  const lines = await page.$$eval('#caption-box p:not(.placeholder):not(.interim)', (els) => els.map((e) => e.textContent));
  console.log('offline captions:', lines);
  const all = lines.join(' ');
  assert.ok(all.includes('ท่าเรือ') || all.includes('อาเซียน'), all);
  await page.screenshot({ path: path.join(shots, 'light-captions-offline.png') });
  await page.click('#caption-toggle');

  // The session was saved; it shows in history and opens.
  await page.click('#open-history');
  await page.waitForSelector('.history-item');
  assert.equal(await page.locator('.history-item').count(), 1);
  await settle(page);
  await page.screenshot({ path: path.join(shots, 'light-history.png') });
  await page.fill('#history-search', 'ไม่มีคำนี้แน่นอน');
  await page.waitForSelector('#history-empty:not([hidden])');
  await page.fill('#history-search', '');
  await page.click('.history-item');
  await page.waitForSelector('#history-detail:not([hidden])');
  const saved = await page.$$eval('#history-lines p', (els) => els.map((e) => e.textContent));
  assert.ok(saved.length >= 2 && saved[0].match(/^\d\d:\d\d:\d\d/), saved.join('|'));
  await settle(page);
  await page.screenshot({ path: path.join(shots, 'light-history-detail.png') });
  // Back to the list, then dark-mode screenshots.
  await page.click('#history-back');
  await page.waitForSelector('#history-list-view:not([hidden])');
  await page.emulateMedia({ colorScheme: 'dark' });
  await settle(page);
  await page.screenshot({ path: path.join(shots, 'dark-history.png') });
  await page.click('.history-item');
  await page.waitForSelector('#history-detail:not([hidden])');
  await settle(page);
  await page.screenshot({ path: path.join(shots, 'dark-history-detail.png') });
  await page.click('#nav-captions');
  await settle(page);
  await page.screenshot({ path: path.join(shots, 'dark-captions-offline.png') });

  // Delete the transcript from its page.
  await page.goto(`${base}#history`);
  await page.click('.history-item');
  page.once('dialog', (d) => d.accept());
  await page.click('#history-delete');
  await page.waitForSelector('#history-empty:not([hidden])');

  await ctx.setOffline(false);
  assert.deepEqual(errors, []);
  console.log('captions e2e passed');
} finally {
  await browser.close();
  server.close();
  hf.close();
}

function makeCert() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fang-tls-'));
  execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', `${dir}/key.pem`, '-out', `${dir}/cert.pem`,
    '-days', '1', '-subj', '/CN=huggingface.co'], { stdio: 'ignore' });
  return { key: fs.readFileSync(`${dir}/key.pem`), cert: fs.readFileSync(`${dir}/cert.pem`) };
}
