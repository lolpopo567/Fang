// Browser smoke test. Usage: node tests/e2e.mjs [fake-camera.y4m]
// With a y4m of a hand, also asserts that hands are detected.
import { chromium } from 'playwright';
import path from 'node:path';
import assert from 'node:assert/strict';
import { serve } from './serve.mjs';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const video = process.argv[2];
const server = await serve(root);
const base = `http://localhost:${server.address().port}/`;
const args = ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'];
if (video) args.push(`--use-file-for-fake-video-capture=${path.resolve(video)}`);
const browser = await chromium.launch({ args });
const errors = [];

try {
  const ctx = await browser.newContext({ permissions: ['camera'], viewport: { width: 390, height: 844 } });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto(base);
  await assert.doesNotReject(page.waitForSelector('text=เริ่มฟัง'));
  await page.screenshot({ path: path.join(root, 'tests/out-captions.png') });

  await page.click('#nav-speak');
  await page.fill('#speak-text', 'สวัสดีครับ');
  await page.click('#speak-show');
  assert.equal(await page.textContent('#bigtext-content'), 'สวัสดีครับ');
  await page.click('#bigtext-close');
  await page.click('#speak-save');
  assert.ok((await page.textContent('#phrase-list')).includes('สวัสดีครับ'));
  await page.screenshot({ path: path.join(root, 'tests/out-speak.png') });

  await page.selectOption('#ui-lang', 'en');
  assert.ok(await page.isVisible('text=Type to speak'));
  await page.selectOption('#ui-lang', 'th');

  await page.click('#nav-sign');
  await page.click('#sign-toggle');
  await page.waitForFunction(() => document.getElementById('stat-pipeline').textContent === 'กำลังทำงาน', null, { timeout: 60000 });
  await page.waitForFunction(() => Number(document.getElementById('stat-fps').textContent) > 0, null, { timeout: 60000 });
  if (video) {
    await page.waitForFunction(() => Number(document.getElementById('stat-hands').textContent) > 0, null, { timeout: 60000 });
  }
  // models/tsl-model.json is bundled, so the sign result shows a prediction or "no sign", never "no model".
  await page.waitForFunction(() => {
    const meta = document.getElementById('result-meta').textContent;
    return meta && !meta.includes('ยังไม่มีโมเดล');
  }, null, { timeout: 60000 });
  console.log('sign result:', await page.textContent('#result-label'), await page.textContent('#result-meta'));

  // Switching to ASL loads models/asl-model.json the same way.
  await page.selectOption('#sign-lang', 'ase');
  await page.waitForFunction(() => {
    const meta = document.getElementById('result-meta').textContent;
    return meta && !meta.includes('ยังไม่มีโมเดล');
  }, null, { timeout: 60000 });
  console.log('ASL result:', await page.textContent('#result-label'), await page.textContent('#result-meta'));
  await page.selectOption('#sign-lang', 'tsl');

  if (video) {
    await page.click('.recorder summary');
    await page.fill('#rec-label', 'ทดสอบ');
    await page.click('#rec-start');
    await page.waitForFunction(() => document.getElementById('rec-status').textContent.includes('1'), null, { timeout: 20000 });
  }
  const stats = await page.$$eval('.stats dd', (els) => els.map((e) => e.textContent));
  console.log('sign stats [status, hands, fps, latency]:', stats);
  await page.screenshot({ path: path.join(root, 'tests/out-sign.png') });

  assert.deepEqual(errors, []);
  console.log('e2e passed');
} finally {
  await browser.close();
  server.close();
}
