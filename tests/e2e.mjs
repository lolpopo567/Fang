// Browser smoke test. Usage: [FANG_E2E_AUDIO=dog-bark.wav] node tests/e2e.mjs [fake-camera.y4m]
// With a y4m of a hand, also asserts that hands are detected. With a wav of a dog barking as the
// fake microphone, also asserts that sound alerts fire for it.
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
const audio = process.env.FANG_E2E_AUDIO;
if (audio) args.push(`--use-file-for-fake-audio-capture=${path.resolve(audio)}`);
const browser = await chromium.launch({ args });
const errors = [];

try {
  const ctx = await browser.newContext({ permissions: ['camera', 'microphone'], viewport: { width: 390, height: 844 } });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto(base);
  // The app opens on the home screen with a card for each mode.
  await page.waitForSelector('#tab-home:not([hidden])');
  assert.equal(await page.locator('.mode-card').count(), 5);
  await page.screenshot({ path: path.join(root, 'tests/out-home.png') });
  await page.click('.mode-card[href="#captions"]');
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

  // Settings: UI language, and the caption size copy drives the real caption slider.
  await page.click('#open-settings');
  await page.selectOption('#ui-lang', 'en');
  assert.ok((await page.textContent('#nav-speak')).includes('Type to speak'));
  await page.$eval('#set-caption-size', (el) => { el.value = '50'; el.dispatchEvent(new Event('input')); });
  assert.equal(await page.inputValue('#caption-size'), '50');
  assert.equal(await page.$eval('#caption-box', (el) => el.style.fontSize), '50px');
  await page.selectOption('#theme', 'dark');
  assert.equal(await page.getAttribute('html', 'data-theme'), 'dark');
  await page.screenshot({ path: path.join(root, 'tests/out-settings.png'), fullPage: true });
  await page.selectOption('#theme', 'auto');
  await page.selectOption('#ui-lang', 'th');
  // The phone back button returns to the previous screen.
  await page.goBack();
  assert.ok(await page.isVisible('#tab-speak'));

  // Sound alerts: the test button shows the full-screen alert, then listening runs YAMNet on the mic.
  await page.click('#nav-alerts');
  await page.waitForSelector('#tab-alerts:not([hidden])');
  assert.equal(await page.locator('#alerts-meters li').count(), 9);
  await page.click('#alerts-test');
  assert.ok(await page.isVisible('#alert-flash'));
  assert.equal(await page.textContent('#alert-flash-name'), 'กริ่งหรือเคาะประตู');
  await page.click('#alert-flash-close');
  assert.equal(await page.locator('#alerts-log li').count(), 1);
  await page.click('#alerts-toggle');
  await page.waitForFunction(() => document.getElementById('alerts-status').textContent.startsWith('กำลังฟัง'), null, { timeout: 60000 });
  assert.ok(await page.$eval('#nav-alerts', (el) => el.classList.contains('is-live')));
  if (audio) {
    await page.waitForSelector('#alert-flash:not([hidden])', { timeout: 30000 });
    console.log('sound alert:', await page.textContent('#alert-flash-name'));
    await page.screenshot({ path: path.join(root, 'tests/out-alert-flash.png') });
    await page.click('#alert-flash-close');
  } else {
    await page.waitForTimeout(2500);
  }
  await page.screenshot({ path: path.join(root, 'tests/out-alerts.png'), fullPage: true });
  await page.click('#alerts-toggle');
  assert.equal(await page.textContent('#alerts-status'), 'หยุดแล้ว');
  // Turning a sound off in settings removes its meter.
  await page.click('#open-settings');
  await page.waitForSelector('#tab-settings:not([hidden])');
  await page.click('#alerts-sounds .check-row:has-text("สุนัขเห่า") input');
  await page.click('#nav-alerts');
  await page.waitForSelector('#tab-alerts:not([hidden])');
  assert.equal(await page.locator('#alerts-meters li').count(), 8);
  await page.click('#alerts-log-clear');

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
  assert.equal(await page.inputValue('#set-sign-lang'), 'ase');
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
