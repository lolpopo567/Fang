// Scores the sound-alert model on ESC-50 clips (https://github.com/karolpiczak/ESC-50, CC BY-NC;
// used only to measure, never bundled). Runs YAMNet in Chromium exactly as the app does, with
// the app's own 1 s window and 0.5 s hop, and writes the per-window group scores to a JSON file.
// Usage: node tests/eval_alerts.mjs path/to/esc50.csv path/to/audio-dir out.json
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { serve } from './serve.mjs';

const [csvPath, audioDir, outPath] = process.argv.slice(2);
const root = path.resolve(new URL('..', import.meta.url).pathname);
const rows = fs.readFileSync(csvPath, 'utf8').trim().split('\n').slice(1).map((l) => l.split(','));
const clips = rows
  .map(([filename, , , category]) => ({ filename, category }))
  .filter((c) => fs.existsSync(path.join(audioDir, c.filename)));

const server = await serve(root);
const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.route('**/esc50/*', (route) => route.fulfill({ path: path.join(audioDir, path.basename(new URL(route.request().url()).pathname)) }));
  await page.goto(`http://localhost:${server.address().port}/tests/serve.mjs`);
  await page.evaluate(async () => {
    const { FilesetResolver, AudioClassifier } = await import('/vendor/mediapipe/audio_bundle.mjs');
    const fileset = await FilesetResolver.forAudioTasks('/vendor/mediapipe/wasm');
    window.classifier = await AudioClassifier.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: '/models/yamnet.tflite' }, maxResults: -1
    });
    window.sounds = await import('/js/alerts/sounds.js');
  });
  const out = [];
  for (const [i, clip] of clips.entries()) {
    const windows = await page.evaluate(async (file) => {
      const buf = await (await fetch(`/esc50/${file}`)).arrayBuffer();
      const audio = await new OfflineAudioContext(1, 1, 16000).decodeAudioData(buf);
      const x = audio.getChannelData(0);
      const res = [];
      for (let s = 0; s + 16000 <= x.length; s += 8000) {
        const r = window.classifier.classify(x.slice(s, s + 16000), 16000)[0];
        res.push(window.sounds.groupScores(r.classifications[0].categories));
      }
      return res;
    }, clip.filename);
    out.push({ ...clip, windows });
    if (i % 50 === 0) console.log(`${i}/${clips.length}`);
  }
  fs.writeFileSync(outPath, JSON.stringify(out));
  console.log(`wrote ${out.length} clips to ${outPath}`);
} finally {
  await browser.close();
  server.close();
}
