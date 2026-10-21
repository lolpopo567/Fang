# Fang ฟัง

An app for deaf and hard-of-hearing people in Thailand. It's one installable web app (PWA) that runs on Android and iPhone browsers and on computers. Thai comes first, and the UI already switches to English. To add a language, add one block in `js/i18n.js`.

## What works now (implemented and tested)

| Feature | Status | How |
|---|---|---|
| **Live captions** (Thai/English speech to large text) | Works in Chrome (Android/desktop) and Safari (iPhone). Firefox has no support. | Browser Web Speech API with `th-TH`. Restarts itself after pauses, has adjustable text size, and lets you copy or save the transcript. |
| **Type to speak** | Works everywhere. A Thai voice needs the device to have one installed. If none is installed it falls back to large text. | `speechSynthesis` plus a full-screen large-text card and quick phrases (built-in and your own saved ones). |
| **Sign camera: hand tracking** | Works. Tested with a fake camera feeding a real hand photo, and 1 hand was detected. | MediaPipe Hand Landmarker (v1.1.1) runs **on the device**. Its files are bundled in `vendor/` and `models/`, so after the first load no network is needed. |
| **Recognizer interface** (offline / online / hybrid) | Wiring works, but **no Thai Sign Language model exists yet**, so the screen says so. | `js/sign/recognizer.js`. Online mode posts landmarks only (never video) to a URL you configure. |
| **Sign model support** | Works. Tested with synthetic data only; there is no real model yet. | `training/train.py` (numpy only) trains a small classifier. The app loads `models/tsl-model.json` if it exists. Datasets are reviewed in [docs/datasets.md](docs/datasets.md). |
| **Dataset recorder** | Works. | Records 2-second landmark sequences with a label and an optional signer code, and exports JSON (`fang-handframes-v1`). No video is stored. |

## What it does NOT do yet

- **It does not translate Thai Sign Language.** Detecting 21 hand points is not translation. That needs a model trained on licensed TSL data (candidates are TSL-ONE and TSL-51, but their licenses still need checking) and testing with deaf TSL signers.
- No accuracy or speed claims. In the cloud test machine (software graphics, no GPU), detection took about 750 to 900 ms per frame. Real phones still need to be measured.
- On most browsers, live captions use the browser vendor's speech service. Audio may go to Google or Apple servers, and it needs internet. Fully offline Thai speech-to-text is a later step (see roadmap).

## Run it

```bash
npm install          # only needed for tests (Playwright)
node tests/serve.mjs # then open http://localhost:8080
```

Camera and microphone need **HTTPS** (or `localhost`). To use it on a phone, host the folder on any static HTTPS host (GitHub Pages, Netlify, Cloudflare Pages) and use "Add to Home Screen" to install it.

## Tests

```bash
npm test                                   # unit tests: features, recognizers, hybrid fallback
node tests/e2e.mjs path/to/hand-video.y4m  # browser test with a fake camera
```

## Structure

```
index.html, css/app.css      UI (3 tabs, Thai-first, large touch targets, dark mode)
js/captions.js               live captions
js/speak.js                  type-to-speak, quick phrases, large text
js/sign/sign.js              camera, MediaPipe, overlay, stats, recorder
js/sign/features.js          landmarks -> normalized 128-value frame vectors (pure, unit-tested)
js/sign/recognizer.js        Offline/Online/Hybrid recognizer interface
sw.js, manifest.webmanifest  offline cache + installable app
vendor/mediapipe, models     bundled MediaPipe runtime + hand model (Apache-2.0)
```

## Roadmap

1. **Now:** put it on an HTTPS host, test it on real Thai phones, and get feedback from deaf users.
2. **Data:** check the TSL-ONE and TSL-51 licenses. Use the recorder to collect consented samples with Thai deaf signers.
3. **First model:** isolated-sign classifier on the `frameVector` sequences (Python), exported to TFLite/LiteRT and plugged into `OfflineRecognizer`. Report per-class accuracy on signers held out of training.
4. **Online backend:** an API that implements the `OnlineRecognizer` request format, with no data retention by default.
5. **Native Android (Kotlin + CameraX + MediaPipe)** as the project brief describes, if real-device performance in the browser isn't good enough. The feature format and recognizer interface carry over unchanged.
6. **Offline Thai captions:** an on-device Thai speech model (for example a Whisper variant fine-tuned for Thai) for privacy and use without internet.
7. Continuous signing, face and body cues, sentence translation: long-term research, evaluated with qualified TSL users.
