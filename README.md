# Fang ฟัง

An app for deaf and hard-of-hearing people, starting in Thailand. It's one installable web app (PWA) that runs on Android and iPhone browsers and on computers. Thai comes first, and the UI already switches to English. To add a language, add one block in `js/i18n.js`.

## What works now (implemented and tested)

| Feature | Status | How |
|---|---|---|
| **Live captions** (Thai/English speech to large text) | Works in Chrome (Android/desktop) and Safari (iPhone). Firefox has no support. | Browser Web Speech API with `th-TH`. Restarts itself after pauses, has adjustable text size, and lets you copy or save the transcript. |
| **Offline Thai captions** | Works in Chrome and Safari (needs WebAssembly and AudioWorklet). Tested in desktop Chromium with a fake microphone and the network switched off. Not yet tested on a real phone. | Optional pack, downloaded once from Settings (about 174 MB) and kept in its own cache: a [Thai Zipformer](https://huggingface.co/yfyeung/icefall-asr-gigaspeech2-th-zipformer-2024-06-20) from the k2/icefall team, trained on GigaSpeech2 (Apache-2.0), run with ONNX Runtime Web, plus the Silero voice detector (MIT) to find pauses. Text appears a sentence at a time after the speaker pauses, with a rough preview every 1.5 s. Thai only. 3% character errors on the model's own 3 sample clips (a sanity check, not a benchmark). In the cloud test machine it decodes about 5x faster than real time on one CPU thread; phones will be slower. "Automatic" uses the browser's online recogniser when there is internet and this one when there isn't, and switches mid-session if the connection drops. |
| **Caption history** | Works. | Each listening session is saved on the device (IndexedDB) as it goes, with a time on each line. A history screen lists, searches, opens, copies, downloads (.txt) and deletes them. Auto-save can be turned off in Settings. |
| **Type to speak** | Works everywhere. A Thai voice needs the device to have one installed. If none is installed it falls back to large text. | `speechSynthesis` plus a full-screen large-text card and quick phrases (built-in and your own saved ones). |
| **Sign camera: hand tracking** | Works. Tested with a fake camera feeding a real hand photo, and 1 hand was detected. | MediaPipe Hand Landmarker (v1.1.1) runs **on the device**. Its files are bundled in `vendor/` and `models/`, so after the first load no network is needed. |
| **Recognizer interface** (offline / online / hybrid) | Works. Offline mode uses the bundled TSL-51 model below. | `js/sign/recognizer.js`. Online mode posts landmarks only (never video) to a URL you configure. |
| **Thai sign model (demo, 51 signs)** | Works as a demo. On a signer it never saw, run through the app's own hand tracker, it gets 68–70% of single signs right (83–84% in its top 3), at any camera frame rate and for left-handed signing; 52–81% on the other held-out signers. Treat it as a guess. Not yet tested on a real phone camera. | `models/tsl-model.json`, a small classifier trained on the [TSL-51](https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51) landmarks by `training/train_tsl51.py`. **TSL-51 is CC BY-NC-SA 4.0, so this model is for non-commercial use only.** Details in [docs/datasets.md](docs/datasets.md). |
| **ASL sign model (demo, 250 signs)** | Works as a demo. Pick "American Sign Language" in the sign tab. On 8 Deaf signers it never saw, it gets 64% of single signs right (82% in its top 3), counting only clips where the tracker saw a hand; on the careful one-sign recordings alone, 60% (77%). It averages 4 small networks. Not yet tested on a real phone camera. | `models/asl-model.json`, the same kind of classifier trained on [PopSign ASL v1.0](https://huggingface.co/datasets/sign/popsign-images) by `training/train_popsign.py`, using hand points from the app's own tracker. **PopSign is CC BY 4.0, so commercial use is allowed with credit** (Starner et al., NeurIPS 2023). Details in [docs/datasets.md](docs/datasets.md). |
| **Dataset recorder** | Works. | Records 2-second landmark sequences with a label and an optional signer code, and exports JSON (`fang-handframes-v1`). No video is stored. |

## What it does NOT do yet

- **It does not translate Thai Sign Language or ASL.** It guesses one isolated sign (from 51 Thai or 250 ASL signs) from about 1.5 seconds of hand movement. It does not use the face or body, and has not been tested with deaf users of the app. The Thai model was trained on only 3 signers. Sentences are not supported.
- No accuracy or speed claims. In the cloud test machine (software graphics, no GPU), detection took about 750 to 900 ms per frame. Real phones still need to be measured.
- Without the offline pack, live captions use the browser vendor's speech service. Audio may go to Google or Apple servers, and it needs internet. The offline pack keeps audio on the device but is Thai only, and it is not streaming: words show after each pause rather than as they are spoken.

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
# offline captions: needs the pack files (see js/offline/pack.js) in a folder, and a 16 kHz wav of speech
FANG_SPEECH_DIR=path/to/pack npm test       # also runs the real Thai model on its sample clips
node tests/e2e-captions.mjs path/to/pack speech.wav
```

## Structure

```
index.html, css/app.css      UI (3 tabs, Thai-first, large touch targets, dark mode)
js/captions.js               live captions, engine choice, auto-save
js/transcripts.js, history.js saved transcripts (IndexedDB) and the history screen
js/offline/                  offline Thai speech: pack download, mic capture, voice detection,
                             log-mel features, transducer decoding (worker)
vendor/onnxruntime           ONNX Runtime Web (MIT), runs the speech model
models/silero_vad_v5.onnx    Silero voice activity detector (MIT)
js/speak.js                  type-to-speak, quick phrases, large text
js/sign/sign.js              camera, MediaPipe, overlay, stats, recorder
js/sign/features.js          landmarks -> normalized 128-value frame vectors (pure, unit-tested)
js/sign/recognizer.js        Offline/Online/Hybrid recognizer interface
sw.js, manifest.webmanifest  offline cache + installable app
vendor/mediapipe, models     bundled MediaPipe runtime + hand model (Apache-2.0)
vendor/fonts                 IBM Plex Sans Thai, bundled for offline use (SIL OFL 1.1)
css/icons.css                line icons from Lucide (ISC), drawn as CSS masks
```

## Roadmap

1. **Now:** put it on an HTTPS host, test it on real Thai phones, and get feedback from deaf users.
2. **Data:** use the recorder to collect consented samples with Thai deaf signers; more signers is what will raise accuracy most. Ask about TSL-ONE-S and NADT data.
3. **First models (done, demos):** TSL-51 isolated-sign classifier and a PopSign ASL classifier, both scored on held-out signers. Next: test on phones, add more signers, try a sequence model and body/face points.
4. **Online backend:** an API that implements the `OnlineRecognizer` request format, with no data retention by default.
5. **Native Android (Kotlin + CameraX + MediaPipe)** as the project brief describes, if real-device performance in the browser isn't good enough. The feature format and recognizer interface carry over unchanged.
6. **Offline Thai captions (done, first version):** next, measure speed and accuracy on real phones and in noisy rooms, and consider a smaller or streaming model if phones lag.
7. Continuous signing, face and body cues, sentence translation: long-term research, evaluated with qualified TSL users.
