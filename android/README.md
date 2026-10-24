# Fang ฟัง for Android

A native Android app (Kotlin, Jetpack Compose, CameraX, MediaPipe) with the same features and design as the
web app in the repo root. It shares the web app's sign models, hand model, UI text and icons, so both apps stay
in step.

| Feature | How |
|---|---|
| Live captions | Android `SpeechRecognizer`, restarted after each pause until you stop it. Settings has **Recognise on this phone (offline)**: with the phone's Thai language pack installed, no audio leaves the phone. Copy, save as a file, clear, text size. Keeps the screen on while listening. |
| Type to speak | Android `TextToSpeech` (Thai voice for Thai text), quick phrases, your own saved phrases, full-screen large text. |
| Sign camera (prototype) | CameraX (640×480) → MediaPipe Hand Landmarker 1.1.1 on the device (GPU, falls back to CPU) → the same feature code and TSL/ASL models as the web app, every 500 ms. Offline / online / hybrid recognizer, confidence and "uncertain" warnings, landmark overlay, status/FPS/latency. |
| Sample recorder | 2-second landmark samples with a label and signer code, exported as `fang-handframes-v1` JSON, the same format as the web app, so the training scripts take both. No video is stored. |
| Settings | Language (Thai/English), theme (auto/light/dark), caption language, offline captions, text size, speech speed, sign language, recognition mode, online API URL. |

The sign camera has the same limits as in the web app: it guesses one sign at a time from a small vocabulary
and is often wrong. It is not a translator.

## Build

Open `android/` in Android Studio, or:

```bash
cd android
./gradlew assembleDebug        # app/build/outputs/apk/debug/app-debug.apk
./gradlew testDebugUnitTest    # tests
./gradlew testDebugUnitTest -Pscreenshots   # also writes screen images to app/build/screenshots/
```

Needs JDK 17+ and the Android SDK (platform 36). The release build is signed with the debug key so it can be
installed for testing; use your own key before publishing it.

## How it fits with the web app

- `models/` in the repo root is the app's `assets/` folder (see `app/build.gradle.kts`), so a retrained model
  reaches both apps.
- `app/src/main/java/app/fang/sign/Features.kt` and `SignModel.kt` are ports of `js/sign/features.js` and
  `js/sign/model.js`. `ParityTest` checks they give the same numbers as the JS code on recorded-style clips,
  including both models' predictions (`tools/make-parity-fixture.mjs` writes the expected values).
- The frame given to MediaPipe is upright but **not mirrored**, even from the front camera, because the web
  app feeds the raw camera frame and the models learned their Left/Right hand slots from it.
- UI text is generated from `js/i18n.js` (`node tools/gen-strings.mjs`; Android-only text is in that script)
  and icons from `css/icons.css` (`node tools/gen-icons.mjs`). CI fails if the generated files are stale.

## Not tested yet

It has been built and its screens rendered on a computer, but it has **not yet run on a real phone**: camera,
hand tracking speed, speech recognition and text-to-speech need checking on a device.

## Licences

IBM Plex Sans Thai (SIL OFL 1.1, `licenses/`), Lucide icons (ISC), MediaPipe (Apache 2.0). The TSL model is
trained on TSL-51 (CC BY-NC-SA 4.0, non-commercial only); the ASL model on PopSign ASL (CC BY 4.0).
