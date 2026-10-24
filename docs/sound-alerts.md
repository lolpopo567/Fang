# Sound alerts

The phone listens for important sounds and, when it hears one, flashes the whole screen, vibrates
(Android; iPhone browsers cannot vibrate) and can send a notification if another app is open.

## How it works

- **Model:** [YAMNet](https://www.kaggle.com/models/google/yamnet) (Apache-2.0), the MediaPipe
  `audio_classifier/yamnet/float32` build, 4 MB, bundled as `models/yamnet.tflite`. It runs **on the
  device** with MediaPipe Tasks Audio 1.1.1 (`vendor/mediapipe/audio_bundle.mjs` and the
  `audio_wasm_*` files). No audio leaves the phone and none is stored.
- **Windows:** the app keeps the last 1 s of microphone audio and classifies it every 0.5 s.
  Phone voice processing (echo cancellation, noise suppression, auto gain) is turned off, because
  it treats alarms and doorbells as noise.
- **Sounds:** each alert groups a few of YAMNet's 521 AudioSet classes (`js/alerts/sounds.js`).
  Running water and "someone talking" are off by default.
- **When to alert:** the average score over the last 2 windows must reach the threshold
  (Low 0.5, Normal 0.3, High 0.15); after an alert that sound stays quiet for 10 s.

## How well it works

Measured with `tests/eval_alerts.mjs` on [ESC-50](https://github.com/karolpiczak/ESC-50)
(CC BY-NC, used only for measuring, not bundled): 5-second recordings, 40 per sound, run through
the same windows and rules as the app. "Other sounds" are 1,048 clips of 26 other ESC-50 classes,
chosen to be hard (cats, cows, laughing, footsteps, creaking doors, vacuum cleaner…).

| Sensitivity | Target sounds caught | Other clips that raised any alert |
|---|---|---|
| High (0.15) | 90% | 7.0% |
| **Normal (0.3)** | **80%** | **2.8%** |
| Low (0.5) | 72% | 1.0% |

Per sound at Normal: dog 100%, siren 98%, baby crying 90%, alarm clock 83%, door knock 80%,
glass breaking 63%, car horn 48%. ESC-50 has no doorbell, smoke alarm, phone or shouting class,
so those were not measured. Most false alerts were cats or pigs read as a dog, footsteps read as
a knock, and creaking doors or cats read as a crying baby.

**Not measured yet:** real phones, real homes, Thai doorbells and alarms, and battery use.

## Limits

- The app must stay open with the screen on (it holds a screen wake lock while listening).
  Browsers pause the microphone when the phone sleeps.
- It is a helper. It is not a replacement for a smoke alarm made for deaf people
  (strobe light or bed shaker).
- The flash pulses about once a second, under the 3 flashes per second limit in WCAG 2.3.1, and
  can be turned off.
