# Sign language datasets

Thai Sign Language (TSL) first, then American Sign Language (ASL) [below](#american-sign-language-asl).

## Thai Sign Language

Checked 2026-10-21. Licenses can change, so re-check the source page before using any of these.

| Dataset | What it has | License / access | Fit for Fang |
|---|---|---|---|
| **TSL-51** ([Hugging Face](https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51)) | 51 everyday signs (สวัสดี, ขอบคุณ, หูหนวก, กิน, ที่ไหน…) plus a `null_act` "no sign" class. Videos come from 2 expert interpreters and 1 researcher. Has 1,155 original expert videos with 44k augmented copies, 547 user videos and 252 sentence videos. It also ships **MediaPipe Holistic landmarks as CSV** (54 points × xyz per frame). 2.5 GB. | **CC BY-NC-SA 4.0, non-commercial research only.** The card says the expert clips are used under "fair use" and the web-scraped clips may have copyright limits. Ask the maintainers before any commercial use. | **Best starting point.** Same landmark family as our app (MediaPipe), real words, and it includes a "no sign" class. Weakness: only about 3 signers, so expect poor accuracy on new people. |
| **TSL-ONE-S** ([IEEE, 2026](https://ieeexplore.ieee.org/document/11421897/)) | "Real-world" word-level Thai Sign Language dataset with deep-learning benchmarks ([related paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5603879)). | Not stated anywhere I could read. Probably needs a request to the authors (Chalotonpised, Chen, Vijitkunsawat, Lin). | Worth emailing the authors. It could add more signers. |
| **NADT Thai Sign Language Database** ([th-sl.com](https://www.th-sl.com/)) | Dictionary of 8,086 words from the National Association of the Deaf in Thailand. | © NADT, and the terms reportedly forbid bots and data collection. Needs written permission. | Don't scrape it. It's the best partner to ask for data and for help reviewing the vocabulary. |
| **One-Stage-TFS** ([Mendeley](https://data.mendeley.com/datasets/rknd3wbz42)) | 7,200 photos of 15 Thai fingerspelling consonants, from Rajabhat Maha Sarakham University. | License not shown on the page. | Fingerspelling only, and still photos. Possible later add-on. |
| **Thai digit videos** ([paper](https://www.scitepress.org/publishedPapers/2023/116437/pdf/index.html)) | Digits 1–9, 567 videos, 21 signers. | No public release stated. | Small. Ask the authors if needed. |
| Kaggle "Thai Sign Language Multi Signer" ([link](https://www.kaggle.com/datasets/kornkachornkhumyato/km-research-multi-signer-dataset)) | 6.5 GB, reportedly MIT-licensed. I couldn't read the contents. | Check on Kaggle. | Look at it once Kaggle is reachable. More signers would help most. |
| ThaiSignVis (Hugging Face) | Clips taken from parliament broadcasts. | Tagged Apache-2.0, but the video rights are unclear. | **Don't use.** |

### Plan

1. **Done (2026-10-21):** first model trained on TSL-51's landmark CSVs (hand points only), see Training below. Treat it as a demo of ~50 signs, not a reliable translator.
2. **In parallel:** collect our own consented data with the in-app recorder. Use a signer code and no real names, aiming for 10+ deaf signers. This is the only way to get a fair score on people the model hasn't seen.
3. Contact NADT, Ratchasuda College (Mahidol) and the TSL-ONE-S authors about data sharing and vocabulary review by deaf signers.
4. Before any commercial release, replace or license any non-commercial data.

### Training

#### On TSL-51 (the bundled model)

```bash
# landmark CSVs only (about 1.2 GB), plus the researcher's 547 short videos for the app-tracker test
B=https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51/resolve/main
mkdir -p data/tsl51 && cd data/tsl51
curl -L --create-dirs -o metadata/expert_metadata.csv $B/metadata/expert_metadata.csv
curl -L --create-dirs -o metadata/user_sign_metadata.csv $B/metadata/user_sign_metadata.csv
curl -LO $B/landmarks/expert_primary_02.zip && unzip -q expert_primary_02.zip -d e02
curl -LO $B/landmarks/expert_primary_03.zip && unzip -q expert_primary_03.zip -d e03
# plus the files listed in user_sign_metadata.csv: $B/<landmark_path> and $B/<video_path>
cd ../..
pip install numpy pandas torch                          # torch: CPU build is enough
python training/tsl51.py prepare data/tsl51 --out data/tsl51.npz
python training/tsl51.py selftest data/tsl51.npz        # numpy features == features.py
python training/tsl51.py track data/tsl51 --out data/tsl51-user-app.npz   # needs mediapipe + opencv, ~5 min
python training/train_tsl51.py data/tsl51.npz --test-signers ""   # the shipped model, ~15 min on 4 CPUs
python training/train_tsl51.py data/tsl51.npz --test-signers user_01 --app-test data/tsl51-user-app.npz  # a score
```

Results (2026-10-24, third model), 52 classes (51 signs + `null_act`), one prediction per clip, centred on the movement:

| Held-out signer | Trained on | Clips | Top-1 | Top-3 | Second model | First model |
|---|---|---|---|---|---|---|
| user_01 (researcher, 640x480 webcam) | both interpreters | 547 | 76% | 90% | 69% / 90% | 60% / 84% |
| user_01, landmarks from **the app's own tracker** | both interpreters | 547 | 67% | 83% | 63% / 83% | **7% / 13%** |
| expert_02 (interpreter) | expert_01 + user_01 | 552 | 81% | 94% | 72% / 91% | 55% / 78% |
| expert_01 (interpreter, 46 signs) | expert_02 + user_01 | 481 | 52% | 81% | 45% / 76% | 42% / 63% |

The app-tracker row is the closest to real use: the researcher's videos run through `models/hand_landmarker.task`, the
app's `HandLabeler` (replayed by `tsl51.label_hands`), then the same steps as `MlpModel.predict` (`train.model_probs`).
Through that exact path the third model gets 67.5% / 69.7% / 68.7% top-1 with the camera at 30 / 15 / 10 fps (second
model 62.5 / 63.3 / 64.0), and the same for mirrored (left-handed) signing. Top-3 stays around 83-84% for both, so the
gain is mostly in getting the first guess right. The shipped `models/tsl-model.json` uses these settings trained on all
three signers; its `report` holds these numbers. Random guessing would be 2%. With only 3 signers, a new person should
expect roughly half to three quarters right, lower with a different camera angle or lighting.

**Guarding against overfitting (third model).** Every setting was chosen on the average of all three leave-one-signer-out
folds, each run with 2 random seeds (one fold moves by about 2 points between seeds, so smaller differences were treated
as noise), never on training accuracy. The shipped model is then retrained on all signers with the chosen settings
unchanged. What the third model changed, averaged over the three folds (61.6% -> 70.3% top-1):

1. **`handPosition`** (added for the ASL model): where each wrist is in the frame and how big the hand looks, so signs
   made at the chin and at the chest no longer look alike. Training randomly shifts and zooms the whole frame (about
   ±2.5% and ±5%) so the model doesn't learn one camera's framing: +4.4 points.
2. **Dropout 0.5** instead of 0.3, and **30 epochs** instead of 10: +3.5. 40 or 60 epochs (with dropout 0.6) were no
   better.
3. **Simulated cameras down to 5 fps** instead of 8, for slow phones: no change in score, kept for robustness.

Tried without gain: averaging 2-6 networks (under 1 point, and each doubles the download), 512-unit layers, stronger
weight decay, random time warping, more landmark noise, input dropout.

What the second model changed from the first, in order of effect:

1. **Hand mix-up fixed.** The first adapter put the signer's right hand in the app's `Left` slot, based on MediaPipe's
   docs. Running the app's tracker on 40 raw TSL-51 videos showed it names Holistic's left hand `Left` 97% of the time,
   so in the app the signing hand arrived in the slot the model had learned for the other hand. Scores on the
   dataset's own landmarks never showed this, because training and test were wrong the same way.
2. **Same time span at any frame rate.** The data is 30 fps; phones track hands at maybe 8-20 fps, so the first
   model's 45-frame window covered 2-5 seconds on a phone instead of 1.5. Models now carry `fps` (15) and `window`
   (24): the app keeps the last 1.6 s of frames and resamples them to 24 by timestamp, and training simulates cameras
   at random 8-30 fps.
3. **Training on the dataset's 30 augmented copies** (rotation, scale, speed, crops) plus our own random rotation,
   stretch, jitter and dropped hands: +14 points on the app-tracker test.
4. **`features: 2`**: besides mean, spread and first-half/second-half change, the mean of each quarter of the window,
   which keeps the order of movements (+5). Frames without hands are skipped (`skipEmptyFrames`, +4).
5. **Two hidden layers** of 256 with dropout, trained with PyTorch and exported as `layers` (+3). 512,256 was no better.
6. **`canonicalHand`**: when the `Left` slot hand leads a window, the window is mirrored, so left-handed signers get the
   same accuracy (about 1 point cost for right-handed ones). Plain mirror augmentation instead cost 6 points.

Tried without gain: a small temporal conv net on the 24 frames (same top-1, lower top-3), windows of 1.6-2 s at 10 or
15 fps (all within a point), 20 epochs instead of 10.

Things to know:

- Only the 2 x 21 hand points are used, because the app runs the hand tracker only. Where the hands are is known only
  relative to the camera frame (`handPosition`), not to the face or body, so signs that differ mainly by body location
  are still the most confused.
- Coordinates are corrected for the frame's aspect ratio (new `aspect` field on each HandFrame), so a portrait phone and
  a landscape webcam give the same hand shape. Old recordings without the field are read as aspect 1.
- When fewer than 30% of the frames in a window contain a hand, the app shows "no sign" without asking the model.
- **Touching hands (2026-10-22).** When the hands touch or cross, the tracker often labels both 'Left' or both
  'Right' (6% of all frames on the 547 user_01 clips, about a quarter of the two-hand frames), and the app then kept
  only one of them. `HandLabeler` gives each hand its own label by where each was last seen. On the 196 clips where the
  hands touch: 79.1% -> 82.7% top-1 (all clips 82.8% -> 84.3%; this model trained on user_01's Holistic landmarks, so
  these are higher than the held-out score above, compare only the change). Tried without gain: carrying a lost hand
  forward for 150-500 ms (-1 to -6 points) and lower tracker thresholds (0.3 instead of 0.5: no more two-hand frames).
  A hand fully hidden behind the other is still lost by the tracker itself.
- **License:** TSL-51 is CC BY-NC-SA 4.0. The model is a derivative, so it is non-commercial and share-alike, and must
  credit Namonpas Viriyaroj, TSL-51 (2026). Replace or license this data before any commercial release.

#### On your own recordings

```bash
python training/train.py exported-samples.json [more.json ...] --out models/tsl-model.json
```

The trainer prints accuracy and per-class results. When samples carry a `signer` code, it holds out whole signers for testing, so the score reflects people the model never saw. When `models/tsl-model.json` exists, the app loads it and shows predictions with a confidence score. Low-confidence predictions are labeled "may be wrong".

## American Sign Language (ASL)

Checked 2026-10-21. ASL is the English-speaking sign language with by far the most public data. (British Sign Language
is a different language; it would need its own data.)

| Dataset | What it has | License / access | Fit for Fang |
|---|---|---|---|
| **PopSign ASL v1.0** ([paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract.html), [Hugging Face copy](https://huggingface.co/datasets/sign/popsign-images)) | 250 signs for young children's vocabulary, about 210k clips by **47 Deaf adults** on Pixel phone selfie cameras, split by signer (31 train / 8 validation / 8 test). The Hugging Face copy has the signing part of each clip as 256x256 frames at about 5 fps; a "non-game" subset (39k clips, 19 GB) and a "game" subset (174k clips, 71 GB). | **CC BY 4.0**: commercial use allowed with credit. | **Used.** Phone cameras like ours, many Deaf signers, clean license. Weakness: the Hugging Face copy is low frame rate and square-cropped, so the tracker misses hands in about half the frames. The original videos are at signdata.cc.gatech.edu, which this cloud environment can't reach. |
| Google "Isolated Sign Language Recognition" (Kaggle `asl-signs`) | The same 250 signs from the same collection, 21 signers, as MediaPipe Holistic landmarks. | Reportedly CC BY 4.0; check the Kaggle page. Needs a Kaggle account and accepting the competition rules. | Good later source. Needs a Kaggle API token added to the environment. |
| ASL Citizen ([Microsoft](https://www.microsoft.com/en-us/research/project/asl-citizen/)) | 2,731 signs, 83k webcam videos by 52 Deaf/hard-of-hearing signers, split by signer. | **Microsoft Research License: non-commercial research only, and you may not distribute the data or modifications of it** or offer it as a hosted solution. | **Not used.** Shipping a model trained on it inside a public web app is a legal grey area under those terms. Ask ASL_Citizen@microsoft.com before using it. Unofficial copies on Hugging Face should not be used. |
| WLASL | 2,000 signs from web videos. | Computational Use of Data Agreement, research only; video rights belong to the original sites. | Not used. |

### Training on PopSign

```bash
pip install mediapipe huggingface_hub pyarrow pillow numpy torch
# Runs the app's own hand tracker (models/hand_landmarker.task) over every frame and streams the parquet files:
# ~35 min for non-game, ~2 h 20 min for game on 4 CPUs
python training/popsign.py prepare --subset non-game --out data/popsign.npz
python training/popsign.py prepare --subset game --out data/popsign-game.npz
# score on the 8 validation signers (for comparing settings)
python training/train_popsign.py data/popsign.npz data/popsign-game.npz --repeat-first 4
# the shipped model: train + validation signers, scored on the 8 test signers (~30 min, needs ~10 GB RAM)
python training/train_popsign.py data/popsign.npz data/popsign-game.npz --repeat-first 4 --final
```

Results (2026-10-22). Every number is for signers the model never trained on, and counts only clips where the
tracker saw a hand in at least 2 frames (about 75% of clips). In the app the others would show "no sign".
"Non-game" clips are the careful ones (one sign per recording); "game" clips were recorded while playing a game.

| Model | Trained on | Scored on | Signs | Top-1 | Top-3 |
|---|---|---|---|---|---|
| v1 (2026-10-21) | non-game, train + validation | non-game test clips (2,412) | 246 | 45% | 64% |
| **v2, shipped** | non-game + game, train + validation | non-game test clips (2,412) | 250 | **57%** | **76%** |
| **v2, shipped** | non-game + game, train + validation | all test clips (27,281) | 250 | **62%** | **80%** |
| v1 settings | non-game, train | non-game validation (6,218) | 244 | 32% | 48% |
| v2 settings | non-game + game, train | non-game validation (6,220) | 250 | 40% | 59% |

Random guessing would be under 0.5%. The Hugging Face copy's non-game test split covers only 115 of the signs; the
validation rows cover 208, so they are the fairer estimate for the whole vocabulary. Signs that look alike from the
hands alone are the most confused (nap/sleepy, dog/puppy, cheek/think, ear/hear), because the app only tracks hands.

What changed in v2, measured on the validation signers (non-game clips):

- **Where the hands are** (`handPosition` in the model file): each frame also gets the wrist's place in the frame and
  the hand's size, so signs made at the chin, chest or side can be told apart: +4 points (36% → 40%).
- **Quarter-by-quarter features** (`features: 2`, shared with the Thai model) and a **two-layer network**
  (512 + 256 units, dropout), trained with PyTorch.
- **Fresh augmentation every batch:** mirrored (left-handed) copies, small rotations, zoom and shifts of the whole
  frame, landmark jitter, shorter spans, dropped frames and dropped hands. Together with the network: +4 points.
- **The game subset** (174k more clips): a big gain on game clips (57% on validation), but on the careful non-game
  clips it only helps when those are weighted 4x (`--repeat-first 4`), and then it roughly matches non-game alone. The
  shipped model uses both, since it does better on the mixed test set and has seen more ways of signing.
- **Frame rate:** the model has `fps: 5` and `window: 8`, so the app resamples its last 1.6 s of camera frames to 5 fps,
  the rate of the training clips, whatever rate the phone tracks at.
- Tried, no gain: 1024 + 512 units (39% vs 40%), 80 instead of 40 epochs, upscaling or padding the 256x256 frames
  before tracking (hands found in 59% vs 58% of frames).
- No "no sign" class: PopSign has none. With hands in view but not signing, the app will still name a sign, usually
  with low confidence ("may be wrong").
- **Next:** the original full-rate videos (the tracker would see hands in far more frames), body and face points so
  the model knows where the hand is relative to the face, and testing on real phones.
- **License:** PopSign ASL v1.0 is CC BY 4.0. Credit: Thad Starner et al., "PopSign ASL v1.0: An Isolated American Sign
  Language Dataset Collected via Smartphones", NeurIPS 2023 Datasets and Benchmarks. The model file carries this in its
  `license` field.
