# Thai Sign Language datasets

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

## Plan

1. **Done (2026-10-21):** first model trained on TSL-51's landmark CSVs (hand points only), see Training below. Treat it as a demo of ~50 signs, not a reliable translator.
2. **In parallel:** collect our own consented data with the in-app recorder. Use a signer code and no real names, aiming for 10+ deaf signers. This is the only way to get a fair score on people the model hasn't seen.
3. Contact NADT, Ratchasuda College (Mahidol) and the TSL-ONE-S authors about data sharing and vocabulary review by deaf signers.
4. Before any commercial release, replace or license any non-commercial data.

## Training

### On TSL-51 (the bundled model)

```bash
# landmark CSVs only (about 1.2 GB), not the videos
B=https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51/resolve/main
mkdir -p data/tsl51 && cd data/tsl51
curl -L --create-dirs -o metadata/expert_metadata.csv $B/metadata/expert_metadata.csv
curl -L --create-dirs -o metadata/user_sign_metadata.csv $B/metadata/user_sign_metadata.csv
curl -LO $B/landmarks/expert_primary_02.zip && unzip -q expert_primary_02.zip -d e02
curl -LO $B/landmarks/expert_primary_03.zip && unzip -q expert_primary_03.zip -d e03
# plus the 547 files listed in user_sign_metadata.csv, from $B/landmarks/user_sign/<name>.csv
cd ../..
python training/tsl51.py prepare data/tsl51 --out data/tsl51.npz
python training/tsl51.py selftest data/tsl51.npz        # numpy features == features.py
python training/train_tsl51.py data/tsl51.npz --window 45 --hidden 256 --test-signers ""   # the shipped model
python training/train_tsl51.py data/tsl51.npz --window 45 --hidden 256 --test-signers user_01  # a score
```

Results (2026-10-21), 52 classes (51 signs + `null_act`), one 45-frame window centred on the movement of each clip:

| Held-out signer | Trained on | Clips | Top-1 | Top-3 |
|---|---|---|---|---|
| user_01 (researcher, 640x480 webcam) | both interpreters | 547 | 60% | 84% |
| expert_02 (interpreter) | expert_01 + user_01 | 552 | 55% | 78% |
| expert_01 (interpreter, 46 signs) | expert_02 + user_01 | 481 | 42% | 63% |

The shipped `models/tsl-model.json` uses the same settings trained on all three signers; its `report` holds these numbers.
Random guessing would be 2%. With only 3 signers, expect a new person to land in the same 40–60% range, lower with a
different camera angle or lighting. What was tried: windows of 30/45/60 frames (45 best), 128 vs 256 hidden units (256
slightly better), stronger weight decay (no gain), and training on mirrored clips for left-handed signers
(`--mirror`, about 2 points lower on right-handed test signers, so off by default).

Things to know:

- Only the 2 x 21 hand points are used, because the app runs the hand tracker only. Signs that differ mainly by where
  the hand is on the body (face vs chest) are the most confused.
- Hands: Holistic names hands from the signer's side; the app's tracker labels them as if the camera image were mirrored.
  The adapter maps the signer's right hand to the app's `Left` slot. This follows MediaPipe's documentation and has not
  yet been checked on a phone; if predictions look mirrored on a real camera, swap the slots in `tsl51.frame_vectors`.
- Coordinates are corrected for the frame's aspect ratio (new `aspect` field on each HandFrame), so a portrait phone and
  a landscape webcam give the same hand shape. Old recordings without the field are read as aspect 1.
- When fewer than 30% of the frames in a window contain a hand, the app shows "no sign" without asking the model.
- **License:** TSL-51 is CC BY-NC-SA 4.0. The model is a derivative, so it is non-commercial and share-alike, and must
  credit Namonpas Viriyaroj, TSL-51 (2026). Replace or license this data before any commercial release.

### On your own recordings

```bash
python training/train.py exported-samples.json [more.json ...] --out models/tsl-model.json
```

The trainer prints accuracy and per-class results. When samples carry a `signer` code, it holds out whole signers for testing, so the score reflects people the model never saw. When `models/tsl-model.json` exists, the app loads it and shows predictions with a confidence score. Low-confidence predictions are labeled "may be wrong".
