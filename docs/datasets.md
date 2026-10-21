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

1. **Now:** train a first model on TSL-51's landmark CSVs. Convert them to our `fang-handframes-v1` format (hand points only), then run `training/train.py`. Treat it as a demo of ~50 signs, not a reliable translator.
2. **In parallel:** collect our own consented data with the in-app recorder. Use a signer code and no real names, aiming for 10+ deaf signers. This is the only way to get a fair score on people the model hasn't seen.
3. Contact NADT, Ratchasuda College (Mahidol) and the TSL-ONE-S authors about data sharing and vocabulary review by deaf signers.
4. Before any commercial release, replace or license any non-commercial data.

## Training

```bash
python training/train.py exported-samples.json [more.json ...] --out models/tsl-model.json
```

The trainer prints accuracy and per-class results. When samples carry a `signer` code, it holds out whole signers for testing, so the score reflects people the model never saw. When `models/tsl-model.json` exists, the app loads it and shows predictions with a confidence score. Low-confidence predictions are labeled "may be wrong".
