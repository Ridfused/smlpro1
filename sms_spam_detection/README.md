# SMS Spam & Scam Detection

Statistical Machine Learning project: TF-IDF (word + character n-grams) → calibrated Linear SVM,
with independent warning-sign rules, link-safety checks and a Streamlit app.

## Run it (Windows / Mac / Linux)

```bash
cd sms_spam_detection
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

python train.py                  # ~1-2 min: EDA charts, model comparison, grid search, final model
streamlit run app.py             # opens http://localhost:8501
```

`python train.py --fast` skips the model comparison and grid search and reuses their winner, so it produces the
same final model and numbers as the full run (the comparison table is kept from the last full run).
The zip already contains trained models: you only need to retrain if the app says a model can't be loaded.
Always run `train.py` once on your own laptop: saved models depend on your scikit-learn version.

## Project structure

| File | What it does | Report section |
|---|---|---|
| `src/data.py` | Loads UCI data, cleans text (NFKC, HTML entities, whitespace), exact de-dup, MinHash near-duplicate groups, shared-phone-number groups, label-conflict quarantine | 3, 5.3 |
| `train.py` | EDA figures, grouped + frozen train/test split, 5-model comparison, 24-setting grid search, calibrated final model, threshold from out-of-fold predictions, latency, per-language results | 4, 5, 6 |
| `src/model.py` | TF-IDF word 1–2-grams + char 1–6-grams, Linear SVM + sigmoid (Platt) calibration, word-level explanations, embedding features | 5, 7, 9 |
| `src/rules.py` | Warning signs: KYC, OTP/PIN, UPI “enter PIN to receive”, prize, delivery fee, fake job, investment, premium-rate, urgency (English, Hinglish, Hindi) | 9 |
| `src/links.py` | Link safety as text only: shorteners, IP hosts, punycode, risky TLDs, brand look-alikes, defanging; optional Google Safe Browsing | 9 |
| `src/risk.py` | Risk score = 60 × probability + warning-sign points (−20…+40); low < 30 ≤ medium < 60 ≤ high | 9 |
| `app.py` | Streamlit: Scan, Batch (CSV/TXT, safe CSV download), Live dashboard (counts only), Model evaluation, model selector | 9 |
| `experiments/embeddings_experiment.py` | Sentence-embedding and hybrid models on the same frozen test set | 7 |
| `samples/sample_batch.csv` | 10 messages for the batch demo | – |

## Results (frozen test set of 4,115 messages, 4 languages)

Data: UCI SMS Spam Collection (5,574 → 5,160 unique) + French/German/Hindi translations
(`data/raw/data-augmented.csv`, already included) → 20,571 unique messages (18,023 ham, 2,548 spam).

| Metric | Value |
|---|---|
| Accuracy | 0.9876 |
| Spam precision | 0.9811 |
| Spam recall | 0.9176 |
| Spam F1 | 0.9483 |
| TN / FP / FN / TP | 3,596 / 9 / 42 / 468 |
| Latency (median, one message) | ~4 ms |

Model comparison (grouped 5-fold CV F1): Linear SVM word+char 0.943 > LogReg word+char 0.942 >
Linear SVM word-only 0.905 > LogReg word-only 0.902 > Naive Bayes 0.834.
Grid search best: C = 0.3, word 1–2-grams, char 2–5-grams. Threshold 0.55 (from training-data CV).
By language (English-only training → all languages): EN 0.972 → 0.972, FR 0.939 → 0.972,
DE 0.939 → 0.964, HI 0.591 → 0.881.

Full training takes about 6 minutes on a 2-core machine (`--fast`: about 1 minute).
`--no-translations` trains on English UCI data only.

## Website (`web/`)

`web/index.html` + `web/engine.js` + `web/model.json` is a static site that runs a compact export of the
model in the browser (F1 0.943 on the same test set; probabilities match Python to within 0.00001).
Open it with any static server: `cd web && python -m http.server 8000`, then visit http://localhost:8000.
Rebuild `model.json` with `python -m tools.export_web_model`.

## Report

`report/SMS_Spam_Detection_Project_Report_v2.pdf`: the project report with the numbers this code produces.

## Optional: sentence embeddings (Section 7)

```bash
pip install sentence-transformers
python -m experiments.embeddings_experiment
```
Downloads the multilingual MiniLM model (~470 MB) on first run, prints the comparison table and adds
“Sentence Embeddings” and “Hybrid” to the app's model selector.

## Optional: Google Safe Browsing

Set an API key (`set GOOGLE_SAFE_BROWSING_KEY=...` on Windows, or put it in `.streamlit/secrets.toml`).
The app only sends URLs, and only after the user ticks the consent box.

## Not included in this code

Section 8 of the report (the consolidated 513,005-message dataset from 8 sources and its retrained models)
is not part of this folder.
