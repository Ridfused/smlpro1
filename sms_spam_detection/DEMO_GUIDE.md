# Demo guide (about 5 minutes)

**Before you present:** run `python train.py` once (about 6 min; or `--fast`), then `streamlit run app.py`, and keep the app open in the browser.
Backup if your laptop fails: the website version runs on any phone or laptop browser.

1. **Problem (30 s).** SMS scams such as fake prizes, KYC phishing and OTP theft. This is binary text classification: spam vs ham.
   Spam is only 12.4% of the data, so accuracy alone is misleading. That is why we report precision, recall and F1.
2. **Model evaluation tab (1 min).** Show the class-balance and message-length charts, then the model comparison table
   (adding character n-grams gave the biggest improvement), then the confusion matrix.
3. **Scan tab (2 min).** Pick these examples in order:
   - *Prize spam (classic)*: the model gives about 100%, so it is HIGH risk.
   - *KYC phishing*: HIGH risk, shows the look-alike SBI link (defanged and never opened) and the recommended actions.
   - *UPI scam (Hinglish)*: the model says spam (~83%) and the UPI-PIN rule adds points, giving HIGH risk (90).
   - *Hinglish OTP fraud*: the model gives only 0.2% because it never saw such scams. The OTP-request rule
     still raises it to MEDIUM. This is why the rules are added on top of the model.
   - *Genuine OTP*: the model is unsure (43%) because the 2011 data has no OTP messages. The protective-wording
     rule ("do not share") keeps it at LOW risk (11).
   - *Friend message*: LOW risk, near 0%.
   - Point to the **"Why the model decided this"** panel, which shows the words that pushed the decision.
4. **Batch tab (45 s).** Upload `samples/sample_batch.csv`. Show the summary charts and the CSV download.
5. **Live dashboard (15 s).** It shows counts only. No message text is stored.

# Likely viva questions

- **Why TF-IDF and not plain word counts?** IDF down-weights words that appear everywhere ("the", "you"),
  and sub-linear TF stops one repeated word from dominating.
- **Why character n-grams?** They handle misspellings and shorthand ("txt"/"text", "congratulation(s)"),
  obfuscation ("fr€e") and Devanagari script.
- **Why Linear SVM?** Text features are high-dimensional and sparse. Linear models are fast, accurate and interpretable
  (one weight per feature). It had the best CV F1.
- **Why calibration?** An SVM outputs a distance from the margin, not a probability. Platt (sigmoid) scaling, fitted
  on out-of-fold predictions, turns that into a probability for the app.
- **class_weight="balanced"?** It raises the penalty for mistakes on the minority spam class, so the model doesn't
  just predict ham.
- **How did you avoid data leakage?** The split is made before fitting, and TF-IDF sits inside the Pipeline.
  Near-duplicates (MinHash) and messages sharing a phone number are grouped and never split across train and test.
  The test set is frozen (stored as hashes), and the threshold is chosen from cross-validation on training data only.
- **Precision vs recall here?** A false positive hides a real message (bad UX). A false negative lets a scam through.
  We report both, plus F1.
- **Limitations?** 2011 UK/Singapore data, so the model misses modern Indian scams (rules help). There is no real
  labelled Hindi/Hinglish dataset, and the probability reflects the training data, not real-world scam rates.
- **Future work?** Real consented Indian SMS, fine-tuning MuRIL/XLM-R, and a supervised scam-category classifier.
