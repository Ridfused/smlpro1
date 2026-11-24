"""Section 7: sentence embeddings vs TF-IDF on the SAME frozen test set.

    pip install sentence-transformers
    python -m experiments.embeddings_experiment      (run from the project folder, after train.py)

First run downloads paraphrase-multilingual-MiniLM-L12-v2 (~470 MB) from Hugging Face.
Saves models/embeddings.joblib and models/hybrid.joblib (they then appear in the app's model selector)
and reports/embeddings_results.json.
"""
import json
import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import ROOT, build_dataset
from src.model import EmbeddingFeatures, char_vec, word_vec
from train import frozen_split, spam_metrics


def lr():
    return CalibratedClassifierCV(LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
                                  method="sigmoid", cv=5)


def main():
    df, _ = build_dataset(verbose=False)
    train, test = frozen_split(df)
    base = joblib.load(ROOT / "models" / "original.joblib")
    emb = EmbeddingFeatures()
    print("Encoding messages once (this is the slow part)...")
    models = {
        "Sentence Embeddings + Logistic Regression": Pipeline([("features", emb), ("clf", lr())]),
        "Hybrid TF-IDF + Sentence Embeddings": Pipeline([
            ("features", FeatureUnion([("word", word_vec()), ("char", char_vec()), ("emb", emb)])),
            ("clf", lr())]),
    }
    rows = []

    def evaluate(name, pipe, threshold):
        proba = pipe.predict_proba(test.message)[:, 1]
        m = spam_metrics(test.y, (proba >= threshold).astype(int))
        lat = []
        for s in test.message.sample(100, random_state=0):
            t = time.perf_counter(); pipe.predict_proba([s]); lat.append((time.perf_counter() - t) * 1000)
        m["latency_ms_mean"] = float(np.mean(lat))
        rows.append({"model": name, **m})
        print(f"{name:45s} F1 {m['f1']:.4f}  false alarms {m['fp']}  missed {m['fn']}  {m['latency_ms_mean']:.2f} ms")
        return m

    evaluate("TF-IDF + Linear SVM", base["pipeline"], base["threshold"])
    for (name, pipe), key in zip(models.items(), ["embeddings", "hybrid"]):
        pipe.fit(train.message, train.y)
        m = evaluate(name, pipe, 0.5)
        joblib.dump({"pipeline": pipe, "threshold": 0.5, "name": name,
                     "trained_on": base["trained_on"], "metrics": m}, ROOT / "models" / f"{key}.joblib")
    (ROOT / "reports" / "embeddings_results.json").write_text(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
