"""Train the compact model used by the website and export it to web/model.json.

    python -m tools.export_web_model

Same pipeline as the main model (TF-IDF word 1-2 + char_wb 2-5, Linear SVM C=0.3, balanced, sigmoid
calibration) but limited to the 20,000 most frequent word n-grams and 40,000 character n-grams and a
single calibrated SVM, so the whole model fits in a ~1.5 MB file the browser can run.
"""
import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.data import ROOT, build_dataset
from src.model import char_vec, word_vec
from train import frozen_split, spam_metrics

OUT = ROOT / "web" / "model.json"


def main():
    df, _ = build_dataset(verbose=False)
    train, test = frozen_split(df)
    w = word_vec((1, 2)); w.max_features = 20000
    c = char_vec((2, 5)); c.max_features = 40000
    pipe = Pipeline([("features", FeatureUnion([("word", w), ("char", c)])),
                     ("clf", CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", max_iter=20000),
                                                    method="sigmoid", cv=5, ensemble=False))])
    pipe.fit(train.message, train.y)
    proba = pipe.predict_proba(test.message)[:, 1]
    m = spam_metrics(test.y, (proba >= 0.5).astype(int))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    cc = pipe.named_steps["clf"].calibrated_classifiers_[0]
    svm, cal = cc.estimator, cc.calibrators[0]
    coef = svm.coef_.ravel()
    blocks, start = {}, 0
    for name, vec in pipe.named_steps["features"].transformer_list:
        vocab = vec.vocabulary_
        terms = sorted(vocab, key=vocab.get)
        n = len(terms)
        blocks[name] = {"terms": terms,
                        "idf": [round(float(x), 4) for x in vec.idf_],
                        "coef": [round(float(x), 5) for x in coef[start:start + n]]}
        start += n
    model = {"version": 1, "intercept": float(svm.intercept_[0]),
             "calibration": {"a": float(cal.a_), "b": float(cal.b_)}, "threshold": 0.5,
             "word": blocks["word"], "char": blocks["char"],
             "char_ngram": [2, 5], "word_ngram": [1, 2],
             "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
             "test_messages": len(test)}
    OUT.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # parity samples for testing the JavaScript port
    sample = test.sample(400, random_state=1)
    (ROOT / "web" / "parity_samples.json").write_text(json.dumps(
        [{"m": t, "p": float(p)} for t, p in zip(sample.message, pipe.predict_proba(sample.message)[:, 1])],
        ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
