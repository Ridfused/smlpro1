"""TF-IDF (word + character n-grams) + calibrated Linear SVM, and explanations."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def word_vec(ngram=(1, 2)):
    return TfidfVectorizer(analyzer="word", ngram_range=ngram, lowercase=True,
                           sublinear_tf=True, min_df=1,
                           token_pattern=r"(?u)\b\w+\b|[£$€₹]")


def char_vec(ngram=(1, 6)):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, lowercase=True,
                           sublinear_tf=True, min_df=2)


def features(word_ngram=(1, 2), char_ngram=(1, 6), use_char=True):
    parts = [("word", word_vec(word_ngram))]
    if use_char:
        parts.append(("char", char_vec(char_ngram)))
    return FeatureUnion(parts)


def candidate_models():
    """The five models compared in Section 5.4 (uncalibrated, for CV comparison)."""
    svm = lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)
    lr = lambda: LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000)
    return {
        "Linear SVM (word+char)": Pipeline([("features", features()), ("clf", svm())]),
        "Logistic Regression (word+char)": Pipeline([("features", features()), ("clf", lr())]),
        "Linear SVM": Pipeline([("features", features(use_char=False)), ("clf", svm())]),
        "Logistic Regression": Pipeline([("features", features(use_char=False)), ("clf", lr())]),
        "Multinomial Naive Bayes": Pipeline([("features", features(word_ngram=(1, 1), use_char=False)),
                                             ("clf", MultinomialNB(alpha=0.1))]),
    }


def final_pipeline(C=1.0, word_ngram=(1, 2), char_ngram=(1, 6)):
    """Linear SVM with sigmoid (Platt) calibration fitted on internal out-of-fold predictions."""
    svm = LinearSVC(C=C, class_weight="balanced", max_iter=20000)
    clf = CalibratedClassifierCV(svm, method="sigmoid", cv=5)
    return Pipeline([("features", features(word_ngram, char_ngram)), ("clf", clf)])


def _linear_coef(pipe):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "calibrated_classifiers_"):
        return np.mean([c.estimator.coef_.ravel() for c in clf.calibrated_classifiers_], axis=0)
    return clf.coef_.ravel()


def explain(pipe, message: str, top_k: int = 8):
    """Top features pushing towards spam (+) and ham (-) for one message.
    Character n-grams are mapped back to the words that contain them."""
    feats = pipe.named_steps["features"]
    X = feats.transform([message]).tocsr()
    coef = _linear_coef(pipe)
    names = feats.get_feature_names_out()
    idx = X.indices
    contrib = X.data * coef[idx]
    # aggregate char n-gram evidence per word of the message
    words = {}
    import re
    tokens = [t for t in (re.sub(r"^\W+|\W+$", "", w).lower() for w in message.split()) if t]
    for j, c in zip(idx, contrib):
        name = names[j]
        block, term = name.split("__", 1)
        if block == "word":
            key = term
        else:
            t = term.strip().lower()
            if len(t) < 3:
                continue
            owner = next((w for w in tokens if t in w), None)
            if owner is None:
                continue
            key = owner
        words[key] = words.get(key, 0.0) + float(c)
    items = sorted(words.items(), key=lambda kv: kv[1])
    spam = [(k, v) for k, v in reversed(items) if v > 0][:top_k]
    ham = [(k, v) for k, v in items if v < 0][:top_k]
    return spam, ham


# ----------------------------------------------------------------------------- sentence embeddings
from sklearn.base import BaseEstimator, TransformerMixin

_ENCODERS = {}


def _encoder(name):
    if name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer  # optional dependency
        _ENCODERS[name] = SentenceTransformer(name)
    return _ENCODERS[name]


class EmbeddingFeatures(BaseEstimator, TransformerMixin):
    """Maps each message to a 384-d multilingual sentence embedding (model loaded lazily, not pickled)."""

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", batch_size=64):
        self.model_name = model_name
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _encoder(self.model_name).encode(list(X), batch_size=self.batch_size,
                                                normalize_embeddings=True, show_progress_bar=False)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"emb{i}" for i in range(384)], dtype=object)
