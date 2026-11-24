"""Train and evaluate the SMS spam & scam detector.

    python train.py            # full run (EDA, model comparison, 24-setting grid search, final model)
    python train.py --fast     # skip comparison + grid search, reuse the grid-search winner (same final model)

Outputs: models/*.joblib, reports/metrics.json, reports/figures/*.png, data/frozen_test_hashes.txt
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict, cross_val_score

from src.data import ROOT, build_dataset
from src.model import candidate_models, final_pipeline, features
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FROZEN = ROOT / "data" / "frozen_test_hashes.txt"
SEED = 42
HAM_C, SPAM_C, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})

STOP = set("""a an the and or but if to of in on at for with is are was were be been am i you u me my
your it its this that we he she they them his her our ur so do dont don't im i'm 2 not no can will just
have has had get got go going what when how all up out now from by as there call""".split())


def spam_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "fpr": fp / max(1, fp + tn), "fnr": fn / max(1, fn + tp)}


# ----------------------------------------------------------------------------- EDA
def eda(df: pd.DataFrame) -> dict:
    u = df[df.source == "uci_sms_spam"].copy()
    u["length"] = u.message.str.len()
    stats = {
        "ham": int((u.y == 0).sum()), "spam": int((u.y == 1).sum()),
        "ham_pct": float((u.y == 0).mean() * 100),
        "median_len_spam": float(u[u.y == 1].length.median()),
        "median_len_ham": float(u[u.y == 0].length.median()),
        "pct_spam_5digit": float(u[u.y == 1].message.str.contains(r"\d{5,}").mean() * 100),
        "pct_ham_5digit": float(u[u.y == 0].message.str.contains(r"\d{5,}").mean() * 100),
        "pct_spam_currency": float(u[u.y == 1].message.str.contains(r"[£$]").mean() * 100),
        "pct_ham_currency": float(u[u.y == 0].message.str.contains(r"[£$]").mean() * 100),
    }

    # Fig 1 class balance
    fig, ax = plt.subplots(figsize=(5, 3.4))
    counts = [stats["ham"], stats["spam"]]
    bars = ax.bar(["Ham (legitimate)", "Spam"], counts, color=[HAM_C, SPAM_C], width=0.55)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 60, f"{c:,} ({c / sum(counts):.1%})",
                ha="center", color=INK)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_ylabel("Messages"); ax.set_title("Figure 1. Class balance after de-duplication", loc="left")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "fig1_class_balance.png"); plt.close(fig)

    # Fig 2 length distribution
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.arange(0, 401, 10)
    ax.hist(u[u.y == 0].length.clip(upper=400), bins=bins, color=HAM_C, alpha=0.75, label="Ham",
            density=True, edgecolor="#fcfcfb", linewidth=1)
    ax.hist(u[u.y == 1].length.clip(upper=400), bins=bins, color=SPAM_C, alpha=0.75, label="Spam",
            density=True, edgecolor="#fcfcfb", linewidth=1)
    ax.axvline(160, color=MUTED, ls="--", lw=1); ax.text(163, ax.get_ylim()[1] * 0.9, "160-char SMS limit",
                                                       color=MUTED, fontsize=8)
    ax.set_xlabel("Message length (characters, capped at 400)"); ax.set_ylabel("Share of messages")
    ax.set_title("Figure 2. Message length by label", loc="left"); ax.legend(frameon=False)
    fig.savefig(FIG / "fig2_length_by_label.png"); plt.close(fig)

    # Fig 3 top words
    def top(sub):
        c = Counter()
        for m in sub.message.str.lower():
            c.update(w for w in re.findall(r"[a-z£$]+", m) if w not in STOP and len(w) > 1)
        return c.most_common(15)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6))
    for ax, sub, col, name in [(axes[0], u[u.y == 0], HAM_C, "Ham"), (axes[1], u[u.y == 1], SPAM_C, "Spam")]:
        w, n = zip(*top(sub)[::-1])
        ax.barh(w, n, color=col, height=0.7); ax.set_title(name, loc="left"); ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 3. Most frequent words per class (stop words removed for this chart only)",
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_top_words.png"); plt.close(fig)
    return stats


# ----------------------------------------------------------------------------- split
def frozen_split(df: pd.DataFrame):
    if FROZEN.exists():
        hashes = set(FROZEN.read_text().split())
        is_test = df.hash.isin(hashes)
        # keep whole groups together even if data changed
        test_groups = set(df.loc[is_test, "group"])
        is_test = df.group.isin(test_groups)
        print(f"  using frozen test set ({is_test.sum()} messages)")
    else:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        _, test_idx = next(sgkf.split(df.message, df.y, df.group))
        is_test = df.index.isin(test_idx)
        FROZEN.write_text("\n".join(df.loc[is_test, "hash"]))
        print(f"  created and froze a new test set ({is_test.sum()} messages)")
    train, test = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    crossing = set(train.group) & set(test.group)
    assert not crossing, "group leakage between train and test"
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip model comparison and grid search")
    ap.add_argument("--no-translations", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)
    report = {}

    print("1) Loading and cleaning data")
    df, stats = build_dataset(use_translations=not args.no_translations)
    report["data"] = stats
    report["languages"] = df.language.value_counts().to_dict()

    print("2) Exploratory data analysis")
    report["eda"] = eda(df)
    print("  ", {k: round(v, 1) for k, v in report["eda"].items()})

    print("3) Grouped train/test split")
    train, test = frozen_split(df)
    report["split"] = {"train": len(train), "test": len(test), "test_spam": int(test.y.sum()),
                       "groups_crossing_boundary": 0}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xtr, ytr, gtr = train.message, train.y, train.group

    # result of the full 24-setting grid search on the 4-language data (used as-is by --fast)
    best = {"C": 0.3, "word": (1, 2), "char": (2, 5)}
    if not args.fast:
        print("4) Model comparison (grouped 5-fold CV)")
        rows = []
        for name, pipe in candidate_models().items():
            t = time.time()
            cvf1 = cross_val_score(pipe, Xtr, ytr, groups=gtr, cv=cv, scoring="f1", n_jobs=-1).mean()
            m = clone(pipe).fit(Xtr, ytr)
            r = spam_metrics(test.y, m.predict(test.message))
            rows.append({"model": name, "cv_f1": cvf1, "test_precision": r["precision"],
                         "test_recall": r["recall"], "test_f1": r["f1"]})
            print(f"   {name:34s} CV F1 {cvf1:.4f} | test P {r['precision']:.4f} R {r['recall']:.4f} "
                  f"F1 {r['f1']:.4f}  ({time.time() - t:.0f}s)")
        report["model_comparison"] = rows

        print("5) Grid search, 24 settings (grouped 5-fold CV)")
        base = Pipeline([("features", features()),
                         ("clf", LinearSVC(class_weight="balanced", max_iter=20000))])
        grid = {"clf__C": [0.3, 1.0, 3.0],
                "features__char__ngram_range": [(1, 5), (1, 6), (2, 5), (2, 6)],
                "features__word__ngram_range": [(1, 1), (1, 2)]}
        gs = GridSearchCV(base, grid, scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr, groups=gtr)
        bp = gs.best_params_
        best = {"C": bp["clf__C"], "word": bp["features__word__ngram_range"],
                "char": bp["features__char__ngram_range"]}
        default_cv = next(r["cv_f1"] for r in rows if r["model"] == "Linear SVM (word+char)")
        report["grid_search"] = {"settings": 24, "best_params": {k: str(v) for k, v in best.items()},
                                 "best_cv_f1": gs.best_score_, "default_cv_f1": default_cv}
        print(f"   best {best}  CV F1 {gs.best_score_:.4f} (defaults {default_cv:.4f})")

    print("6) Final model: TF-IDF word+char + calibrated Linear SVM")
    pipe = final_pipeline(best["C"], best["word"], best["char"])
    # decision threshold from out-of-fold probabilities on TRAINING data only
    oof = cross_val_predict(pipe, Xtr, ytr, groups=gtr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    ths = np.arange(0.2, 0.81, 0.01)
    f1s = [f1_score(ytr, (oof >= t).astype(int)) for t in ths]
    threshold = float(round(ths[int(np.argmax(f1s))], 2))
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(test.message)[:, 1]
    final = spam_metrics(test.y, (proba >= threshold).astype(int))
    lat = []
    for m in test.message.sample(min(300, len(test)), random_state=SEED):
        t = time.perf_counter(); pipe.predict_proba([m]); lat.append((time.perf_counter() - t) * 1000)
    final.update({"threshold": threshold, "latency_ms_median": float(np.median(lat)),
                  "latency_ms_mean": float(np.mean(lat)), "test_messages": len(test),
                  "train_messages": len(train)})
    report["final"] = final
    print(f"   threshold {threshold} | acc {final['accuracy']:.4f} P {final['precision']:.4f} "
          f"R {final['recall']:.4f} F1 {final['f1']:.4f} | TN/FP/FN/TP {final['tn']}/{final['fp']}/"
          f"{final['fn']}/{final['tp']} | {final['latency_ms_median']:.2f} ms")

    # Fig 4 confusion matrix
    cm = np.array([[final["tn"], final["fp"]], [final["fn"], final["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() * 0.5 else INK)
    ax.set_xticks([0, 1], ["Ham", "Spam"]); ax.set_yticks([0, 1], ["Ham", "Spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Figure 4. Confusion matrix ({len(test):,} test messages)", loc="left", fontsize=10)
    fig.savefig(FIG / "fig4_confusion_matrix.png"); plt.close(fig)

    # per-language results (+ English-only training comparison when translations exist)
    langs = sorted(test.language.unique())
    if len(langs) > 1:
        print("7) Per-language results")
        en_pipe = final_pipeline(best["C"], best["word"], best["char"]).fit(
            Xtr[train.language == "en"], ytr[train.language == "en"])
        rows = []
        for lg in langs:
            s = test.language == lg
            rows.append({"language": lg, "test": int(s.sum()), "spam": int(test.y[s].sum()),
                         "f1_english_only": f1_score(test.y[s], en_pipe.predict(test.message[s])),
                         "f1_all_languages": f1_score(test.y[s], (proba[s.values] >= threshold).astype(int))})
            print(f"   {lg}: {rows[-1]}")
        report["per_language"] = rows
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = np.arange(len(rows)); w = 0.36
        ax.bar(x - w / 2 - 0.01, [r["f1_english_only"] for r in rows], w, color=HAM_C, label="English-only training")
        ax.bar(x + w / 2 + 0.01, [r["f1_all_languages"] for r in rows], w, color=SPAM_C, label="All languages")
        ax.set_xticks(x, [r["language"].upper() for r in rows]); ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Spam F1")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax.set_title("Figure 5. Spam F1 by language (test set)", loc="left"); ax.grid(axis="x", visible=False)
        fig.savefig(FIG / "fig5_f1_by_language.png"); plt.close(fig)

    joblib.dump({"pipeline": pipe, "threshold": threshold,
                 "name": "TF-IDF + Linear SVM (original, default)",
                 "trained_on": f"UCI SMS Spam Collection{' + FR/DE/HI translations' if len(langs) > 1 else ''}",
                 "metrics": final}, MODELS / "original.joblib")

    # alternative model for the app's model selector
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    lr = Pipeline([("features", features(best["word"], best["char"])),
                   ("clf", CalibratedClassifierCV(LogisticRegression(C=10, class_weight="balanced",
                                                                     max_iter=5000), method="sigmoid", cv=5))])
    lr.fit(Xtr, ytr)
    lr_m = spam_metrics(test.y, lr.predict(test.message))
    joblib.dump({"pipeline": lr, "threshold": 0.5, "name": "TF-IDF + Logistic Regression",
                 "trained_on": "same data as original", "metrics": lr_m}, MODELS / "logreg.joblib")
    report["logreg_final"] = lr_m

    prev = ROOT / "reports" / "metrics.json"
    if args.fast and prev.exists():
        old = json.loads(prev.read_text())
        for k in ("model_comparison", "grid_search"):
            if k in old and k not in report:
                report[k] = old[k]
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print("Saved models/ and reports/metrics.json")


if __name__ == "__main__":
    main()
