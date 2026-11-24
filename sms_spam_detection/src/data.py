"""Data loading, cleaning, duplicate control and leakage-safe grouping."""
from __future__ import annotations

import csv
import hashlib
import html
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UCI_FILE = RAW / "sms.tsv"
# Optional: French/German/Hindi machine translations of UCI
# (Kaggle "rajnathpatel/multilingual-spam-data" or the Hugging Face copy).
# Expected columns: labels, text, text_hi, text_de, text_fr
TRANSLATIONS_FILE = RAW / "data-augmented.csv"
LANG_COLS = {"text": "en", "text_fr": "fr", "text_de": "de", "text_hi": "hi"}


def clean_text(s: str) -> str:
    """Unicode NFKC, HTML entities decoded, whitespace collapsed. Digits/symbols kept."""
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def msg_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_uci() -> pd.DataFrame:
    df = pd.read_csv(UCI_FILE, sep="\t", header=None, names=["label", "message"],
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    df["message"] = df["message"].map(clean_text)
    df["language"] = "en"
    df["source"] = "uci_sms_spam"
    return df


def load_translations() -> pd.DataFrame | None:
    if not TRANSLATIONS_FILE.exists():
        return None
    t = pd.read_csv(TRANSLATIONS_FILE)
    label_col = "labels" if "labels" in t.columns else "label"
    rows = []
    for i, r in t.iterrows():
        for col, lang in LANG_COLS.items():
            if col in t.columns and isinstance(r[col], str) and r[col].strip():
                rows.append({"label": r[label_col], "message": clean_text(r[col]),
                             "language": lang, "source": "translations_uci_fr_de_hi",
                             "row_link": i})
    return pd.DataFrame(rows)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _norm_for_near_dup(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d", "0", s)
    return re.sub(r"[^\w0 ]+", "", s)


def near_duplicate_pairs(texts: list[str], threshold: float = 0.8, num_perm: int = 128):
    """MinHash LSH over character 5-gram shingles. Returns list of (i, j) pairs."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mhs = []
    for i, t in enumerate(texts):
        t = _norm_for_near_dup(t)
        m = MinHash(num_perm=num_perm)
        shingles = {t[k:k + 5] for k in range(max(1, len(t) - 4))}
        for sh in shingles:
            m.update(sh.encode("utf-8"))
        mhs.append(m)
        lsh.insert(str(i), m)
    pairs = []
    for i, m in enumerate(mhs):
        for j in lsh.query(m):
            j = int(j)
            if j > i:
                pairs.append((i, j))
    return pairs


PHONE_RE = re.compile(r"\d{8,}")


def build_dataset(use_translations: bool = True, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Returns cleaned, de-duplicated data with a `group` column and a stats dict."""
    stats = {}
    uci = load_uci()
    stats["uci_raw"] = len(uci)
    parts = [uci]
    trans = load_translations() if use_translations else None
    if trans is not None:
        # give each UCI message the row id of its translations, so they share a group
        en_link = dict(zip(trans.loc[trans.language == "en", "message"],
                           trans.loc[trans.language == "en", "row_link"]))
        uci["row_link"] = uci["message"].map(en_link)
        parts.append(trans)
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    stats["raw_records"] = len(df)

    df = df.drop_duplicates(subset=["message", "label"]).reset_index(drop=True)
    stats["after_exact_dedup"] = len(df)

    # Same text with conflicting labels -> quarantine
    conflict = df.groupby("message")["label"].transform("nunique") > 1
    stats["quarantined_conflicting_exact"] = int(conflict.sum())
    df = df[~conflict].drop_duplicates(subset=["message"]).reset_index(drop=True)
    stats["unique_messages"] = len(df)

    # --- Grouping so related messages never cross train/test ---
    uf = UnionFind(len(df))
    # 1) translations of the same original row (and the English original itself)
    # (English copies in the translation file were merged with UCI by exact de-dup,
    #  so the UCI original joins the group through rule 2/3 or the shared row link.)
    if "row_link" in df.columns:
        link_first = {}
        for i, rl in enumerate(df["row_link"]):
            if pd.notna(rl):
                rl = int(rl)
                if rl in link_first:
                    uf.union(link_first[rl], i)
                else:
                    link_first[rl] = i
    # 2) shared long phone numbers / codes (links translations and campaign copies)
    first_seen = {}
    for i, msg in enumerate(df["message"]):
        for num in set(PHONE_RE.findall(msg)):
            if num in first_seen:
                uf.union(first_seen[num], i)
            else:
                first_seen[num] = i
    # 3) near duplicates
    pairs = near_duplicate_pairs(df["message"].tolist())
    near_rows = set()
    for i, j in pairs:
        uf.union(i, j)
        near_rows.update((i, j))
    df["group"] = [uf.find(i) for i in range(len(df))]
    sizes = df["group"].value_counts()
    stats["near_duplicate_pairs"] = len(pairs)
    stats["rows_in_near_duplicate_pairs"] = len(near_rows)
    stats["groups"] = int(sizes.size)
    stats["largest_group"] = int(sizes.max())

    # Groups whose members disagree on label -> quarantine (ambiguous)
    g_conflict = df.groupby("group")["label"].transform("nunique") > 1
    near_conf = g_conflict & df.index.isin(list(near_rows))
    # only quarantine rows whose disagreement comes from near-duplicate text
    stats["quarantined_near_dup_conflicts"] = int(near_conf.sum())
    df = df[~near_conf].reset_index(drop=True)

    df["y"] = (df["label"] == "spam").astype(int)
    df["hash"] = df["message"].map(msg_hash)
    stats["usable"] = len(df)
    stats["ham"] = int((df.y == 0).sum())
    stats["spam"] = int((df.y == 1).sum())
    if verbose:
        for k, v in stats.items():
            print(f"  {k:32s} {v}")
    return df, stats
