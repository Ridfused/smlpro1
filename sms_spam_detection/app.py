"""SMS Spam & Scam Detector — Streamlit app.   Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import os
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.links import defang_text, safe_browsing_check
from src.model import explain
from src.risk import assess

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MAX_BATCH_ROWS, MAX_UPLOAD_MB = 5000, 5
LEVEL_ORDER = {"low": "1 · low", "medium": "2 · medium", "high": "3 · high"}  # keeps bar order
LEVEL_COLOR = {"low": "#0ca30c", "medium": "#c98500", "high": "#d03b3b"}

EXAMPLES = {
    "— choose an example —": "",
    "UPI scam (Hinglish)": "Congratulations! Aapko Rs 5000 cashback mila hai. Paise receive karne ke liye UPI PIN daalein: bit.ly/cb-paytm5000",
    "KYC phishing": "Dear Customer, your SBI account will be blocked today due to pending KYC. Update immediately: http://sbi-kyc-update.xyz/login",
    "Genuine OTP": "483920 is your OTP for login to HDFC Bank NetBanking. Valid for 5 mins. Do not share it with anyone. -HDFC Bank",
    "Prize spam (classic)": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "Delivery scam": "IndiaPost: Your parcel is on hold due to incomplete address. Pay Rs 25 redelivery fee within 24 hrs: indiapost-redeliver.top/pay",
    "Fake job": "Part time job! Earn Rs 3000 daily from home by liking YouTube videos. Join our Telegram task group now.",
    "Hindi KYC scam": "प्रिय ग्राहक, आपका बैंक खाता आज ब्लॉक कर दिया जाएगा। तुरंत केवाईसी अपडेट करें: http://kyc-sbi-update.top",
    "Hinglish OTP fraud": "Sir aapka refund approve ho gaya hai, abhi aaye hue OTP ko batayein taaki paise account me credit ho sake",
    "Friend message": "Hey, are we still meeting at 6 for dinner? I'll bring the notes for tomorrow's exam.",
}

st.set_page_config(page_title="SMS Spam & Scam Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_models():
    out = {}
    for p in sorted(MODELS.glob("*.joblib")):
        try:
            out[p.stem] = joblib.load(p)
        except Exception as e:  # version mismatch etc.
            out[p.stem] = {"error": str(e)}
    return out


def badge(text, color):
    return (f"<span style='background:{color};color:white;padding:3px 10px;border-radius:12px;"
            f"font-weight:600'>{text}</span>")


def predict(bundle, messages):
    return bundle["pipeline"].predict_proba(list(messages))[:, 1]


def safe_cell(v):
    """Prevent CSV/Excel formula injection in downloads."""
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


# ---------------------------------------------------------------- session stats (no text stored)
if "stats" not in st.session_state:
    st.session_state.stats = {"scans": 0, "levels": Counter(), "types": Counter(), "verdicts": Counter()}


def record(result):
    s = st.session_state.stats
    s["scans"] += 1
    s["levels"][result["risk_level"]] += 1
    s["verdicts"][result["model_verdict"]] += 1
    if result["scam_type"]:
        s["types"][result["scam_type"]] += 1


# ---------------------------------------------------------------- sidebar
models = load_models()
usable = {k: v for k, v in models.items() if "pipeline" in v}
with st.sidebar:
    st.header("🛡️ SMS Scam Detector")
    if not usable:
        st.error("No trained model found. Run `python train.py` first.")
        st.stop()
    keys = sorted(usable, key=lambda k: (k != "original", k))
    choice = st.selectbox("Model", keys, format_func=lambda k: usable[k]["name"])
    bundle = usable[choice]
    threshold = bundle.get("threshold", 0.5)
    m = bundle.get("metrics", {})
    st.caption(f"Trained on: {bundle.get('trained_on', '?')}")
    if m:
        st.caption(f"Test F1 {m.get('f1', 0):.3f} · precision {m.get('precision', 0):.3f} · "
                   f"recall {m.get('recall', 0):.3f} · threshold {threshold}")
    for k, v in models.items():
        if "error" in v:
            st.warning(f"Could not load `{k}.joblib` (re-run `python train.py`).")
    st.divider()
    st.markdown("**Privacy**\n- Messages are processed in memory only and never saved.\n"
                "- Links are analysed as text and **never opened**.\n"
                "- Dashboard keeps counts only, no message text.")
    gsb_key = os.environ.get("GOOGLE_SAFE_BROWSING_KEY")
    try:
        gsb_key = gsb_key or st.secrets.get("GOOGLE_SAFE_BROWSING_KEY")
    except Exception:
        pass

st.title("SMS Spam & Scam Detection")
st.caption("TF-IDF word + character n-grams · calibrated Linear SVM · independent warning-sign rules · link safety")

tab_scan, tab_batch, tab_dash, tab_eval = st.tabs(["🔍 Scan", "📄 Batch", "📊 Live dashboard", "🧪 Model evaluation"])

# ---------------------------------------------------------------- Scan
with tab_scan:
    ex = st.selectbox("Try an example", list(EXAMPLES))
    msg = st.text_area("Paste an SMS message", value=EXAMPLES[ex], height=120, max_chars=2000)
    if st.button("Scan message", type="primary") and msg.strip():
        p = float(predict(bundle, [msg])[0])
        r = assess(msg, p, threshold)
        record(r)
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Model verdict**<br>" + badge(r["model_verdict"].upper(),
                    "#d03b3b" if r["model_verdict"] == "spam" else "#0ca30c"), unsafe_allow_html=True)
        c2.metric("Model probability (spam)", f"{p:.1%}")
        c3.markdown(f"**Risk score**<br><span style='font-size:2rem;font-weight:700'>{r['risk_score']}</span>/100 "
                    + badge(r["risk_level"].upper(), LEVEL_COLOR[r["risk_level"]]), unsafe_allow_html=True)
        st.progress(r["risk_score"] / 100)
        if r["scam_type"]:
            st.markdown(f"**Possible scam type:** {r['scam_type']}")

        left, right = st.columns(2)
        with left:
            st.subheader("Warning signs")
            if r["warning_signs"]:
                for s in r["warning_signs"]:
                    sign = "+" if s["points"] > 0 else ""
                    st.markdown(f"- **{s['label']}** ({sign}{s['points']}) — `{defang_text(s['evidence'])}`")
            else:
                st.write("No rule-based warning signs found.")
            st.subheader("Links (never opened)")
            if r["links"]:
                for l in r["links"]:
                    st.code(l["defanged"], language=None)
                    st.markdown("\n".join(f"- {x}" for x in l["reasons"]) or "- no obvious red flags in the text of this link")
                if gsb_key:
                    if st.checkbox("Check these links with Google Safe Browsing (sends only the URLs to Google)"):
                        try:
                            flagged = safe_browsing_check([l["url"] for l in r["links"]], gsb_key)
                            if flagged:
                                st.error(f"Flagged: {flagged}")
                            else:
                                st.success("Not on Google's blocklists.")
                        except Exception as e:
                            st.warning(f"Safe Browsing check failed: {e}")
            else:
                st.write("No links found.")
        with right:
            st.subheader("Why the model decided this")
            try:
                spam_w, ham_w = explain(bundle["pipeline"], msg)
                spam_w = [(defang_text(w), v) for w, v in spam_w]
                ham_w = [(defang_text(w), v) for w, v in ham_w]
                if spam_w:
                    st.markdown("Words pushing towards **spam**: " + ", ".join(f"`{w}`" for w, _ in spam_w))
                if ham_w:
                    st.markdown("Words pushing towards **legitimate**: " + ", ".join(f"`{w}`" for w, _ in ham_w))
                chart = pd.DataFrame(spam_w[:6] + ham_w[:6], columns=["word", "influence"]).set_index("word")
                if not chart.empty:
                    st.bar_chart(chart, horizontal=True, color="#2a78d6")
            except Exception:
                st.write("Explanation not available for this model.")
            st.subheader("Recommended actions")
            for a in r["actions"] or ["Looks normal. Still, never share OTPs or PINs."]:
                st.markdown(f"- {a}")
        st.caption("Risk score = 60 × model probability + warning-sign points (limited to −20…+40). "
                   "Low < 30 ≤ medium < 60 ≤ high. It is a triage aid, not a probability.")

# ---------------------------------------------------------------- Batch
with tab_batch:
    st.write("Upload a **CSV** with a `message` column, or a **TXT** file with one message per line "
             f"(max {MAX_BATCH_ROWS:,} messages, {MAX_UPLOAD_MB} MB).")
    up = st.file_uploader("Upload file", type=["csv", "txt"])
    if up is not None:
        raw = up.getvalue()
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error("File too large."); st.stop()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        if up.name.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.StringIO(text), dtype=str)
            except Exception as e:
                st.error(f"Could not read CSV: {e}"); st.stop()
            col = next((c for c in df.columns if c.strip().lower() in ("message", "text", "sms", "v2")), None)
            if col is None:
                st.error(f"No `message` column found. Columns: {list(df.columns)}"); st.stop()
            msgs = df[col].fillna("").astype(str)
        else:
            msgs = pd.Series([l for l in text.splitlines() if l.strip()])
        msgs = msgs[msgs.str.strip() != ""].str.slice(0, 2000).reset_index(drop=True)
        if len(msgs) > MAX_BATCH_ROWS:
            st.warning(f"Only the first {MAX_BATCH_ROWS:,} messages are checked.")
            msgs = msgs.iloc[:MAX_BATCH_ROWS]
        probs = predict(bundle, msgs)
        rows = []
        for mtext, p in zip(msgs, probs):
            r = assess(mtext, float(p), threshold)
            rows.append({"message": mtext, "verdict": r["model_verdict"], "probability": round(float(p), 4),
                         "risk_score": r["risk_score"], "risk_level": r["risk_level"],
                         "scam_type": r["scam_type"] or "",
                         "warning_signs": "; ".join(s["label"] for s in r["warning_signs"]),
                         "links_defanged": " ".join(l["defanged"] for l in r["links"])})
        res = pd.DataFrame(rows)
        a, b, c = st.columns(3)
        a.metric("Messages", len(res)); b.metric("Model: spam", int((res.verdict == "spam").sum()))
        c.metric("High risk", int((res.risk_level == "high").sum()))
        c1, c2 = st.columns(2)
        c1.caption("Risk level")
        c1.bar_chart(res.risk_level.value_counts().reindex(["low", "medium", "high"]).fillna(0)
                     .rename(index=LEVEL_ORDER), color="#2a78d6")
        c2.caption("Possible scam type")
        st_counts = res[res.scam_type != ""].scam_type.value_counts()
        if len(st_counts):
            c2.bar_chart(st_counts, horizontal=True, color="#eb6834")
        else:
            c2.write("None detected.")
        st.dataframe(res.drop(columns=["message"]).assign(message=res.message.str.slice(0, 80)),
                     width="stretch", hide_index=True)
        safe = res.map(safe_cell) if hasattr(res, "map") else res.applymap(safe_cell)
        st.download_button("Download results (CSV)", safe.to_csv(index=False).encode("utf-8"),
                           "sms_scan_results.csv", "text/csv")

# ---------------------------------------------------------------- Live dashboard
with tab_dash:
    s = st.session_state.stats
    st.caption("Counts from scans in this browser session only. No message text is stored.")
    a, b, c = st.columns(3)
    a.metric("Messages scanned", s["scans"])
    b.metric("Flagged as spam", s["verdicts"].get("spam", 0))
    c.metric("High risk", s["levels"].get("high", 0))
    if s["scans"]:
        c1, c2 = st.columns(2)
        c1.caption("Risk level")
        c1.bar_chart(pd.Series({LEVEL_ORDER[k]: s["levels"].get(k, 0) for k in ["low", "medium", "high"]}),
                     color="#2a78d6")
        if s["types"]:
            c2.caption("Scam types")
            c2.bar_chart(pd.Series(dict(s["types"])), horizontal=True, color="#eb6834")
    else:
        st.info("Scan some messages to see statistics here.")

# ---------------------------------------------------------------- Offline evaluation
with tab_eval:
    st.caption("Offline evaluation on the frozen, leakage-controlled test set (from `python train.py`).")
    mpath = REPORTS / "metrics.json"
    if mpath.exists():
        rep = json.loads(mpath.read_text())
        f = rep["final"]
        cols = st.columns(5)
        for col, (k, lbl) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Spam precision"),
                                        ("recall", "Spam recall"), ("f1", "Spam F1"),
                                        ("latency_ms_median", "Latency (ms)")]):
            col.metric(lbl, f"{f[k]:.4f}" if k != "latency_ms_median" else f"{f[k]:.2f}")
        st.write(f"Test messages: **{f['test_messages']:,}** (trained on {f['train_messages']:,}). "
                 f"Confusion matrix TN/FP/FN/TP = **{f['tn']} / {f['fp']} / {f['fn']} / {f['tp']}**")
        if "model_comparison" in rep:
            st.subheader("Model comparison")
            st.dataframe(pd.DataFrame(rep["model_comparison"]).round(4), hide_index=True, width="stretch")
        if "per_language" in rep:
            st.subheader("By language")
            st.dataframe(pd.DataFrame(rep["per_language"]).round(4), hide_index=True, width="stretch")
        figs = sorted((REPORTS / "figures").glob("*.png"))
        for i in range(0, len(figs), 2):
            cs = st.columns(2)
            for c, fpath in zip(cs, figs[i:i + 2]):
                c.image(str(fpath), width="stretch")
    else:
        st.info("Run `python train.py` to generate the evaluation report.")
