"""Combine model probability + warning signs into a 0-100 triage risk score."""
from __future__ import annotations

from .links import extract_links
from .rules import warning_signs

ACTIONS = {
    "KYC / bank phishing": ["Do not click the link or call the number in the SMS.",
                            "Open your bank's official app or website yourself to check.",
                            "Banks never ask for KYC updates through SMS links."],
    "OTP / PIN theft": ["Never share an OTP, PIN, CVV or password with anyone.",
                        "No bank, delivery or support agent needs your OTP."],
    "UPI fraud": ["You never need to enter a UPI PIN to RECEIVE money.",
                  "Decline unknown collect requests in your UPI app."],
    "Prize / lottery scam": ["You cannot win a contest you did not enter.",
                             "Do not pay any 'processing' or 'tax' fee to claim a prize."],
    "Delivery scam": ["Track parcels only on the courier's official website/app.",
                      "Do not pay small 're-delivery' fees through SMS links."],
    "Fake job scam": ["Genuine employers do not ask you to pay to get a job.",
                      "Avoid 'like videos and earn' tasks on Telegram/WhatsApp."],
    "Investment scam": ["Guaranteed high returns are a red flag.",
                        "Check SEBI registration before investing."],
    "Premium-rate spam": ["Do not reply or call back; it may be charged at premium rates."],
    "Generic spam / promotion": ["Ignore or block the sender if you did not opt in."],
}
REPORT = "In India you can report fraud at cybercrime.gov.in or call 1930; forward spam to 1909 / use Sanchar Saathi (Chakshu)."


def assess(message: str, prob_spam: float, threshold: float = 0.5) -> dict:
    signs = warning_signs(message)
    links = extract_links(message)
    sign_pts = sum(s["points"] for s in signs)
    link_pts = max([l["points"] for l in links], default=0)
    extra = max(-20, min(40, sign_pts + link_pts))
    score = int(round(max(0, min(100, 60 * prob_spam + extra))))
    level = "high" if score >= 60 else "medium" if score >= 30 else "low"
    typed = [s for s in signs if s["scam_type"]]
    typed.sort(key=lambda s: -s["points"])
    is_spam = prob_spam >= threshold
    if typed:
        scam_type = typed[0]["scam_type"]
    elif level != "low":
        scam_type = "Generic spam / promotion"
    else:
        scam_type = None
    actions = ACTIONS.get(scam_type, []) if scam_type else []
    if level == "high":
        actions = actions + [REPORT]
    return {"model_verdict": "spam" if is_spam else "legitimate", "prob_spam": prob_spam,
            "risk_score": score, "risk_level": level, "scam_type": scam_type,
            "warning_signs": signs, "links": links, "actions": actions,
            "rule_points": extra}
