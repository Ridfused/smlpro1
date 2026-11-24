"""Hand-written warning signs, independent of the ML model (English, Hinglish, Hindi)."""
from __future__ import annotations

import re
from dataclasses import dataclass

F = re.IGNORECASE | re.UNICODE


@dataclass
class Rule:
    key: str
    label: str
    points: int
    patterns: list
    scam_type: str | None = None


RULES = [
    Rule("kyc", "KYC / account-block threat", 20, [
        r"\bkyc\b.{0,40}(update|expire|pending|suspend|block|verify|complete)",
        r"(update|complete|verify).{0,25}\b(kyc|pan|aadhaa?r)\b",
        r"(account|a/c|card|sim|wallet).{0,30}(will be |is |has been )?(blocked|suspended|deactivated|closed|frozen)",
        r"(account|khata).{0,20}band (ho|kar)",
        r"(खाता|अकाउंट).{0,30}(बंद|ब्लॉक)", r"केवाईसी",
    ], "KYC / bank phishing"),
    Rule("otp", "Asks you to share OTP / PIN / password", 25, [
        r"(share|send|tell|forward|give|provide|reply with|bata\w*|bhej\w*)\s+(\w+\s+){0,3}(otp|pin|cvv|password|passcode)",
        r"(otp|pin|cvv)\s+(\w+\s+){0,3}(share|send|bata\w*|bhej\w*|batayein|bataye)\b",
        r"(ओटीपी|पिन).{0,20}(बताएं|बताइए|भेजें|शेयर)",
    ], "OTP / PIN theft"),
    Rule("upi", "UPI 'enter PIN to receive money'", 25, [
        r"(enter|put|daal\w*|dale)\s+(your\s+)?(upi\s+)?pin.{0,30}(receive|get|credit|claim|paise|paisa|refund)",
        r"(receive|get|claim).{0,30}(enter|daal\w*).{0,10}\bpin\b",
        r"(receive|paise|paisa|refund|cashback|credit).{0,40}\bpin\s+(daal\w*|dal\w*|enter|daale\w*)",
        r"collect request.{0,40}(approve|accept)",
    ], "UPI fraud"),
    Rule("prize", "Prize / lottery / reward claim", 15, [
        r"\b(won|winner|win|congratulations|congrats|selected|claim)\b.{0,60}(prize|reward|cash|lottery|£|\$|rs\.?\s?\d|₹|voucher|gift|bonus)",
        r"(cashback|reward|prize|inaam)\s+(\w+\s+){0,2}(mila|jeeta|won|credited|unlocked)",
        r"lucky (draw|winner|customer)", r"\blottery\b", r"\bjackpot\b",
        r"(इनाम|लॉटरी|जीत\w*)", r"(aap|apne).{0,20}(jeet|jeete|jita)",
    ], "Prize / lottery scam"),
    Rule("delivery", "Parcel / delivery fee or address problem", 15, [
        r"(parcel|package|courier|delivery|shipment|consignment).{0,50}(fee|pay|charge|held|on hold|failed|unsuccessful|address|customs|redeliver)",
        r"(india ?post|fedex|dhl|bluedart|usps).{0,50}(pay|fee|update|confirm)",
    ], "Delivery scam"),
    Rule("job", "Fake job / easy money offer", 15, [
        r"(work from home|part[- ]?time (job|work)|online job|daily (income|earning|payment))",
        r"earn\s+(up to\s+)?(rs\.?|₹|inr)?\s?\d[\d,]*.{0,30}(per|a|daily|day|week|hour|month)",
        r"(like|subscribe).{0,20}(youtube|videos).{0,30}(earn|pay|₹|rs)", r"telegram.{0,20}task",
        r"ghar (baithe|se).{0,30}(kama|earn|income)",
    ], "Fake job scam"),
    Rule("invest", "Investment / guaranteed-returns offer", 15, [
        r"(guaranteed|assured|fixed|double)\s+(\w+\s+)?(returns?|profit|income|money)",
        r"(crypto|bitcoin|forex|stock|trading)\s+(tips|signals|investment|profit|group)",
        r"(invest|investment).{0,40}(\d+\s?%|daily|guaranteed|returns)",
    ], "Investment scam"),
    Rule("premium", "Premium-rate number / text-to-claim", 10, [
        r"\b09\d{8,9}\b", r"\b(text|txt|reply|send)\s+[A-Z]{2,10}\s+to\s+\d{4,6}\b",
        r"\b\d{1,3}p\s?/\s?(min|msg)\b|£\d+(\.\d+)?\s?/\s?(msg|wk|week)",
    ], "Premium-rate spam"),
    Rule("urgency", "Pressure / urgency language", 5, [
        r"\b(urgent|immediately|act now|within 24 ?(hrs?|hours)|today only|expires? (today|soon|in)|last chance|final (notice|reminder))\b",
        r"(turant|jaldi|abhi)\s", r"(तुरंत|जल्दी)",
    ]),
]

PROTECTIVE = Rule("protective", "Protective wording (e.g. 'do not share your OTP')", -15, [
    r"(do not|don't|dont|never|not to)\s+(\w+\s+){0,3}(share|disclose|tell)",
    r"(kisi ke saath|kisi se).{0,20}(share na|mat|na kare)", r"(साझा|शेयर) न करें",
    r"(bank|we) (will )?never ask",
])

_COMPILED = [(r, [re.compile(p, F) for p in r.patterns]) for r in RULES + [PROTECTIVE]]


def warning_signs(message: str):
    """Returns list of dicts: key, label, points, scam_type, evidence."""
    hits = []
    protective = False
    for rule, pats in _COMPILED:
        for p in pats:
            m = p.search(message)
            if m:
                if rule.key == "protective":
                    protective = True
                hits.append({"key": rule.key, "label": rule.label, "points": rule.points,
                             "scam_type": rule.scam_type, "evidence": m.group(0)[:80]})
                break
    # A protective sentence cancels the OTP-request rule only when the request
    # itself is negated (e.g. "never share your OTP").
    if protective:
        hits = [h for h in hits if not (h["key"] == "otp" and
                re.search(r"(not|never|don't|dont|na|mat)\s+(\w+\s+){0,3}" + re.escape(h["evidence"].split()[0]),
                          message, F))]
    return hits
