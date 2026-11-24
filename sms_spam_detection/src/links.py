"""Link safety: URLs are analysed as text only and never opened."""
from __future__ import annotations

import ipaddress
import json
import re
import urllib.request

URL_RE = re.compile(
    r"((?:https?://|hxxps?://|www\.)[^\s<>\"']+|"
    r"\b(?:[a-z0-9-]+\.)+(?:com|in|net|org|co|xyz|top|info|ly|me|io|link|click|online|site|live|app|"
    r"icu|buzz|shop|club|vip|cc|tk|ml|ga|cf|gq|gd|at|to)\b(?:/[^\s<>\"']*)?)",
    re.IGNORECASE)

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly",
              "shorturl.at", "tiny.cc", "s.id", "rebrand.ly", "t.ly", "bitly.com", "v.gd", "shorte.st",
              "wa.me", "tinu.be", "short.gy"}
RISKY_TLDS = {"xyz", "top", "click", "live", "online", "site", "info", "icu", "buzz", "shop",
              "club", "vip", "tk", "ml", "ga", "cf", "gq", "cc", "link"}
# brand -> official registrable domains
BRANDS = {
    "sbi": ["sbi.co.in", "onlinesbi.sbi", "sbi"], "hdfc": ["hdfcbank.com", "hdfc.com"],
    "icici": ["icicibank.com", "icici.com"], "axis": ["axisbank.com"], "kotak": ["kotak.com"],
    "paytm": ["paytm.com"], "phonepe": ["phonepe.com"], "amazon": ["amazon.in", "amazon.com"],
    "flipkart": ["flipkart.com"], "indiapost": ["indiapost.gov.in"], "airtel": ["airtel.in"],
    "jio": ["jio.com"], "google": ["google.com"], "apple": ["apple.com"], "netflix": ["netflix.com"],
    "paypal": ["paypal.com"], "incometax": ["incometax.gov.in"], "uidai": ["uidai.gov.in"],
    "npci": ["npci.org.in"], "irctc": ["irctc.co.in"],
}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def defang(url: str) -> str:
    return re.sub(r"^http", "hxxp", url, flags=re.I).replace(".", "[.]")


def _host(url: str) -> str:
    u = re.sub(r"^(https?|hxxps?)://", "", url, flags=re.I)
    u = u.split("/")[0].split("?")[0].split("#")[0]
    u = u.split("@")[-1].split(":")[0]
    return u.lower().strip(".")


def _edit1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(a == b[:i] + b[i + 1:] for i in range(len(b)))


def analyse_url(url: str) -> dict:
    host = _host(url)
    reasons, points = [], 0
    try:
        ipaddress.ip_address(host)
        reasons.append("uses a raw IP address instead of a domain"); points += 15
    except ValueError:
        pass
    if host in SHORTENERS:
        reasons.append("link shortener hides the real destination"); points += 8
    if "xn--" in host:
        reasons.append("punycode (look-alike characters) in domain"); points += 15
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in RISKY_TLDS:
        reasons.append(f"cheap/high-abuse top-level domain .{tld}"); points += 8
    if "@" in url:
        reasons.append("'@' in link can hide the real host"); points += 10
    if url.lower().startswith(("http://", "hxxp://")):
        reasons.append("not HTTPS"); points += 3
    if host.count("-") >= 2 or host.count(".") >= 4:
        reasons.append("many hyphens/sub-domains"); points += 5
    labels = re.split(r"[.-]", host)
    norm = host.translate(LEET).replace("rn", "m")
    for brand, official in BRANDS.items():
        is_official = any(host == d or host.endswith("." + d) for d in official)
        if is_official:
            continue
        if brand in norm or any(_edit1(lbl.translate(LEET), brand) for lbl in labels if len(brand) >= 4):
            reasons.append(f"mentions '{brand}' but is not its official domain"); points += 15
            break
    return {"url": url, "defanged": defang(url), "host": host, "reasons": reasons,
            "points": min(points, 25)}


def extract_links(message: str) -> list[dict]:
    seen, out = set(), []
    for m in URL_RE.finditer(message):
        u = m.group(0).rstrip(".,);:!?")
        if u.lower() in seen:
            continue
        seen.add(u.lower())
        out.append(analyse_url(u))
    return out


def safe_browsing_check(urls: list[str], api_key: str, timeout: float = 5.0) -> dict:
    """Optional Google Safe Browsing v4 lookup. Sends ONLY the URLs, only with user consent."""
    body = {
        "client": {"clientId": "sms-spam-detector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE",
                            "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"], "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls],
        },
    }
    req = urllib.request.Request(
        f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    flagged = {m["threat"]["url"]: m["threatType"] for m in data.get("matches", [])}
    return flagged


def defang_text(text: str) -> str:
    """Defang every URL-like substring so nothing in the UI is clickable."""
    text = re.sub(r"https?://", lambda m: m.group(0).replace("http", "hxxp"), text, flags=re.I)
    return URL_RE.sub(lambda m: m.group(0).replace(".", "[.]"), text)
