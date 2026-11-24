/* SMS Spam & Scam Detection: browser engine.
   A JavaScript port of src/model.py (TF-IDF + calibrated Linear SVM), src/rules.py,
   src/links.py and src/risk.py. Runs fully in the browser: messages never leave the page. */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ TF-IDF + SVM
  function buildIndex(block) {
    const idx = new Map();
    for (let i = 0; i < block.terms.length; i++) idx.set(block.terms[i], i);
    return idx;
  }

  const WORD_TOKEN = /[\p{L}\p{N}_]+|[£$€₹]/gu;

  function wordTerms(text, nmin, nmax) {
    const toks = text.toLowerCase().match(WORD_TOKEN) || [];
    const out = [];
    for (let n = nmin; n <= nmax; n++)
      for (let i = 0; i + n <= toks.length; i++) out.push(toks.slice(i, i + n).join(" "));
    return out;
  }

  // sklearn analyzer="char_wb"
  function charTerms(text, nmin, nmax) {
    const doc = text.toLowerCase().replace(/\s\s+/gu, " ");
    const out = [];
    for (const raw of doc.split(/\s+/u)) {
      if (!raw) continue;
      const w = Array.from(" " + raw + " ");
      for (let n = nmin; n <= nmax; n++) {
        let offset = 0;
        out.push(w.slice(offset, offset + n).join(""));
        while (offset + n < w.length) { offset++; out.push(w.slice(offset, offset + n).join("")); }
        if (offset === 0) break;
      }
    }
    return out;
  }

  function vectorize(terms, block, index) {
    const counts = new Map();
    for (const t of terms) {
      const j = index.get(t);
      if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
    }
    const feats = [];
    let norm = 0;
    for (const [j, c] of counts) {
      const v = (1 + Math.log(c)) * block.idf[j];
      feats.push([j, v]); norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    for (const f of feats) f[1] /= norm;
    return feats;
  }

  function Model(json) {
    this.m = json;
    this.wIdx = buildIndex(json.word);
    this.cIdx = buildIndex(json.char);
  }

  Model.prototype.analyse = function (text) {
    const m = this.m;
    const wf = vectorize(wordTerms(text, m.word_ngram[0], m.word_ngram[1]), m.word, this.wIdx);
    const cf = vectorize(charTerms(text, m.char_ngram[0], m.char_ngram[1]), m.char, this.cIdx);
    let score = m.intercept;
    const contrib = [];
    for (const [j, v] of wf) { const c = v * m.word.coef[j]; score += c; contrib.push(["word", m.word.terms[j], c]); }
    for (const [j, v] of cf) { const c = v * m.char.coef[j]; score += c; contrib.push(["char", m.char.terms[j], c]); }
    const p = 1 / (1 + Math.exp(m.calibration.a * score + m.calibration.b));
    return { prob: p, explain: explain(text, contrib) };
  };

  function explain(text, contrib, topK) {
    topK = topK || 8;
    const tokens = text.split(/\s+/).map(w => w.replace(/^[^\p{L}\p{N}_]+|[^\p{L}\p{N}_]+$/gu, "").toLowerCase()).filter(Boolean);
    const words = new Map();
    for (const [block, term, c] of contrib) {
      let key;
      if (block === "word") key = term;
      else {
        const t = term.trim();
        if (Array.from(t).length < 3) continue;
        key = tokens.find(w => w.includes(t));
        if (!key) continue;
      }
      words.set(key, (words.get(key) || 0) + c);
    }
    const items = [...words.entries()].sort((a, b) => a[1] - b[1]);
    return {
      spam: items.filter(x => x[1] > 0).reverse().slice(0, topK),
      ham: items.filter(x => x[1] < 0).slice(0, topK),
    };
  }

  // ------------------------------------------------------------------ warning-sign rules
  const RULES = [
    { key: "kyc", label: "KYC / account-block threat", points: 20, type: "KYC / bank phishing", pats: [
      String.raw`\bkyc\b.{0,40}(update|expire|pending|suspend|block|verify|complete)`,
      String.raw`(update|complete|verify).{0,25}\b(kyc|pan|aadhaa?r)\b`,
      String.raw`(account|a/c|card|sim|wallet).{0,30}(will be |is |has been )?(blocked|suspended|deactivated|closed|frozen)`,
      String.raw`(account|khata).{0,20}band (ho|kar)`,
      String.raw`(खाता|अकाउंट).{0,30}(बंद|ब्लॉक)`, String.raw`केवाईसी`] },
    { key: "otp", label: "Asks you to share OTP / PIN / password", points: 25, type: "OTP / PIN theft", pats: [
      String.raw`(share|send|tell|forward|give|provide|reply with|bata\w*|bhej\w*)\s+(\w+\s+){0,3}(otp|pin|cvv|password|passcode)`,
      String.raw`(otp|pin|cvv)\s+(\w+\s+){0,3}(share|send|bata\w*|bhej\w*|batayein|bataye)\b`,
      String.raw`(ओटीपी|पिन).{0,20}(बताएं|बताइए|भेजें|शेयर)`] },
    { key: "upi", label: "UPI 'enter PIN to receive money'", points: 25, type: "UPI fraud", pats: [
      String.raw`(enter|put|daal\w*|dale)\s+(your\s+)?(upi\s+)?pin.{0,30}(receive|get|credit|claim|paise|paisa|refund)`,
      String.raw`(receive|get|claim).{0,30}(enter|daal\w*).{0,10}\bpin\b`,
      String.raw`(receive|paise|paisa|refund|cashback|credit).{0,40}\bpin\s+(daal\w*|dal\w*|enter|daale\w*)`,
      String.raw`collect request.{0,40}(approve|accept)`] },
    { key: "prize", label: "Prize / lottery / reward claim", points: 15, type: "Prize / lottery scam", pats: [
      String.raw`\b(won|winner|win|congratulations|congrats|selected|claim)\b.{0,60}(prize|reward|cash|lottery|£|\$|rs\.?\s?\d|₹|voucher|gift|bonus)`,
      String.raw`(cashback|reward|prize|inaam)\s+(\w+\s+){0,2}(mila|jeeta|won|credited|unlocked)`,
      String.raw`lucky (draw|winner|customer)`, String.raw`\blottery\b`, String.raw`\bjackpot\b`,
      String.raw`(इनाम|लॉटरी|जीत)`, String.raw`(aap|apne).{0,20}(jeet|jeete|jita)`] },
    { key: "delivery", label: "Parcel / delivery fee or address problem", points: 15, type: "Delivery scam", pats: [
      String.raw`(parcel|package|courier|delivery|shipment|consignment).{0,50}(fee|pay|charge|held|on hold|failed|unsuccessful|address|customs|redeliver)`,
      String.raw`(india ?post|fedex|dhl|bluedart|usps).{0,50}(pay|fee|update|confirm)`] },
    { key: "job", label: "Fake job / easy money offer", points: 15, type: "Fake job scam", pats: [
      String.raw`(work from home|part[- ]?time (job|work)|online job|daily (income|earning|payment))`,
      String.raw`earn\s+(up to\s+)?(rs\.?|₹|inr)?\s?\d[\d,]*.{0,30}(per|a|daily|day|week|hour|month)`,
      String.raw`(like|subscribe).{0,20}(youtube|videos).{0,30}(earn|pay|₹|rs)`, String.raw`telegram.{0,20}task`,
      String.raw`ghar (baithe|se).{0,30}(kama|earn|income)`] },
    { key: "invest", label: "Investment / guaranteed-returns offer", points: 15, type: "Investment scam", pats: [
      String.raw`(guaranteed|assured|fixed|double)\s+(\w+\s+)?(returns?|profit|income|money)`,
      String.raw`(crypto|bitcoin|forex|stock|trading)\s+(tips|signals|investment|profit|group)`,
      String.raw`(invest|investment).{0,40}(\d+\s?%|daily|guaranteed|returns)`] },
    { key: "premium", label: "Premium-rate number / text-to-claim", points: 10, type: "Premium-rate spam", pats: [
      String.raw`\b09\d{8,9}\b`, String.raw`\b(text|txt|reply|send)\s+[A-Z]{2,10}\s+to\s+\d{4,6}\b`,
      String.raw`\b\d{1,3}p\s?/\s?(min|msg)\b|£\d+(\.\d+)?\s?/\s?(msg|wk|week)`] },
    { key: "urgency", label: "Pressure / urgency language", points: 5, type: null, pats: [
      String.raw`\b(urgent|immediately|act now|within 24 ?(hrs?|hours)|today only|expires? (today|soon|in)|last chance|final (notice|reminder))\b`,
      String.raw`(turant|jaldi|abhi)\s`, String.raw`(तुरंत|जल्दी)`] },
    { key: "protective", label: "Protective wording (e.g. 'do not share your OTP')", points: -15, type: null, pats: [
      String.raw`(do not|don't|dont|never|not to)\s+(\w+\s+){0,3}(share|disclose|tell)`,
      String.raw`(kisi ke saath|kisi se).{0,20}(share na|mat|na kare)`, String.raw`(साझा|शेयर) न करें`,
      String.raw`(bank|we) (will )?never ask`] },
  ];
  for (const r of RULES) r.re = r.pats.map(p => new RegExp(p, "iu"));

  function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  function warningSigns(message) {
    let hits = [], protective = false;
    for (const r of RULES) {
      for (const re of r.re) {
        const m = message.match(re);
        if (m) {
          if (r.key === "protective") protective = true;
          hits.push({ key: r.key, label: r.label, points: r.points, type: r.type, evidence: m[0].slice(0, 80) });
          break;
        }
      }
    }
    if (protective) {
      hits = hits.filter(h => !(h.key === "otp" && new RegExp(
        String.raw`(not|never|don't|dont|na|mat)\s+(\w+\s+){0,3}` + escapeRe(h.evidence.split(/\s+/)[0]), "iu").test(message)));
    }
    return hits;
  }

  // ------------------------------------------------------------------ link safety (text only)
  const URL_RE = /((?:https?:\/\/|hxxps?:\/\/|www\.)[^\s<>"']+|\b(?:[a-z0-9-]+\.)+(?:com|in|net|org|co|xyz|top|info|ly|me|io|link|click|online|site|live|app|icu|buzz|shop|club|vip|cc|tk|ml|ga|cf|gq|gd|at|to)\b(?:\/[^\s<>"']*)?)/gi;
  const SHORTENERS = new Set(["bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly", "shorturl.at",
    "tiny.cc", "s.id", "rebrand.ly", "t.ly", "bitly.com", "v.gd", "shorte.st", "wa.me", "tinu.be", "short.gy"]);
  const RISKY_TLDS = new Set(["xyz", "top", "click", "live", "online", "site", "info", "icu", "buzz", "shop", "club", "vip", "tk", "ml", "ga", "cf", "gq", "cc", "link"]);
  const BRANDS = {
    sbi: ["sbi.co.in", "onlinesbi.sbi", "sbi"], hdfc: ["hdfcbank.com", "hdfc.com"], icici: ["icicibank.com", "icici.com"],
    axis: ["axisbank.com"], kotak: ["kotak.com"], paytm: ["paytm.com"], phonepe: ["phonepe.com"], amazon: ["amazon.in", "amazon.com"],
    flipkart: ["flipkart.com"], indiapost: ["indiapost.gov.in"], airtel: ["airtel.in"], jio: ["jio.com"], google: ["google.com"],
    apple: ["apple.com"], netflix: ["netflix.com"], paypal: ["paypal.com"], incometax: ["incometax.gov.in"], uidai: ["uidai.gov.in"],
    npci: ["npci.org.in"], irctc: ["irctc.co.in"],
  };
  const LEET = { "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s" };
  const leet = s => s.replace(/[013457@$]/g, ch => LEET[ch]);

  function defang(url) { return url.replace(/^http/i, "hxxp").replace(/\./g, "[.]"); }
  function defangText(text) {
    return text.replace(/https?:\/\//gi, m => m.replace(/http/i, "hxxp")).replace(URL_RE, m => m.replace(/\./g, "[.]"));
  }
  function hostOf(url) {
    let u = url.replace(/^(https?|hxxps?):\/\//i, "");
    u = u.split("/")[0].split("?")[0].split("#")[0];
    u = u.split("@").pop().split(":")[0];
    return u.toLowerCase().replace(/^\.+|\.+$/g, "");
  }
  function edit1(a, b) {
    if (Math.abs(a.length - b.length) > 1 || a === b) return false;
    if (a.length === b.length) { let d = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++; return d === 1; }
    if (a.length > b.length) [a, b] = [b, a];
    for (let i = 0; i < b.length; i++) if (a === b.slice(0, i) + b.slice(i + 1)) return true;
    return false;
  }
  function analyseUrl(url) {
    const host = hostOf(url), reasons = []; let points = 0;
    if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) { reasons.push("uses a raw IP address instead of a domain"); points += 15; }
    if (SHORTENERS.has(host)) { reasons.push("link shortener hides the real destination"); points += 8; }
    if (host.includes("xn--")) { reasons.push("punycode (look-alike characters) in domain"); points += 15; }
    const tld = host.includes(".") ? host.split(".").pop() : "";
    if (RISKY_TLDS.has(tld)) { reasons.push(`cheap/high-abuse top-level domain .${tld}`); points += 8; }
    if (url.includes("@")) { reasons.push("'@' in link can hide the real host"); points += 10; }
    if (/^(http|hxxp):\/\//i.test(url)) { reasons.push("not HTTPS"); points += 3; }
    if ((host.match(/-/g) || []).length >= 2 || (host.match(/\./g) || []).length >= 4) { reasons.push("many hyphens/sub-domains"); points += 5; }
    const labels = host.split(/[.-]/), norm = leet(host).replace(/rn/g, "m");
    for (const [brand, official] of Object.entries(BRANDS)) {
      if (official.some(d => host === d || host.endsWith("." + d))) continue;
      if (norm.includes(brand) || (brand.length >= 4 && labels.some(l => edit1(leet(l), brand)))) {
        reasons.push(`mentions '${brand}' but is not its official domain`); points += 15; break;
      }
    }
    return { url, defanged: defang(url), host, reasons, points: Math.min(points, 25) };
  }
  function extractLinks(message) {
    const seen = new Set(), out = [];
    for (const m of message.matchAll(URL_RE)) {
      const u = m[0].replace(/[.,);:!?]+$/, "");
      if (seen.has(u.toLowerCase())) continue;
      seen.add(u.toLowerCase()); out.push(analyseUrl(u));
    }
    return out;
  }

  // ------------------------------------------------------------------ risk
  const ACTIONS = {
    "KYC / bank phishing": ["Do not click the link or call the number in the SMS.", "Open your bank's official app or website yourself to check.", "Banks never ask for KYC updates through SMS links."],
    "OTP / PIN theft": ["Never share an OTP, PIN, CVV or password with anyone.", "No bank, delivery or support agent needs your OTP."],
    "UPI fraud": ["You never need to enter a UPI PIN to RECEIVE money.", "Decline unknown collect requests in your UPI app."],
    "Prize / lottery scam": ["You cannot win a contest you did not enter.", "Do not pay any 'processing' or 'tax' fee to claim a prize."],
    "Delivery scam": ["Track parcels only on the courier's official website/app.", "Do not pay small 're-delivery' fees through SMS links."],
    "Fake job scam": ["Genuine employers do not ask you to pay to get a job.", "Avoid 'like videos and earn' tasks on Telegram/WhatsApp."],
    "Investment scam": ["Guaranteed high returns are a red flag.", "Check SEBI registration before investing."],
    "Premium-rate spam": ["Do not reply or call back; it may be charged at premium rates."],
    "Generic spam / promotion": ["Ignore or block the sender if you did not opt in."],
  };
  const REPORT = "In India you can report fraud at cybercrime.gov.in or call 1930; forward spam to 1909 / use Sanchar Saathi (Chakshu).";

  function assess(message, prob, threshold) {
    const signs = warningSigns(message), links = extractLinks(message);
    const signPts = signs.reduce((s, x) => s + x.points, 0);
    const linkPts = links.reduce((m, l) => Math.max(m, l.points), 0);
    const extra = Math.max(-20, Math.min(40, signPts + linkPts));
    const score = Math.round(Math.max(0, Math.min(100, 60 * prob + extra)));
    const level = score >= 60 ? "high" : score >= 30 ? "medium" : "low";
    const typed = signs.filter(s => s.type).sort((a, b) => b.points - a.points);
    const scamType = typed.length ? typed[0].type : (level !== "low" ? "Generic spam / promotion" : null);
    let actions = scamType ? ACTIONS[scamType].slice() : [];
    if (level === "high") actions.push(REPORT);
    return { verdict: prob >= threshold ? "spam" : "legitimate", prob, score, level, scamType, signs, links, actions, rulePoints: extra };
  }

  const api = { Model, assess, warningSigns, extractLinks, defangText, wordTerms, charTerms };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SpamEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
