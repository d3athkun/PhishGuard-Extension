"""
predict.py
----------
Prediction utility for the Flask API.
Loads the trained MNB model + dual TF-IDF vectorizers once at startup.
"""

import re
import os
import math
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR           = os.path.dirname(os.path.abspath(__file__))
model          = joblib.load(os.path.join(_DIR, "model.pkl"))
char_vec       = joblib.load(os.path.join(_DIR, "char_vectorizer.pkl"))
word_vec       = joblib.load(os.path.join(_DIR, "word_vectorizer.pkl"))
print("[✓] Model and vectorizers loaded.")

# ── Trusted base domains (bypass model — always safe) ─────────────────────────
TRUSTED_DOMAINS = {
    "google.com","google.co.uk","youtube.com","gmail.com",
    "microsoft.com","live.com","outlook.com","office.com","bing.com",
    "apple.com","icloud.com",
    "amazon.com","amazon.co.uk",
    "facebook.com","instagram.com","whatsapp.com",
    "twitter.com","x.com","linkedin.com",
    "reddit.com","wikipedia.org","github.com","stackoverflow.com",
    "paypal.com","ebay.com","netflix.com","spotify.com",
    "yahoo.com","cloudflare.com","amazonaws.com",
}

# ── Regex constants ───────────────────────────────────────────────────────────
SHORTENERS = re.compile(
    r"(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|buff\.ly|"
    r"adf\.ly|tiny\.cc|is\.gd|cli\.gs|bc\.vc|short\.io)", re.I
)
SUSPICIOUS_TLDS = re.compile(
    r"\.(xyz|tk|ml|ga|cf|gq|pw|top|club|online|site|info|"
    r"link|live|stream|download|win|racing|date|review)(\b|/|$)", re.I
)
# Only flag brand keyword when used suspiciously (with hyphen or as subdomain on another domain)
BRAND_CONTEXT = re.compile(
    r"(paypal|amazon|apple|microsoft|google|facebook|instagram|"
    r"netflix|ebay|bank|secure|login|verify|account|update|confirm)-", re.I
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_base_domain(host: str) -> str:
    parts = host.lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    probs = [v / len(s) for v in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def _tokenize(url: str) -> str:
    return " ".join(re.split(r"[^a-z0-9]", str(url).lower()))


def _hand_features(url: str) -> np.ndarray:
    url  = str(url)
    lo   = url.lower()
    no_scheme    = re.sub(r"^https?://", "", lo)
    parts        = no_scheme.split("/", 1)
    host         = parts[0]
    path         = parts[1] if len(parts) > 1 else ""
    host_no_port = host.split(":")[0]
    subdomains   = host_no_port.split(".")
    query        = path.split("?", 1)[1] if "?" in path else ""

    feats = [
        len(url),
        len(host_no_port),
        len(path),
        url.count("."),
        url.count("-"),
        url.count("@"),
        url.count("/"),
        url.count("?"),
        url.count("="),
        url.count("%"),
        url.count("_"),
        sum(c.isdigit() for c in host_no_port),
        max(0, len(subdomains) - 2),
        path.count("/"),
        len(query.split("&")) if query else 0,
        1 if lo.startswith("https") else 0,
        1 if re.search(r"\d{1,3}(\.\d{1,3}){3}", host_no_port) else 0,
        1 if SHORTENERS.search(host_no_port) else 0,
        1 if SUSPICIOUS_TLDS.search(lo) else 0,
        # Only flag brand name if used in suspicious hyphenated context
        1 if BRAND_CONTEXT.search(host_no_port) else 0,
        round(_entropy(host_no_port), 4),
    ]
    return np.array(feats, dtype=float).reshape(1, -1)


# ── Public API ────────────────────────────────────────────────────────────────
def predict_url(url: str) -> dict:
    """
    Returns:
        {"label": "phishing"|"safe", "confidence": float, "url": str}
    """
    lo           = url.lower()
    no_scheme    = re.sub(r"^https?://", "", lo)
    host         = no_scheme.split("/")[0].split(":")[0]
    base_domain  = _get_base_domain(host)

    # Fast path: known-trusted domains
    if base_domain in TRUSTED_DOMAINS:
        return {"label": "safe", "confidence": 1.0, "url": url}

    token_str = _tokenize(url)
    char_feat = char_vec.transform([token_str])
    word_feat = word_vec.transform([token_str])
    hand_feat = csr_matrix(_hand_features(url))
    combined  = hstack([char_feat, word_feat, hand_feat])

    pred_class = model.predict(combined)[0]
    proba      = model.predict_proba(combined)[0]

    label      = "phishing" if pred_class == 1 else "safe"
    confidence = float(round(float(max(proba)), 4))

    return {"label": label, "confidence": confidence, "url": url}
