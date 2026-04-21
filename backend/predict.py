"""
predict.py
----------
Prediction utility for the Flask API.
Loads all 3 trained models + shared dual TF-IDF vectorizers once at startup.

Public API:
    predict_url(url, model="mnb")  -> single result dict
    predict_all(url)               -> dict with results from all 3 models
"""

import re
import os
import math
import time
import threading
import numpy as np
import joblib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from scipy.sparse import hstack, csr_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))

# Shared vectorizers
char_vec = joblib.load(os.path.join(_DIR, "char_vectorizer.pkl"))
word_vec = joblib.load(os.path.join(_DIR, "word_vectorizer.pkl"))

# Load all 3 models
_MODEL_FILES = {
    "mnb": "model_mnb.pkl",
    "lr":  "model_lr.pkl",
    "rf":  "model_rf.pkl",
}

_MODELS = {}
for key, fname in _MODEL_FILES.items():
    fpath = os.path.join(_DIR, fname)
    if os.path.exists(fpath):
        _MODELS[key] = joblib.load(fpath)
        print(f"[✓] Loaded model: {key} ({fname})")
    else:
        print(f"[!] Model file not found, skipping: {fname}")

# Fallback: if individual files missing, use model.pkl for mnb
if "mnb" not in _MODELS:
    fallback = os.path.join(_DIR, "model.pkl")
    if os.path.exists(fallback):
        _MODELS["mnb"] = joblib.load(fallback)
        print("[✓] Loaded model.pkl as MNB fallback")

print(f"[✓] Vectorizers loaded. Available models: {list(_MODELS.keys())}")

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
    "yahoo.com","cloudflare.com","amazonaws.com","amazon.in","amazon.co.in","amazon.com","claude.ai","chatgpt.ai"
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

# ── Human-readable model names ────────────────────────────────────────────────
MODEL_LABELS = {
    "mnb": "Multinomial Naive Bayes",
    "lr":  "Logistic Regression",
    "rf":  "Random Forest",
}

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


def _build_features(url: str):
    """Build the combined feature vector for a URL."""
    token_str = _tokenize(url)
    char_feat = char_vec.transform([token_str])
    word_feat = word_vec.transform([token_str])
    hand_feat = csr_matrix(_hand_features(url))
    return hstack([char_feat, word_feat, hand_feat])


def _run_model(mdl, combined, url: str) -> dict:
    """Run a single model and return a result dict."""
    pred_class = mdl.predict(combined)[0]
    proba      = mdl.predict_proba(combined)[0]
    label      = "phishing" if pred_class == 1 else "safe"
    confidence = float(round(float(max(proba)), 4))
    return {"label": label, "confidence": confidence, "url": url}


def _explain(url: str) -> list:
    """
    Return a list of human-readable reasons why a URL looks suspicious.
    Based purely on the 21 hand-crafted features — no ML needed.
    Returns at most 5 most relevant signals.
    """
    lo          = url.lower()
    no_scheme   = re.sub(r"^https?://", "", lo)
    parts       = no_scheme.split("/", 1)
    host        = parts[0]
    path        = parts[1] if len(parts) > 1 else ""
    host_np     = host.split(":")[0]          # host without port
    subdomains  = host_np.split(".")
    query       = path.split("?", 1)[1] if "?" in path else ""
    entropy     = _entropy(host_np)

    reasons = []

    if SUSPICIOUS_TLDS.search(lo):
        tld = re.search(r"\.(\w+)(?:/|$)", host_np)
        tld_str = f".{tld.group(1)}" if tld else "suspicious TLD"
        reasons.append(f"Uses a high-risk domain extension ({tld_str}) commonly associated with phishing")

    if BRAND_CONTEXT.search(host_np):
        reasons.append("Contains a trusted brand name in a suspicious hyphenated context (e.g. paypal-login.xyz)")

    if re.search(r"\d{1,3}(\.\d{1,3}){3}", host_np):
        reasons.append("Hostname is a raw IP address — legitimate sites use domain names")

    if not lo.startswith("https"):
        reasons.append("Connection is not encrypted (HTTP, not HTTPS)")

    if SHORTENERS.search(host_np):
        reasons.append("URL uses a shortener service that hides the real destination")

    if entropy > 4.2:
        reasons.append(f"Hostname looks randomly generated (high entropy: {entropy:.2f}) — typical of auto-created phishing domains")

    if url.count("-") > 4:
        reasons.append(f"Unusually high number of hyphens ({url.count('-')}) in the URL")

    if max(0, len(subdomains) - 2) > 2:
        reasons.append(f"Deep subdomain structure ({max(0, len(subdomains)-2)} levels) used to disguise the real domain")

    if url.count("@") > 0:
        reasons.append("Contains @ symbol — can be used to redirect to a different host")

    if url.count("%") > 3:
        reasons.append(f"Contains {url.count('%')} hex-encoded characters — often used to obfuscate malicious URLs")

    if len(url) > 100:
        reasons.append(f"Unusually long URL ({len(url)} characters) — often used to hide the real destination")

    if len(query.split("&")) > 5 if query else False:
        reasons.append(f"Large number of query parameters ({len(query.split('&'))}) which may be used for tracking or obfuscation")

    # Return top 5 most relevant (first matched = highest priority)
    return reasons[:5] if reasons else ["URL pattern matches known phishing characteristics based on ML analysis"]


# ── Domain Age — WHOIS lookup ─────────────────────────────────────────────────
_AGE_CACHE      = {}          # {host: (age_days_or_None, fetched_at)}
_AGE_CACHE_LOCK = threading.Lock()
_AGE_CACHE_TTL  = 3600        # seconds (1 hour)
_WHOIS_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="whois")


def _whois_lookup(host: str):
    """Raw WHOIS call — runs in a thread so we can apply a timeout."""
    try:
        import whois                        # lazy import — not always installed
        w = whois.whois(host)
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0]
        if cd is None:
            return None
        if cd.tzinfo is None:
            cd = cd.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - cd).days
    except Exception:
        return None


def _get_domain_age_days(host: str) -> int | None:
    """
    Returns domain age in days via WHOIS, or None if lookup fails/times out.
    Results are cached for 1 hour.
    """
    now = time.time()
    with _AGE_CACHE_LOCK:
        if host in _AGE_CACHE:
            age_days, fetched_at = _AGE_CACHE[host]
            if now - fetched_at < _AGE_CACHE_TTL:
                return age_days

    try:
        future = _WHOIS_EXECUTOR.submit(_whois_lookup, host)
        age_days = future.result(timeout=5)
    except (FuturesTimeout, Exception):
        age_days = None

    with _AGE_CACHE_LOCK:
        _AGE_CACHE[host] = (age_days, time.time())

    return age_days


def _apply_domain_age(result: dict, host: str, url: str) -> dict:
    """
    Post-ML adjustment based on domain age.
    Modifies result in-place and returns it.

    Rules:
      < 30 days  → override to phishing (strong signal) regardless of ML verdict
      30–90 days → if already phishing, add age as supporting evidence
      > 90 days  → no change
      None       → WHOIS failed, ignore
    """
    age = _get_domain_age_days(host)
    if age is None:
        # WHOIS unavailable — still build explanation from URL features if phishing
        if result["label"] == "phishing":
            result.setdefault("explanation", _explain(url))
        return result

    result["domain_age_days"] = age

    if age < 30:
        # Very new domain — high confidence phishing signal
        if result["label"] == "safe":
            result["label"]      = "phishing"
            result["confidence"] = round(max(result["confidence"], 0.78), 4)
        age_reason = (
            f"Domain registered only {age} day{'s' if age != 1 else ''} ago — "
            f"newly registered domains are a primary indicator of phishing attacks"
        )
        existing = _explain(url)
        result["explanation"] = [age_reason] + existing[:4]   # age first, then URL signals

    elif age < 90:
        # Relatively new — add as supporting evidence if ML already flagged it
        if result["label"] == "phishing":
            existing = _explain(url)
            age_reason = (
                f"Domain is relatively new ({age} days old) — "
                f"phishing sites frequently use recently registered domains"
            )
            result["explanation"] = existing[:4] + [age_reason]
    else:
        # Established domain — ML result stands, build explanation normally
        if result["label"] == "phishing":
            result.setdefault("explanation", _explain(url))

    return result


# ── Public API ────────────────────────────────────────────────────────────────
def predict_url(url: str, model: str = "mnb") -> dict:
    """
    Predict a single URL with the chosen model.

    Args:
        url:   URL to classify
        model: "mnb" | "lr" | "rf"  (default: "mnb")

    Returns:
        {"label": "phishing"|"safe", "confidence": float, "url": str, "model": str}
    """
    model = model.lower().strip()
    if model not in _MODELS:
        available = list(_MODELS.keys())
        raise ValueError(f"Unknown model '{model}'. Available: {available}")

    lo          = url.lower()
    no_scheme   = re.sub(r"^https?://", "", lo)
    host        = no_scheme.split("/")[0].split(":")[0]
    base_domain = _get_base_domain(host)

    # Fast path: known-trusted domains
    if base_domain in TRUSTED_DOMAINS:
        return {
            "label": "safe",
            "confidence": 1.0,
            "url": url,
            "model": model,
            "model_name": MODEL_LABELS.get(model, model),
        }

    combined   = _build_features(url)
    result     = _run_model(_MODELS[model], combined, url)
    result["model"]      = model
    result["model_name"] = MODEL_LABELS.get(model, model)
    # Domain age check — may override label or enrich explanation
    result = _apply_domain_age(result, host, url)
    return result


def predict_all(url: str) -> dict:
    """
    Run all available models on a URL and return all results.

    Returns:
        {
          "url": str,
          "trusted": bool,
          "results": {
             "mnb": {"label": ..., "confidence": ..., "model_name": ..., "explanation": [...]},
             "lr":  {...},
             "rf":  {...}
          }
        }
    """
    lo          = url.lower()
    no_scheme   = re.sub(r"^https?://", "", lo)
    host        = no_scheme.split("/")[0].split(":")[0]
    base_domain = _get_base_domain(host)

    # Fast path: known-trusted domains
    if base_domain in TRUSTED_DOMAINS:
        return {
            "url": url,
            "trusted": True,
            "results": {
                key: {
                    "label": "safe",
                    "confidence": 1.0,
                    "model_name": MODEL_LABELS.get(key, key),
                }
                for key in _MODELS
            },
        }

    combined    = _build_features(url)
    out = {"url": url, "trusted": False, "results": {}}
    for key, mdl in _MODELS.items():
        r = _run_model(mdl, combined, url)
        r["model"]      = key
        r["model_name"] = MODEL_LABELS.get(key, key)
        r = _apply_domain_age(r, host, url)
        out["results"][key] = {
            "label":      r["label"],
            "confidence": r["confidence"],
            "model_name": r["model_name"],
        }
        if r["label"] == "phishing" and "explanation" in r:
            out["results"][key]["explanation"] = r["explanation"]
    return out
