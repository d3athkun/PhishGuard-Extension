"""
app.py
------
Flask REST API for the phishing website detector.

Endpoints:
    POST /predict
        Body: {"url": "http://...", "model": "mnb"|"lr"|"rf"}   (model optional, default=mnb)
        -> {"label": "phishing"|"safe", "confidence": 0.97, "url": "...", "model": "mnb", "model_name": "..."}

    POST /predict-all
        Body: {"url": "http://..."}
        -> {"url": "...", "trusted": bool, "results": {"mnb": {...}, "lr": {...}, "rf": {...}}}

    GET  /health -> {"status": "ok", "models": ["mnb", "lr", "rf"]}

Start:
    python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import predict  # loads models + vectorizers once at startup

app = Flask(__name__)
CORS(app)  # allow extension (different origin) and web frontend to call this API


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_url(data):
    """Parse and validate URL from request JSON."""
    if not data or "url" not in data:
        return None, (jsonify({"error": "Request body must be JSON with a 'url' field."}), 400)
    url = str(data["url"]).strip()
    if not url:
        return None, (jsonify({"error": "'url' must not be empty."}), 400)
    # Prefix bare domains so they look like a proper URL to the feature extractor
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url, None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": list(predict._MODELS.keys())})


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True)
    url, err = _extract_url(data)
    if err:
        return err

    model_key = str(data.get("model", "mnb")).lower().strip()

    try:
        result = predict.predict_url(url, model=model_key)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict-all", methods=["POST"])
def predict_all_route():
    """Run all available models and return comparison results. Used by the website only."""
    data = request.get_json(silent=True)
    url, err = _extract_url(data)
    if err:
        return err

    try:
        result = predict.predict_all(url)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[*] Starting Phishing Detector API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
