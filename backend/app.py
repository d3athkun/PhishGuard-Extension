"""
app.py
------
Flask REST API for the phishing website detector.

Endpoints:
    POST /predict   {"url": "http://..."}
                 -> {"label": "phishing"|"safe", "confidence": 0.97, "url": "..."}
    GET  /health -> {"status": "ok"}

Start:
    python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import predict  # loads model + vectorizer once at startup

app = Flask(__name__)
CORS(app)  # allow extension (different origin) to call this API


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True)

    if not data or "url" not in data:
        return jsonify({"error": "Request body must be JSON with a 'url' field."}), 400

    url = str(data["url"]).strip()
    if not url:
        return jsonify({"error": "'url' must not be empty."}), 400

    # Prefix bare domains so they look like a proper URL to the feature extractor
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        result = predict.predict_url(url)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[*] Starting Phishing Detector API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
