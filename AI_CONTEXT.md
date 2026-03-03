# PhishGuard — AI Context & Development Notes

This file exists so any AI assistant picking up this project can understand the full context, decisions made, and rules to follow.

---

## What This Project Is

A **Chrome/Edge browser extension** (Manifest V3) that detects phishing websites using a machine learning backend.

- **Auto-scan**: Every tab load is scanned automatically via `background.js`
- **Manual check**: User can paste any URL in the popup
- **Backend**: Flask REST API serving an MNB model trained on 549K URLs
- **Model accuracy**: 94.69% | AUC-ROC: 0.9884

---

## Project Structure

```
backend/
  app.py                  Flask API (POST /predict, GET /health)
  predict.py              Inference — loads model + vectorizers, feature extraction
  train_model.py          Trains the model from scratch (run once)
  model.pkl               Trained MultinomialNB model (BEST performer)
  char_vectorizer.pkl     TF-IDF char n-grams (3-5), 60K features
  word_vectorizer.pkl     TF-IDF word n-grams (1-2), 30K features
  requirements.txt        Python dependencies

extension/
  manifest.json           Chrome MV3 — permissions: tabs, notifications, storage
  background.js           Service worker — scans tabs, updates badge, fires notifications
  content.js              Minimal — sends page URL to background on load
  popup.html/css/js       Dark-theme popup UI with auto-scan + manual check
  icons/                  icon16.png, icon48.png, icon128.png

phishing website/
  phishing_website_detection.ipynb   Original FYP notebook (reference only)
  phishing_site_urls.csv             Dataset — excluded from git (549K rows)
```

---

## Model Details

### Feature Vector (90,021 total)
1. **Char TF-IDF** — `char_wb`, n-gram (3,5), 60K features, sublinear TF
2. **Word TF-IDF** — `word`, n-gram (1,2), 30K features, sublinear TF
3. **21 hand-crafted URL features:**
   - URL/host/path length
   - Dot, hyphen, @, slash, ?, =, %, _ counts
   - Digit count in host
   - Subdomain depth, path depth, query param count
   - HTTPS flag, IP-in-host, URL shortener flag
   - Suspicious TLD flag (.xyz, .tk, .ml, .ga etc.)
   - Brand keyword in suspicious context (only if hyphenated: `paypal-something`)
   - Host Shannon entropy

### Model Comparison (test set, 20% of 549K)
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **MultinomialNB** (saved) | **94.69%** | **94.98%** | **85.89%** | **90.21%** | **98.84%** |
| LogisticRegression | 90.54% | 88.95% | 76.24% | 82.10% | 95.46% |

---

## Key Design Decisions

### Why MNB over Logistic Regression?
MNB outperforms LR on this task across all metrics. LR also didn't converge in 200 iterations (needs ~1000+, which is very slow on 90K sparse features). MNB is also much faster at inference.

### Why Dual TF-IDF?
- **Char n-grams** capture morphological patterns (e.g. `pay`, `paypа`, `-login`)
- **Word n-grams** capture semantic tokens (e.g. `secure`, `verify`, `account update`)
- Combined with hand features: better than either alone

### Trusted Domain Whitelist (in `predict.py`)
`google.com`, `paypal.com`, etc. are short-circuited to safe **before** the model runs.  
**Why**: The BRAND_KEYWORDS feature was causing false positives (google.com → phishing).  
**Rule**: Only flag brand keywords when used in hyphenated context (`paypal-login.xyz`).

### Chrome Extension — Why MV3?
MV3 is the current Chrome standard. Background scripts are Service Workers (no persistent background page). `chrome.storage.local` is used to cache scan results so the popup reads instantly without an extra API call.

---

## What TO DO Next

- [ ] **Deploy backend to Render.com** (free) — update `API_BASE` in `background.js`
- [ ] **Package extension** — zip `extension/` folder for Chrome Web Store submission
- [ ] **Improve false negatives** — `amazon-offer-winner.tk` slips through (~14% miss rate). Options:
  - Add a rule-based pre-check: suspicious TLD + brand name = auto-flag
  - Retrain with increased `alpha` for better calibration
  - Try a GradientBoostingClassifier (slower to train but may be more accurate)
- [ ] **Add feature: Page content scanning** — scan page HTML/JS for suspicious patterns beyond just URL
- [ ] **Add allowlist/blocklist** — let user mark sites as always-safe or always-blocked
- [ ] **Tests** — write `backend/tests/test_predict.py` with pytest

---

## What NOT TO DO

- ❌ **Do NOT retrain without reading the training notes** — LR with `saga` solver takes 10+ minutes; use `lbfgs` with `max_iter=200` or stick with MNB
- ❌ **Do NOT increase TF-IDF features above 60K+30K** without testing speed — 150K combined features caused the LR training to hang indefinitely
- ❌ **Do NOT remove the trusted domain whitelist** from `predict.py` — it prevents critical false positives on major brands
- ❌ **Do NOT commit the dataset CSV** — it's 549K rows (~50MB), excluded via `.gitignore`
- ❌ **Do NOT use Manifest V2** — Chrome is deprecating it; stick with MV3
- ❌ **Do NOT change the feature order** in `_hand_features()` without retraining — the model was trained on a specific 21-feature vector

---

## Running the Project

```bash
# 1. Install dependencies
cd backend && pip install -r requirements.txt

# 2. Start API
python app.py
# → http://localhost:5000

# 3. Load extension in Chrome
# chrome://extensions/ → Developer Mode → Load unpacked → select extension/

# 4. To retrain model (optional)
python train_model.py
# Requires: kaggle API key at ~/.kaggle/kaggle.json
```

---

## API Contract

```
POST http://localhost:5000/predict
Content-Type: application/json
Body: { "url": "http://example.com" }

Response 200:
{ "label": "phishing" | "safe", "confidence": 0.95, "url": "http://example.com" }

Response 400: { "error": "..." }

GET http://localhost:5000/health
Response 200: { "status": "ok" }
```

---

## Tech Stack
- Python 3.11+, scikit-learn 1.x, Flask, Flask-CORS, joblib, scipy, numpy, pandas, matplotlib
- Chrome Extension: Manifest V3, Vanilla JS (no frameworks)
- Dataset: [Kaggle — taruntiwarihp/phishing-site-urls](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
