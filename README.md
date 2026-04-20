# PhishGuard — Phishing Website Detector

> Real-time phishing detection — available as a **website** and a **Chrome/Edge extension**, both powered by the same ML backend.

![PhishGuard Extension](extension/icons/icon128.png)

---

## What It Does

PhishGuard uses a machine learning model trained on **549,000+ URLs** to detect phishing websites in real time.

| Mode | How It Works |
|---|---|
| 🌐 **Website** | Visit the web app, paste any URL, get instant results |
| 🧩 **Extension** | Auto-scans every tab you open; badge turns 🔴/🟢; manual check in popup |

Both modes call the same **Flask REST API** — switch between them freely.

---

## Architecture

```
┌──────────────────────┐     ┌──────────────────────┐
│   Website            │     │   Chrome Extension   │
│   (frontend/)        │     │   (extension/)       │
│   GitHub Pages /     │     │   Manifest V3        │
│   Vercel             │     │   popup + bg worker  │
└──────────┬───────────┘     └──────────┬───────────┘
           │  POST /predict             │  POST /predict
           ▼                            ▼
   ┌─────────────────────────────────────────┐
   │         Flask API  (backend/)           │
   │         localhost:5000  OR  Render.com  │
   │                                         │
   │   MultinomialNB · Dual TF-IDF           │
   │   · 21 hand-crafted URL features        │
   └─────────────────────────────────────────┘
```

---

## Project Structure

```
PhishGuard-Extension/
│
├── frontend/                        ← Web App (NEW)
│   ├── index.html                   # Main website page
│   ├── style.css                    # Dark-theme styles
│   └── app.js                       # fetch() → /predict, renders result
│
├── backend/                         ← Flask REST API
│   ├── app.py                       # POST /predict, GET /health
│   ├── predict.py                   # Model inference + feature extraction
│   ├── train_model.py               # Dataset download + model training
│   ├── model.pkl                    # Trained MultinomialNB model
│   ├── char_vectorizer.pkl          # Char n-gram TF-IDF (3–5, 60K features)
│   ├── word_vectorizer.pkl          # Word n-gram TF-IDF (1–2, 30K features)
│   └── requirements.txt
│
├── extension/                       ← Chrome/Edge Extension (MV3)
│   ├── manifest.json
│   ├── background.js                # Service worker — auto-scan + badge
│   ├── content.js                   # Sends URL to background on page load
│   ├── popup.html / popup.css / popup.js
│   └── icons/                       # icon16, icon48, icon128
│
└── phishing website/                ← Research Notebook
    └── phishing_website_detection.ipynb   # Training experiments (RF + LR + MNB)
```

---

## Setup & Running

### Prerequisites

- Python 3.11+
- Google Chrome or Microsoft Edge

### 1. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Flask API

```bash
python app.py
# → Running on http://localhost:5000
```

### 3a. Use the Web App

Open `frontend/index.html` in your browser — no server needed, it runs as a static page.

> Make sure the Flask API is running on `localhost:5000` first.

### 3b. Use the Chrome Extension

1. Go to `chrome://extensions/`
2. Enable **Developer Mode** (top-right toggle)
3. Click **Load unpacked** → select the `extension/` folder
4. Pin **PhishGuard** to your toolbar ✓

---

## Model Details

### Feature Vector (90,021 total features)

| Feature Type | Details | Count |
|---|---|---|
| Char TF-IDF | 3–5 char n-grams (`char_wb`) | 60,000 |
| Word TF-IDF | 1–2 word n-grams | 30,000 |
| Hand-crafted | URL/host/path length, dot/hyphen/@/slash counts, subdomain depth, HTTPS flag, IP-in-host, URL shortener flag, suspicious TLD, Shannon entropy… | 21 |

### Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **MultinomialNB** *(deployed)* | **94.69%** | **94.98%** | **85.89%** | **90.21%** | **98.84%** |
| Random Forest *(notebook)* | ~97–98% | — | — | — | — |
| Logistic Regression | 90.54% | 88.95% | 76.24% | 82.10% | 95.46% |

> The Random Forest model was trained in the research notebook using a simpler CountVectorizer pipeline.  
> The deployed backend uses the higher-quality dual TF-IDF + 21 features MNB pipeline.

---

## API Reference

```
POST /predict
Content-Type: application/json
Body: { "url": "https://example.com" }

Response 200:
{
  "label": "phishing" | "safe",
  "confidence": 0.97,
  "url": "https://example.com"
}

GET /health  →  { "status": "ok" }
```

---

## Deployment (Online)

### Backend → Render.com (free)
1. Push the repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set **Root Directory** to `backend/`, **Start Command** to `gunicorn app:app`
4. Update `API_BASE` in `extension/background.js` and `frontend/app.js` to your Render URL

### Frontend → Vercel / GitHub Pages (free)
- The `frontend/` folder is pure static HTML — deploy to any static host
- GitHub Pages: Settings → Pages → Source: `main` branch, `/frontend` folder

---

## Dataset

[Phishing Site URLs — Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)  
549,346 URLs — 156,422 phishing / 392,924 safe

> The CSV is excluded from git (`.gitignore`). To retrain: `python backend/train_model.py`  
> Requires a [Kaggle API key](https://www.kaggle.com/settings) at `~/.kaggle/kaggle.json`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML / Training | scikit-learn (MultinomialNB, RandomForest, TF-IDF), pandas, numpy |
| API | Python 3.11+, Flask, Flask-CORS, gunicorn |
| Web Frontend | HTML5, CSS3 (dark theme), Vanilla JS |
| Extension | Chrome Manifest V3, Vanilla JS |
| Deployment | Render.com (API), Vercel / GitHub Pages (frontend) |

---

## Roadmap

- [x] MNB model — 94.69% accuracy on 549K URLs
- [x] Dual TF-IDF + 21 hand-crafted URL features  
- [x] Chrome/Edge extension (MV3) — auto-scan + manual check  
- [ ] Web frontend (website mode)
- [ ] Deploy backend to Render.com
- [ ] Deploy frontend to Vercel/GitHub Pages
- [ ] Add Random Forest to backend pipeline
- [ ] Page content scanning (HTML/JS analysis)
- [ ] User allowlist/blocklist
- [ ] Chrome Web Store submission.
