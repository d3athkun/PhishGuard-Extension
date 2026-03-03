# PhishGuard — Phishing Website Detector Extension

A Chrome/Edge browser extension that uses **machine learning** to detect phishing websites in real time — auto-scanning every page you visit and letting you manually check any URL.

![PhishGuard Extension](extension/icons/icon128.png)

---

## Features

- 🔴 **Auto-scan** — detects phishing on every page load and updates the badge
- 🟢 **Safe indicator** — green badge on legitimate sites
- 🔍 **Manual check** — paste any URL in the popup to check it instantly
- 🔔 **Notifications** — browser alert when a phishing site is detected
- ⚡ **94.69% accuracy** — Multinomial Naive Bayes trained on 549K URLs
- **AUC-ROC: 0.9884**

---

## Architecture

```
Chrome Extension (MV3)
    │  POST /predict {"url": "..."}
    ▼
Flask REST API  ←  MNB model + Dual TF-IDF (char + word n-grams) + 21 URL features
```

---

## Project Structure

```
├── backend/
│   ├── app.py              # Flask REST API
│   ├── predict.py          # Model inference + feature extraction
│   ├── train_model.py      # Dataset download + model training script
│   ├── model.pkl           # Trained MNB model
│   ├── char_vectorizer.pkl # Char n-gram TF-IDF
│   ├── word_vectorizer.pkl # Word n-gram TF-IDF
│   └── requirements.txt
│
├── extension/
│   ├── manifest.json       # Chrome MV3 manifest
│   ├── background.js       # Service worker (auto-scan + notifications)
│   ├── content.js          # Content script
│   ├── popup.html/css/js   # Extension popup UI
│   └── icons/              # 16, 48, 128px icons
│
└── phishing website/
    └── phishing_website_detection.ipynb  # Original notebook
```

---

## Setup & Running

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the model (optional — pre-trained pkl files included)
```bash
python train_model.py
```
This downloads the dataset from Kaggle (~549K URLs) and trains the model.  
Requires a [Kaggle API key](https://www.kaggle.com/settings) at `~/.kaggle/kaggle.json`.

### 3. Start the Flask API
```bash
python app.py
# → Running on http://localhost:5000
```

### 4. Load the extension in Chrome
1. Go to `chrome://extensions/`
2. Enable **Developer Mode** (top-right toggle)
3. Click **Load unpacked** → select the `extension/` folder
4. Pin **PhishGuard** to your toolbar ✓

---

## Model Details

| Feature Type | Details | Count |
|---|---|---|
| Char TF-IDF | 3–5 char n-grams | 60,000 |
| Word TF-IDF | 1–2 word n-grams | 30,000 |
| Hand-crafted | URL length, entropy, IP detection, TLD, brand keywords… | 21 |

| Metric | Score |
|---|---|
| Accuracy | 94.69% |
| Precision | 94.98% |
| Recall | 85.89% |
| F1 Score | 90.21% |
| AUC-ROC | 98.84% |

---

## Dataset

[Phishing Site URLs — Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)  
549,346 URLs — 156,422 phishing / 392,924 safe

---

## Tech Stack

- **ML:** scikit-learn (Multinomial Naive Bayes, TF-IDF)
- **API:** Flask + Flask-CORS
- **Extension:** Chrome Manifest V3 (Vanilla JS)
- **Language:** Python 3.11+
