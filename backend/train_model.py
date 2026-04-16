"""
train_model.py  (v3 — Multi-Model)
------------------------------------
Features:
  1. Dual TF-IDF  — char n-grams (3-5) + word n-grams (1-2) on tokenized URL
  2. 21 hand-crafted URL features (entropy, subdomain depth, brand keywords, etc.)
  3. Trains Logistic Regression + Multinomial Naive Bayes + Random Forest
  4. Saves all 3 models individually + best as model.pkl
  5. Generates evaluation plots -> backend/plots/

Run:
    python train_model.py
"""

import os, re, math, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves files without a display
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)

# ==============================================================================
# 1.  LOAD DATASET
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

csv_path = None
try:
    import kagglehub
    print("[*] Attempting Kaggle download ...")
    dpath    = kagglehub.dataset_download("taruntiwarihp/phishing-site-urls")
    csv_path = os.path.join(dpath, "phishing_site_urls.csv")
    print(f"[+] Downloaded to: {csv_path}")
except Exception as e:
    print(f"[!] kagglehub skipped ({e}), checking local copies ...")

if not csv_path or not os.path.exists(csv_path):
    candidates = [
        os.path.join(HERE, "..", "phishing website", "phishing_site_urls.csv"),
        r"C:\Users\HP 840 g5\OneDrive\Documents\Saud\PhishGuard-Extension\phishing website\phishing_site_urls.csv",
        r"C:\Users\HP 840 g5\.cache\kagglehub\datasets\taruntiwarihp\phishing-site-urls\versions\1\phishing_site_urls.csv",
        r"C:\Users\HP\Desktop\phishing website\phishing_site_urls.csv",
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)

if not csv_path:
    raise FileNotFoundError("Dataset CSV not found. Place phishing_site_urls.csv in the project folder.")

print(f"[*] Loading: {csv_path}")
df = pd.read_csv(csv_path)
df.dropna(subset=["URL", "Label"], inplace=True)
df["label_bin"] = (df["Label"].str.lower() == "bad").astype(int)
print(f"[*] Rows: {len(df):,}  |  Phishing: {df.label_bin.sum():,}  |  Safe: {(~df.label_bin.astype(bool)).sum():,}")


# ==============================================================================
# 2.  FEATURE ENGINEERING
# ==============================================================================

SHORTENERS = re.compile(
    r"(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|buff\.ly|"
    r"adf\.ly|tiny\.cc|is\.gd|cli\.gs|bc\.vc|short\.io)", re.I
)
SUSPICIOUS_TLDS = re.compile(
    r"\.(xyz|tk|ml|ga|cf|gq|pw|top|club|online|site|info|"
    r"link|live|stream|download|win|racing|date|review)(\b|/|$)", re.I
)
BRAND_KEYWORDS = re.compile(
    r"(paypal|amazon|apple|microsoft|google|facebook|instagram|"
    r"netflix|ebay|bank|secure|login|verify|account|update|confirm)", re.I
)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    probs = [v / len(s) for v in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(url: str) -> list:
    url  = str(url)
    lo   = url.lower()
    no_scheme    = re.sub(r"^https?://", "", lo)
    parts        = no_scheme.split("/", 1)
    host         = parts[0]
    path         = parts[1] if len(parts) > 1 else ""
    host_no_port = host.split(":")[0]
    subdomains   = host_no_port.split(".")
    query        = path.split("?", 1)[1] if "?" in path else ""

    return [
        len(url),                                                               #  1 total length
        len(host_no_port),                                                      #  2 host length
        len(path),                                                              #  3 path length
        url.count("."),                                                         #  4 dots
        url.count("-"),                                                         #  5 hyphens
        url.count("@"),                                                         #  6 @ signs
        url.count("/"),                                                         #  7 slashes
        url.count("?"),                                                         #  8 question marks
        url.count("="),                                                         #  9 equals
        url.count("%"),                                                         # 10 hex encodings
        url.count("_"),                                                         # 11 underscores
        sum(c.isdigit() for c in host_no_port),                               # 12 digits in host
        max(0, len(subdomains) - 2),                                            # 13 subdomain depth
        path.count("/"),                                                        # 14 path depth
        len(query.split("&")) if query else 0,                                 # 15 query params
        1 if lo.startswith("https") else 0,                                    # 16 has HTTPS
        1 if re.search(r"\d{1,3}(\.\d{1,3}){3}", host_no_port) else 0,       # 17 IP in host
        1 if SHORTENERS.search(host_no_port) else 0,                           # 18 URL shortener
        1 if SUSPICIOUS_TLDS.search(lo) else 0,                                # 19 suspicious TLD
        1 if BRAND_KEYWORDS.search(host_no_port) else 0,                       # 20 brand keyword in host
        round(_entropy(host_no_port), 4),                                      # 21 host entropy
    ]


def tokenize_url(url: str) -> str:
    return " ".join(re.split(r"[^a-z0-9]", str(url).lower()))


print("[*] Extracting hand-crafted features ...")
hand_feats = np.array([extract_features(u) for u in df["URL"]], dtype=float)

print("[*] Tokenising URLs for TF-IDF ...")
df["tokenized"] = df["URL"].map(tokenize_url)


# ==============================================================================
# 3.  TRAIN / TEST SPLIT
# ==============================================================================
y = df["label_bin"].values

(X_tok_tr, X_tok_te,
 y_tr,     y_te,
 hf_tr,    hf_te) = train_test_split(
    df["tokenized"].values, y, hand_feats,
    test_size=0.2, random_state=42, stratify=y
)


# ==============================================================================
# 4.  DUAL TF-IDF  (char + word n-grams)
# ==============================================================================
print("[*] Fitting char-level TF-IDF (3-5 grams, 60K feats) ...")
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              max_features=60_000, sublinear_tf=True)
char_tr = char_tfidf.fit_transform(X_tok_tr)
char_te = char_tfidf.transform(X_tok_te)

print("[*] Fitting word-level TF-IDF (1-2 grams, 30K feats) ...")
word_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                              max_features=30_000, sublinear_tf=True)
word_tr = word_tfidf.fit_transform(X_tok_tr)
word_te = word_tfidf.transform(X_tok_te)

X_tr = hstack([char_tr, word_tr, csr_matrix(hf_tr)])
X_te = hstack([char_te, word_te, csr_matrix(hf_te)])
print(f"[*] Combined feature matrix: {X_tr.shape[1]:,} features")


# ==============================================================================
# 5.  TRAIN & EVALUATE
# ==============================================================================
results = {}

print("\n[*] Training Logistic Regression ...")
lr = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
lr.fit(X_tr, y_tr)
lr_preds = lr.predict(X_te)
lr_acc   = accuracy_score(y_te, lr_preds)
results["LogisticRegression"] = (lr, lr_acc)
print(f"    Accuracy: {lr_acc:.4f}")
print(classification_report(y_te, lr_preds, target_names=["safe", "phishing"]))

print("[*] Training Multinomial Naive Bayes ...")
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_tr, y_tr)
mnb_preds = mnb.predict(X_te)
mnb_acc   = accuracy_score(y_te, mnb_preds)
results["MultinomialNB"] = (mnb, mnb_acc)
print(f"    Accuracy: {mnb_acc:.4f}")
print(classification_report(y_te, mnb_preds, target_names=["safe", "phishing"]))

print("[*] Training Random Forest (100 trees, n_jobs=-1) ...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
rf.fit(X_tr, y_tr)
rf_preds = rf.predict(X_te)
rf_acc   = accuracy_score(y_te, rf_preds)
results["RandomForest"] = (rf, rf_acc)
print(f"    Accuracy: {rf_acc:.4f}")
print(classification_report(y_te, rf_preds, target_names=["safe", "phishing"]))


# ==============================================================================
# 6.  SAVE ALL MODELS + BEST
# ==============================================================================
best_name, (best_model, best_acc) = max(results.items(), key=lambda x: x[1][1])
print(f"\n[OK] Best model: {best_name}  (accuracy = {best_acc:.4f})")

# Save vectorizers (shared by all models)
joblib.dump(char_tfidf, os.path.join(HERE, "char_vectorizer.pkl"))
joblib.dump(word_tfidf, os.path.join(HERE, "word_vectorizer.pkl"))

# Save each model individually
joblib.dump(mnb, os.path.join(HERE, "model.pkl"))           # default / best
joblib.dump(mnb, os.path.join(HERE, "model_mnb.pkl"))       # explicit MNB
joblib.dump(lr,  os.path.join(HERE, "model_lr.pkl"))        # Logistic Regression
joblib.dump(rf,  os.path.join(HERE, "model_rf.pkl"))        # Random Forest
print(f"[OK] model.pkl, model_mnb.pkl, model_lr.pkl, model_rf.pkl saved -> {HERE}")
print(f"[OK] char_vectorizer.pkl, word_vectorizer.pkl saved -> {HERE}")


# ==============================================================================
# 7.  EVALUATION PLOTS
# ==============================================================================
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = ["#1565c0", "#6a1b9a", "#00695c"]
CLASSES = ["safe", "phishing"]

# -- helper: dark axes --
def dark_ax(ax, title):
    ax.set_facecolor("#161b22")
    ax.set_title(title, fontsize=12, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("white")


# 7a. Confusion matrices side-by-side
print("[*] Generating confusion matrix plot ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
fig.suptitle("Confusion Matrices — Phishing URL Classifier",
             fontsize=14, fontweight="bold", color="white")
for ax, (name, (mdl, _)) in zip(axes, results.items()):
    preds = mdl.predict(X_te)
    cm    = confusion_matrix(y_te, preds)
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    dark_ax(ax, f"{name}\nTP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for t in disp.text_.ravel():
        t.set_color("white"); t.set_fontsize(12)
fig.patch.set_facecolor("#0d1117")
plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"    Saved -> {cm_path}")

# 7b. ROC Curves
print("[*] Generating ROC curve plot ...")
fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
for (name, (mdl, _)), color in zip(results.items(), COLORS):
    proba = mdl.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name}  (AUC={auc(fpr,tpr):.4f})")
ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate", color="white")
ax.set_ylabel("True Positive Rate", color="white")
ax.legend(facecolor="#0d1117", labelcolor="white")
dark_ax(ax, "ROC Curve — Phishing vs Safe")
fig.patch.set_facecolor("#0d1117")
plt.tight_layout()
roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"    Saved -> {roc_path}")

# 7c. Precision-Recall
print("[*] Generating precision-recall plot ...")
fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
for (name, (mdl, _)), color in zip(results.items(), COLORS):
    proba = mdl.predict_proba(X_te)[:, 1]
    prec, rec, _ = precision_recall_curve(y_te, proba)
    ap = average_precision_score(y_te, proba)
    ax.plot(rec, prec, color=color, lw=2.5, label=f"{name}  (AP={ap:.4f})")
ax.set_xlabel("Recall", color="white")
ax.set_ylabel("Precision", color="white")
ax.legend(facecolor="#0d1117", labelcolor="white")
dark_ax(ax, "Precision-Recall Curve — Phishing Detection")
fig.patch.set_facecolor("#0d1117")
plt.tight_layout()
pr_path = os.path.join(PLOTS_DIR, "precision_recall.png")
plt.savefig(pr_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"    Saved -> {pr_path}")

# 7d. Model Comparison Bar Chart
print("[*] Generating model comparison bar chart ...")
metrics_data = {}
for name, (mdl, _) in results.items():
    preds = mdl.predict(X_te)
    proba = mdl.predict_proba(X_te)[:, 1]
    fpr2, tpr2, _ = roc_curve(y_te, proba)
    metrics_data[name] = {
        "Accuracy":  accuracy_score(y_te, preds),
        "Precision": precision_score(y_te, preds),
        "Recall":    recall_score(y_te, preds),
        "F1 Score":  f1_score(y_te, preds),
        "AUC-ROC":   auc(fpr2, tpr2),
    }

metric_names = list(list(metrics_data.values())[0].keys())
x = np.arange(len(metric_names))
n_models = len(results)
w = 0.25
fig, ax = plt.subplots(figsize=(13, 5), facecolor="#0d1117")
offsets = np.linspace(-(n_models-1)*w/2, (n_models-1)*w/2, n_models)
bars_all = []
for (name, _), offset, color in zip(results.items(), offsets, COLORS):
    bars = ax.bar(x + offset, [metrics_data[name][m] for m in metric_names],
                  w, label=name, color=color, alpha=0.9)
    bars_all.append(bars)
for bars in bars_all:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", color="white", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0.80, 1.04)
ax.set_ylabel("Score", color="white")
ax.legend(facecolor="#0d1117", labelcolor="white")
dark_ax(ax, "Model Performance Comparison — LR vs MNB vs Random Forest")
fig.patch.set_facecolor("#0d1117")
plt.tight_layout()
cmp_path = os.path.join(PLOTS_DIR, "model_comparison.png")
plt.savefig(cmp_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"    Saved -> {cmp_path}")

# Summary
print("\n" + "=" * 60)
print("  EVALUATION SUMMARY")
print("=" * 60)
for name, mdict in metrics_data.items():
    print(f"\n  {name}")
    for k, v in mdict.items():
        print(f"    {k:<12}: {v:.4f}")
print("=" * 60)
print(f"\n[OK] All plots saved to: {PLOTS_DIR}")
print("Done! Run:  python app.py  to start the API.")
