/**
 * app.js — PhishGuard Web Frontend
 * Handles model selection, API calls, and result rendering.
 */

const API_BASE = "http://localhost:5000"; // Update to Render.com URL after deployment

// ── DOM refs ──────────────────────────────────────────────────────────────────
const urlInput    = document.getElementById("urlInput");
const checkBtn    = document.getElementById("checkBtn");
const checkText   = document.getElementById("checkBtnText");
const checkSpinner= document.getElementById("checkSpinner");
const clearBtn    = document.getElementById("clearBtn");
const resultArea  = document.getElementById("resultArea");

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(on) {
  checkBtn.disabled   = on;
  checkText.style.display  = on ? "none" : "";
  checkSpinner.style.display = on ? "inline-block" : "none";
}

function getSelectedModel() {
  return document.querySelector('input[name="model"]:checked').value;
}

urlInput.addEventListener("input", () => {
  clearBtn.style.display = urlInput.value ? "block" : "none";
});
clearBtn.addEventListener("click", () => {
  urlInput.value = "";
  clearBtn.style.display = "none";
  resultArea.innerHTML = "";
  urlInput.focus();
});

// ── Run on Enter key ──────────────────────────────────────────────────────────
urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") checkBtn.click();
});

// ── Also allow Enter from anywhere in checker ─────────────────────────────────
checkBtn.addEventListener("click", runCheck);

// ── Main logic ────────────────────────────────────────────────────────────────
async function runCheck() {
  const raw = urlInput.value.trim();
  if (!raw) { urlInput.focus(); return; }

  const model = getSelectedModel();
  setLoading(true);
  resultArea.innerHTML = "";

  try {
    if (model === "all") {
      await runAll(raw);
    } else {
      await runSingle(raw, model);
    }
  } catch (err) {
    showError("Could not reach the PhishGuard API. Make sure the backend is running on localhost:5000.");
    console.error(err);
  } finally {
    setLoading(false);
  }
}

// ── Single model ──────────────────────────────────────────────────────────────
async function runSingle(url, model) {
  const res  = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, model }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || "API error");
  renderSingle(data);
}

function renderSingle(data) {
  const { label, confidence, url, model_name } = data;
  const isSafe    = label === "safe";
  const pct       = Math.round(confidence * 100);
  const icon      = isSafe ? "✅" : "🚨";
  const verdict   = isSafe ? "Safe" : "Phishing Detected";
  const cls       = isSafe ? "safe" : "phishing";

  resultArea.innerHTML = `
    <div class="result-card ${cls}">
      <div class="result-icon">${icon}</div>
      <div class="result-body">
        <div class="result-verdict">${verdict}</div>
        <div class="result-model-tag">via ${escHtml(model_name)}</div>
        <div class="result-url" title="${escHtml(url)}">${escHtml(url)}</div>
        <div class="conf-bar-wrap">
          <div class="conf-label">
            <span>Confidence</span>
            <span>${pct}%</span>
          </div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width: 0%"></div>
          </div>
        </div>
      </div>
    </div>`;

  // Animate the bar after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      resultArea.querySelector(".conf-bar-fill").style.width = pct + "%";
    });
  });
}

// ── Run All ───────────────────────────────────────────────────────────────────
async function runAll(url) {
  const res  = await fetch(`${API_BASE}/predict-all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || "API error");
  renderAll(data);
}

function renderAll(data) {
  const { url, results } = data;
  const order   = ["mnb", "rf", "lr"];
  const icons   = { mnb: "🧠", rf: "🌲", lr: "📈" };

  // Tally votes
  const votes   = Object.values(results).map(r => r.label);
  const phCount = votes.filter(v => v === "phishing").length;
  const safeCount = votes.length - phCount;
  const majority  = phCount >= safeCount ? "phishing" : "safe";
  const summaryIcon = majority === "phishing" ? "⚠️" : "✅";
  const summaryTxt  = majority === "phishing"
    ? `${phCount} of ${votes.length} models flagged this URL as <strong style="color:var(--red)">phishing</strong>.`
    : `${safeCount} of ${votes.length} models consider this URL <strong style="color:var(--green)">safe</strong>.`;

  const cards = order.filter(k => results[k]).map(key => {
    const r       = results[key];
    const isSafe  = r.label === "safe";
    const pct     = Math.round(r.confidence * 100);
    const cls     = isSafe ? "safe" : "phishing";
    const verdict = isSafe ? "Safe" : "Phishing";
    const icon    = isSafe ? "✅" : "🚨";
    return `
      <div class="ra-card ${cls}" data-pct="${pct}">
        <div class="ra-icon">${icons[key]}</div>
        <div class="ra-model-name">${escHtml(r.model_name)}</div>
        <div class="ra-verdict">${icon} ${verdict}</div>
        <div class="ra-conf">${pct}% confidence</div>
        <div class="ra-bar-bg">
          <div class="ra-bar-fill" style="width: 0%"></div>
        </div>
      </div>`;
  }).join("");

  resultArea.innerHTML = `
    <div class="run-all-wrap">
      <div class="run-all-title">Results for: <code style="color:var(--accent-1)">${escHtml(url)}</code></div>
      <div class="run-all-grid">${cards}</div>
      <div class="ra-summary">${summaryIcon} ${summaryTxt}</div>
    </div>`;

  // Animate bars
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      resultArea.querySelectorAll(".ra-bar-fill").forEach(bar => {
        const pct = bar.closest(".ra-card").dataset.pct;
        bar.style.width = pct + "%";
      });
    });
  });
}

// ── Error ─────────────────────────────────────────────────────────────────────
function showError(msg) {
  resultArea.innerHTML = `<div class="result-error">⚠️ ${escHtml(msg)}</div>`;
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
