/**
 * popup.js
 * ---------
 * Handles both the auto-scan result display (reads cached storage)
 * and the manual URL check (sends message to background.js).
 */

const API_BASE = "http://localhost:5000";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const autoIcon = document.getElementById("autoIcon");
const autoLabel = document.getElementById("autoLabel");
const autoUrl = document.getElementById("autoUrl");
const autoBarWrap = document.getElementById("autoBarWrap");
const autoBarFill = document.getElementById("autoBarFill");
const autoConfValue = document.getElementById("autoConfValue");

const urlInput = document.getElementById("urlInput");
const checkBtn = document.getElementById("checkBtn");
const manualResultWrap = document.getElementById("manualResultWrap");
const manualLoading = document.getElementById("manualLoading");
const manualIcon = document.getElementById("manualIcon");
const manualLabel = document.getElementById("manualLabel");
const manualUrl = document.getElementById("manualUrl");
const manualBarFill = document.getElementById("manualBarFill");
const manualConfValue = document.getElementById("manualConfValue");
const apiDot = document.getElementById("apiDot");

// ── SVG icons ─────────────────────────────────────────────────────────────────
const ICON_SAFE = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2L3 7V12C3 16.55 7.08 20.74 12 22C16.92 20.74 21 16.55 21 12V7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
  <path d="M9 12L11 14L15 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

const ICON_PHISHING = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2L3 7V12C3 16.55 7.08 20.74 12 22C16.92 20.74 21 16.55 21 12V7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
  <path d="M12 8V12M12 16H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
</svg>`;

const ICON_OFFLINE = `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><path d="M8 12h8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;

const ICON_SCANNING = `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" stroke-dasharray="4 2"/></svg>`;

// ── Render helpers ─────────────────────────────────────────────────────────────
function renderResult({ iconEl, labelEl, urlEl, barFillEl, confEl, barWrapEl }, result) {
    const isPhishing = result.label === "phishing";
    const isOffline = result.label === "offline";
    const pct = Math.round(result.confidence * 100);

    iconEl.innerHTML = isPhishing ? ICON_PHISHING : isOffline ? ICON_OFFLINE : ICON_SAFE;
    iconEl.className = `result-icon ${isPhishing ? "danger" : isOffline ? "offline" : "safe"}`;

    labelEl.textContent = isPhishing ? "⚠ Phishing Detected"
        : isOffline ? "API Offline"
            : "✓ Safe";
    labelEl.className = `result-label ${isPhishing ? "danger" : isOffline ? "" : "safe"}`;

    if (urlEl && result.url) {
        urlEl.textContent = result.url.length > 40
            ? result.url.slice(0, 37) + "…"
            : result.url;
    }

    if (!isOffline && barWrapEl) {
        barWrapEl.style.display = "flex";
        barFillEl.style.width = `${pct}%`;
        barFillEl.className = `confidence-bar-fill ${isPhishing ? "danger" : "safe"}`;
        confEl.textContent = `${pct}%`;
    }
}

// ── Check API health ───────────────────────────────────────────────────────────
async function checkApiHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
        apiDot.className = res.ok ? "status-dot online" : "status-dot offline";
    } catch {
        apiDot.className = "status-dot offline";
    }
}

// ── Auto-scan: load from storage ───────────────────────────────────────────────
async function loadAutoScan() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) return;

    autoUrl.textContent = tab.url?.length > 40 ? tab.url.slice(0, 37) + "…" : tab.url || "";

    chrome.storage.local.get(`tab_${tab.id}`, (data) => {
        const result = data[`tab_${tab.id}`];
        if (!result) {
            autoIcon.innerHTML = ICON_SCANNING;
            autoIcon.className = "result-icon loading";
            autoLabel.textContent = "Scanning…";
            return;
        }
        renderResult({
            iconEl: autoIcon,
            labelEl: autoLabel,
            urlEl: null,
            barFillEl: autoBarFill,
            confEl: autoConfValue,
            barWrapEl: autoBarWrap,
        }, result);
    });
}

// ── Manual check ───────────────────────────────────────────────────────────────
async function doManualCheck() {
    let url = urlInput.value.trim();
    if (!url) return;
    if (!url.startsWith("http://") && !url.startsWith("https://")) url = "http://" + url;

    checkBtn.disabled = true;
    manualLoading.style.display = "flex";
    manualResultWrap.style.display = "none";

    chrome.runtime.sendMessage({ type: "MANUAL_SCAN", url }, (response) => {
        manualLoading.style.display = "none";
        checkBtn.disabled = false;

        if (!response || !response.ok) {
            manualResultWrap.style.display = "block";
            renderResult({
                iconEl: manualIcon, labelEl: manualLabel, urlEl: manualUrl,
                barFillEl: manualBarFill, confEl: manualConfValue, barWrapEl: null,
            }, { label: "offline", confidence: 0, url });
            return;
        }

        manualResultWrap.style.display = "block";
        renderResult({
            iconEl: manualIcon, labelEl: manualLabel, urlEl: manualUrl,
            barFillEl: manualBarFill, confEl: manualConfValue,
            barWrapEl: manualResultWrap.querySelector(".confidence-bar-wrap"),
        }, response.result);
    });
}

// ── Event listeners ────────────────────────────────────────────────────────────
checkBtn.addEventListener("click", doManualCheck);
urlInput.addEventListener("keydown", (e) => { if (e.key === "Enter") doManualCheck(); });

// ── Init ───────────────────────────────────────────────────────────────────────
checkApiHealth();
loadAutoScan();
