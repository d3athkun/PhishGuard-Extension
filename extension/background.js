/**
 * background.js  (Service Worker — Manifest V3)
 * -----------------------------------------------
 * - Listens for tab URL changes.
 * - Calls the Flask /predict API.
 * - Updates the extension badge & shows a notification when phishing is detected.
 * - Stores the last result per tab so popup.js can read it instantly.
 */

// ── Config ───────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:5000";  // ← change to your Render URL in production

// ── Badge helpers ─────────────────────────────────────────────────────────────
function setBadge(tabId, label, confidence) {
    const isPhishing = label === "phishing";
    chrome.action.setBadgeBackgroundColor({
        tabId,
        color: isPhishing ? "#e53935" : "#43a047",
    });
    chrome.action.setBadgeText({
        tabId,
        text: isPhishing ? "⚠" : "✓",
    });
    chrome.action.setTitle({
        tabId,
        title: isPhishing
            ? `PhishGuard ⚠ PHISHING DETECTED (${Math.round(confidence * 100)}% confidence)`
            : `PhishGuard ✓ Safe (${Math.round(confidence * 100)}% confidence)`,
    });
}

function setBadgeOffline(tabId) {
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#9e9e9e" });
    chrome.action.setBadgeText({ tabId, text: "?" });
    chrome.action.setTitle({ tabId, title: "PhishGuard — API offline" });
}

function clearBadge(tabId) {
    chrome.action.setBadgeText({ tabId, text: "" });
    chrome.action.setTitle({ tabId, title: "PhishGuard" });
}

// ── Notification ──────────────────────────────────────────────────────────────
function notifyPhishing(url, confidence) {
    chrome.notifications.create({
        type: "basic",
        iconUrl: "icons/icon128.png",
        title: "⚠ Phishing Website Detected!",
        message: `This site looks dangerous (${Math.round(confidence * 100)}% confidence).\n${url}`,
        priority: 2,
    });
}

// ── Core scan function ────────────────────────────────────────────────────────
async function scanUrl(tabId, url) {
    // Skip non-http pages (chrome://, about:, file:// etc.)
    if (!url || (!url.startsWith("http://") && !url.startsWith("https://"))) {
        clearBadge(tabId);
        return;
    }

    // Show loading state
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#1565c0" });
    chrome.action.setBadgeText({ tabId, text: "…" });

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const result = await response.json();

        // Cache result so popup can read it without making another API call
        chrome.storage.local.set({ [`tab_${tabId}`]: result });

        setBadge(tabId, result.label, result.confidence);

        if (result.label === "phishing") {
            notifyPhishing(url, result.confidence);
        }
    } catch (err) {
        console.warn("[PhishGuard] API unreachable:", err.message);
        setBadgeOffline(tabId);
        chrome.storage.local.set({ [`tab_${tabId}`]: { label: "offline", confidence: 0, url } });
    }
}

// ── Listeners ─────────────────────────────────────────────────────────────────
// Fires when a tab finishes loading
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url) {
        scanUrl(tabId, tab.url);
    }
});

// Clean up storage when a tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
    chrome.storage.local.remove(`tab_${tabId}`);
});

// Listen for manual scan requests from popup.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "MANUAL_SCAN") {
        fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: message.url }),
        })
            .then((r) => r.json())
            .then((result) => sendResponse({ ok: true, result }))
            .catch((err) => sendResponse({ ok: false, error: err.message }));
        return true; // keep message channel open for async response
    }

    if (message.type === "GET_API_BASE") {
        sendResponse({ apiBase: API_BASE });
        return false;
    }
});
