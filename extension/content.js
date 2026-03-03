/**
 * content.js
 * ----------
 * Injected into every http/https page.
 * Sends the current page URL to the background service worker for scanning.
 * Also listens for messages from the background to update any in-page UI (future).
 */

(function () {
    "use strict";

    const currentUrl = window.location.href;

    // Notify background that this page has loaded (used as redundancy to tabs.onUpdated)
    chrome.runtime.sendMessage({
        type: "PAGE_LOADED",
        url: currentUrl,
    });
})();
