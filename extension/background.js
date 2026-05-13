/**
 * background.js — service worker da extensão.
 * Recebe mensagens do content script e roteia para o servidor Python local.
 */

const SERVER_BASE = "http://localhost:5043";
const NETWORK_CAPTURE_ENABLED = true;
const NETWORK_CAPTURE_FLUSH_MS = 1000;
const NETWORK_CAPTURE_MAX_QUEUE = 1000;
const EXTENSION_DEBUG_LOGS = false;
let _networkCaptureQueue = [];
let _networkCaptureTimer = null;
const _webRequestStarted = new Map();

function _debugLog(...args) {
  if (EXTENSION_DEBUG_LOGS) console.log(...args);
}

function _debugError(...args) {
  if (EXTENSION_DEBUG_LOGS) console.error(...args);
}

chrome.runtime.onMessage.addListener((message) => {
  switch (message.type) {

    case "TINDER_RECS":
      // Leva de perfis recebida → envia para classificação
      _post("/profiles", message.payload);
      break;

    case "PROFILE_VISIBLE":
      // Perfil atualmente visível na tela → sincroniza com o swiper
      _post("/current", {
        name: message.name,
        tinder_id: message.tinder_id || "",
        age: message.age || 0,
        super_like_available: message.super_like_available,
        super_like_reason: message.super_like_reason || "",
      });
      break;

    case "NETWORK_CAPTURE":
      _enqueueNetworkCapture({
        source: "page_hook",
        ...message.payload,
      });
      break;

    case "OUT_OF_PROFILES":
      // Apenas log — o reload já é feito pelo content.js
      _debugLog("[local] Sem perfis — recarregando...");
      break;
  }
});

function _isLocalServerUrl(url) {
  return /^https?:\/\/(localhost|127\.0\.0\.1):5043\//i.test(String(url || ""));
}

function _networkCaptureUrlAllowed(url) {
  try {
    const parsed = new URL(String(url || ""));
    const host = parsed.hostname.toLowerCase();
    const path = parsed.pathname || "";
    if (host !== "api.gotinder.com") return false;
    if (path.includes("/recs/") || path.includes("/v2/recs")) return true;
    if (/^\/(?:like|pass)\//.test(path)) return true;
    if (path === "/updates" || path.startsWith("/updates/")) return true;
    if (path === "/v2/profile" || path.startsWith("/v2/profile/")) return true;
    if (path === "/v2/fast-match/teaser" || path.startsWith("/v2/fast-match/teaser/")) return true;
  } catch (_) {}
  return false;
}

function _isTinderInitiated(details) {
  const url = String(details?.url || "");
  const initiator = String(details?.initiator || "");
  return (
    !_isLocalServerUrl(url) &&
    _networkCaptureUrlAllowed(url) &&
    (/(^https?:\/\/([^/]+\.)?tinder\.com\b)/i.test(url) ||
      /(^https?:\/\/([^/]+\.)?gotinder\.com\b)/i.test(url) ||
      /(^https?:\/\/([^/]+\.)?gotinder\.com\b)/i.test(initiator) ||
      /(^https?:\/\/([^/]+\.)?tinder\.com\b)/i.test(initiator))
  );
}

function _headersArrayToObject(headers) {
  const out = {};
  for (const item of headers || []) {
    if (!item || !item.name) continue;
    out[String(item.name).toLowerCase()] = String(item.value || "");
  }
  return out;
}

function _requestBodySummary(requestBody) {
  if (!requestBody) return {};
  try {
    if (requestBody.formData) return { body_json: requestBody.formData, body_type: "form_data" };
    if (requestBody.raw && requestBody.raw.length) {
      const total = requestBody.raw.reduce((sum, item) => sum + (item.bytes?.byteLength || 0), 0);
      return { body_type: "raw_bytes", body_length: total, body_omitted: "webrequest_raw_bytes" };
    }
  } catch (err) {
    return { body_error: String(err?.message || err) };
  }
  return {};
}

function _enqueueNetworkCapture(event) {
  if (!NETWORK_CAPTURE_ENABLED || !event) return;
  _networkCaptureQueue.push({
    extension_received_at: new Date().toISOString(),
    ...event,
  });
  if (_networkCaptureQueue.length > NETWORK_CAPTURE_MAX_QUEUE) {
    _networkCaptureQueue.splice(0, _networkCaptureQueue.length - NETWORK_CAPTURE_MAX_QUEUE);
  }
  if (_networkCaptureTimer) return;
  _networkCaptureTimer = setTimeout(_flushNetworkCapture, NETWORK_CAPTURE_FLUSH_MS);
}

function _flushNetworkCapture() {
  _networkCaptureTimer = null;
  if (!_networkCaptureQueue.length) return;
  const events = _networkCaptureQueue.splice(0, _networkCaptureQueue.length);
  _post("/network-capture", { events });
}

if (chrome.webRequest) {
  chrome.webRequest.onBeforeRequest.addListener(
    (details) => {
      if (!_isTinderInitiated(details)) return;
      const started = {
        source: "webrequest",
        capture_type: "webrequest",
        request_id: details.requestId,
        phase: "request",
        url: details.url,
        method: details.method,
        tab_id: details.tabId,
        frame_id: details.frameId,
        parent_frame_id: details.parentFrameId,
        type: details.type,
        initiator: details.initiator || "",
        time_stamp: details.timeStamp,
        request: _requestBodySummary(details.requestBody),
      };
      _webRequestStarted.set(details.requestId, started);
      _enqueueNetworkCapture(started);
    },
    { urls: ["<all_urls>"] },
    ["requestBody"]
  );

  chrome.webRequest.onCompleted.addListener(
    (details) => {
      if (!_isTinderInitiated(details)) return;
      const started = _webRequestStarted.get(details.requestId) || {};
      _webRequestStarted.delete(details.requestId);
      _enqueueNetworkCapture({
        source: "webrequest",
        capture_type: "webrequest",
        request_id: details.requestId,
        phase: "complete",
        url: details.url,
        method: details.method || started.method || "",
        tab_id: details.tabId,
        frame_id: details.frameId,
        type: details.type,
        initiator: details.initiator || started.initiator || "",
        status: details.statusCode,
        from_cache: Boolean(details.fromCache),
        ip: details.ip || "",
        time_stamp: details.timeStamp,
        duration_ms:
          started.time_stamp && details.timeStamp ? Math.round(details.timeStamp - started.time_stamp) : undefined,
        response_headers: _headersArrayToObject(details.responseHeaders),
      });
    },
    { urls: ["<all_urls>"] },
    ["responseHeaders"]
  );

  chrome.webRequest.onErrorOccurred.addListener(
    (details) => {
      if (!_isTinderInitiated(details)) return;
      const started = _webRequestStarted.get(details.requestId) || {};
      _webRequestStarted.delete(details.requestId);
      _enqueueNetworkCapture({
        source: "webrequest",
        capture_type: "webrequest",
        request_id: details.requestId,
        phase: "error",
        url: details.url,
        method: details.method || started.method || "",
        tab_id: details.tabId,
        frame_id: details.frameId,
        type: details.type,
        initiator: details.initiator || started.initiator || "",
        error: details.error || "",
        time_stamp: details.timeStamp,
        duration_ms:
          started.time_stamp && details.timeStamp ? Math.round(details.timeStamp - started.time_stamp) : undefined,
      });
    },
    { urls: ["<all_urls>"] }
  );
}

function _post(path, body) {
  fetch(SERVER_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then((res) => {
      if (!res.ok)
        _debugError(`[local] ${path} → erro ${res.status}`);
    })
    .catch((err) => {
      if (path === "/profiles") {
        _debugError(
          "[local] Servidor offline. Rode: python3 src/server.py",
          err.message
        );
      }
    });
}
