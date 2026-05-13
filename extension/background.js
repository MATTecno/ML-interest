/**
 * background.js — service worker da extensão.
 * Recebe mensagens do content script e roteia para o servidor Python local.
 */

const SERVER_BASE = "http://localhost:5043";
const CLOUD_DEFAULTS = {
  cloudUrl: "http://127.0.0.1:8080",
  deviceId: "local-pc",
  token: "",
  cloudEnabled: false,
  autoswipeEnabled: false,
};
const NETWORK_CAPTURE_ENABLED = true;
const NETWORK_CAPTURE_FLUSH_MS = 1000;
const NETWORK_CAPTURE_MAX_QUEUE = 1000;
const EXTENSION_DEBUG_LOGS = false;
let _networkCaptureQueue = [];
let _networkCaptureTimer = null;
const _webRequestStarted = new Map();
let _cloudSessionId = "";
let _cloudModeEnabledCache = false;

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
      _handleTinderRecs(message.payload);
      break;

    case "PROFILE_VISIBLE":
      // Perfil atualmente visível na tela → sincroniza com o swiper
      _handleProfileVisible({
        name: message.name,
        tinder_id: message.tinder_id || "",
        age: message.age || 0,
        super_like_available: message.super_like_available,
        super_like_reason: message.super_like_reason || "",
      });
      break;

    case "NETWORK_CAPTURE":
      _handleNetworkCapture({
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

function _cloudSession() {
  if (!_cloudSessionId) {
    _cloudSessionId =
      "ext_" +
      Date.now().toString(36) +
      "_" +
      Math.random().toString(36).slice(2, 10);
  }
  return _cloudSessionId;
}

function _settings() {
  return chrome.storage.sync.get(CLOUD_DEFAULTS).then((saved) => ({
    cloudUrl: String(saved.cloudUrl || CLOUD_DEFAULTS.cloudUrl).replace(/\/+$/, ""),
    deviceId: String(saved.deviceId || CLOUD_DEFAULTS.deviceId),
    token: String(saved.token || ""),
    cloudEnabled: Boolean(saved.cloudEnabled),
    autoswipeEnabled: Boolean(saved.autoswipeEnabled),
  })).then((cfg) => {
    _cloudModeEnabledCache = cfg.cloudEnabled;
    return cfg;
  });
}

chrome.storage.sync.get(CLOUD_DEFAULTS)
  .then((saved) => {
    _cloudModeEnabledCache = Boolean(saved.cloudEnabled);
  })
  .catch(() => {});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.cloudEnabled) {
    _cloudModeEnabledCache = Boolean(changes.cloudEnabled.newValue);
  }
});

async function _handleTinderRecs(payload) {
  const cfg = await _settings();
  if (!cfg.cloudEnabled) {
    _post("/profiles", payload);
    return;
  }
  const profiles = _profilesForCloud(payload);
  if (!profiles.length) return;
  for (const profile of profiles) {
    await _postCloud("/api/profiles", {
      session_id: _cloudSession(),
      device_id: cfg.deviceId,
      autoswipe_enabled: cfg.autoswipeEnabled,
      profile,
    }, cfg);
  }
}

async function _handleProfileVisible(visible) {
  const cfg = await _settings();
  if (!cfg.cloudEnabled) {
    _post("/current", visible);
    return;
  }
  await _postCloud("/api/commands", {
    device_id: cfg.deviceId,
    session_id: _cloudSession(),
    command: "set_current",
    visible_profile: visible,
  }, cfg);
}

async function _handleNetworkCapture(event) {
  const cfg = await _settings();
  if (cfg.cloudEnabled) {
    _debugLog("[cloud] Network capture bruto ignorado no modo cloud.");
    return;
  }
  _enqueueNetworkCapture(event);
}

function _cloudHeaders(cfg) {
  const headers = { "Content-Type": "application/json" };
  if (cfg.token) headers.Authorization = `Bearer ${cfg.token}`;
  return headers;
}

function _postCloud(path, body, cfg) {
  return fetch(cfg.cloudUrl + path, {
    method: "POST",
    headers: _cloudHeaders(cfg),
    body: JSON.stringify(body || {}),
  })
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json().catch(() => ({}));
    })
    .catch((err) => {
      _debugError(`[cloud] ${path} → erro`, err.message || err);
    });
}

function _calcAge(birthDate) {
  if (!birthDate) return 0;
  try {
    const birth = new Date(birthDate);
    if (Number.isNaN(birth.getTime())) return 0;
    const today = new Date();
    let age = today.getFullYear() - birth.getFullYear();
    const monthDelta = today.getMonth() - birth.getMonth();
    if (monthDelta < 0 || (monthDelta === 0 && today.getDate() < birth.getDate())) age -= 1;
    return age;
  } catch (_) {
    return 0;
  }
}

function _valueText(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map(_valueText).filter(Boolean).join(", ");
  if (typeof value === "object") {
    if (Array.isArray(value.choice_selections)) {
      const text = _valueText(value.choice_selections.map((item) => item?.name || item));
      if (text) return text;
    }
    if (value.measurable_selection && value.measurable_selection.value !== undefined) {
      return `${value.measurable_selection.value} ${value.measurable_selection.unit_of_measure || ""}`.trim();
    }
    for (const key of ["name", "value", "display_value", "display_text", "body_text", "subtitle", "description", "text", "title_text"]) {
      const text = _valueText(value[key]);
      if (text) return text;
    }
  }
  return "";
}

function _extractDescriptors(user) {
  const out = {};
  for (const item of user?.selected_descriptors || []) {
    const key = String(item?.name || item?.section_name || item?.prompt || item?.id || "").trim();
    const value = _valueText(item);
    if (key && value && key !== value) out[key] = value;
  }
  const intent = user?.relationship_intent || {};
  if (intent?.body_text) {
    out[String(intent.title_text || "Objetivo").trim() || "Objetivo"] = String(intent.body_text || "").trim();
  }
  return out;
}

function _profilesForCloud(payload) {
  const results = payload?.data?.results || [];
  const profiles = [];
  for (const result of results) {
    if (result?.type !== "user") continue;
    const user = result.user || {};
    const interests = (result.experiment_info?.user_interests?.selected_interests || [])
      .map((item) => item?.name)
      .filter(Boolean);
    const distanceMi = result.distance_mi;
    const distanceKm = distanceMi === null || distanceMi === undefined || distanceMi === ""
      ? ""
      : Math.round(Number(distanceMi) * 1.609);
    profiles.push({
      profile_id: user._id || result.content_hash || `${user.name || "perfil"}_${Date.now()}`,
      tinder_id: user._id || "",
      name: user.name || "",
      age: _calcAge(user.birth_date || ""),
      distance_km: Number.isFinite(distanceKm) ? distanceKm : "",
      bio: user.bio || "",
      interests,
      descriptors: _extractDescriptors(user),
      captured_at: new Date().toISOString(),
    });
  }
  return profiles;
}

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
  if (_cloudModeEnabledCache) {
    _debugLog("[cloud] Network capture bruto ignorado no modo cloud.");
    return;
  }
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
