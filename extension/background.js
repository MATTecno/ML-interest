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
const CLOUD_PHOTO_MAX_PER_PROFILE = 3;
const CLOUD_PHOTO_MAX_BYTES = 2 * 1024 * 1024;
const CLOUD_PHOTO_FETCH_TIMEOUT_MS = 8000;
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
  const items = _profilesForCloud(payload);
  if (!items.length) return;
  for (const item of items) {
    const photos = await _fetchPhotosForCloud(item.photoUrls);
    const body = {
      session_id: _cloudSession(),
      device_id: cfg.deviceId,
      autoswipe_enabled: cfg.autoswipeEnabled,
      profile: item.profile,
    };
    if (photos.length) body.photos = photos;
    await _postCloud("/api/profiles", body, cfg);
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

function _photoArea(item) {
  const width = Number(item?.width || item?.w || 0);
  const height = Number(item?.height || item?.h || 0);
  return width * height;
}

function _photoUrlsForCloud(result) {
  const urls = [];
  const seen = new Set();
  const pushUrl = (url) => {
    const clean = String(url || "").trim();
    if (!clean || seen.has(clean)) return;
    seen.add(clean);
    urls.push(clean);
  };

  for (const photo of result?.user?.photos || []) {
    const processed = Array.isArray(photo?.processedFiles)
      ? [...photo.processedFiles].sort((a, b) => _photoArea(b) - _photoArea(a))
      : [];
    for (const item of processed) pushUrl(item?.url);
    pushUrl(photo?.url);
    if (urls.length >= CLOUD_PHOTO_MAX_PER_PROFILE) break;
  }
  return urls.slice(0, CLOUD_PHOTO_MAX_PER_PROFILE);
}

function _arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

async function _fetchPhotoForCloud(url, index) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CLOUD_PHOTO_FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      credentials: "include",
      signal: controller.signal,
      headers: { Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8" },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const mime = String(response.headers.get("content-type") || "image/jpeg")
      .split(";")[0]
      .trim()
      .toLowerCase();
    if (!mime.startsWith("image/")) throw new Error(`conteudo nao e imagem: ${mime}`);

    const expectedBytes = Number(response.headers.get("content-length") || 0);
    if (expectedBytes > CLOUD_PHOTO_MAX_BYTES) {
      throw new Error(`foto acima do limite: ${expectedBytes} bytes`);
    }

    const buffer = await response.arrayBuffer();
    if (buffer.byteLength > CLOUD_PHOTO_MAX_BYTES) {
      throw new Error(`foto acima do limite: ${buffer.byteLength} bytes`);
    }

    return {
      index,
      mime,
      content_base64: _arrayBufferToBase64(buffer),
    };
  } catch (err) {
    _debugError("[cloud] Foto ignorada no upload:", err?.message || err);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function _fetchPhotosForCloud(urls) {
  const limited = (urls || []).slice(0, CLOUD_PHOTO_MAX_PER_PROFILE);
  const photos = await Promise.all(limited.map((url, index) => _fetchPhotoForCloud(url, index)));
  return photos.filter(Boolean);
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
      profile: {
        profile_id: user._id || result.content_hash || `${user.name || "perfil"}_${Date.now()}`,
        tinder_id: user._id || "",
        name: user.name || "",
        age: _calcAge(user.birth_date || ""),
        distance_km: Number.isFinite(distanceKm) ? distanceKm : "",
        bio: user.bio || "",
        interests,
        descriptors: _extractDescriptors(user),
        captured_at: new Date().toISOString(),
      },
      photoUrls: _photoUrlsForCloud(result),
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
