/**
 * background.js — service worker da extensão.
 * Recebe mensagens do content script e roteia para o servidor Python local.
 */

const SERVER_BASE = "http://localhost:5043";
const NETWORK_CAPTURE_ENABLED = true;
const NETWORK_CAPTURE_FLUSH_MS = 1000;
const NETWORK_CAPTURE_MAX_QUEUE = 1000;
const WEBREQUEST_CAPTURE_REQUEST_BODY = false;
const CONTROL_POLL_MS = 10000;
const CONTROL_RETRY_SAME_COMMAND_MS = 5 * 60 * 1000;

const ENABLE_BACKGROUND_CONTROL_NAVIGATE = true;
const ENABLE_BACKGROUND_NON_SYNC_RELOAD = false;
const ENABLE_BACKGROUND_SYNC_RELOAD = false;
const ENABLE_BACKGROUND_STALE_RELOAD = true;
const BACKGROUND_STALE_RELOAD_AFTER_MS = 90 * 1000;

const SYNC_RELOAD_DELAY_MS = 3 * 60 * 1000;
const SYNC_RELOAD_REPEAT_COOLDOWN_MS = 5 * 60 * 1000;

const TINDER_TAB_URLS = ["*://tinder.com/*", "*://*.tinder.com/*"];
const EXTENSION_DEBUG_LOGS = false;

let _networkCaptureQueue = [];
let _networkCaptureTimer = null;
let _controlPollTimer = null;
let _lastControlCommandKey = "";
let _lastControlCommandAt = 0;

let _syncReloadPendingKey = "";
let _syncReloadFirstSeenAt = 0;
let _lastSyncReloadAppliedAt = 0;
const _pendingReloadTabs = new Map();
const _extensionLogLast = new Map();

const _webRequestStarted = new Map();

function _debugLog(...args) {
  if (EXTENSION_DEBUG_LOGS) console.log(...args);
}

function _debugError(...args) {
  if (EXTENSION_DEBUG_LOGS) console.error(...args);
}

function _postExtensionLog(event, message, details = {}, level = "info", throttleMs = 0) {
  try {
    if (throttleMs > 0) {
      const key = `${event}|${message}`;
      const now = Date.now();
      const last = _extensionLogLast.get(key) || 0;
      if (now - last < throttleMs) return;
      _extensionLogLast.set(key, now);
    }
    fetch(SERVER_BASE + "/extension-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: "background",
        event,
        message: String(message || ""),
        level,
        details,
      }),
    }).catch(() => {});
  } catch (_) {}
}

chrome.runtime.onMessage.addListener((message) => {
  _pollControlFromBackground();

  switch (message.type) {
    case "TINDER_RECS":
      _post("/profiles", message.payload);
      break;

    case "PROFILE_VISIBLE":
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

    case "BLOCKING_MODAL":
      _post("/modal", message.payload || {});
      break;

    case "OUT_OF_PROFILES":
      _debugLog("[local] Sem perfis — reload tratado pelo content.js");
      break;
  }
});

chrome.runtime.onStartup?.addListener(() => {
  _startControlPolling();
});

chrome.runtime.onInstalled?.addListener(() => {
  _startControlPolling();
});

chrome.tabs?.onUpdated?.addListener((tabId, changeInfo, tab) => {
  if (changeInfo?.status !== "complete") return;
  if (!_pendingReloadTabs.has(tabId)) return;
  if (!_isTinderTabUrl(tab?.url || "")) return;

  const pending = _pendingReloadTabs.get(tabId) || {};
  _pendingReloadTabs.delete(tabId);
  _resetSyncReloadPending();
  _postExtensionLog("reload_ack_sent", pending.reason || "reload confirmado pelo background", {
    tab_id: tabId,
    mode: pending.mode || "reload",
    command_key: pending.commandKey || "",
    age_ms: pending.startedAt ? Date.now() - Number(pending.startedAt || 0) : 0,
    url: tab?.url || "",
  });
  _post("/reloaded", {
    ok: true,
    source: "background",
    reason: pending.reason || "reload confirmado pelo background",
  });
});

if (chrome.alarms) {
  chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm?.name === "tinder_ia_control_poll") {
      _pollControlFromBackground();
    }
  });
}

_startControlPolling();

function _isLocalServerUrl(url) {
  return /^https?:\/\/(localhost|127\.0\.0\.1):5043\//i.test(String(url || ""));
}

function _isTinderTabUrl(url) {
  try {
    const parsed = new URL(String(url || ""));
    const host = parsed.hostname.toLowerCase();
    return host === "tinder.com" || host.endsWith(".tinder.com");
  } catch (_) {
    return false;
  }
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
    if (requestBody.formData) {
      return {
        body_json: requestBody.formData,
        body_type: "form_data",
      };
    }

    if (requestBody.raw && requestBody.raw.length) {
      const total = requestBody.raw.reduce(
        (sum, item) => sum + (item.bytes?.byteLength || 0),
        0
      );

      return {
        body_type: "raw_bytes",
        body_length: total,
        body_omitted: "webrequest_raw_bytes",
      };
    }
  } catch (err) {
    return {
      body_error: String(err?.message || err),
    };
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

function _startControlPolling() {
  if (!_controlPollTimer) {
    _controlPollTimer = setInterval(_pollControlFromBackground, CONTROL_POLL_MS);
  }

  try {
    chrome.alarms?.create("tinder_ia_control_poll", { periodInMinutes: 0.5 });
  } catch (_) {}

  _pollControlFromBackground();
}

function _controlCommandKey(data) {
  if (!data) return "";

  return [
    data.reload ? "r" : "",
    data.requested_at || 0,
    data.navigate ? "n" : "",
    data.navigate_generation || 0,
    data.navigate_url || "",
  ].join("|");
}

function _controlReasonText(data) {
  return String(data?.navigate_reason || data?.reason || "").toLowerCase();
}

function _isSyncControlCommand(data) {
  const reason = _controlReasonText(data);

  return (
    reason.includes("sincronização") ||
    reason.includes("sincronizacao") ||
    reason.includes("sync") ||
    reason.includes("fora da fila") ||
    reason.includes("perfil visível") ||
    reason.includes("perfil visivel")
  );
}

function _syncReloadStableKey(data) {
  const reason = _controlReasonText(data);

  return [
    data?.reload ? "reload" : "",
    reason,
    data?.navigate ? "navigate" : "",
    data?.navigate_url || "",
  ].join("|");
}

function _resetSyncReloadPending() {
  _syncReloadPendingKey = "";
  _syncReloadFirstSeenAt = 0;
}

function _pollControlFromBackground() {
  fetch(SERVER_BASE + "/control")
    .then((res) => (res.ok ? res.json() : null))
    .then((data) => {
      if (!data || (!data.reload && !data.navigate)) {
        _resetSyncReloadPending();
        return;
      }

      _applyControlCommandToTinderTabs(data);
    })
    .catch((error) => {
      _postExtensionLog("control_poll_error", error?.message || "falha ao consultar /control", {}, "warning");
    });
}

function _applyControlCommandToTinderTabs(data) {
  const now = Date.now();
  const isSyncCommand = _isSyncControlCommand(data);
  const reason = data.navigate_reason || data.reason || "reload solicitado pelo servidor";
  const requestedAtMs = Number(data.requested_at || 0) * 1000;
  const reloadAgeMs = requestedAtMs > 0 ? Math.max(0, now - requestedAtMs) : 0;
  const staleReload = ENABLE_BACKGROUND_STALE_RELOAD && reloadAgeMs >= BACKGROUND_STALE_RELOAD_AFTER_MS;

  if (data.navigate && !ENABLE_BACKGROUND_CONTROL_NAVIGATE) {
    _debugLog("[local] Navigate do servidor ignorado pelo background:", reason);
    _postExtensionLog("navigate_ignored", reason, { cause: "background_navigate_disabled" });

    if (!data.reload) return;
  }

  if (data.reload) {
    if (isSyncCommand) {
      if (!ENABLE_BACKGROUND_SYNC_RELOAD && !staleReload) {
        _postExtensionLog("sync_reload_waiting_for_content", reason, {
          reload_age_ms: reloadAgeMs,
          stale_after_ms: BACKGROUND_STALE_RELOAD_AFTER_MS,
        }, "info", 30_000);
        return;
      }

      const syncKey = _syncReloadStableKey(data);

      if (!staleReload) {
        if (_syncReloadPendingKey !== syncKey) {
          _syncReloadPendingKey = syncKey;
          _syncReloadFirstSeenAt = now;
          _debugLog("[local] Sync fora de sincronia detectado; aguardando 3 min antes de recarregar.");
          _postExtensionLog("sync_reload_delay_started", reason, {
            sync_key: syncKey,
            delay_ms: SYNC_RELOAD_DELAY_MS,
          });
          return;
        }

        const pendingAge = now - _syncReloadFirstSeenAt;

        if (pendingAge < SYNC_RELOAD_DELAY_MS) {
          _debugLog(
            `[local] Sync fora de sincronia há ${Math.round(pendingAge / 1000)}s; aguardando 180s para reload.`
          );
          return;
        }
      } else {
        _postExtensionLog("stale_sync_reload_bypass", reason, {
          reload_age_ms: reloadAgeMs,
          stale_after_ms: BACKGROUND_STALE_RELOAD_AFTER_MS,
          sync_reload_enabled: ENABLE_BACKGROUND_SYNC_RELOAD,
        }, "warning");
      }

      if (now - _lastSyncReloadAppliedAt < SYNC_RELOAD_REPEAT_COOLDOWN_MS) {
        _debugLog("[local] Reload de sync ignorado por cooldown pós-reload.");
        _postExtensionLog("sync_reload_cooldown", reason, {
          age_ms: now - _lastSyncReloadAppliedAt,
          cooldown_ms: SYNC_RELOAD_REPEAT_COOLDOWN_MS,
        });
        return;
      }

      _lastSyncReloadAppliedAt = now;
    } else if (!ENABLE_BACKGROUND_NON_SYNC_RELOAD && !staleReload) {
      _debugLog("[local] Reload não-sync do servidor ignorado pelo background:", reason);
      _postExtensionLog("non_sync_reload_ignored", reason, {
        reload_age_ms: reloadAgeMs,
        stale_after_ms: BACKGROUND_STALE_RELOAD_AFTER_MS,
      });
      return;
    }
  }

  const commandKey = _controlCommandKey(data);

  if (
    commandKey &&
    commandKey === _lastControlCommandKey &&
    now - _lastControlCommandAt < CONTROL_RETRY_SAME_COMMAND_MS
  ) {
    return;
  }

  _lastControlCommandKey = commandKey;
  _lastControlCommandAt = now;

  chrome.tabs.query({ url: TINDER_TAB_URLS }, (tabs) => {
    if (chrome.runtime.lastError) {
      _debugError("[local] Falha ao procurar abas do Tinder:", chrome.runtime.lastError.message);
      return;
    }

    if (!tabs || !tabs.length) {
      _debugLog("[local] Servidor pediu reload, mas nenhuma aba do Tinder foi encontrada.");
      _postExtensionLog("control_no_tinder_tabs", reason, {
        reload: Boolean(data.reload),
        navigate: Boolean(data.navigate),
      }, "warning");
      return;
    }

    _post("/queue-reset", {
      reason: isSyncCommand
        ? `background reload após 3min de sync perdido: ${reason}`
        : `background reload: ${reason}`,
    });

    for (const tab of tabs) {
      if (!tab || tab.id == null) continue;

      const currentUrl = String(tab.url || "");

      if (
        ENABLE_BACKGROUND_CONTROL_NAVIGATE &&
        data.navigate &&
        data.navigate_url &&
        currentUrl !== data.navigate_url
      ) {
        _postExtensionLog("navigate_firing", reason, {
          tab_id: tab.id,
          from: currentUrl,
          to: data.navigate_url,
          navigate_generation: data.navigate_generation || 0,
        });
        _pendingReloadTabs.set(tab.id, {
          reason,
          commandKey,
          startedAt: Date.now(),
          mode: "navigate",
        });
        chrome.tabs.update(tab.id, { url: data.navigate_url }, () => {
          if (chrome.runtime.lastError) {
            _pendingReloadTabs.delete(tab.id);
            _postExtensionLog("navigate_failed", chrome.runtime.lastError.message, {
              tab_id: tab.id,
              to: data.navigate_url,
            }, "error");
            _debugError("[local] Falha ao navegar aba do Tinder:", chrome.runtime.lastError.message);
          }
        });

        continue;
      }

      if (data.reload) {
        _postExtensionLog("reload_firing", reason, {
          tab_id: tab.id,
          mode: staleReload ? "stale_reload" : "reload",
          reload_generation: data.reload_generation || 0,
          reload_age_ms: reloadAgeMs,
        });
        _pendingReloadTabs.set(tab.id, {
          reason,
          commandKey,
          startedAt: Date.now(),
          mode: staleReload ? "stale_reload" : "reload",
        });
        chrome.tabs.reload(tab.id, { bypassCache: true }, () => {
          if (chrome.runtime.lastError) {
            _pendingReloadTabs.delete(tab.id);
            _postExtensionLog("reload_failed", chrome.runtime.lastError.message, {
              tab_id: tab.id,
              reload_generation: data.reload_generation || 0,
            }, "error");
            _debugError("[local] Falha ao recarregar aba do Tinder:", chrome.runtime.lastError.message);
          }
        });
      }
    }
  });
}

// Captura eventos de rede do Tinder
if (chrome.webRequest) {
  chrome.webRequest.onBeforeRequest.addListener(
    (details) => {
      if (!_isTinderInitiated(details)) return;
      if (String(details.method || "").toUpperCase() === "OPTIONS") return;

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
        request: WEBREQUEST_CAPTURE_REQUEST_BODY ? _requestBodySummary(details.requestBody) : undefined,
      };

      _webRequestStarted.set(details.requestId, started);
    },
    { urls: ["<all_urls>"] },
    WEBREQUEST_CAPTURE_REQUEST_BODY ? ["requestBody"] : []
  );

  chrome.webRequest.onCompleted.addListener(
    (details) => {
      if (!_isTinderInitiated(details)) return;
      if (String(details.method || "").toUpperCase() === "OPTIONS") return;

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
          started.time_stamp && details.timeStamp
            ? Math.round(details.timeStamp - started.time_stamp)
            : undefined,
        response_headers: _headersArrayToObject(details.responseHeaders),
      });
    },
    { urls: ["<all_urls>"] },
    ["responseHeaders"]
  );

  chrome.webRequest.onErrorOccurred.addListener(
    (details) => {
      if (!_isTinderInitiated(details)) return;
      if (String(details.method || "").toUpperCase() === "OPTIONS") return;

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
          started.time_stamp && details.timeStamp
            ? Math.round(details.timeStamp - started.time_stamp)
            : undefined,
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
      if (!res.ok) {
        _debugError(`[local] ${path} → erro ${res.status}`);
      }
    })
    .catch((err) => {
      if (path === "/profiles") {
        _debugError("[local] Servidor offline. Rode: python3 src/server.py", err.message);
      }
    });
}
