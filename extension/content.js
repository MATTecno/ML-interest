/**
 * content.js — roda no contexto do content script (isolado da página).
 *
 * Responsabilidades:
 *  1. Injeta injected.js na página para interceptar o fetch do Tinder
 *  2. Identifica o perfil visível usando URLs de fotos (mais confiável que ler texto)
 *  3. Detecta quando acabam os perfis e recarrega a página automaticamente
 */

// ─── 1. Injeta o interceptor de fetch ────────────────────────────────────────

const PAGE_BRIDGE_SOURCE =
  "__bridge_" +
  Date.now().toString(36) +
  "_" +
  Math.random().toString(36).slice(2);
const PAGE_MSG_RECS = "r";
const PAGE_MSG_PROFILE_MAP = "m";
const PAGE_MSG_NETWORK_CAPTURE = "n";
const EXTENSION_DEBUG_LOGS = false;

function _debugLog(...args) {
  if (EXTENSION_DEBUG_LOGS) console.log(...args);
}

function _debugWarn(...args) {
  if (EXTENSION_DEBUG_LOGS) console.warn(...args);
}

const script = document.createElement("script");
script.src =
  chrome.runtime.getURL("injected.js") +
  "?bridge=" +
  encodeURIComponent(PAGE_BRIDGE_SOURCE);
script.onload = () => script.remove();
(document.head || document.documentElement).appendChild(script);

const SERVER_BASE = "http://localhost:5043";
const PAUSE_CAPTURE_WHEN_UNFOCUSED = false;
const AUTO_DISMISS_SUPERLIKE_UPSELL = true;
const SUPERLIKE_UPSELL_POLL_MS = 350;
const SUPERLIKE_UPSELL_REPORT_MIN_INTERVAL_MS = 350;
const RECS_DEDUPE_TTL_MS = 10 * 60 * 1000;
const RECS_DEDUPE_MAX_KEYS = 80;
const SWIPE_CANONICAL_URL = "https://tinder.com/app/recs";
const SWIPE_ALLOWED_PATH_PREFIXES = ["/app/recs"];
const RELOAD_SCHEDULE_STALE_MS = 40_000;
const RELOAD_ACK_STORAGE_KEY = "tinder_ia_pending_reload_ack";
const PAYWALL_URL_PATTERN =
  /(paywall|purchase|checkout|payment|subscribe|subscription|plus|gold|platinum|super[-_]?like|superswipe|boost)/i;

// Controle fino dos reloads.
// O content script e o executor unico de reload/navegacao pedido pelo servidor.
const ENABLE_SERVER_CONTROL_RELOAD = true;
const ENABLE_SERVER_CONTROL_NAVIGATE = true;
const ENABLE_OUT_OF_PROFILES_AUTO_RELOAD = true;
const ENABLE_OUT_OF_PROFILES_QUEUE_RESET_ON_RELOAD = true;
const CURRENT_REPORT_DEBOUNCE_MS = 220;
const SAME_NAME_ID_FLIPFLOP_IGNORE_MS = 1200;

// ─── 2. Mapa folder_id → {name, tinderId, age} ───────────────────────────────
// Preenchido pelo injected.js quando intercepta o batch de perfis.
// Chave: folder_id extraído das URLs das fotos do Tinder (ex: "8tQt8Wt9CWV4cx9p1cXCFP")
// Valor: { name: "Ana", tinderId: "60c96a6c..." }

let _profileMap = {};
let _lastReportedTinderId = "";
let _lastReportedName = "";
let _lastReportedAge = 0;
let _lastReportedAt = 0;
let _previousReportedTinderId = "";
let _previousReportedName = "";
let _previousReportedAge = 0;
let _previousReportedAt = 0;
let _redetectTimers = [];
let _captureActive = false;
let _lastCaptureStateKey = "";
let _recentRecsBatches = new Map();
let _reloadScheduled = false;
let _reloadScheduledAt = 0;
let _lastControlLogKey = "";
let _lastControlLogAt = 0;
const CONTROL_POLL_MS = 10_000;

function _isTinderHost() {
  const host = location.hostname || "";
  return host === "tinder.com" || host.endsWith(".tinder.com");
}

function _isAllowedSwipeUrl() {
  if (!_isTinderHost()) return false;
  const path = location.pathname || "/";
  return SWIPE_ALLOWED_PATH_PREFIXES.some((prefix) => {
    const clean = (prefix || "").replace(/\/+$/, "") || "/";
    return path === clean || path.startsWith(clean + "/");
  });
}

function _isLikelyPaywallUrl() {
  return PAYWALL_URL_PATTERN.test(location.href || "");
}

function _postServer(path, body) {
  return fetch(SERVER_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  })
    .then((res) => res.ok)
    .catch(() => false);
}

function _postServerJson(path, body) {
  return fetch(SERVER_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  })
    .then(async (res) => {
      let data = null;
      try {
        data = await res.json();
      } catch (_) {}
      return { ok: res.ok, status: res.status, data };
    })
    .catch((error) => ({ ok: false, status: 0, data: null, error }));
}

function _postExtensionLog(event, message, details = {}, level = "info") {
  try {
    fetch(SERVER_BASE + "/extension-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: "content",
        event,
        message: String(message || ""),
        level,
        details: {
          href: location.href,
          visibility: document.visibilityState,
          focused: document.hasFocus(),
          ...details,
        },
      }),
      keepalive: true,
    }).catch(() => {});
  } catch (_) {}
}

function _controlLogAllowed(key, minIntervalMs = 30_000) {
  const now = Date.now();
  if (key && key === _lastControlLogKey && now - _lastControlLogAt < minIntervalMs) return false;
  _lastControlLogKey = key || "";
  _lastControlLogAt = now;
  return true;
}

function _postQueueReset(reason) {
  return _postServer("/queue-reset", { reason: reason || "fila atual invalidada pela extensão" });
}

function _postQueueResetKeepAlive(reason) {
  const payload = JSON.stringify({ reason: reason || "pagina sendo descarregada" });
  try {
    if (navigator.sendBeacon) {
      const blob = new Blob([payload], { type: "application/json" });
      if (navigator.sendBeacon(SERVER_BASE + "/queue-reset", blob)) return;
    }
  } catch (_) {}

  try {
    fetch(SERVER_BASE + "/queue-reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      keepalive: true,
    }).catch(() => {});
  } catch (_) {}
}

function _rememberPendingReloadAck(reason, mode) {
  try {
    sessionStorage.setItem(
      RELOAD_ACK_STORAGE_KEY,
      JSON.stringify({
        reason: String(reason || "reload solicitado"),
        mode: String(mode || "reload"),
        href: location.href,
        at: Date.now(),
      })
    );
  } catch (_) {}
}

function _consumePendingReloadAck() {
  let raw = "";
  try {
    raw = sessionStorage.getItem(RELOAD_ACK_STORAGE_KEY) || "";
    sessionStorage.removeItem(RELOAD_ACK_STORAGE_KEY);
  } catch (_) {
    raw = "";
  }
  if (!raw) return false;

  let pending = {};
  try {
    pending = JSON.parse(raw) || {};
  } catch (_) {
    pending = {};
  }

  _postServer("/reloaded", {
    ok: true,
    source: "content",
    reason: pending.reason || "reload confirmado pelo content script",
    mode: pending.mode || "reload",
  });
  _postExtensionLog("reload_ack_sent", pending.reason || "reload confirmado pelo content script", {
    mode: pending.mode || "reload",
    previous_href: pending.href || "",
    age_ms: pending.at ? Date.now() - Number(pending.at || 0) : 0,
  });
  return true;
}

function _markReloadScheduled(reason, mode) {
  _reloadScheduled = true;
  _reloadScheduledAt = Date.now();
  _rememberPendingReloadAck(reason, mode);
}

function _reloadScheduleIsStale() {
  return _reloadScheduled && _reloadScheduledAt > 0 && Date.now() - _reloadScheduledAt > RELOAD_SCHEDULE_STALE_MS;
}

function _schedulePageReload(reason, delayMs, options = {}) {
  if (_reloadScheduled && !_reloadScheduleIsStale()) {
    _debugLog("[local] Reload ja agendado; ignorando novo pedido:", reason || "resync");
    _postExtensionLog("reload_schedule_ignored", reason || "resync", {
      scheduled_age_ms: Date.now() - _reloadScheduledAt,
    });
    return false;
  }

  if (_reloadScheduled && _reloadScheduleIsStale()) {
    _debugWarn("[local] Reload agendado nao descarregou a pagina; liberando nova tentativa.");
    _postExtensionLog(
      "reload_schedule_stale",
      "reload agendado nao descarregou a pagina; liberando nova tentativa",
      { scheduled_age_ms: Date.now() - _reloadScheduledAt },
      "warning"
    );
  }

  _markReloadScheduled(reason, options.mode || "reload");

  if (options.queueResetReason) {
    _postQueueResetKeepAlive(options.queueResetReason);
  }

  if (options.notifyBackgroundOutOfProfiles) {
    chrome.runtime.sendMessage({ type: "OUT_OF_PROFILES" });
  }

  const delay = Math.max(0, Number(delayMs) || 0);
  _debugLog("[local] Reload agendado:", reason || "resync", `delay=${delay}ms`);
  _postExtensionLog("reload_scheduled", reason || "resync", {
    delay_ms: delay,
    mode: options.mode || "reload",
  });
  setTimeout(() => {
    _postExtensionLog("reload_firing", reason || "resync", { mode: options.mode || "reload" });
    location.reload();
  }, delay);
  return true;
}

function _pollControl() {
  if (!_isTinderHost()) return;
  fetch(SERVER_BASE + "/control")
    .then((res) => (res.ok ? res.json() : null))
    .then((data) => {
      if (!data) return;
      if (data.reload || data.navigate) {
        const controlKey = [
          data.reload ? "r" : "",
          data.reload_generation || 0,
          data.navigate ? "n" : "",
          data.navigate_generation || 0,
          data.navigate_url || "",
        ].join("|");
        if (_controlLogAllowed(controlKey)) {
          _postExtensionLog("control_received", data.navigate_reason || data.reason || "controle recebido", {
            reload: Boolean(data.reload),
            reload_generation: data.reload_generation || 0,
            navigate: Boolean(data.navigate),
            navigate_generation: data.navigate_generation || 0,
            navigate_url: data.navigate_url || "",
            requested_at: data.requested_at || 0,
          });
        }
      }
      if (_reloadScheduled) {
        if (!_reloadScheduleIsStale()) return;
        _debugWarn("[local] Reload agendado nao descarregou a pagina; liberando nova tentativa.");
        _postExtensionLog(
          "reload_schedule_stale",
          "reload agendado nao descarregou a pagina; liberando nova tentativa",
          { scheduled_age_ms: Date.now() - _reloadScheduledAt },
          "warning"
        );
        _reloadScheduled = false;
        _reloadScheduledAt = 0;
      }

      if (
        ENABLE_SERVER_CONTROL_NAVIGATE &&
        data.navigate &&
        data.navigate_url &&
        location.href !== data.navigate_url
      ) {
        _debugLog("[local] Navegação solicitada pelo servidor:", data.navigate_reason || data.navigate_url);
        _postExtensionLog("navigate_scheduled", data.navigate_reason || data.navigate_url, {
          navigate_generation: data.navigate_generation || 0,
          target: data.navigate_url || SWIPE_CANONICAL_URL,
        });
        _markReloadScheduled(data.navigate_reason || data.navigate_url, "navigate");
        _postQueueResetKeepAlive(data.navigate_reason || "voltando para a tela de swipes");
        setTimeout(() => {
          _postExtensionLog("navigate_firing", data.navigate_reason || data.navigate_url, {
            target: data.navigate_url || SWIPE_CANONICAL_URL,
          });
          location.assign(data.navigate_url || SWIPE_CANONICAL_URL);
        }, 3000);
        return;
      }

      if (!ENABLE_SERVER_CONTROL_RELOAD || !data.reload) return;
      _debugLog("[local] Reload solicitado pelo servidor:", data.reason || "resync");
      _schedulePageReload(data.reason || "resync", 3000);
    })
    .catch(() => {});
}
function _captureState() {
  const isTinder = _isTinderHost();
  const allowedSwipeUrl = _isAllowedSwipeUrl();
  const visible = document.visibilityState === "visible";
  const focused = document.hasFocus();
  const active = Boolean(
    isTinder && allowedSwipeUrl && (!PAUSE_CAPTURE_WHEN_UNFOCUSED || (visible && focused))
  );
  let reason = "active";
  if (!isTinder) reason = "not_tinder_url";
  else if (!allowedSwipeUrl) reason = _isLikelyPaywallUrl() ? "paywall_url" : "wrong_tinder_url";
  else if (PAUSE_CAPTURE_WHEN_UNFOCUSED && !visible) reason = "tab_hidden";
  else if (PAUSE_CAPTURE_WHEN_UNFOCUSED && !focused) reason = "window_unfocused";
  else if (!visible) reason = "tab_hidden_ignored";
  else if (!focused) reason = "window_unfocused_ignored";
  return {
    active,
    reason,
    url: location.href,
    visible,
    focused,
  };
}

function _isCaptureActive() {
  return _captureState().active;
}

document.addEventListener(
  "keydown",
  (event) => {
    if (event.key === "F10") {
      event.preventDefault();
      event.stopPropagation();
      _postServer("/hotkey", { action: "stop", source: "browser" });
      _debugWarn("[local] F10 capturado no navegador — solicitando parada dos swipes");
      return;
    }
    if (event.key === "F8") {
      event.preventDefault();
      event.stopPropagation();
      _postServer("/hotkey", { action: "pause", source: "browser" });
      _debugWarn("[local] F8 capturado no navegador — alternando pausa dos swipes");
    }
  },
  true
);

function _notifyCaptureState(force = false) {
  const state = _captureState();
  const key = `${state.active}|${state.reason}|${state.url}`;
  _captureActive = state.active;
  if (!state.active) {
    _previousReportedTinderId = _lastReportedTinderId;
    _previousReportedName = _lastReportedName;
    _previousReportedAge = _lastReportedAge;
    _previousReportedAt = _lastReportedAt;
    _lastReportedTinderId = "";
    _lastReportedName = "";
    _lastReportedAge = 0;
    _lastReportedAt = 0;
  }
  if (!force && key === _lastCaptureStateKey) return;
  _lastCaptureStateKey = key;
  _postServer("/browser-state", state);
  _debugLog(`[local] captura ${state.active ? "ativa" : "pausada"}: ${state.reason}`);
}

function _norm(s) {
  return (s || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function _recsBatchKey(payload) {
  try {
    const results = payload?.data?.results || [];
    const parts = [];
    for (const result of results) {
      if (result?.type !== "user") continue;
      const user = result.user || {};
      const id = String(user._id || "").trim();
      const hash = String(result.content_hash || "").trim();
      const sNumber = String(result.s_number || "").trim();
      const name = String(user.name || "").trim();
      const birthDate = String(user.birth_date || "").trim();
      if (id || hash || sNumber || name) {
        parts.push([id, hash, sNumber, name, birthDate].join(":"));
      }
    }
    if (!parts.length) return "";
    parts.sort();
    return parts.join("|");
  } catch (_) {
    return "";
  }
}

function _cleanupRecentRecsBatches(now) {
  try {
    for (const [key, seenAt] of _recentRecsBatches.entries()) {
      if (now - seenAt > RECS_DEDUPE_TTL_MS) _recentRecsBatches.delete(key);
    }
    while (_recentRecsBatches.size > RECS_DEDUPE_MAX_KEYS) {
      const first = _recentRecsBatches.keys().next().value;
      if (!first) break;
      _recentRecsBatches.delete(first);
    }
  } catch (_) {}
}

function _shouldForwardTinderRecs(payload) {
  const key = _recsBatchKey(payload);
  if (!key) return true;
  const now = Date.now();
  _cleanupRecentRecsBatches(now);
  const seenAt = _recentRecsBatches.get(key);
  if (seenAt && now - seenAt <= RECS_DEDUPE_TTL_MS) return false;
  _recentRecsBatches.set(key, now);
  return true;
}

// Recebe mensagens do injected.js (que roda no contexto da página)
window.addEventListener("message", (event) => {
  if (event.source !== window || event.data?.source !== PAGE_BRIDGE_SOURCE) return;

  if (event.data.kind === PAGE_MSG_RECS) {
    _notifyCaptureState();
    if (!_captureActive) {
      _debugLog("[local] Leva ignorada: aba/janela do Tinder não está ativa.");
      return;
    }
    if (!_shouldForwardTinderRecs(event.data.payload)) {
      _debugLog("[local] Leva repetida ignorada antes de enviar ao servidor.");
      return;
    }
    chrome.runtime.sendMessage({ type: "TINDER_RECS", payload: event.data.payload });
  }

  if (event.data.kind === PAGE_MSG_PROFILE_MAP) {
    // Acumula o mapa — novos perfis chegam em levas
    Object.assign(_profileMap, event.data.payload);
    _notifyCaptureState();
    if (!_captureActive) return;
    _scheduleRedetectBurst(true);
  }

  if (event.data.kind === PAGE_MSG_NETWORK_CAPTURE) {
    _notifyCaptureState();
    if (!_captureActive) return;
    chrome.runtime.sendMessage({
      type: "NETWORK_CAPTURE",
      payload: event.data.payload,
    });
  }
});

function _scheduleRedetectBurst(forceFirst = false) {
  if (!_isCaptureActive()) return;
  for (const timer of _redetectTimers) clearTimeout(timer);
  _redetectTimers = [];

  for (const delay of [0, 80, 180, 320, 520]) {
    const timer = setTimeout(() => {
      _reportVisibleProfile(forceFirst && delay === 0);
    }, delay);
    _redetectTimers.push(timer);
  }
}

// Extrai o folder_id de uma URL do Tinder: /u/{folder_id}/...
function _extractFolderId(url) {
  const m = url.match(/\/u\/([A-Za-z0-9]+)\//);
  return m ? m[1] : null;
}

function _folderIdFromValue(value) {
  if (!value) return null;
  const fid = _extractFolderId(String(value));
  return fid && _profileMap[fid] ? fid : null;
}

function _extractFolderIdFromElement(el) {
  if (!el) return null;

  const directValues = [
    el.src,
    el.currentSrc,
    el.getAttribute?.("src"),
    el.getAttribute?.("style"),
  ];

  try {
    directValues.push(getComputedStyle(el).backgroundImage);
  } catch (_) {}

  for (const value of directValues) {
    const fid = _folderIdFromValue(value);
    if (fid) return fid;
  }

  if (typeof el.querySelectorAll === "function") {
    const descendants = el.querySelectorAll("img, div, section, article, span");
    for (let i = 0; i < Math.min(descendants.length, 12); i++) {
      const child = descendants[i];
      const childValues = [
        child.src,
        child.currentSrc,
        child.getAttribute?.("src"),
        child.getAttribute?.("style"),
      ];
      try {
        childValues.push(getComputedStyle(child).backgroundImage);
      } catch (_) {}

      for (const value of childValues) {
        const fid = _folderIdFromValue(value);
        if (fid) return fid;
      }
    }
  }

  return null;
}

function _visibleScore(el) {
  const rect = el.getBoundingClientRect();
  if (rect.width < 120 || rect.height < 120) return -1;
  if (rect.bottom < 0 || rect.top > window.innerHeight) return -1;
  if (rect.right < 0 || rect.left > window.innerWidth) return -1;

  const visibleW = Math.min(rect.right, window.innerWidth) - Math.max(rect.left, 0);
  const visibleH = Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0);
  if (visibleW <= 0 || visibleH <= 0) return -1;

  const area = visibleW * visibleH;
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  const distX = Math.abs(centerX - window.innerWidth / 2);
  const distY = Math.abs(centerY - window.innerHeight / 2);

  return area - (distX * 2 + distY * 3);
}

function _extractNameAgeFromText(text) {
  const normalized = (text || "").replace(/\s+/g, " ").trim();
  if (!normalized) return null;

  const matches = [
    ...normalized.matchAll(
      /([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'`.-]+(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'`.-]+){0,3})\s+(\d{2})\b/g
    ),
  ];
  if (!matches.length) return null;

  const last = matches[matches.length - 1];
  return {
    name: last[1].trim(),
    age: parseInt(last[2], 10) || 0,
  };
}

function _profileFromNameAge(name, age) {
  const key = `${_norm(name)}::${parseInt(age || 0, 10) || 0}`;
  for (const profile of Object.values(_profileMap)) {
    const profileKey = `${_norm(profile.name)}::${parseInt(profile.age || 0, 10) || 0}`;
    if (profileKey === key) return profile;
  }
  return null;
}

function _cardScore(el) {
  if (!el || typeof el.getBoundingClientRect !== "function") return -1;
  const rect = el.getBoundingClientRect();
  if (rect.width < 220 || rect.height < 320) return -1;
  if (rect.bottom < 0 || rect.top > window.innerHeight) return -1;
  if (rect.right < 0 || rect.left > window.innerWidth) return -1;

  const aspect = rect.width / Math.max(rect.height, 1);
  if (aspect < 0.35 || aspect > 0.9) return -1;

  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  if (centerX < window.innerWidth * 0.40 || centerX > window.innerWidth * 0.78) return -1;
  if (centerY < window.innerHeight * 0.28 || centerY > window.innerHeight * 0.78) return -1;

  return (
    _visibleScore(el) -
    Math.abs(centerX - window.innerWidth * 0.60) * 0.8 -
    Math.abs(centerY - window.innerHeight * 0.48) * 0.8
  );
}

function _closestCardRoot(el) {
  let node = el;
  let depth = 0;
  while (node && depth < 10) {
    if (
      node.matches?.('div[data-keyboard-gamepad="true"][aria-hidden="false"]')
    ) {
      return node;
    }
    node = node.parentElement;
    depth += 1;
  }
  return null;
}

function _findFrontCardRoot() {
  const samplePoints = [
    [0.58, 0.84],
    [0.58, 0.79],
    [0.54, 0.84],
    [0.62, 0.84],
    [0.58, 0.74],
    [0.58, 0.64],
  ];

  let bestPointRoot = null;
  let bestPointScore = -Infinity;

  for (const [pointIdx, [px, py]] of samplePoints.entries()) {
    const x = Math.round(window.innerWidth * px);
    const y = Math.round(window.innerHeight * py);
    const stack = document.elementsFromPoint(x, y) || [];
    const seen = new Set();

    for (let stackIdx = 0; stackIdx < Math.min(stack.length, 8); stackIdx++) {
      const el = stack[stackIdx];
      const root = _closestCardRoot(el);
      if (!root) continue;
      if (seen.has(root)) continue;
      seen.add(root);

      const baseScore = _cardScore(root);
      if (baseScore <= 0) continue;
      const score = baseScore + (8 - stackIdx) * 50 - pointIdx * 3;
      if (score > bestPointScore) {
        bestPointRoot = root;
        bestPointScore = score;
      }
    }
  }

  if (bestPointRoot && bestPointScore > 0) return bestPointRoot;

  const selectorRoots = document.querySelectorAll(
    'div[data-keyboard-gamepad="true"][aria-hidden="false"]'
  );
  let bestSelectorRoot = null;
  let bestSelectorScore = -Infinity;

  for (const root of selectorRoots) {
    const score = _cardScore(root);
    if (score > bestSelectorScore) {
      bestSelectorRoot = root;
      bestSelectorScore = score;
    }
  }

  return bestSelectorRoot;
}

function _extractNameAgeFromCardRoot(root) {
  if (!root) return null;

  const nameEl =
    root.querySelector('button.focus-button-style [itemprop="name"]') ||
    root.querySelector('[itemprop="name"]');
  const ageEl =
    root.querySelector('button.focus-button-style [itemprop="age"]') ||
    root.querySelector('[itemprop="age"]');
  const name = (nameEl?.textContent || "").replace(/\s+/g, " ").trim();
  const age = parseInt((ageEl?.textContent || "").trim(), 10) || 0;

  if (!name || !age) return null;
  return _profileFromNameAge(name, age);
}

function _detectVisibleName(scope = document) {
  let bestProfile = null;
  let bestScore = -Infinity;
  const profiles = Object.values(_profileMap);
  if (!profiles.length) return null;
  const candidates = (scope || document).querySelectorAll("div, span, h1, h2, h3, p");
  for (const el of candidates) {
    const text = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
    if (!text || text.length < 4 || text.length > 120) continue;

    const rect = el.getBoundingClientRect();
    if (rect.width < 30 || rect.height < 10) continue;
    if (rect.bottom < 0 || rect.top > window.innerHeight) continue;
    if (rect.right < 0 || rect.left > window.innerWidth) continue;

    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    // Nome do card tende a ficar no terço central/direito e parte inferior do card.
    if (centerX < window.innerWidth * 0.40 || centerX > window.innerWidth * 0.78) continue;
    if (centerY < window.innerHeight * 0.55 || centerY > window.innerHeight * 0.92) continue;

    const extracted = _extractNameAgeFromText(text);
    if (!extracted) continue;
    const candidateProfile = _profileFromNameAge(extracted.name, extracted.age);
    if (!candidateProfile) continue;

    const style = getComputedStyle(el);
    const fontSize = parseFloat(style.fontSize || "0");
    const fontWeight = parseInt(style.fontWeight || "400", 10) || 400;

    const score =
      fontSize * 14 +
      fontWeight / 20 -
      Math.abs(centerX - window.innerWidth * 0.58) * 0.7 -
      Math.abs(centerY - window.innerHeight * 0.74) * 1.2 -
      text.length * 0.5 +
      extracted.name.length * 2;

    if (score > bestScore) {
      bestProfile = candidateProfile;
      bestScore = score;
    }
  }

  return bestProfile;
}

function _isActuallyVisible(el) {
  if (!el || typeof el.getBoundingClientRect !== "function") return false;
  const rect = el.getBoundingClientRect();
  if (rect.width < 8 || rect.height < 8) return false;
  if (rect.bottom < 0 || rect.top > window.innerHeight) return false;
  if (rect.right < 0 || rect.left > window.innerWidth) return false;
  const style = getComputedStyle(el);
  return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity || 1) > 0.05;
}

function _isActionDisabled(el) {
  if (!el) return true;
  return Boolean(
    el.disabled ||
    el.getAttribute("aria-disabled") === "true" ||
    el.getAttribute("disabled") !== null ||
    _norm(String(el.className || "")).includes("disabled")
  );
}

function _superLikeButtonText(el) {
  const attrs = [
    el.innerText,
    el.textContent,
    el.getAttribute?.("aria-label"),
    el.getAttribute?.("title"),
    el.getAttribute?.("data-testid"),
    el.getAttribute?.("data-test-id"),
    el.getAttribute?.("class"),
  ];
  return _norm(attrs.filter(Boolean).join(" "));
}

function _detectSuperLikeAvailability() {
  const upsellOpen = Array.from(
    document.querySelectorAll('[role="dialog"][aria-modal="true"], [role="dialog"]')
  ).some(_isSuperLikeUpsellDialog);
  if (upsellOpen) {
    return { available: false, reason: "modal_upgrade_super_like_aberto" };
  }

  const buttons = Array.from(document.querySelectorAll('button, [role="button"]'))
    .filter(_isActuallyVisible);
  const candidates = buttons.filter((button) => {
    const text = _superLikeButtonText(button);
    return (
      text.includes("super like") ||
      text.includes("superlike") ||
      text.includes("super curt") ||
      text.includes("supercurt")
    );
  });

  if (!candidates.length) {
    return { available: false, reason: "botao_super_like_nao_detectado" };
  }

  const enabled = candidates.find((button) => !_isActionDisabled(button));
  if (enabled) {
    return { available: true, reason: "botao_super_like_ativo" };
  }

  return { available: false, reason: "botao_super_like_desabilitado" };
}

let _lastSuperLikeUpsellReportAt = 0;

function _elementText(el) {
  return _norm(
    [
      el?.innerText,
      el?.textContent,
      el?.getAttribute?.("aria-label"),
      el?.getAttribute?.("aria-labelledby"),
      el?.getAttribute?.("title"),
      el?.getAttribute?.("data-testid"),
      el?.getAttribute?.("data-test-id"),
    ]
      .filter(Boolean)
      .join(" ")
  );
}

function _dialogText(dialog) {
  return _elementText(dialog);
}

function _isNoThanksText(text) {
  return (
    text.includes("nao, obrigado") ||
    text.includes("nao, obrigada") ||
    text.includes("nao obrigado") ||
    text.includes("nao obrigada") ||
    text.includes("obrigado(a)") ||
    text.includes("agora nao") ||
    text.includes("no thanks") ||
    text.includes("not now") ||
    text.includes("maybe later")
  );
}

function _classifySuperLikeUpgradeText(text) {
  const mentionsSuperLike = text.includes("super like") || text.includes("superlike");
  const isPopularProfileUpgrade =
    mentionsSuperLike &&
    (
      text.includes("fazer upgrade") ||
      text.includes("perfil popular") ||
      text.includes("popular profile") ||
      text.includes("mandar um super like") ||
      text.includes("send a super like")
    );
  if (isPopularProfileUpgrade) {
    return { kind: "popular_profile_upgrade", reason: "popular_profile_super_like_prompt" };
  }

  if (!mentionsSuperLike) return false;

  const isDepletedUpsell =
    text.includes("nao tem mais super likes") ||
    text.includes("nao tem mais super like") ||
    text.includes("sem super likes") ||
    text.includes("sem super like") ||
    text.includes("nao quer esperar") ||
    text.includes("descolar mais super likes") ||
    text.includes("descolar mais super like") ||
    text.includes("upgrade");
  if (isDepletedUpsell) {
    return { kind: "super_like_upsell", reason: "super_likes_depleted_prompt" };
  }

  return false;
}

function _classifySuperLikeUpgradeDialog(dialog) {
  if (!AUTO_DISMISS_SUPERLIKE_UPSELL || !_isActuallyVisible(dialog)) return false;
  return _classifySuperLikeUpgradeText(_dialogText(dialog));
}

function _findUpgradeRootForDecline(decline) {
  let node = decline;
  for (let depth = 0; node && node !== document.documentElement && depth < 14; depth++) {
    if (node.nodeType === Node.ELEMENT_NODE && _isActuallyVisible(node)) {
      const modal = _classifySuperLikeUpgradeText(_elementText(node));
      if (modal) return { root: node, modal };
    }
    node = node.parentElement;
  }

  const bodyModal = _classifySuperLikeUpgradeText(_elementText(document.body));
  if (bodyModal) {
    return { root: document.body, modal: bodyModal };
  }
  return null;
}

function _isSuperLikeUpsellDialog(dialog) {
  return Boolean(_classifySuperLikeUpgradeDialog(dialog));
}

function _findDialogDeclineButton(dialog) {
  const buttons = Array.from(dialog.querySelectorAll('button, [role="button"], a, [tabindex]'))
    .filter(_isActuallyVisible);

  return buttons.find((button) => {
    const text = _elementText(button);
    return _isNoThanksText(text);
  });
}

function _findGlobalUpgradeDeclineTarget() {
  const candidates = Array.from(
    document.querySelectorAll(
      'button, [role="button"], a, [tabindex], div.c9iqosj, span'
    )
  ).filter(_isActuallyVisible);

  for (const candidate of candidates) {
    if (!_isNoThanksText(_elementText(candidate))) continue;

    const clickable =
      candidate.closest?.('button, [role="button"], a, [tabindex]') ||
      candidate;
    if (!_isActuallyVisible(clickable)) continue;

    const context = _findUpgradeRootForDecline(clickable);
    if (!context) continue;

    return {
      root: context.root,
      modal: context.modal,
      decline: clickable,
      source: candidate.matches?.("div.c9iqosj")
        ? "global_no_thanks_c9iqosj"
        : "global_no_thanks_text",
    };
  }

  const bodyText = _elementText(document.body);
  const modal = _classifySuperLikeUpgradeText(bodyText);
  if (!modal || !_isNoThanksText(bodyText)) return null;

  const textNodes = [];
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        return _isNoThanksText(_norm(node.nodeValue || ""))
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_SKIP;
      },
    }
  );

  while (textNodes.length < 8) {
    const node = walker.nextNode();
    if (!node) break;
    textNodes.push(node);
  }

  for (const textNode of textNodes) {
    const parent = textNode.parentElement;
    const clickable =
      parent?.closest?.('button, [role="button"], a, [tabindex]') ||
      parent;
    if (clickable && _isActuallyVisible(clickable)) {
      return {
        root: document.body,
        modal,
        decline: clickable,
        source: "global_no_thanks_textnode",
      };
    }
  }

  return null;
}

function _findReportedUpgradeModalTarget() {
  const dialogs = Array.from(
    document.querySelectorAll('[role="dialog"][aria-modal="true"], [role="dialog"], [aria-modal="true"]')
  ).filter(_isActuallyVisible);

  for (const dialog of dialogs) {
    const modal = _classifySuperLikeUpgradeDialog(dialog);
    if (!modal) continue;

    const decline = _findDialogDeclineButton(dialog);
    return {
      root: dialog,
      modal,
      decline,
      source: decline ? "dialog_decline_button" : "dialog_backdrop",
    };
  }

  return _findGlobalUpgradeDeclineTarget();
}

function _targetViewportPoint(target) {
  const targetRect = target.decline?.getBoundingClientRect();
  if (targetRect) {
    return {
      x: targetRect.left + targetRect.width / 2,
      y: targetRect.top + targetRect.height / 2,
    };
  }
  return _dialogBackdropPoint(target.root);
}

function _targetDialogText(target) {
  if (target.root === document.body) {
    return _elementText(document.body).slice(0, 220);
  }
  return _dialogText(target.root).slice(0, 220);
}

function _targetLabel(target) {
  if (target.decline) {
    return target.source || "decline_button_no_thanks";
  }
  return "backdrop";
}

function _reportBlockingModalTarget(target, viewportPoint) {
  const screenPoint = _viewportPointToScreen(viewportPoint.x, viewportPoint.y);
  const payload = {
    kind: target.modal.kind,
    target: _targetLabel(target),
    reason: target.modal.reason,
    dialog_text: _targetDialogText(target),
    screen_x: screenPoint.x,
    screen_y: screenPoint.y,
    device_pixel_ratio: screenPoint.device_pixel_ratio,
  };

  chrome.runtime.sendMessage({ type: "BLOCKING_MODAL", payload });
}

function _viewportPointToScreen(x, y) {
  const borderX = Math.max(0, (window.outerWidth - window.innerWidth) / 2);
  const chromeTop = Math.max(0, window.outerHeight - window.innerHeight - borderX);
  return {
    x: Math.round(window.screenX + borderX + x),
    y: Math.round(window.screenY + chromeTop + y),
    device_pixel_ratio: window.devicePixelRatio || 1,
  };
}

function _dialogBackdropPoint(dialog) {
  const rect = dialog.getBoundingClientRect();
  const candidates = [
    [Math.max(8, rect.left - 48), Math.min(window.innerHeight - 8, rect.top + rect.height / 2)],
    [Math.min(window.innerWidth - 8, rect.right + 48), Math.min(window.innerHeight - 8, rect.top + rect.height / 2)],
    [Math.min(window.innerWidth - 8, window.innerWidth / 2), Math.max(8, rect.top - 48)],
  ];

  for (const [x, y] of candidates) {
    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) continue;
    return { x, y };
  }

  return null;
}

function _reportSuperLikeUpsellForPyAutoGui() {
  if (!AUTO_DISMISS_SUPERLIKE_UPSELL || !_isCaptureActive()) return false;

  const now = Date.now();
  if (now - _lastSuperLikeUpsellReportAt < SUPERLIKE_UPSELL_REPORT_MIN_INTERVAL_MS) {
    return false;
  }

  const target = _findReportedUpgradeModalTarget();
  if (!target) return false;

  const viewportPoint = _targetViewportPoint(target);
  if (!viewportPoint) return false;

  _lastSuperLikeUpsellReportAt = now;
  _reportBlockingModalTarget(target, viewportPoint);
  _debugLog("[local] Modal de upgrade/Super Like reportado para clique via PyAutoGUI.");
  return true;
}

// Busca o folder_id do card realmente visível na tela
function _detectVisibleFolderId(scope = null) {
  if (scope) {
    const activePanel =
      scope.querySelector('.keen-slider__slide[aria-hidden="false"]') ||
      scope.querySelector('[role="tabpanel"][aria-hidden="false"]');

    const scopedImage =
      activePanel?.querySelector('.sentry-block[style*="gotinder.com/u/"]') ||
      activePanel?.querySelector('[style*="gotinder.com/u/"]') ||
      activePanel;

    const scopedId = _extractFolderIdFromElement(scopedImage || activePanel || scope);
    if (scopedId) return scopedId;
  }

  const samplePoints = [
    [0.58, 0.72],
    [0.58, 0.62],
    [0.58, 0.82],
    [0.48, 0.72],
    [0.68, 0.72],
  ];

  for (const [px, py] of samplePoints) {
    const x = Math.round(window.innerWidth * px);
    const y = Math.round(window.innerHeight * py);
    const stack = document.elementsFromPoint(x, y) || [];

    for (const el of stack.slice(0, 8)) {
      let node = el;
      let depth = 0;
      while (node && depth < 6) {
        const fid = _extractFolderIdFromElement(node);
        if (fid) return fid;
        node = node.parentElement;
        depth += 1;
      }
    }
  }

  let bestFolderId = null;
  let bestScore = -1;

  // Estratégia 1: <img src="...gotinder.com/u/{folder_id}/...">
  for (const img of document.querySelectorAll("img")) {
    const src = img.src || "";
    if (src.includes("gotinder.com/u/")) {
      const fid = _extractFolderId(src);
      const score = _visibleScore(img);
      if (fid && _profileMap[fid] && score > bestScore) {
        bestFolderId = fid;
        bestScore = score;
      }
    }
  }

  // Estratégia 2: background-image em divs (Tinder usa muito isso nos cards)
  for (const el of document.querySelectorAll("[style]")) {
    const style = el.getAttribute("style") || "";
    if (style.includes("gotinder.com/u/")) {
      const fid = _extractFolderId(style);
      const score = _visibleScore(el);
      if (fid && _profileMap[fid] && score > bestScore) {
        bestFolderId = fid;
        bestScore = score;
      }
    }
  }

  return bestFolderId;
}

function _reportVisibleProfile(force = false) {
  _notifyCaptureState();
  if (!_captureActive) return;

  const frontCardRoot = _findFrontCardRoot();
  const folderId = _detectVisibleFolderId(frontCardRoot);
  const folderProfile = folderId ? _profileMap[folderId] : null;
  const visibleProfile =
    _extractNameAgeFromCardRoot(frontCardRoot) ||
    _detectVisibleName(frontCardRoot || document);
  const nameProfile = visibleProfile || null;

  let chosenProfile = null;
  if (nameProfile && folderProfile) {
    chosenProfile = {
      ...folderProfile,
      ...nameProfile,
      tinderId: folderProfile.tinderId || nameProfile.tinderId || "",
    };
  } else if (nameProfile) {
    chosenProfile = nameProfile;
  } else if (folderProfile) {
    chosenProfile = folderProfile;
  }

  if (!chosenProfile) return;
  const visibleAge = parseInt(chosenProfile.age || 0, 10) || 0;
  const visibleName = chosenProfile.name || "";
  const visibleId = chosenProfile.tinderId || "";
  const now = Date.now();
  const sameVisibleProfile =
    visibleName === _lastReportedName &&
    visibleId === _lastReportedTinderId &&
    visibleAge === _lastReportedAge;
  if (sameVisibleProfile && !force) return;

  const sameNameAgeAsLast =
    _norm(visibleName) === _norm(_lastReportedName) && visibleAge === _lastReportedAge;
  const sameNameAgeAsPrevious =
    _norm(visibleName) === _norm(_previousReportedName) && visibleAge === _previousReportedAge;

  if (
    !force &&
    visibleId &&
    _previousReportedTinderId &&
    visibleId === _previousReportedTinderId &&
    visibleId !== _lastReportedTinderId &&
    sameNameAgeAsLast &&
    sameNameAgeAsPrevious &&
    now - _lastReportedAt <= SAME_NAME_ID_FLIPFLOP_IGNORE_MS
  ) {
    _debugLog("[local] /current ignorado por flip-flop curto de mesmo nome/idade:", visibleName, visibleAge);
    return;
  }

  if (
    !force &&
    visibleId &&
    _lastReportedTinderId &&
    visibleId !== _lastReportedTinderId &&
    sameNameAgeAsLast &&
    now - _lastReportedAt <= CURRENT_REPORT_DEBOUNCE_MS
  ) {
    _debugLog("[local] /current segurado por debounce curto de mesmo nome/idade:", visibleName, visibleAge);
    return;
  }

  if (!sameVisibleProfile) {
    _previousReportedTinderId = _lastReportedTinderId;
    _previousReportedName = _lastReportedName;
    _previousReportedAge = _lastReportedAge;
    _previousReportedAt = _lastReportedAt;
  }
  _lastReportedTinderId = visibleId;
  _lastReportedName = visibleName;
  _lastReportedAge = visibleAge;
  _lastReportedAt = now;
  const superLikeState = _detectSuperLikeAvailability();
  chrome.runtime.sendMessage({
    type: "PROFILE_VISIBLE",
    name: visibleName,
    tinder_id: visibleId,
    age: visibleAge,
    super_like_available: superLikeState.available,
    super_like_reason: superLikeState.reason,
  });
}

// Observer: dispara quando imagens ou estilos mudam no DOM.
// Como o content script roda em document_start, o body pode ainda não existir.
// Então esperamos o DOM ficar pronto antes de iniciar a observação.
const _observer = new MutationObserver(() => {
  if (!_isCaptureActive()) return;
  _reportSuperLikeUpsellForPyAutoGui();
  _scheduleRedetectBurst();
});

let _trackingStarted = false;

function _startVisibleTracking() {
  if (_trackingStarted) return;
  if (!document.body) return;

  _trackingStarted = true;
  _notifyCaptureState(true);
  const ackedPendingReload = _consumePendingReloadAck();
  if (_captureActive && !ackedPendingReload) {
    _postServer("/reloaded", { ok: true });
  }

  _observer.observe(document.body, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["src", "style"],
  });

  _scheduleRedetectBurst();
  _reportSuperLikeUpsellForPyAutoGui();
  setInterval(() => _reportVisibleProfile(false), 500);
  setInterval(() => _reportVisibleProfile(true), 5_000);
  setInterval(_reportSuperLikeUpsellForPyAutoGui, SUPERLIKE_UPSELL_POLL_MS);
  setInterval(_pollControl, CONTROL_POLL_MS);
}

document.addEventListener("visibilitychange", () => {
  _notifyCaptureState(true);
  if (_captureActive) {
    _scheduleRedetectBurst(true);
  }
});
window.addEventListener("focus", () => {
  _notifyCaptureState(true);
  if (_captureActive) {
    _scheduleRedetectBurst(true);
  }
});
window.addEventListener("blur", () => _notifyCaptureState(true));
window.addEventListener("pageshow", () => _notifyCaptureState(true));
window.addEventListener("pagehide", () => {
  _notifyCaptureState(true);
  _postQueueResetKeepAlive("pagehide/reload do Tinder: descartar fila antiga");
});
window.addEventListener("beforeunload", () => {
  _postQueueResetKeepAlive("beforeunload do Tinder: descartar fila antiga");
});

if (document.body) {
  _startVisibleTracking();
} else {
  document.addEventListener("DOMContentLoaded", _startVisibleTracking, { once: true });
  const _bootInterval = setInterval(() => {
    if (!document.body) return;
    clearInterval(_bootInterval);
    _startVisibleTracking();
  }, 100);
}

// ─── 3. Detecção de "sem perfis" + sleep 5 min + auto-reload ─────────────────
// Quando o Tinder mostra "sem perfis", aguardamos 5 min antes de recarregar
// para não fazer reloads excessivos quando é só falta temporária de cards.

const OUT_OF_PROFILES_TEXTS = [
  "você viu todos",
  "voce viu todos",
  "acabaram as fichas",
  "aumentar o raio",
  "back in gold",
  "expand your radius",
  "you've seen everyone",
  "you have seen all",
  "não há novos membros",
  "nao ha novos membros",
  "nao conseguimos encontrar",
  "não conseguimos encontrar",
  "matches em potencial",
];

let _noCardCounter = 0;
let _lastOfflineReloadNoticeAt = 0;
let _outOfProfilesAt = 0;
let _queueResetSentForOutOfProfiles = false;
const OUT_OF_PROFILES_SLEEP_MS = 5 * 60 * 1000; // 5 minutos antes de recarregar

function _isOutOfProfiles() {
  if (!_isCaptureActive()) return false;
  const bodyText = document.body.innerText.toLowerCase();
  if (OUT_OF_PROFILES_TEXTS.some((t) => bodyText.includes(t))) return true;

  // Detecção conservadora: só considera "sem card" quando não há folder_id,
  // nem root de card válido, nem nome/idade visível. Isso evita limpar/recarregar
  // enquanto o card existe mas a leitura por foto falhou momentaneamente.
  const frontCardRoot = _findFrontCardRoot();
  const folderId = _detectVisibleFolderId(frontCardRoot);
  const visibleByName =
    _extractNameAgeFromCardRoot(frontCardRoot) ||
    _detectVisibleName(frontCardRoot || document);

  const hasAnyKnownBatch = Object.keys(_profileMap).length > 0;
  const hasVisibleCard = Boolean(folderId || frontCardRoot || visibleByName);

  if (!hasVisibleCard && hasAnyKnownBatch) {
    _noCardCounter++;
    return _noCardCounter >= 8;
  }

  _noCardCounter = 0;
  return false;
}
setInterval(() => {
  if (!_isCaptureActive()) return;
  if (_reloadScheduled) return;

  const outNow = _isOutOfProfiles();

  if (!outNow) {
    // Perfis voltaram — cancela o timer de 5 min
    if (_outOfProfilesAt > 0) {
      _debugLog("[local] Perfis detectados novamente. Timer de reload cancelado.");
      _outOfProfilesAt = 0;
      _queueResetSentForOutOfProfiles = false;
    }
    return;
  }

  const now = Date.now();
  if (_outOfProfilesAt === 0) {
    _outOfProfilesAt = now;
    _queueResetSentForOutOfProfiles = true;
    _postQueueReset("sem perfis/card visível detectado pela extensão").then((ok) => {
      if (ok) {
        _debugLog("[local] Sem perfis detectado. Fila antiga limpa; aguardando novos perfis ou reload em 5 min.");
      } else {
        _debugLog("[local] Sem perfis detectado. Servidor offline; aguardando 5 min antes de recarregar...");
        _queueResetSentForOutOfProfiles = false;
      }
    });
    return;
  }

  if (!_queueResetSentForOutOfProfiles) {
    _queueResetSentForOutOfProfiles = true;
    _postQueueReset("sem perfis/card visível detectado pela extensão");
  }

  const elapsed = now - _outOfProfilesAt;
  if (elapsed < OUT_OF_PROFILES_SLEEP_MS) {
    const remainMin = Math.ceil((OUT_OF_PROFILES_SLEEP_MS - elapsed) / 60_000);
    _debugLog(`[local] Sem perfis — ${remainMin} min restante(s) para reload automático`);
    return;
  }

  if (!ENABLE_OUT_OF_PROFILES_AUTO_RELOAD) return;

  // 5 minutos aguardados — hora de recarregar por ausência real de perfil/card.
  const reason = "sem perfis/card visível detectado pela extensão (após 5min de espera)";
  const resetPromise = ENABLE_OUT_OF_PROFILES_QUEUE_RESET_ON_RELOAD
    ? _postQueueReset(reason)
    : Promise.resolve(true);

  resetPromise.finally(() => {
    _postServerJson("/reload-start", { reason }).then((response) => {
      if (!response.ok) {
        const offlineNow = Date.now();
        if (offlineNow - _lastOfflineReloadNoticeAt > 60_000) {
          _debugLog("[local] Sem perfis, mas o servidor está offline. Auto-reload pausado.");
          _lastOfflineReloadNoticeAt = offlineNow;
        }
        _outOfProfilesAt = now; // reseta o timer, tenta de novo na próxima janela
        return;
      }

      if (response.data?.ignored) {
        _debugLog("[local] Backend ignorou reload de sem perfis:", response.data.ignored);
        _outOfProfilesAt = now; // tenta novamente depois de outra janela
        return;
      }

      _debugLog("[local] 5 min sem perfil/card. Recarregando a página agora...");
      _schedulePageReload(reason, 500, { notifyBackgroundOutOfProfiles: true });
    });
  });
}, 15_000);
