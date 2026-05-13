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
const PAYWALL_URL_PATTERN =
  /(paywall|purchase|checkout|payment|subscribe|subscription|plus|gold|platinum|super[-_]?like|superswipe|boost)/i;

// ─── 2. Mapa folder_id → {name, tinderId, age} ───────────────────────────────
// Preenchido pelo injected.js quando intercepta o batch de perfis.
// Chave: folder_id extraído das URLs das fotos do Tinder (ex: "8tQt8Wt9CWV4cx9p1cXCFP")
// Valor: { name: "Ana", tinderId: "60c96a6c..." }

let _profileMap = {};
let _lastReportedTinderId = "";
let _lastReportedName = "";
let _lastReportedAge = 0;
let _redetectTimers = [];
let _captureActive = false;
let _lastCaptureStateKey = "";
let _recentRecsBatches = new Map();
const CONTROL_POLL_MS = 5_000;

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

function _pollControl() {
  if (!_isTinderHost()) return;
  fetch(SERVER_BASE + "/control")
    .then((res) => (res.ok ? res.json() : null))
    .then((data) => {
      if (!data || _reloadScheduled) return;
      if (data.navigate && data.navigate_url && location.href !== data.navigate_url) {
        _debugLog("[local] Navegação solicitada pelo servidor:", data.navigate_reason || data.navigate_url);
        _reloadScheduled = true;
        _postQueueResetKeepAlive(data.navigate_reason || "voltando para a tela de swipes");
        setTimeout(() => location.assign(data.navigate_url || SWIPE_CANONICAL_URL), 300);
        return;
      }
      if (!data.reload) return;
      _debugLog("[local] Reload solicitado pelo servidor:", data.reason || "resync");
      _reloadScheduled = true;
      setTimeout(() => location.reload(), 300);
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
    _lastReportedTinderId = "";
    _lastReportedName = "";
    _lastReportedAge = 0;
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

  for (const [px, py] of samplePoints) {
    const x = Math.round(window.innerWidth * px);
    const y = Math.round(window.innerHeight * py);
    const stack = document.elementsFromPoint(x, y) || [];
    const seen = new Set();

    for (const el of stack.slice(0, 6)) {
      const root = _closestCardRoot(el);
      if (!root) continue;
      if (seen.has(root)) continue;
      seen.add(root);

      const score = _cardScore(root);
      if (score > 0) {
        return root;
      }
    }
  }

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

function _dialogText(dialog) {
  return _norm(
    [
      dialog?.innerText,
      dialog?.textContent,
      dialog?.getAttribute?.("aria-label"),
      dialog?.getAttribute?.("aria-labelledby"),
    ]
      .filter(Boolean)
      .join(" ")
  );
}

function _isSuperLikeUpsellDialog(dialog) {
  if (!AUTO_DISMISS_SUPERLIKE_UPSELL || !_isActuallyVisible(dialog)) return false;

  const text = _dialogText(dialog);
  if (!text.includes("super like") && !text.includes("superlike")) return false;

  return (
    text.includes("nao tem mais super likes") ||
    text.includes("nao tem mais super like") ||
    text.includes("sem super likes") ||
    text.includes("sem super like") ||
    text.includes("nao quer esperar") ||
    text.includes("descolar mais super likes") ||
    text.includes("descolar mais super like") ||
    text.includes("fazer upgrade") ||
    text.includes("perfil popular") ||
    text.includes("popular profile") ||
    text.includes("upgrade")
  );
}

function _findDialogDeclineButton(dialog) {
  const buttons = Array.from(dialog.querySelectorAll('button, [role="button"]'))
    .filter(_isActuallyVisible);

  return buttons.find((button) => {
    const text = _norm(
      [
        button.innerText,
        button.textContent,
        button.getAttribute?.("aria-label"),
        button.getAttribute?.("title"),
        button.getAttribute?.("data-testid"),
        button.getAttribute?.("data-test-id"),
      ]
        .filter(Boolean)
        .join(" ")
    );

    return (
      text.includes("nao, obrigado") ||
      text.includes("nao obrigado") ||
      text.includes("agora nao") ||
      text.includes("no thanks") ||
      text.includes("not now") ||
      text.includes("maybe later")
    );
  });
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

  const dialogs = Array.from(
    document.querySelectorAll('[role="dialog"][aria-modal="true"], [role="dialog"]')
  ).filter(_isActuallyVisible);

  for (const dialog of dialogs) {
    if (!_isSuperLikeUpsellDialog(dialog)) continue;

    const decline = _findDialogDeclineButton(dialog);
    const targetRect = decline?.getBoundingClientRect();
    const viewportPoint = targetRect
      ? {
          x: targetRect.left + targetRect.width / 2,
          y: targetRect.top + targetRect.height / 2,
        }
      : _dialogBackdropPoint(dialog);
    if (!viewportPoint) continue;

    const screenPoint = _viewportPointToScreen(viewportPoint.x, viewportPoint.y);
    _lastSuperLikeUpsellReportAt = now;
    _postServer("/modal", {
      kind: "super_like_upsell",
      target: decline ? "decline_button" : "backdrop",
      dialog_text: _dialogText(dialog).slice(0, 220),
      screen_x: screenPoint.x,
      screen_y: screenPoint.y,
      device_pixel_ratio: screenPoint.device_pixel_ratio,
    });
    _debugLog("[local] Modal de upgrade/Super Like reportado para clique via PyAutoGUI.");
    return true;
  }

  return false;
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
  const sameVisibleProfile =
    (chosenProfile.name || "") === _lastReportedName &&
    (chosenProfile.tinderId || "") === _lastReportedTinderId &&
    visibleAge === _lastReportedAge;
  if (sameVisibleProfile && !force) return;

  _lastReportedTinderId = chosenProfile.tinderId || "";
  _lastReportedName = chosenProfile.name || "";
  _lastReportedAge = visibleAge;
  const superLikeState = _detectSuperLikeAvailability();
  chrome.runtime.sendMessage({
    type: "PROFILE_VISIBLE",
    name: chosenProfile.name,
    tinder_id: chosenProfile.tinderId || "",
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
  if (_captureActive) {
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
let _reloadScheduled = false;
let _lastOfflineReloadNoticeAt = 0;
let _outOfProfilesAt = 0;
let _queueResetSentForOutOfProfiles = false;
const OUT_OF_PROFILES_SLEEP_MS = 5 * 60 * 1000; // 5 minutos antes de recarregar

function _isOutOfProfiles() {
  if (!_isCaptureActive()) return false;
  const bodyText = document.body.innerText.toLowerCase();
  if (OUT_OF_PROFILES_TEXTS.some((t) => bodyText.includes(t))) return true;

  // Sem foto reconhecida do mapa por várias verificações seguidas = sem card
  if (_detectVisibleFolderId() === null && Object.keys(_profileMap).length > 0) {
    _noCardCounter++;
    return _noCardCounter >= 4;
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

  // 5 minutos aguardados — hora de recarregar
  _postServer("/reload-start", {
    reason: "sem perfis detectados pela extensão (após 5min de espera)",
  }).then((ok) => {
    if (!ok) {
      const offlineNow = Date.now();
      if (offlineNow - _lastOfflineReloadNoticeAt > 60_000) {
        _debugLog("[local] Sem perfis, mas o servidor está offline. Auto-reload pausado.");
        _lastOfflineReloadNoticeAt = offlineNow;
      }
      _outOfProfilesAt = now; // reseta o timer, tenta de novo na próxima janela
      return;
    }

    _debugLog("[local] 5 min sem perfis. Recarregando a página agora...");
    chrome.runtime.sendMessage({ type: "OUT_OF_PROFILES" });
    _reloadScheduled = true;
    setTimeout(() => location.reload(), 500);
  });
}, 15_000);
