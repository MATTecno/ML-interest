from pathlib import Path
import re

CONTENT_PATH = Path("content.js")
BACKGROUND_PATH = Path("background.js")

if not CONTENT_PATH.exists():
    raise SystemExit("ERRO: content.js não encontrado nesta pasta")

if not BACKGROUND_PATH.exists():
    raise SystemExit("ERRO: background.js não encontrado nesta pasta")


def replace_function(src: str, function_name: str, replacement: str) -> str:
    marker = f"function {function_name}("
    start = src.find(marker)
    if start < 0:
        raise SystemExit(f"ERRO: função {function_name} não encontrada")

    brace = src.find("{", start)
    if brace < 0:
        raise SystemExit(f"ERRO: abertura da função {function_name} não encontrada")

    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                return src[:start] + replacement.rstrip() + src[end:]

    raise SystemExit(f"ERRO: fechamento da função {function_name} não encontrado")


def replace_bottom_out_of_profiles_interval(src: str, replacement: str) -> str:
    marker = "setInterval(() => {"
    start = src.rfind(marker)
    if start < 0:
        raise SystemExit("ERRO: setInterval final de out-of-profiles não encontrado")

    # Procura o fechamento do setInterval final: }, 15_000);
    tail = src[start:]
    m = re.search(r"\n\},\s*15_000\);\s*$", tail)
    if not m:
        raise SystemExit("ERRO: fechamento do setInterval final não encontrado")

    end = start + m.end()
    return src[:start] + replacement.rstrip() + "\n" + src[end:]


# ─────────────────────────────────────────────────────────────
# content.js
# ─────────────────────────────────────────────────────────────

content = CONTENT_PATH.read_text(encoding="utf-8")

# 1) Desliga reload/navigate imediato via /control no content.
content_poll_control = r'''
function _pollControl() {
  // Reload/navigate por /control fica centralizado no background.js.
  // Isso evita dois lugares diferentes recarregando a página ao mesmo tempo.
  return;
}
'''

content = replace_function(content, "_pollControl", content_poll_control)

# 2) Evita queue-reset em pagehide/beforeunload. O reset deve acontecer quando o reload for aceito.
content = re.sub(
    r'''window\.addEventListener\("pagehide",\s*\(\)\s*=>\s*\{\s*
\s*_notifyCaptureState\(true\);\s*
\s*_postQueueResetKeepAlive\("pagehide/reload do Tinder: descartar fila antiga"\);\s*
\s*\}\);''',
    '''window.addEventListener("pagehide", () => {
  _notifyCaptureState(true);
});''',
    content,
    flags=re.MULTILINE,
)

content = re.sub(
    r'''window\.addEventListener\("beforeunload",\s*\(\)\s*=>\s*\{\s*
\s*_postQueueResetKeepAlive\("beforeunload do Tinder: descartar fila antiga"\);\s*
\s*\}\);''',
    '''window.addEventListener("beforeunload", () => {
  // Não limpa fila automaticamente no unload; o reload aceito já faz a limpeza.
});''',
    content,
    flags=re.MULTILINE,
)

# 3) Substitui o intervalo final de “sem perfis”.
# Agora:
# - não limpa fila no começo;
# - só chama /reload-start depois de 5 minutos;
# - só recarrega se o servidor responder OK sem ignored.
content_out_interval = r'''
setInterval(() => {
  if (!_isCaptureActive()) return;
  if (_reloadScheduled) return;

  const outNow = _isOutOfProfiles();

  if (!outNow) {
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
    _queueResetSentForOutOfProfiles = false;
    _debugLog("[local] Sem perfil/card detectado. Aguardando 5 min antes de recarregar.");
    return;
  }

  const elapsed = now - _outOfProfilesAt;

  if (elapsed < OUT_OF_PROFILES_SLEEP_MS) {
    const remainMin = Math.ceil((OUT_OF_PROFILES_SLEEP_MS - elapsed) / 60_000);
    _debugLog(`[local] Sem perfis — ${remainMin} min restante(s) para reload automático`);
    return;
  }

  _postServer("/reload-start", {
    reason: "sem perfis detectados pela extensão (após 5min de espera)",
  })
    .then((ok) => {
      if (!ok) {
        const offlineNow = Date.now();
        if (offlineNow - _lastOfflineReloadNoticeAt > 60_000) {
          _debugLog("[local] Sem perfis, mas o servidor está offline. Auto-reload pausado.");
          _lastOfflineReloadNoticeAt = offlineNow;
        }
        _outOfProfilesAt = now;
        return null;
      }

      return fetch(SERVER_BASE + "/control")
        .then((res) => (res.ok ? res.json() : null))
        .catch(() => null);
    })
    .then((control) => {
      // Se o backend ignorou o reload por cooldown, não recarrega no braço.
      if (control && control.ignored) {
        _debugLog("[local] Reload de sem perfis ignorado pelo servidor:", control.ignored);
        _outOfProfilesAt = Date.now();
        return;
      }

      _debugLog("[local] 5 min sem perfis. Recarregando a página agora...");
      chrome.runtime.sendMessage({ type: "OUT_OF_PROFILES" });
      _markReloadScheduled();
      setTimeout(() => location.reload(), 500);
    });
}, 15_000);
'''

content = replace_bottom_out_of_profiles_interval(content, content_out_interval)

CONTENT_PATH.with_suffix(".js.bak").write_text(CONTENT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
CONTENT_PATH.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# background.js
# ─────────────────────────────────────────────────────────────

background = BACKGROUND_PATH.read_text(encoding="utf-8")

# 1) Insere constantes/estado do sync reload.
insert_after = 'let _lastControlCommandAt = 0;\n'
insert_block = r'''
const SYNC_RELOAD_MIN_PERSIST_MS = 3 * 60 * 1000;
const SYNC_RELOAD_COOLDOWN_MS = 3 * 60 * 1000;

let _syncReloadCandidateKey = "";
let _syncReloadCandidateSince = 0;
let _lastSyncReloadAppliedAt = 0;

function _isSyncFailureReason(reason) {
  const text = String(reason || "").toLowerCase();
  return (
    text.includes("sincronização") ||
    text.includes("sincronizacao") ||
    text.includes("sync") ||
    text.includes("fora da fila")
  );
}

function _syncReasonKey(data) {
  return [
    data?.requested_at || 0,
    data?.reason || "",
    data?.navigate_reason || "",
    data?.reload ? "reload" : "",
  ].join("|");
}

'''

if "SYNC_RELOAD_MIN_PERSIST_MS" not in background:
    if insert_after not in background:
        raise SystemExit("ERRO: ponto de inserção do background não encontrado")
    background = background.replace(insert_after, insert_after + insert_block, 1)

# 2) Substitui _applyControlCommandToTinderTabs.
background_apply_control = r'''
function _applyControlCommandToTinderTabs(data) {
  const reason = data.navigate_reason || data.reason || "reload solicitado pelo servidor";

  // Não faz navegação automática pelo background.
  // Navegação automática era uma fonte de reload/reset agressivo.
  if (data.navigate) {
    _debugLog("[local] Navegação automática ignorada pelo background:", reason);
    return;
  }

  // Só aceitamos reload por perda de sincronia.
  if (!data.reload || !_isSyncFailureReason(reason)) {
    _debugLog("[local] Reload não-sync ignorado pelo background:", reason);
    return;
  }

  const now = Date.now();
  const syncKey = _syncReasonKey(data);

  if (syncKey !== _syncReloadCandidateKey) {
    _syncReloadCandidateKey = syncKey;
    _syncReloadCandidateSince = now;
    _debugLog("[local] Sync reload candidato iniciado:", reason);
    return;
  }

  const persistentFor = now - _syncReloadCandidateSince;
  if (persistentFor < SYNC_RELOAD_MIN_PERSIST_MS) {
    _debugLog(
      `[local] Sync fora de fila ainda recente — aguardando ${Math.ceil(
        (SYNC_RELOAD_MIN_PERSIST_MS - persistentFor) / 1000
      )}s antes de reload`
    );
    return;
  }

  if (now - _lastSyncReloadAppliedAt < SYNC_RELOAD_COOLDOWN_MS) {
    _debugLog("[local] Sync reload ignorado por cooldown:", reason);
    return;
  }

  const key = _controlCommandKey(data);
  if (key && key === _lastControlCommandKey && now - _lastControlCommandAt < CONTROL_RETRY_SAME_COMMAND_MS) {
    return;
  }

  _lastControlCommandKey = key;
  _lastControlCommandAt = now;
  _lastSyncReloadAppliedAt = now;
  _syncReloadCandidateKey = "";
  _syncReloadCandidateSince = 0;

  chrome.tabs.query({ url: TINDER_TAB_URLS }, (tabs) => {
    if (chrome.runtime.lastError) {
      _debugError("[local] Falha ao procurar abas do Tinder:", chrome.runtime.lastError.message);
      return;
    }

    if (!tabs || !tabs.length) {
      _debugLog("[local] Servidor pediu reload de sync, mas nenhuma aba foi encontrada.");
      return;
    }

    _post("/queue-reset", { reason: `background reload após 3min de sync perdido: ${reason}` });

    for (const tab of tabs) {
      if (!tab || tab.id == null) continue;

      chrome.tabs.reload(tab.id, { bypassCache: true }, () => {
        if (chrome.runtime.lastError) {
          _debugError("[local] Falha ao recarregar aba do Tinder:", chrome.runtime.lastError.message);
        }
      });
    }
  });
}
'''

background = replace_function(background, "_applyControlCommandToTinderTabs", background_apply_control)

BACKGROUND_PATH.with_suffix(".js.bak").write_text(BACKGROUND_PATH.read_text(encoding="utf-8"), encoding="utf-8")
BACKGROUND_PATH.write_text(background, encoding="utf-8")

print("OK: content.js e background.js corrigidos.")
print("Backups criados:")
print(" - content.js.bak")
print(" - background.js.bak")
