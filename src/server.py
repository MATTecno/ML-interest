"""
Servidor local que recebe dados da extensão do Chrome.

Endpoints:
  POST /profiles   — leva recebida de perfis do Tinder
  POST /current    — perfil atualmente visível na tela (para sincronização)
  POST /browser-state — informa se a aba/janela do Tinder está ativa
  POST /modal     — modal bloqueante detectado pela extensão
  POST /queue-reset — limpa filas quando a tela ficou sem card ou vai recarregar
  POST /network-capture — salva eventos de rede capturados pela extensão
  POST /reloaded   — página do Tinder recarregada, limpa flags de resync
  GET  /control    — instruções simples para a extensão (ex: recarregar página)

Execute: python3 src/server.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import time
import hashlib
import threading
import queue
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))

from logging_config import get_logger, setup_logging

LOG_PATH = setup_logging()
logger = get_logger(__name__)

from import_json import process_response
from synthetic import ensure_synthetic_exists
import model as mdl
from swiper import handle_hotkey_action, start_pause_hotkey_listener, swiper_queue
from config import load_config
from profile_parser import calc_age
from resource_guard import (
    get_system_resource_snapshot as _resource_snapshot,
    is_memory_pressure,
)
from desktop_notify import notify as desktop_notify
import state

PORT = 5043
_batch_queue: "queue.Queue[dict]" = queue.Queue()
DEBUG_DIR = Path(__file__).parent.parent / "data" / "debug"
LAST_RESPONSE_PATH = DEBUG_DIR / "last_tinder_response.json"
NETWORK_CAPTURE_DIR = Path(__file__).parent.parent / "data" / "network_capture"
NETWORK_CAPTURE_MAX_STRING = 300_000
_recent_profile_batches: dict[str, float] = {}
_recent_profile_batches_lock = threading.Lock()
_memory_pressure_last_notice_at = 0.0
_memory_pressure_notice_lock = threading.Lock()
_MEMORY_PRESSURE_NOTICE_COOLDOWN_SECONDS = 15.0
_URL_GUARD_REASONS = {"wrong_tinder_url", "paywall_url"}
_URL_GUARD_NOTICE_COOLDOWN_SECONDS = 20.0
_url_guard_last_notice_at = 0.0
_url_guard_notice_lock = threading.Lock()


def _load_config() -> dict:
    return load_config()


def _get_max_pending_batches() -> int:
    cfg = _load_config().get("swiper", {})
    return int(cfg.get("max_pending_batches", 3) or 0)


def _get_system_resource_snapshot() -> dict[str, float]:
    return _resource_snapshot()


def _log_resource_snapshot(stage: str) -> dict[str, float]:
    snapshot = _get_system_resource_snapshot()
    logger.info(
        "Resource snapshot [%s]: cpu=%.1f%% load1=%.2f load5=%.2f load15=%.2f cores=%s mem=%.1f%% used=%.1fMB avail=%.1fMB pending_batches=%s",
        stage,
        snapshot["cpu_pct"],
        snapshot["load1"],
        snapshot["load5"],
        snapshot["load15"],
        int(snapshot["cpu_count"]),
        snapshot["mem_used_pct"],
        snapshot["mem_used_mb"],
        snapshot["mem_avail_mb"],
        _batch_queue.qsize(),
    )
    cfg = _load_config().get("swiper", {})
    cpu_threshold = float(cfg.get("cpu_alert_threshold_percent", 80) or 80)
    mem_threshold = float(cfg.get("mem_alert_threshold_percent", 85) or 85)
    if snapshot["cpu_pct"] >= cpu_threshold or snapshot["mem_used_pct"] >= mem_threshold:
        logger.warning(
            "Resource high usage [%s]: cpu=%.1f%% mem=%.1f%% load1=%.2f/%s mem_used=%.1fMB avail=%.1fMB",
            stage,
            snapshot["cpu_pct"],
            snapshot["mem_used_pct"],
            snapshot["load1"],
            int(snapshot["cpu_count"]),
            snapshot["mem_used_mb"],
            snapshot["mem_avail_mb"],
        )
        desktop_notify(
            "resource_high",
            "Tinder IA: uso alto do computador",
            f"CPU {snapshot['cpu_pct']:.0f}% · memoria {snapshot['mem_used_pct']:.0f}% · livres {snapshot['mem_avail_mb']:.0f}MB",
            urgency="normal",
        )

    pressure, reason, _ = is_memory_pressure(_load_config(), snapshot=snapshot)
    if pressure:
        logger.warning(
            "Resource memory pressure [%s]: %s mem=%.1f%% avail=%.1fMB",
            stage,
            reason,
            snapshot["mem_used_pct"],
            snapshot["mem_avail_mb"],
        )

    return snapshot


def _memory_pressure_notice_allowed() -> bool:
    global _memory_pressure_last_notice_at
    now = time.time()
    with _memory_pressure_notice_lock:
        if now - _memory_pressure_last_notice_at < _MEMORY_PRESSURE_NOTICE_COOLDOWN_SECONDS:
            return False
        _memory_pressure_last_notice_at = now
        return True


def _handle_memory_pressure(
    stage: str,
    snapshot: dict[str, float] | None = None,
    clear_processing: bool = True,
) -> tuple[bool, dict]:
    pressure, reason, snap = is_memory_pressure(_load_config(), snapshot=snapshot)
    if not pressure:
        return False, {"reason": "", "snapshot": snap}

    if clear_processing:
        cleared = _clear_processing_queues(f"memory_pressure:{reason}")
    else:
        cleared = {
            "pending_batches_cleared": _clear_pending_batches(),
            "pending_swipes_cleared": 0,
        }

    logger.warning(
        "Memory pressure guard [%s]: %s mem=%.1f%% avail=%.1fMB cleared_batches=%s cleared_swipes=%s",
        stage,
        reason,
        snap.get("mem_used_pct", 0.0),
        snap.get("mem_avail_mb", 0.0),
        cleared.get("pending_batches_cleared", 0),
        cleared.get("pending_swipes_cleared", 0),
    )
    if _memory_pressure_notice_allowed():
        with state.terminal_lock:
            print(
                "  [server] Memória crítica "
                f"({snap.get('mem_used_pct', 0.0):.1f}% usada, "
                f"{snap.get('mem_avail_mb', 0.0):.0f}MB livres) — "
                "pausando novos lotes e limpando filas para evitar travamento"
            )
        desktop_notify(
            "memory_pressure",
            "Tinder IA pausou por memoria critica",
            f"{snap.get('mem_used_pct', 0.0):.1f}% usada · {snap.get('mem_avail_mb', 0.0):.0f}MB livres",
            urgency="critical",
        )

    return True, {
        "reason": reason,
        "snapshot": snap,
        **cleared,
    }


def _clear_pending_batches() -> int:
    dropped = 0
    while True:
        try:
            _batch_queue.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    if dropped:
        logger.warning("Batches pendentes descartados: %s", dropped)
    return dropped


def _clear_processing_queues(reason: str) -> dict:
    """Limpa tudo que foi calculado para a tela atual do Tinder."""
    dropped_batches = _clear_pending_batches()
    dropped_swipes = swiper_queue.clear_pending()
    state.clear_active_profiles()
    state.set_current("", "", 0)
    logger.warning(
        "Processamento atual limpo: reason=%r pending_batches=%s pending_swipes=%s generation=%s",
        reason,
        dropped_batches,
        dropped_swipes,
        state.get_reload_generation(),
    )
    if dropped_batches or dropped_swipes:
        if str(reason or "").startswith("memory_pressure:"):
            message = "Memória crítica — limpando fila atual"
        else:
            message = "Tela sem perfis/reload — limpando fila atual"
        with state.terminal_lock:
            print(
                f"  [server] {message} "
                f"({dropped_batches} lote(s), {dropped_swipes} swipe(s))"
            )
        desktop_notify(
            "queue_cleared",
            "Tinder IA limpou a fila atual",
            f"{dropped_batches} lote(s), {dropped_swipes} swipe(s) descartado(s)",
            urgency="normal",
        )
    return {
        "pending_batches_cleared": dropped_batches,
        "pending_swipes_cleared": dropped_swipes,
    }


def _make_room_for_new_batch() -> int:
    max_pending = _get_max_pending_batches()
    if max_pending <= 0:
        return 0

    dropped = 0
    while _batch_queue.qsize() >= max_pending:
        try:
            _batch_queue.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    return dropped


def _get_recs_dedupe_ttl_seconds() -> float:
    cfg = _load_config()
    browser_cfg = cfg.get("browser", {}) or {}
    swiper_cfg = cfg.get("swiper", {}) or {}
    return float(browser_cfg.get("recs_dedupe_ttl_seconds", swiper_cfg.get("recs_dedupe_ttl_seconds", 600)) or 0)


def _browser_url_guard_cfg() -> dict:
    browser_cfg = _load_config().get("browser", {}) or {}
    guard_cfg = browser_cfg.get("enforce_recs_url", {}) or {}
    if not isinstance(guard_cfg, dict):
        guard_cfg = {"enabled": bool(guard_cfg)}
    return {
        **guard_cfg,
        "_browser": browser_cfg,
    }


def _canonical_recs_url(guard_cfg: dict | None = None) -> str:
    cfg = guard_cfg or _browser_url_guard_cfg()
    browser_cfg = cfg.get("_browser", {}) or {}
    return str(
        cfg.get("canonical_url")
        or browser_cfg.get("tinder_url")
        or "https://tinder.com/app/recs"
    ).strip()


def _allowed_recs_prefixes(guard_cfg: dict | None = None) -> list[str]:
    cfg = guard_cfg or _browser_url_guard_cfg()
    raw = cfg.get("allowed_path_prefixes") or ["/app/recs"]
    if isinstance(raw, str):
        raw = [raw]
    prefixes = [str(item or "").strip() for item in raw if str(item or "").strip()]
    return prefixes or ["/app/recs"]


def _is_tinder_url(url: str) -> bool:
    try:
        host = urlparse(url or "").hostname or ""
    except Exception:
        return False
    return host == "tinder.com" or host.endswith(".tinder.com")


def _is_allowed_recs_url(url: str, guard_cfg: dict | None = None) -> bool:
    if not _is_tinder_url(url):
        return False
    try:
        path = urlparse(url or "").path or "/"
    except Exception:
        return False
    for prefix in _allowed_recs_prefixes(guard_cfg):
        clean = prefix.rstrip("/") or "/"
        if path == clean or path.startswith(clean + "/"):
            return True
    return False


def _url_guard_notice_allowed() -> bool:
    global _url_guard_last_notice_at
    now = time.time()
    with _url_guard_notice_lock:
        if now - _url_guard_last_notice_at < _URL_GUARD_NOTICE_COOLDOWN_SECONDS:
            return False
        _url_guard_last_notice_at = now
        return True


def _maybe_request_recs_navigation(active: bool, reason: str, url: str) -> None:
    guard_cfg = _browser_url_guard_cfg()
    if not bool(guard_cfg.get("enabled", False)):
        return

    if active and _is_allowed_recs_url(url, guard_cfg):
        pending, target, _, _, _ = state.peek_navigation_request()
        if pending and target == _canonical_recs_url(guard_cfg):
            state.clear_navigation_request()
        return

    if reason not in _URL_GUARD_REASONS or not _is_tinder_url(url):
        return

    canonical = _canonical_recs_url(guard_cfg)
    if not canonical or _is_allowed_recs_url(url, guard_cfg):
        state.clear_navigation_request()
        return

    try:
        cooldown = float(guard_cfg.get("wrong_url_cooldown_seconds", 20) or 20)
    except Exception:
        cooldown = 20.0

    pending, target, _, requested_at, _ = state.peek_navigation_request()
    now = time.time()
    if pending and target == canonical and requested_at and now - requested_at < cooldown:
        return

    label = "tela de compra/paywall" if reason == "paywall_url" else "URL fora do swipe"
    nav_reason = f"{label}: {url} -> {canonical}"
    generation = state.request_navigation(canonical, nav_reason)
    cleared = _clear_processing_queues(nav_reason)
    logger.warning(
        "URL guard solicitou retorno ao swipe generation=%s url=%r canonical=%r cleared=%s",
        generation,
        url,
        canonical,
        cleared,
    )
    if _url_guard_notice_allowed():
        desktop_notify(
            "wrong_tinder_url",
            "Tinder IA voltando para o swipe",
            str(label)[:120],
            urgency="normal",
        )


def _make_profiles_batch_key(data: dict) -> tuple[str, int]:
    """Chave estável para ignorar a mesma resposta de /v2/recs/core."""
    results_raw = data.get("data", {}).get("results", [])
    parts: list[str] = []
    for result in results_raw:
        if not isinstance(result, dict) or result.get("type") != "user":
            continue
        user = result.get("user") or {}
        if not isinstance(user, dict):
            user = {}
        uid = str(user.get("_id") or "").strip()
        content_hash = str(result.get("content_hash") or "").strip()
        s_number = str(result.get("s_number") or "").strip()
        name = str(user.get("name") or "").strip()
        birth_date = str(user.get("birth_date") or "").strip()
        if uid or content_hash or s_number or name:
            parts.append("|".join([uid, content_hash, s_number, name, birth_date]))

    if not parts:
        return "", 0
    digest = hashlib.sha1("\n".join(sorted(parts)).encode("utf-8")).hexdigest()
    return digest, len(parts)


def _is_duplicate_profiles_batch(data: dict) -> tuple[bool, str, int]:
    ttl = _get_recs_dedupe_ttl_seconds()
    if ttl <= 0:
        return False, "", 0

    key, n_profiles = _make_profiles_batch_key(data)
    if not key:
        return False, "", n_profiles

    now = time.time()
    with _recent_profile_batches_lock:
        expired = [k for k, seen_at in _recent_profile_batches.items() if now - seen_at > ttl]
        for k in expired:
            _recent_profile_batches.pop(k, None)
        if key in _recent_profile_batches:
            _recent_profile_batches[key] = now
            return True, key, n_profiles
        _recent_profile_batches[key] = now
        if len(_recent_profile_batches) > 200:
            ordered = sorted(_recent_profile_batches.items(), key=lambda item: item[1])
            for old_key, _ in ordered[: len(_recent_profile_batches) - 200]:
                _recent_profile_batches.pop(old_key, None)
    return False, key, n_profiles


def _save_last_profiles_payload(data: dict) -> None:
    """Salva a última leva recebida para diagnóstico offline."""
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        LAST_RESPONSE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Ultima resposta do Tinder salva para debug: %s", LAST_RESPONSE_PATH)
    except Exception:
        logger.exception("Falha ao salvar resposta de debug: %s", LAST_RESPONSE_PATH)


def _is_sensitive_network_key(key: object) -> bool:
    normalized = str(key or "").strip().lower()
    return any(part in normalized for part in ("authorization", "cookie", "token", "secret"))


def _trim_network_capture_value(value, depth: int = 0, key: object = ""):
    if depth > 8:
        return "<max_depth>"
    if key and _is_sensitive_network_key(key):
        return "<redacted>"
    if isinstance(value, str):
        if len(value) <= NETWORK_CAPTURE_MAX_STRING:
            return value
        return value[:NETWORK_CAPTURE_MAX_STRING] + f"...<truncated {len(value) - NETWORK_CAPTURE_MAX_STRING} chars>"
    if isinstance(value, list):
        return [_trim_network_capture_value(item, depth + 1) for item in value[:500]]
    if isinstance(value, dict):
        return {
            str(child_key)[:200]: _trim_network_capture_value(item, depth + 1, child_key)
            for child_key, item in list(value.items())[:500]
        }
    return value


def _is_useful_network_capture_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or ""))
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if host != "api.gotinder.com":
        return False
    if "/recs/" in path or "/v2/recs" in path:
        return True
    if path.startswith("/like/") or path.startswith("/pass/"):
        return True
    if path == "/updates" or path.startswith("/updates/"):
        return True
    if path == "/v2/profile" or path.startswith("/v2/profile/"):
        return True
    if path == "/v2/fast-match/teaser" or path.startswith("/v2/fast-match/teaser/"):
        return True
    return False


def _is_useful_network_capture_event(event: dict) -> bool:
    return _is_useful_network_capture_url(str(event.get("url") or ""))


def _network_capture_path() -> Path:
    day = datetime.now().strftime("%Y%m%d")
    return NETWORK_CAPTURE_DIR / f"network_capture_{day}.jsonl"


def _save_network_capture_payload(data: dict) -> tuple[int, str]:
    raw_events = data.get("events")
    if isinstance(raw_events, list):
        events = raw_events
    else:
        events = [data]

    NETWORK_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = _network_capture_path()
    latest = NETWORK_CAPTURE_DIR / "network_capture_latest.jsonl"
    now = datetime.now().isoformat(timespec="milliseconds")
    saved = 0
    with open(path, "a", encoding="utf-8") as f, open(latest, "a", encoding="utf-8") as latest_f:
        for event in events:
            if not isinstance(event, dict):
                continue
            if not _is_useful_network_capture_event(event):
                continue
            record = _trim_network_capture_value({
                "server_received_at": now,
                **event,
            })
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            f.write(line)
            latest_f.write(line)
            saved += 1
    return saved, str(path.relative_to(Path(__file__).parent.parent))


def _iter_network_capture_events(data: dict):
    raw_events = data.get("events")
    if isinstance(raw_events, list):
        for event in raw_events:
            if isinstance(event, dict):
                yield event
    elif isinstance(data, dict):
        yield data


def _parse_swipe_url(url: str) -> tuple[str, str]:
    try:
        parsed = urlparse(str(url or ""))
    except Exception:
        return "", ""
    if (parsed.hostname or "").lower() != "api.gotinder.com":
        return "", ""
    parts = [part for part in (parsed.path or "").split("/") if part]
    if len(parts) >= 2 and parts[0] in {"like", "pass"}:
        return parts[0], parts[1]
    return "", ""


def _parse_capture_json_body(container: dict | None) -> dict:
    if not isinstance(container, dict):
        return {}
    body = container.get("body")
    if not body or not isinstance(body, str):
        return {}
    try:
        parsed = json.loads(body)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_profile_super_likes(event: dict) -> dict | None:
    try:
        parsed = urlparse(str(event.get("url") or ""))
    except Exception:
        return None
    if (parsed.hostname or "").lower() != "api.gotinder.com":
        return None
    if parsed.path != "/v2/profile" and not parsed.path.startswith("/v2/profile/"):
        return None

    response_body = _parse_capture_json_body(event.get("response"))
    data = response_body.get("data") if isinstance(response_body, dict) else {}
    if not isinstance(data, dict):
        return None
    super_likes = data.get("super_likes")
    if not isinstance(super_likes, dict):
        return None

    def pick_int(name: str) -> int | None:
        return _as_int_or_none(super_likes.get(name))

    remaining = pick_int("remaining")
    alc_remaining = pick_int("alc_remaining")
    new_alc_remaining = pick_int("new_alc_remaining")
    total = sum(v for v in (remaining, alc_remaining, new_alc_remaining) if isinstance(v, int) and v > 0)
    return {
        "remaining": remaining,
        "alc_remaining": alc_remaining,
        "new_alc_remaining": new_alc_remaining,
        "allotment": pick_int("allotment"),
        "resets_at": super_likes.get("resets_at") or "",
        "available": total > 0,
        "total_available": total,
        "source": "v2_profile_network",
    }


def _as_int_or_none(value):
    try:
        if value in ("", None):
            return None
        return int(value)
    except Exception:
        return None


def _process_network_capture_events(data: dict) -> int:
    confirmed = 0
    for event in _iter_network_capture_events(data):
        if str(event.get("phase") or "").lower() != "complete":
            continue
        balance = _parse_profile_super_likes(event)
        if balance is not None:
            state.set_super_like_balance(balance)
            if not balance.get("available"):
                desktop_notify(
                    "super_like_empty",
                    "Tinder IA: sem Super Likes",
                    "Saldo de Super Likes veio zerado no network; proximas curtidas fortes viram curtidas normais.",
                    urgency="normal",
                )
            continue
        if str(event.get("method") or "").upper() != "POST":
            continue
        action, tinder_id = _parse_swipe_url(str(event.get("url") or ""))
        if not action or not tinder_id:
            continue

        status = _as_int_or_none(event.get("status"))
        if status is not None and not (200 <= status < 300):
            continue

        response_body = _parse_capture_json_body(event.get("response"))
        match = response_body.get("match")
        if match is not None:
            match = bool(match)
        likes_remaining = _as_int_or_none(response_body.get("likes_remaining"))

        state.mark_recent_swipe(
            tinder_id=tinder_id,
            action=action,
            source=str(event.get("source") or event.get("capture_type") or "network_capture"),
            status=status,
            match=match,
            likes_remaining=likes_remaining,
        )
        removed = swiper_queue.acknowledge_network_swipe(tinder_id, action=action, status=status)
        confirmed += 1
        logger.info(
            "Swipe confirmado via rede: action=%s id=%s status=%s match=%r likes_remaining=%r removed_pending=%s",
            action,
            tinder_id,
            status,
            match,
            likes_remaining,
            removed,
        )
        if match is True or (likes_remaining is not None and likes_remaining <= 5):
            with state.terminal_lock:
                if match is True:
                    print("  [network] Match confirmado pelo Tinder")
                    desktop_notify("match", "Tinder IA: match confirmado", "", urgency="normal")
                if likes_remaining is not None and likes_remaining <= 5:
                    print(f"  [network] Curtidas restantes: {likes_remaining}")
                    desktop_notify(
                        "likes_low",
                        "Tinder IA: poucas curtidas restantes",
                        f"Curtidas restantes: {likes_remaining}",
                        urgency="normal",
                    )
    return confirmed


def _first_prompt_loader(stop_event: threading.Event) -> None:
    """
    Mostra um pequeno loading enquanto o primeiro prompt interativo
    da leva ainda não apareceu.
    """
    frames = ["   ", ".  ", ".. ", "..."]
    i = 0
    logger.debug("Loading do primeiro prompt iniciado")

    while not stop_event.is_set():
        if state.is_prompt_active():
            break

        acquired = state.terminal_lock.acquire(timeout=0.05)
        if acquired:
            try:
                sys.stdout.write(
                    "\r  [loading] preparando primeiro perfil" + frames[i % len(frames)]
                )
                sys.stdout.flush()
            finally:
                state.terminal_lock.release()

        i += 1
        if stop_event.wait(0.35):
            break

    acquired = state.terminal_lock.acquire(timeout=0.2)
    if acquired:
        try:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
        finally:
            state.terminal_lock.release()
    logger.debug("Loading do primeiro prompt finalizado")


def _process_profiles_batch(data: dict) -> None:
    started_at = time.perf_counter()
    batch_generation = int(data.get("_server_generation", state.get_reload_generation()) or 0)
    current_generation = state.get_reload_generation()
    if batch_generation != current_generation:
        logger.warning(
            "Leva descartada por geracao antiga: batch_generation=%s current_generation=%s",
            batch_generation,
            current_generation,
        )
        return

    pressure, _ = _handle_memory_pressure("batch_start", clear_processing=True)
    if pressure:
        logger.warning("Leva descartada antes do processamento por memoria critica")
        return

    results_raw = data.get("data", {}).get("results", [])
    n = len([r for r in results_raw if r.get("type") == "user"])
    batch_ids = []
    batch_names = []
    batch_name_ages = []

    for result in results_raw:
        if result.get("type") != "user":
            continue
        user = result.get("user", {})
        if user.get("_id"):
            batch_ids.append(user["_id"])
        if user.get("name"):
            batch_names.append(user["name"])
            batch_name_ages.append((user["name"], calc_age(user.get("birth_date", ""))))

    cfg = _load_config()
    swiper_cfg = cfg.get("swiper", {})
    swipe_enabled = swiper_cfg.get("enabled", False)
    interactive_mode = swiper_cfg.get("interactive_mode", False)
    enqueued = 0
    loading_stop = threading.Event()
    loading_thread = None
    logger.info(
        "Processando nova leva: profiles=%s swipe_enabled=%s interactive=%s queue_size=%s",
        n,
        swipe_enabled,
        interactive_mode,
        _batch_queue.qsize(),
    )
    logger.debug("Leva ativa ids=%s nomes=%s", batch_ids, batch_names)

    memory_cancelled = False

    def should_cancel_batch() -> bool:
        nonlocal memory_cancelled
        if memory_cancelled:
            return True
        if state.get_reload_generation() != batch_generation:
            return True
        pressure_now, _ = _handle_memory_pressure("batch_process", clear_processing=True)
        if pressure_now:
            memory_cancelled = True
            return True
        return False

    def on_profile_ready(profile, result):
        nonlocal enqueued
        if state.is_swipe_stop_requested():
            logger.warning("Perfil pronto ignorado: swipes encerrados por hotkey name=%r", profile.get("name"))
            return

        if should_cancel_batch():
            logger.warning(
                "Perfil pronto ignorado por cancelamento do lote: name=%r batch_generation=%s current_generation=%s",
                profile.get("name"),
                batch_generation,
                state.get_reload_generation(),
            )
            return

        profile["_skip_prompt"] = result.get("model_type") == "filtro"
        profile["_filter_reason"] = result.get("filter_reason", "")
        logger.info(
            "Perfil pronto: name=%r age=%r decision=%s model=%s skip_prompt=%s reason=%r",
            profile.get("name"),
            profile.get("age"),
            result.get("decision"),
            result.get("model_type"),
            profile.get("_skip_prompt"),
            profile.get("_filter_reason"),
        )
        if swipe_enabled and state.is_scheduler_active():
            swiper_queue.add([(profile, result["decision"])])
            enqueued += 1
            logger.debug("Perfil enfileirado para swipe: name=%r pending=%s", profile.get("name"), swiper_queue.pending())

    with state.terminal_lock:
        print(f"\n{'='*60}")
        print(f"  Nova leva: {n} perfil(is) recebido(s)")
        if not state.is_scheduler_active():
            print("  [scheduler] Fora do horário ativo — classificando mas não swipando")
        print(f"{'='*60}")

    if should_cancel_batch():
        logger.warning("Leva cancelada antes de registrar perfis ativos: generation=%s", batch_generation)
        return

    state.add_active_profiles(batch_ids, batch_names, batch_name_ages)

    if interactive_mode:
        loading_thread = threading.Thread(
            target=_first_prompt_loader,
            args=(loading_stop,),
            daemon=True,
            name="first-prompt-loader",
        )
        loading_thread.start()

    try:
        process_response(
            data,
            show_explanation=not interactive_mode,
            on_profile_ready=on_profile_ready,
            interactive_mode=interactive_mode,
            should_cancel=should_cancel_batch,
        )
        logger.info(
            "Leva processada com sucesso: profiles=%s enqueued=%s elapsed=%.2fs",
            n,
            enqueued,
            time.perf_counter() - started_at,
        )
    finally:
        loading_stop.set()
        if loading_thread is not None:
            loading_thread.join(timeout=1.0)

    if enqueued and not interactive_mode:
        with state.terminal_lock:
            print(f"  [swipe] {enqueued} swipe(s) enfileirado(s)\n")


def _profiles_loop() -> None:
    logger.info("Thread de processamento de levas iniciada")
    while True:
        data = _batch_queue.get()
        try:
            _process_profiles_batch(data)
        except Exception as e:
            logger.exception("Erro ao processar lote")
            with state.terminal_lock:
                print(f"\n  [server] ⚠ Erro ao processar lote: {type(e).__name__}: {e}\n")
        finally:
            _batch_queue.task_done()


# ─── Scheduler ───────────────────────────────────────────────────────────────

_DAY_MAP = {
    "seg": 0, "ter": 1, "qua": 2, "qui": 3,
    "sex": 4, "sab": 5, "sáb": 5, "dom": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}


def _parse_time(t: str) -> tuple[int, int]:
    """Converte '19:30' → (19, 30)."""
    h, m = t.split(":")
    return int(h), int(m)


def _minutes_since_midnight(h: int, m: int) -> int:
    return h * 60 + m


def _is_within_window(window: dict, now: datetime) -> bool:
    """Verifica se 'now' está dentro de uma janela de agendamento."""
    days = [_DAY_MAP.get(d.lower(), -1) for d in window.get("days", [])]
    if now.weekday() not in days:
        return False

    start_h, start_m = _parse_time(window.get("start", "00:00"))
    end_h, end_m = _parse_time(window.get("end", "23:59"))

    now_min = _minutes_since_midnight(now.hour, now.minute)
    start_min = _minutes_since_midnight(start_h, start_m)
    end_min = _minutes_since_midnight(end_h, end_m)

    if end_min <= start_min:  # janela atravessa meia-noite (ex: 22:00 → 01:00)
        return now_min >= start_min or now_min <= end_min
    return start_min <= now_min <= end_min


def _scheduler_loop() -> None:
    """Thread que verifica o horário a cada minuto e atualiza o estado."""
    logger.info("Thread scheduler iniciada")
    while True:
        try:
            cfg = _load_config().get("scheduler", {})
            if not cfg.get("enabled", False):
                state.set_scheduler_active(True)
                time.sleep(60)
                continue

            windows = cfg.get("windows", [])
            now = datetime.now()
            active = any(_is_within_window(w, now) for w in windows)

            was_active = state.is_scheduler_active()
            state.set_scheduler_active(active)

            if active and not was_active:
                logger.info("Scheduler ativado as %s", now.strftime("%H:%M"))
                print(f"\n  [scheduler] {now.strftime('%H:%M')} — INICIANDO (dentro do horário ativo)")
            elif not active and was_active:
                logger.info("Scheduler pausado as %s", now.strftime("%H:%M"))
                print(f"\n  [scheduler] {now.strftime('%H:%M')} — PAUSANDO (fora do horário ativo)")

            time.sleep(60)
        except Exception:
            logger.exception("Erro no scheduler_loop")
            time.sleep(10)


# ─── Handler HTTP ─────────────────────────────────────────────────────────────

class TinderHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        try:
            if self.path == "/control":
                self._handle_control()
            else:
                self._respond(404, {"error": "not found"})
        except Exception as e:
            logger.exception("Erro ao tratar GET %s", self.path)
            self._respond(500, {"error": f"{type(e).__name__}: {e}"})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except Exception as e:
            logger.exception("JSON invalido recebido em %s", self.path)
            self._respond(400, {"error": str(e)})
            return

        try:
            if self.path == "/profiles":
                self._handle_profiles(data)
            elif self.path == "/current":
                self._handle_current(data)
            elif self.path == "/reload-start":
                self._handle_reload_start(data)
            elif self.path == "/queue-reset":
                self._handle_queue_reset(data)
            elif self.path == "/network-capture":
                self._handle_network_capture(data)
            elif self.path == "/reloaded":
                self._handle_reloaded()
            elif self.path == "/browser-state":
                self._handle_browser_state(data)
            elif self.path == "/modal":
                self._handle_modal(data)
            elif self.path == "/hotkey":
                self._handle_hotkey(data)
            else:
                self._respond(404, {"error": "not found"})
        except Exception as e:
            logger.exception("Erro ao tratar POST %s", self.path)
            self._respond(500, {"error": f"{type(e).__name__}: {e}"})

    def _handle_profiles(self, data: dict) -> None:
        if state.is_swipe_stop_requested():
            logger.warning("POST /profiles ignorado: swipes encerrados por hotkey")
            self._respond(200, {"ok": True, "ignored": "swipe_stop_requested"})
            return

        if _load_config().get("browser", {}).get("pause_capture_when_unfocused", True):
            if not state.is_tinder_capture_active():
                active, reason, url, age = state.get_tinder_capture_state()
                logger.warning(
                    "POST /profiles ignorado: captura Tinder inativa reason=%r url=%r age=%.1fs",
                    reason,
                    url,
                    age,
                )
                self._respond(200, {"ok": True, "ignored": "capture_inactive"})
                return

        is_duplicate, batch_key, n_profiles = _is_duplicate_profiles_batch(data)
        if is_duplicate:
            logger.info(
                "POST /profiles ignorado: leva repetida key=%s profiles=%s",
                batch_key[:12],
                n_profiles,
            )
            self._respond(200, {"ok": True, "ignored": "duplicate_recs_batch", "profiles": n_profiles})
            return

        snapshot = _log_resource_snapshot("profiles_receive")
        memory_pressure, pressure_info = _handle_memory_pressure(
            "profiles_receive",
            snapshot=snapshot,
            clear_processing=True,
        )
        if memory_pressure:
            self._respond(
                200,
                {
                    "ok": True,
                    "ignored": "memory_pressure",
                    "reason": pressure_info.get("reason", ""),
                    "mem_used_pct": round(pressure_info["snapshot"].get("mem_used_pct", 0.0), 1),
                    "mem_avail_mb": round(pressure_info["snapshot"].get("mem_avail_mb", 0.0), 1),
                    "pending_batches_cleared": pressure_info.get("pending_batches_cleared", 0),
                    "pending_swipes_cleared": pressure_info.get("pending_swipes_cleared", 0),
                },
            )
            return

        dropped_old = _make_room_for_new_batch()
        if dropped_old:
            logger.warning(
                "POST /profiles: lista de batches cheia, descartando %s lote(s) antigos para manter os perfis novos",
                dropped_old,
            )
            with state.terminal_lock:
                print(
                    f"  [server] Fila cheia — descartando {dropped_old} lote(s) antigos para inserir os perfis mais recentes"
                )
            desktop_notify(
                "queue_overflow",
                "Tinder IA descartou lotes antigos",
                f"Fila de batches cheia: {dropped_old} lote(s) removido(s).",
                urgency="normal",
            )

        _save_last_profiles_payload(data)
        data["_server_generation"] = state.get_reload_generation()
        _batch_queue.put(data)
        logger.info(
            "POST /profiles recebido e enfileirado queue_size=%s generation=%s batch_key=%s",
            _batch_queue.qsize(),
            data["_server_generation"],
            batch_key[:12] if batch_key else "",
        )
        self._respond(200, {"ok": True})

    def _handle_current(self, data: dict) -> None:
        if _load_config().get("browser", {}).get("pause_capture_when_unfocused", True):
            if not state.is_tinder_capture_active():
                logger.debug("POST /current ignorado: captura Tinder inativa")
                self._respond(200, {"ok": True, "ignored": "capture_inactive"})
                return

        name = data.get("name", "").strip()
        tinder_id = data.get("tinder_id", "").strip()
        age = int(data.get("age", 0) or 0)
        super_like_available = data.get("super_like_available")
        if super_like_available is not None:
            super_like_available = bool(super_like_available)
        super_like_reason = str(data.get("super_like_reason") or "").strip()
        if (name or tinder_id) and state.belongs_to_active_profiles(name, tinder_id, age):
            prev_name, prev_id, prev_age, _ = state.get_current_meta()
            state.set_current(
                name,
                tinder_id,
                age,
                super_like_available=super_like_available,
                super_like_reason=super_like_reason,
            )
            if (prev_name, prev_id, prev_age) != (name, tinder_id, age):
                logger.info(
                    "POST /current aceito name=%r age=%s id=%r super_like=%r reason=%r",
                    name,
                    age,
                    tinder_id,
                    super_like_available,
                    super_like_reason,
                )
        else:
            logger.debug("POST /current ignorado name=%r age=%s id=%r", name, age, tinder_id)
        self._respond(200, {"ok": True})

    def _handle_browser_state(self, data: dict) -> None:
        active = bool(data.get("active", False))
        reason = data.get("reason", "")
        url = data.get("url", "")
        if not _load_config().get("browser", {}).get("pause_capture_when_unfocused", True):
            if reason in {"window_unfocused", "tab_hidden"}:
                active = True
                reason = f"{reason}_ignored"
        state.set_tinder_capture_active(active, reason, url)
        if not active:
            state.set_current("", "", 0)
        _maybe_request_recs_navigation(active, reason, url)
        self._respond(200, {"ok": True})

    def _handle_modal(self, data: dict) -> None:
        if _load_config().get("browser", {}).get("pause_capture_when_unfocused", True):
            if not state.is_tinder_capture_active():
                self._respond(200, {"ok": True, "ignored": "capture_inactive"})
                return

        kind = str(data.get("kind") or "").strip()
        target = str(data.get("target") or "").strip()
        dialog_text = str(data.get("dialog_text") or "").strip()
        try:
            screen_x = int(round(float(data.get("screen_x") or 0)))
            screen_y = int(round(float(data.get("screen_y") or 0)))
        except Exception:
            screen_x = 0
            screen_y = 0

        if kind != "super_like_upsell" or (screen_x == 0 and screen_y == 0):
            logger.debug("POST /modal ignorado kind=%r x=%s y=%s", kind, screen_x, screen_y)
            self._respond(200, {"ok": True, "ignored": "invalid_modal"})
            return

        state.set_blocking_modal_target(kind, screen_x, screen_y, target)
        state.mark_super_likes_depleted("modal_super_like_upsell")
        desktop_notify(
            "super_like_modal",
            "Tinder IA fechando popup de Super Like",
            "Sem saldo de Super Likes; o perfil deve receber curtida normal.",
            urgency="normal",
        )
        logger.info(
            "POST /modal aceito kind=%r target=%r x=%s y=%s text=%r",
            kind,
            target,
            screen_x,
            screen_y,
            dialog_text[:160],
        )
        self._respond(200, {"ok": True})

    def _handle_hotkey(self, data: dict) -> None:
        action = str(data.get("action") or "").strip().lower()
        source = str(data.get("source") or "browser").strip().lower() or "browser"
        ok = handle_hotkey_action(action, source)
        logger.warning("POST /hotkey action=%r source=%r ok=%s", action, source, ok)
        self._respond(200, {"ok": ok})

    def _handle_reloaded(self) -> None:
        logger.info("POST /reloaded recebido; limpando estado de reload/perfis ativos")
        reload_now, reason, _ = state.peek_reload_request()
        if reload_now:
            cleared = _clear_processing_queues(f"pagina recarregada apos reload solicitado: {reason}")
            desktop_notify(
                "reload_done",
                "Tinder IA detectou reload concluido",
                str(reason or "")[:180],
                urgency="normal",
            )
        else:
            cleared = {"pending_batches_cleared": 0, "pending_swipes_cleared": 0}
            state.clear_active_profiles()
            state.set_current("", "", 0)
        state.clear_reload_request()
        state.clear_navigation_request()
        self._respond(200, {"ok": True, **cleared})

    def _handle_reload_start(self, data: dict) -> None:
        reason = data.get("reason") or "reload solicitado pela extensão"
        generation = state.request_reload(reason)
        cleared = _clear_processing_queues(reason)
        logger.warning(
            "POST /reload-start recebido generation=%s reason=%r pending_batches=%s pending_swipes=%s",
            generation,
            reason,
            cleared["pending_batches_cleared"],
            cleared["pending_swipes_cleared"],
        )
        desktop_notify(
            "reload_start",
            "Tinder IA vai recarregar o Tinder",
            str(reason or "")[:180],
            urgency="normal",
        )
        self._respond(200, {"ok": True, "generation": generation, **cleared})

    def _handle_queue_reset(self, data: dict) -> None:
        reason = data.get("reason") or "estado da tela invalidou a fila atual"
        cleared = _clear_processing_queues(reason)
        self._respond(200, {"ok": True, **cleared})

    def _handle_network_capture(self, data: dict) -> None:
        saved, path = _save_network_capture_payload(data)
        confirmed_swipes = _process_network_capture_events(data)
        logger.debug("POST /network-capture salvo events=%s path=%s confirmed_swipes=%s", saved, path, confirmed_swipes)
        self._respond(200, {"ok": True, "saved": saved, "path": path, "confirmed_swipes": confirmed_swipes})

    def _handle_control(self) -> None:
        reload_now, reason, requested_at = state.peek_reload_request()
        nav_now, nav_url, nav_reason, nav_requested_at, nav_generation = state.peek_navigation_request()
        self._respond(
            200,
            {
                "reload": reload_now,
                "reason": reason,
                "requested_at": requested_at,
                "navigate": nav_now,
                "navigate_url": nav_url,
                "navigate_reason": nav_reason,
                "navigate_requested_at": nav_requested_at,
                "navigate_generation": nav_generation,
                "capture_active": state.is_tinder_capture_active(),
            },
        )

    def _respond(self, code: int, body: dict) -> None:
        try:
            self.send_response(code)
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
        except BrokenPipeError:
            pass

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")

    def log_message(self, format, *args):
        pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("Servidor iniciando")
    state.clear_swipe_stop_request()
    cfg = _load_config()
    swiper_cfg = cfg.get("swiper", {})
    scheduler_cfg = cfg.get("scheduler", {})
    browser_cfg = cfg.get("browser", {})

    print()
    print("  Tinder-IA — Servidor local")
    print("  " + "-" * 44)
    print(f"  Aguardando em http://localhost:{PORT}")

    swipe_on = swiper_cfg.get("enabled", False)
    print(f"  Swipe automático : {'ATIVO' if swipe_on else 'desativado'}")

    sched_on = scheduler_cfg.get("enabled", False)
    if sched_on:
        windows = scheduler_cfg.get("windows", [])
        print(f"  Scheduler        : {len(windows)} janela(s) configurada(s)")
    else:
        print("  Scheduler        : desativado")

    browser_on = browser_cfg.get("enabled", False)
    print(f"  Chrome automático: {'ATIVO' if browser_on else 'desativado'}")
    capture_pause = browser_cfg.get("pause_capture_when_unfocused", True)
    print(f"  Captura fora do Tinder: {'pausada' if capture_pause else 'permitida'}")
    focus_pause = swiper_cfg.get("auto_pause_on_focus_loss", True)
    print(f"  Pausa por foco      : {'ATIVA' if focus_pause else 'desativada'}")
    mem_pressure = float(swiper_cfg.get("mem_pressure_threshold_percent", 88) or 0)
    min_mem_mb = float(swiper_cfg.get("min_mem_available_mb", 0) or 0)
    if mem_pressure or min_mem_mb:
        print(f"  Proteção memória   : {mem_pressure:.0f}% usada ou < {min_mem_mb:.0f}MB livres")

    print()
    print("  FAILSAFE: mouse no canto superior esquerdo para parar swipes")
    print(f"  Pausar/retomar swipes: {swiper_cfg.get('pause_hotkey', 'f8').upper()}")
    print(f"  Finalizar swipes     : {swiper_cfg.get('stop_hotkey', 'f10').upper()}")
    print("  Ctrl+C para encerrar o servidor")
    print(f"  Logs: {LOG_PATH}")
    print()

    ensure_synthetic_exists()
    model_data = mdl.load_model()
    if model_data is None:
        logger.info("Modelo nao encontrado/invalido; treinando")
        print("  Treinando modelo pela primeira vez...")
        mdl.train_model()
        model_data = mdl.load_model()
    logger.info("Modelo carregado: type=%s samples=%s", model_data.get("model_type"), model_data.get("n_samples"))
    print(f"  Modelo: {model_data['model_type']} | {model_data['n_samples']} exemplos\n")

    # Inicia thread do scheduler
    start_pause_hotkey_listener()
    threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler").start()
    threading.Thread(target=_profiles_loop, daemon=True, name="profiles-queue").start()

    server = HTTPServer(("localhost", PORT), TinderHandler)

    if browser_on:
        try:
            from browser_launcher import launch_in_background
            launch_in_background(on_ready=lambda: logger.info("Browser pronto e Tinder carregado"))
        except Exception:
            logger.exception("Falha ao iniciar browser automático")

    try:
        logger.info("HTTPServer ouvindo em localhost:%s", PORT)
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Servidor encerrado por KeyboardInterrupt")
        print("\n\n  Servidor encerrado.\n")
        server.server_close()


if __name__ == "__main__":
    main()
