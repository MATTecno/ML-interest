"""
Servidor local que recebe dados da extensão do Chrome.

Endpoints:
  POST /profiles   — leva recebida de perfis do Tinder
  POST /current    — perfil atualmente visível na tela (para sincronização)
  POST /browser-state — informa se a aba/janela do Tinder está ativa
  POST /modal     — modal bloqueante detectado pela extensão
  POST /queue-reset — limpa filas quando a tela ficou sem card ou vai recarregar
  POST /network-capture — salva eventos de rede capturados pela extensão
  POST /extension-log — eventos de controle/reload emitidos pela extensão
  POST /reloaded   — página do Tinder recarregada, limpa flags de resync
  GET  /control    — instruções simples para a extensão (ex: recarregar página)

Execute: python3 src/server.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    memory_relief_reached,
)
from desktop_notify import notify as desktop_notify
from reload_controller import request_tinder_reload
import state

PORT = 5043
_batch_queue: "queue.Queue[dict]" = queue.Queue()
DEBUG_DIR = Path(__file__).parent.parent / "data" / "debug"
LAST_RESPONSE_PATH = DEBUG_DIR / "last_tinder_response.json"
NETWORK_CAPTURE_DIR = Path(__file__).parent.parent / "data" / "network_capture"
RUNTIME_DIR = Path(__file__).parent.parent / "data" / "runtime"
MEMORY_RESTART_STATE_PATH = RUNTIME_DIR / "server_memory_restart.json"
NETWORK_CAPTURE_MAX_STRING = 300_000
NETWORK_CAPTURE_LATEST_MAX_BYTES = 25 * 1024 * 1024
_recent_profile_batches: dict[str, float] = {}
_recent_profile_batches_lock = threading.Lock()
_memory_pressure_last_notice_at = 0.0
_memory_pressure_notice_lock = threading.Lock()
_MEMORY_PRESSURE_NOTICE_COOLDOWN_SECONDS = 15.0
_URL_GUARD_REASONS = {"wrong_tinder_url", "paywall_url"}
_URL_GUARD_NOTICE_COOLDOWN_SECONDS = 20.0
_url_guard_last_notice_at = 0.0
_url_guard_notice_lock = threading.Lock()
_reload_ack_notice_lock = threading.Lock()
_reload_ack_last_notice_at = 0.0
_reload_ack_last_generation = 0
_reload_ack_last_navigation_at = 0.0
_reload_ack_last_navigation_generation = 0
_control_pending_log_lock = threading.Lock()
_control_pending_last_key = ""
_control_pending_last_at = 0.0
_memory_restart_lock = threading.Lock()
_memory_restart_in_progress = False
_processing_generation = 0
_processing_generation_lock = threading.Lock()
_startup_warmup_event = threading.Event()
_startup_warmup_lock = threading.Lock()
_startup_warmup_started_at = 0.0
_startup_warmup_completed_at = 0.0
_startup_warmup_status = "not_started"
_startup_warmup_errors: list[str] = []
_startup_warmup_notice_printed = False


_last_swipe_confirmed_at = 0.0
_last_swipe_confirmed_lock = threading.Lock()
_network_swipe_seen_lock = threading.Lock()
_network_swipe_seen: dict[tuple[str, str, int | None], dict] = {}
_NETWORK_SWIPE_DEDUPE_SECONDS = 4.0


def _mark_swipe_confirmed_activity() -> None:
    global _last_swipe_confirmed_at
    with _last_swipe_confirmed_lock:
        _last_swipe_confirmed_at = time.time()


def _seconds_since_last_swipe_confirmed() -> float:
    with _last_swipe_confirmed_lock:
        if _last_swipe_confirmed_at <= 0:
            return float("inf")
        return time.time() - _last_swipe_confirmed_at


def _network_swipe_seen_status(
    action: str,
    tinder_id: str,
    status: int | None,
    has_body_signal: bool = False,
) -> str:
    """Retorna new, duplicate ou duplicate_with_body para webrequest/page_hook."""
    clean_action = str(action or "").strip().lower()
    clean_id = str(tinder_id or "").strip()
    if not clean_action or not clean_id:
        return "new"

    key = (clean_action, clean_id, status)
    now = time.time()
    with _network_swipe_seen_lock:
        expired = [
            item_key for item_key, item in _network_swipe_seen.items()
            if now - float(item.get("seen_at", 0.0) or 0.0) > _NETWORK_SWIPE_DEDUPE_SECONDS
        ]
        for item_key in expired:
            _network_swipe_seen.pop(item_key, None)

        previous = _network_swipe_seen.get(key)
        if previous is None:
            _network_swipe_seen[key] = {"seen_at": now, "has_body_signal": bool(has_body_signal)}
            return "new"

        if has_body_signal and not previous.get("has_body_signal"):
            previous["has_body_signal"] = True
            previous["seen_at"] = now
            return "duplicate_with_body"

        previous["seen_at"] = now
        return "duplicate"


_last_reload_start_accepted_at = 0.0
_last_reload_start_lock = threading.Lock()


def _mark_reload_start_accepted() -> None:
    global _last_reload_start_accepted_at
    with _last_reload_start_lock:
        _last_reload_start_accepted_at = time.time()


def _seconds_since_last_reload_start_accepted() -> float:
    with _last_reload_start_lock:
        if _last_reload_start_accepted_at <= 0:
            return float("inf")
        return time.time() - _last_reload_start_accepted_at


def _get_processing_generation() -> int:
    with _processing_generation_lock:
        return _processing_generation


def _bump_processing_generation(reason: str = "") -> int:
    global _processing_generation
    with _processing_generation_lock:
        _processing_generation += 1
        generation = _processing_generation

    logger.warning(
        "Processing generation incrementada: generation=%s reason=%r",
        generation,
        reason,
    )
    return generation


def _load_config() -> dict:
    return load_config()


def _startup_warmup_cfg() -> dict:
    swiper_cfg = _load_config().get("swiper", {}) or {}
    raw = swiper_cfg.get("startup_warmup", {}) or {}
    if not isinstance(raw, dict):
        raw = {"enabled": bool(raw)}

    def _float(name: str, default: float, min_value: float = 0.0) -> float:
        try:
            return max(min_value, float(raw.get(name, default) or 0))
        except Exception:
            return default

    return {
        "enabled": bool(raw.get("enabled", True)),
        "min_cooldown_seconds": _float("min_cooldown_seconds", 8.0, 0.0),
        "max_wait_seconds": _float("max_wait_seconds", 90.0, 1.0),
        "wait_before_profiles": bool(raw.get("wait_before_profiles", True)),
        "warm_clip_worker": bool(raw.get("warm_clip_worker", True)),
        "warm_prediction": bool(raw.get("warm_prediction", True)),
        "warm_bio_embedding": bool(raw.get("warm_bio_embedding", True)),
    }


def _set_startup_warmup_status(status: str, error: str = "") -> None:
    global _startup_warmup_status, _startup_warmup_completed_at
    with _startup_warmup_lock:
        _startup_warmup_status = status
        if error:
            _startup_warmup_errors.append(error)
        if status in {"ready", "failed", "disabled"}:
            _startup_warmup_completed_at = time.time()


def _startup_warmup_snapshot() -> dict:
    with _startup_warmup_lock:
        started_at = _startup_warmup_started_at
        completed_at = _startup_warmup_completed_at
        status = _startup_warmup_status
        errors = list(_startup_warmup_errors)
    now = time.time()
    return {
        "status": status,
        "ready": _startup_warmup_event.is_set(),
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": round((completed_at or now) - started_at, 2) if started_at else 0.0,
        "errors": errors,
    }


def _start_startup_warmup(model_data: dict | None) -> None:
    cfg = _startup_warmup_cfg()
    if not cfg.get("enabled", True):
        _set_startup_warmup_status("disabled")
        _startup_warmup_event.set()
        return

    global _startup_warmup_started_at, _startup_warmup_completed_at, _startup_warmup_status
    with _startup_warmup_lock:
        _startup_warmup_started_at = time.time()
        _startup_warmup_completed_at = 0.0
        _startup_warmup_errors.clear()
        _startup_warmup_status = "running"
    _startup_warmup_event.clear()

    def _run() -> None:
        started = time.perf_counter()
        try:
            restart_state = _load_memory_restart_state()
            if restart_state.get("pending_recovery"):
                logger.warning("Startup warmup pesado pulado: servidor iniciou em recuperacao de memoria")
                _set_startup_warmup_status("ready")
                return

            if cfg.get("warm_prediction", True) and model_data is not None:
                warm_profile = {
                    "name": "warmup",
                    "age": 25,
                    "distance_km": "",
                    "bio": "gosta de viagens, musica e conversa leve",
                    "interests": ["musica", "viagem"],
                    "_descriptors": {},
                    "_photo_features": {},
                }
                try:
                    mdl.predict(warm_profile, model_data)
                    logger.info("Warmup de predicao concluido")
                except Exception as exc:
                    logger.exception("Warmup de predicao falhou")
                    _set_startup_warmup_status("running", f"prediction:{type(exc).__name__}")

            if cfg.get("warm_clip_worker", True):
                try:
                    from photo_semantic_embeddings import prewarm_worker

                    prewarm_worker("server_start_gate")
                except Exception as exc:
                    logger.exception("Warmup CLIP falhou")
                    _set_startup_warmup_status("running", f"clip:{type(exc).__name__}")

            if cfg.get("warm_bio_embedding", True):
                try:
                    from bio_embedding import get_model

                    if get_model(local_only=True) is None:
                        get_model()
                    logger.info("Warmup de bio embedding concluido")
                except Exception as exc:
                    logger.exception("Warmup de bio embedding falhou")
                    _set_startup_warmup_status("running", f"bio_embedding:{type(exc).__name__}")

            elapsed = time.perf_counter() - started
            _set_startup_warmup_status("ready")
            logger.warning(
                "Startup warmup concluido: elapsed=%.2fs status=%s errors=%s",
                elapsed,
                _startup_warmup_snapshot()["status"],
                _startup_warmup_snapshot()["errors"],
            )
        finally:
            _startup_warmup_event.set()

    threading.Thread(target=_run, daemon=True, name="startup-warmup").start()


def _wait_for_startup_warmup_if_needed(stage: str) -> None:
    cfg = _startup_warmup_cfg()
    if not cfg.get("enabled", True) or not cfg.get("wait_before_profiles", True):
        return

    global _startup_warmup_notice_printed
    max_wait = float(cfg.get("max_wait_seconds", 90.0) or 90.0)
    min_cooldown = float(cfg.get("min_cooldown_seconds", 0.0) or 0.0)
    started_at = _startup_warmup_started_at or time.time()
    deadline = time.time() + max_wait

    while True:
        elapsed_since_start = time.time() - started_at
        ready = _startup_warmup_event.is_set()
        cooldown_ok = elapsed_since_start >= min_cooldown
        if ready and cooldown_ok:
            return

        if time.time() >= deadline:
            logger.warning(
                "Startup warmup liberado por timeout: stage=%s snapshot=%s",
                stage,
                _startup_warmup_snapshot(),
            )
            return

        if not _startup_warmup_notice_printed:
            _startup_warmup_notice_printed = True
            logger.warning(
                "Aguardando startup warmup antes de avaliar perfis: stage=%s min_cooldown=%.1fs max_wait=%.1fs",
                stage,
                min_cooldown,
                max_wait,
            )
            with state.terminal_lock:
                print(
                    "\n  [warmup] Aguardando modelos/caches carregarem antes da primeira avaliacao...\n"
                )

        time.sleep(min(0.5, max(0.05, deadline - time.time())))


def _memory_restart_cfg() -> dict:
    swiper_cfg = _load_config().get("swiper", {}) or {}
    raw = swiper_cfg.get("memory_restart", {}) or {}
    if not isinstance(raw, dict):
        raw = {"enabled": bool(raw)}

    def _float(name: str, default: float, min_value: float = 0.0) -> float:
        try:
            return max(min_value, float(raw.get(name, default) or 0))
        except Exception:
            return default

    return {
        "enabled": bool(raw.get("enabled", True)),
        "check_interval_seconds": _float("check_interval_seconds", 10.0, 5.0),
        "pressure_seconds": _float("pressure_seconds", 45.0, 1.0),
        "min_interval_seconds": _float("min_interval_seconds", 300.0, 0.0),
        "reload_after_restart": bool(raw.get("reload_after_restart", True)),
        "navigate_to_recs": bool(raw.get("navigate_to_recs", True)),
        "wait_for_relief": bool(raw.get("wait_for_relief", True)),
        "relief_check_interval_seconds": _float("relief_check_interval_seconds", 5.0, 1.0),
        "relief_timeout_seconds": _float("relief_timeout_seconds", 340.0, 0.0),
    }


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
    """Limpa tudo que foi calculado para a tela atual e invalida lotes em andamento."""
    processing_generation = _bump_processing_generation(reason)

    dropped_batches = _clear_pending_batches()
    dropped_swipes = swiper_queue.clear_pending()
    state.clear_active_profiles()
    state.set_current("", "", 0)

    logger.warning(
        "Processamento atual limpo: reason=%r pending_batches=%s pending_swipes=%s reload_generation=%s processing_generation=%s",
        reason,
        dropped_batches,
        dropped_swipes,
        state.get_reload_generation(),
        processing_generation,
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
        "processing_generation": processing_generation,
    }


def _compact_resource_snapshot(snapshot: dict | None) -> dict:
    clean: dict[str, float] = {}
    for key, value in (snapshot or {}).items():
        try:
            clean[str(key)] = round(float(value), 3)
        except Exception:
            continue
    return clean


def _load_memory_restart_state() -> dict:
    try:
        if MEMORY_RESTART_STATE_PATH.exists():
            data = json.loads(MEMORY_RESTART_STATE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("Falha ao ler estado de auto-restart de memoria", exc_info=True)
    return {}


def _save_memory_restart_state(data: dict) -> None:
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_RESTART_STATE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Falha ao salvar estado de auto-restart de memoria")


def _memory_restart_cooldown_remaining(cfg: dict | None = None) -> float:
    restart_cfg = cfg or _memory_restart_cfg()
    min_interval = float(restart_cfg.get("min_interval_seconds", 0.0) or 0.0)
    if min_interval <= 0:
        return 0.0
    state_data = _load_memory_restart_state()
    try:
        last_restart_at = float(state_data.get("last_restart_at", 0.0) or 0.0)
    except Exception:
        last_restart_at = 0.0
    if last_restart_at <= 0:
        return 0.0
    elapsed = time.time() - last_restart_at
    return max(0.0, min_interval - elapsed)


def _mark_memory_restart_pending(reason: str, snapshot: dict, pressure_age_seconds: float) -> dict:
    previous = _load_memory_restart_state()
    now = time.time()
    record = {
        **previous,
        "pending_recovery": True,
        "last_restart_at": now,
        "last_restart_iso": datetime.now().isoformat(timespec="seconds"),
        "restart_count": int(previous.get("restart_count", 0) or 0) + 1,
        "pid": os.getpid(),
        "reason": str(reason or ""),
        "pressure_age_seconds": round(float(pressure_age_seconds or 0.0), 3),
        "snapshot": _compact_resource_snapshot(snapshot),
    }
    _save_memory_restart_state(record)
    return record


def _mark_memory_restart_recovery_done(result: str, snapshot: dict | None = None) -> None:
    state_data = _load_memory_restart_state()
    state_data["pending_recovery"] = False
    state_data["last_recovery_result"] = str(result or "")
    state_data["last_recovery_at"] = time.time()
    state_data["last_recovery_iso"] = datetime.now().isoformat(timespec="seconds")
    if snapshot is not None:
        state_data["last_recovery_snapshot"] = _compact_resource_snapshot(snapshot)
    _save_memory_restart_state(state_data)


def _exec_self_after_memory_restart(reason: str, snapshot: dict, pressure_age_seconds: float) -> None:
    global _memory_restart_in_progress
    with _memory_restart_lock:
        if _memory_restart_in_progress:
            return
        _memory_restart_in_progress = True

    record = _mark_memory_restart_pending(reason, snapshot, pressure_age_seconds)
    cleared = _clear_processing_queues(f"memory_autorestart:{reason}")
    logger.critical(
        "Auto-restart por memoria critica: reason=%s pressure_age=%.1fs mem=%.1f%% avail=%.1fMB swap=%.1f%% cleared=%s restart_count=%s",
        reason,
        pressure_age_seconds,
        snapshot.get("mem_used_pct", 0.0),
        snapshot.get("mem_avail_mb", 0.0),
        snapshot.get("swap_used_pct", 0.0),
        cleared,
        record.get("restart_count"),
    )
    with state.terminal_lock:
        print(
            "  [server] Memória crítica persistente — reiniciando servidor "
            f"({snapshot.get('mem_used_pct', 0.0):.1f}% usada, "
            f"{snapshot.get('mem_avail_mb', 0.0):.0f}MB livres)"
        )
    desktop_notify(
        "memory_restart",
        "Tinder IA reiniciando por memoria critica",
        f"{snapshot.get('mem_used_pct', 0.0):.1f}% usada · {snapshot.get('mem_avail_mb', 0.0):.0f}MB livres",
        urgency="critical",
    )
    try:
        from photo_semantic_embeddings import release_model

        release_model("memory_autorestart", include_worker=True)
    except Exception:
        logger.debug("Falha ao encerrar CLIP antes do auto-restart", exc_info=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    argv = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    logger.critical("Executando restart do servidor: %s", argv)
    try:
        os.execv(sys.executable, argv)
    except Exception:
        with _memory_restart_lock:
            _memory_restart_in_progress = False
        _mark_memory_restart_recovery_done("exec_failed", snapshot)
        logger.exception("Falha ao executar auto-restart do servidor")


def _memory_restart_watchdog_loop() -> None:
    pressure_since = 0.0
    last_cooldown_notice_at = 0.0
    logger.info("Watchdog de auto-restart por memoria iniciado")
    while True:
        cfg = _memory_restart_cfg()
        interval = float(cfg.get("check_interval_seconds", 5.0) or 5.0)
        if not cfg.get("enabled", True):
            time.sleep(interval)
            pressure_since = 0.0
            continue

        snapshot = _get_system_resource_snapshot()
        pressure, reason, snapshot = is_memory_pressure(_load_config(), snapshot=snapshot)
        now = time.time()
        if not pressure:
            if pressure_since:
                logger.info("Watchdog memoria: pressao aliviou antes do restart")
            pressure_since = 0.0
            time.sleep(interval)
            continue

        if not pressure_since:
            pressure_since = now
            logger.warning("Watchdog memoria: pressao detectada reason=%s", reason)

        pressure_age = now - pressure_since
        if pressure_age < float(cfg.get("pressure_seconds", 45.0) or 45.0):
            time.sleep(interval)
            continue

        cooldown_remaining = _memory_restart_cooldown_remaining(cfg)
        if cooldown_remaining > 0:
            if now - last_cooldown_notice_at >= 60.0:
                last_cooldown_notice_at = now
                logger.warning(
                    "Watchdog memoria: restart adiado por cooldown %.0fs reason=%s mem=%.1f%% avail=%.1fMB",
                    cooldown_remaining,
                    reason,
                    snapshot.get("mem_used_pct", 0.0),
                    snapshot.get("mem_avail_mb", 0.0),
                )
            time.sleep(interval)
            continue

        _exec_self_after_memory_restart(reason, snapshot, pressure_age)
        return


def _post_memory_restart_recovery_loop(restart_state: dict) -> None:
    cfg = _memory_restart_cfg()
    if not cfg.get("reload_after_restart", True):
        logger.info("Recuperacao apos auto-restart: reload desativado por config")
        _mark_memory_restart_recovery_done("reload_disabled")
        return

    reason = str(restart_state.get("reason") or "memoria critica")
    timeout = float(cfg.get("relief_timeout_seconds", 240.0) or 0.0)
    interval = float(cfg.get("relief_check_interval_seconds", 5.0) or 5.0)
    wait_for_relief = bool(cfg.get("wait_for_relief", True))
    started = time.time()
    warned_timeout = False
    last_log_at = 0.0
    last_snapshot: dict | None = None
    logger.warning(
        "Recuperacao apos auto-restart aguardando memoria estabilizar: reason=%s wait=%s timeout=%.0fs",
        reason,
        wait_for_relief,
        timeout,
    )

    while wait_for_relief:
        snapshot = _get_system_resource_snapshot()
        pressure, pressure_reason, snapshot = is_memory_pressure(_load_config(), snapshot=snapshot)
        relief, relief_reason, snapshot = memory_relief_reached(_load_config(), snapshot=snapshot)
        last_snapshot = snapshot
        if not pressure and relief:
            logger.warning(
                "Memoria estabilizada apos auto-restart: mem=%.1f%% avail=%.1fMB swap=%.1f%% waited=%.1fs",
                snapshot.get("mem_used_pct", 0.0),
                snapshot.get("mem_avail_mb", 0.0),
                snapshot.get("swap_used_pct", 0.0),
                time.time() - started,
            )
            break
        now = time.time()
        if now - last_log_at >= 30.0:
            last_log_at = now
            logger.warning(
                "Aguardando memoria estabilizar apos restart: pressure=%s relief=%s reason=%s%s mem=%.1f%% avail=%.1fMB swap=%.1f%%",
                pressure,
                relief,
                pressure_reason or relief_reason,
                " timeout_exceeded" if warned_timeout else "",
                snapshot.get("mem_used_pct", 0.0),
                snapshot.get("mem_avail_mb", 0.0),
                snapshot.get("swap_used_pct", 0.0),
            )
        if timeout > 0 and not warned_timeout and now - started >= timeout:
            warned_timeout = True
            logger.warning("Memoria ainda nao estabilizou %.0fs apos auto-restart; continuando em observacao", timeout)
        time.sleep(interval)

    recovery_reason = f"memoria estabilizada apos restart automatico: {reason}"
    cleared = _clear_processing_queues(recovery_reason)
    reload_result = request_tinder_reload(
        recovery_reason,
        source="server_memory_restart_recovery",
        navigate_to_recs=bool(cfg.get("navigate_to_recs", True)),
        notify_event="memory_recovered_reload",
        notify_title="Tinder IA estabilizou e vai recarregar o Tinder",
        notify_message=str(reason or "")[:180],
        urgency="normal",
    )
    logger.warning("Reload do Tinder solicitado apos auto-restart: generation=%s cleared=%s", reload_result["generation"], cleared)
    _mark_memory_restart_recovery_done("reload_requested", last_snapshot)


def _start_memory_restart_services() -> None:
    cfg = _memory_restart_cfg()
    if cfg.get("enabled", True):
        threading.Thread(
            target=_memory_restart_watchdog_loop,
            daemon=True,
            name="memory-restart-watchdog",
        ).start()
    restart_state = _load_memory_restart_state()
    if restart_state.get("pending_recovery"):
        threading.Thread(
            target=_post_memory_restart_recovery_loop,
            args=(restart_state,),
            daemon=True,
            name="memory-restart-recovery",
        ).start()


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


def _reload_ack_cfg() -> dict:
    browser_cfg = _load_config().get("browser", {}) or {}
    cfg = browser_cfg.get("reload_ack_watchdog", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "check_interval_seconds": float(cfg.get("check_interval_seconds", 5) or 5),
        "ack_timeout_seconds": float(cfg.get("ack_timeout_seconds", 25) or 25),
        "notice_repeat_seconds": float(cfg.get("notice_repeat_seconds", 60) or 60),
        "navigate_to_recs": bool(cfg.get("navigate_to_recs", True)),
    }


def _start_reload_ack_watchdog() -> None:
    cfg = _reload_ack_cfg()
    if not cfg.get("enabled", True):
        return
    threading.Thread(target=_reload_ack_watchdog_loop, daemon=True, name="reload-ack-watchdog").start()


def _reload_ack_watchdog_loop() -> None:
    global _reload_ack_last_notice_at, _reload_ack_last_generation
    global _reload_ack_last_navigation_at, _reload_ack_last_navigation_generation
    logger.info("Watchdog de confirmacao de reload iniciado")
    while True:
        try:
            cfg = _reload_ack_cfg()
            interval = max(1.0, float(cfg.get("check_interval_seconds", 5) or 5))
            if not cfg.get("enabled", True):
                time.sleep(interval)
                continue

            reload_now, reason, requested_at = state.peek_reload_request()
            if not reload_now or not requested_at:
                time.sleep(interval)
                continue

            age = time.time() - requested_at
            timeout = max(1.0, float(cfg.get("ack_timeout_seconds", 25) or 25))
            if age < timeout:
                time.sleep(interval)
                continue

            generation = state.get_reload_generation()
            repeat = max(timeout, float(cfg.get("notice_repeat_seconds", 60) or 60))
            now = time.time()
            with _reload_ack_notice_lock:
                already_noticed = (
                    generation == _reload_ack_last_generation
                    and now - _reload_ack_last_notice_at < repeat
                )
                if not already_noticed:
                    _reload_ack_last_generation = generation
                    _reload_ack_last_notice_at = now

            if not already_noticed:
                logger.warning(
                    "Reload pendente sem confirmacao /reloaded: generation=%s age=%.1fs reason=%r",
                    generation,
                    age,
                    reason,
                )
                desktop_notify(
                    "reload_not_acknowledged",
                    "Tinder IA pediu reload, mas a aba nao confirmou",
                    str(reason or "")[:180],
                    urgency="normal",
                )
            if cfg.get("navigate_to_recs", True):
                should_request_navigation = False
                canonical = _canonical_recs_url()
                nav_pending, nav_target, _, nav_requested_at, _ = state.peek_navigation_request()
                with _reload_ack_notice_lock:
                    nav_recent = (
                        generation == _reload_ack_last_navigation_generation
                        and now - _reload_ack_last_navigation_at < repeat
                    )
                    if not nav_recent:
                        same_pending = (
                            nav_pending
                            and nav_target == canonical
                            and nav_requested_at
                            and now - nav_requested_at < repeat
                        )
                        if not same_pending:
                            _reload_ack_last_navigation_generation = generation
                            _reload_ack_last_navigation_at = now
                            should_request_navigation = True
                if should_request_navigation:
                    state.request_navigation(canonical, f"reload sem confirmacao: {reason}")
            time.sleep(interval)
        except Exception:
            logger.debug("Erro no watchdog de confirmacao de reload", exc_info=True)
            time.sleep(5.0)


def _log_control_pending_snapshot(
    *,
    reload_now: bool,
    reason: str,
    requested_at: float,
    reload_generation: int,
    nav_now: bool,
    nav_url: str,
    nav_reason: str,
    nav_requested_at: float,
    nav_generation: int,
) -> None:
    global _control_pending_last_key, _control_pending_last_at
    if not reload_now and not nav_now:
        return

    now = time.time()
    reload_age = round(now - requested_at, 1) if requested_at else 0.0
    nav_age = round(now - nav_requested_at, 1) if nav_requested_at else 0.0
    key = "|".join(
        [
            "r" if reload_now else "-",
            str(reload_generation),
            "n" if nav_now else "-",
            str(nav_generation),
            str(nav_url or ""),
        ]
    )
    with _control_pending_log_lock:
        if key == _control_pending_last_key and now - _control_pending_last_at < 30.0:
            return
        _control_pending_last_key = key
        _control_pending_last_at = now

    logger.info(
        "GET /control com comando pendente: reload=%s reload_generation=%s reload_age=%.1fs "
        "navigate=%s nav_generation=%s nav_age=%.1fs nav_url=%r reason=%r nav_reason=%r",
        reload_now,
        reload_generation,
        reload_age,
        nav_now,
        nav_generation,
        nav_age,
        nav_url,
        reason,
        nav_reason,
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


def _compact_network_capture_event(event: dict) -> dict:
    capture_type = str(event.get("capture_type") or event.get("source") or "")
    if capture_type != "webrequest":
        compact = dict(event)
        try:
            parsed = urlparse(str(compact.get("url") or ""))
            path = parsed.path or ""
        except Exception:
            path = ""
        if path == "/updates" or path.startswith("/updates/"):
            compact.pop("request", None)
            compact["response"] = {"body_omitted": "network_capture_updates_body_skipped"}
        elif path == "/v2/fast-match/teaser" or path.startswith("/v2/fast-match/teaser/"):
            compact.pop("request", None)
            compact["response"] = {"body_omitted": "network_capture_fast_match_body_skipped"}
        elif path == "/v2/profile" or path.startswith("/v2/profile/"):
            balance = _parse_profile_super_likes(compact)
            compact["profile_super_likes"] = balance or {}
            compact.pop("request", None)
            compact["response"] = {"body_omitted": "network_capture_profile_body_summarized"}
        return compact

    # O page_hook/fetch salva o corpo completo. Para evitar duplicidade pesada,
    # webRequest fica como trilha leve de headers/status/timing.
    keep_keys = {
        "capture_type",
        "source",
        "request_id",
        "phase",
        "url",
        "method",
        "status",
        "type",
        "from_cache",
        "duration_ms",
        "time_stamp",
        "extension_received_at",
        "initiator",
        "tab_id",
        "response_headers",
    }
    return {key: value for key, value in event.items() if key in keep_keys}


def _should_save_network_capture_event(event: dict) -> bool:
    if not _is_useful_network_capture_event(event):
        return False
    capture_type = str(event.get("capture_type") or event.get("source") or "")
    method = str(event.get("method") or "").upper()
    phase = str(event.get("phase") or "").lower()
    if method == "OPTIONS":
        return False
    if capture_type == "webrequest" and phase not in {"complete", "error"}:
        return False
    return True


def _network_capture_path() -> Path:
    day = datetime.now().strftime("%Y%m%d")
    return NETWORK_CAPTURE_DIR / f"network_capture_{day}.jsonl"


def _latest_network_capture_path() -> Path:
    latest = NETWORK_CAPTURE_DIR / "network_capture_latest.jsonl"
    try:
        if latest.exists() and latest.stat().st_size > NETWORK_CAPTURE_LATEST_MAX_BYTES:
            latest.write_text("", encoding="utf-8")
            logger.info(
                "Network capture latest truncado: max_mb=%.1f",
                NETWORK_CAPTURE_LATEST_MAX_BYTES / 1024 / 1024,
            )
    except Exception:
        logger.debug("Falha ao limitar network_capture_latest", exc_info=True)
    return latest


def _save_network_capture_payload(data: dict) -> tuple[int, str]:
    raw_events = data.get("events")
    if isinstance(raw_events, list):
        events = raw_events
    else:
        events = [data]

    NETWORK_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = _network_capture_path()
    latest = _latest_network_capture_path()
    now = datetime.now().isoformat(timespec="milliseconds")
    saved = 0
    with open(path, "a", encoding="utf-8") as f, open(latest, "a", encoding="utf-8") as latest_f:
        for event in events:
            if not isinstance(event, dict):
                continue
            if not _should_save_network_capture_event(event):
                continue
            compact_event = _compact_network_capture_event(event)
            record = _trim_network_capture_value({
                "server_received_at": now,
                **compact_event,
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
        source = str(event.get("source") or event.get("capture_type") or "network_capture")
        seen_status = _network_swipe_seen_status(
            action,
            tinder_id,
            status,
            has_body_signal=match is not None or likes_remaining is not None,
        )
        if seen_status == "duplicate":
            logger.debug(
                "Swipe confirmado duplicado ignorado via rede: action=%s id=%s status=%s source=%s",
                action,
                tinder_id,
                status,
                source,
            )
            continue

        state.mark_recent_swipe(
            tinder_id=tinder_id,
            action=action,
            source=source,
            status=status,
            match=match,
            likes_remaining=likes_remaining,
        )
        removed = 0
        if seen_status == "new":
            removed = swiper_queue.acknowledge_network_swipe(tinder_id, action=action, status=status)
            confirmed += 1
        _mark_swipe_confirmed_activity()
        logger.info(
            "Swipe confirmado via rede: action=%s id=%s status=%s match=%r likes_remaining=%r removed_pending=%s source=%s duplicate_body_update=%s",
            action,
            tinder_id,
            status,
            match,
            likes_remaining,
            removed,
            source,
            seen_status == "duplicate_with_body",
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

    batch_generation = int(data.get("_processing_generation", _get_processing_generation()) or 0)
    current_generation = _get_processing_generation()
    reload_generation = int(data.get("_reload_generation", state.get_reload_generation()) or 0)

    if batch_generation != current_generation:
        logger.warning(
            "Leva descartada por geracao de processamento antiga: batch_generation=%s current_generation=%s reload_generation=%s",
            batch_generation,
            current_generation,
            reload_generation,
        )
        return

    pressure, _ = _handle_memory_pressure("batch_start", clear_processing=True)
    if pressure:
        logger.warning("Leva descartada antes do processamento por memoria critica")
        return

    _wait_for_startup_warmup_if_needed("batch_start")

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
        "Processando nova leva: profiles=%s swipe_enabled=%s interactive=%s queue_size=%s processing_generation=%s reload_generation=%s",
        n,
        swipe_enabled,
        interactive_mode,
        _batch_queue.qsize(),
        batch_generation,
        reload_generation,
    )
    logger.debug("Leva ativa ids=%s nomes=%s", batch_ids, batch_names)

    memory_cancelled = False

    def should_cancel_batch() -> bool:
        nonlocal memory_cancelled

        if memory_cancelled:
            return True

        current_processing_generation = _get_processing_generation()
        if current_processing_generation != batch_generation:
            logger.warning(
                "Cancelando lote por geracao de processamento obsoleta: batch_generation=%s current_generation=%s",
                batch_generation,
                current_processing_generation,
            )
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
                _get_processing_generation(),
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
        logger.warning(
            "Leva cancelada antes de registrar perfis ativos: processing_generation=%s current_generation=%s",
            batch_generation,
            _get_processing_generation(),
        )
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
        if _get_processing_generation() != batch_generation:
            logger.warning(
                "Leva encerrada apos cancelamento por geracao obsoleta: profiles=%s enqueued=%s batch_generation=%s current_generation=%s elapsed=%.2fs",
                n,
                enqueued,
                batch_generation,
                _get_processing_generation(),
                time.perf_counter() - started_at,
            )
            return

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

class TinderHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64


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
            elif self.path == "/extension-log":
                self._handle_extension_log(data)
            elif self.path == "/reloaded":
                self._handle_reloaded(data)
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
        data["_processing_generation"] = _get_processing_generation()
        data["_reload_generation"] = state.get_reload_generation()

        _batch_queue.put(data)

        logger.info(
            "POST /profiles recebido e enfileirado queue_size=%s processing_generation=%s reload_generation=%s batch_key=%s",
            _batch_queue.qsize(),
            data["_processing_generation"],
            data["_reload_generation"],
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

        accepted_kinds = {"super_like_upsell", "popular_profile_upgrade"}
        if kind not in accepted_kinds or (screen_x == 0 and screen_y == 0):
            logger.debug("POST /modal ignorado kind=%r x=%s y=%s", kind, screen_x, screen_y)
            self._respond(200, {"ok": True, "ignored": "invalid_modal"})
            return

        state.set_blocking_modal_target(kind, screen_x, screen_y, target)
        if kind == "super_like_upsell":
            state.mark_super_likes_depleted("modal_super_like_upsell")
        desktop_notify(
            "super_like_modal",
            "Tinder IA fechando popup de Super Like",
            "Clique em Não, obrigado(a) agendado para sair do modal.",
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

    def _handle_reloaded(self, data: dict | None = None) -> None:
        data = data or {}
        logger.info(
            "POST /reloaded recebido; limpando estado de reload/perfis ativos source=%r reason=%r mode=%r",
            data.get("source", ""),
            data.get("reason", ""),
            data.get("mode", ""),
        )
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
            processing_generation = _bump_processing_generation("pagina recarregada sem reload pendente")
            cleared = {
                "pending_batches_cleared": 0,
                "pending_swipes_cleared": 0,
                "processing_generation": processing_generation,
            }
            state.clear_active_profiles()
            state.set_current("", "", 0)
        state.clear_reload_request()
        state.clear_navigation_request()
        self._respond(200, {"ok": True, **cleared})

    def _handle_reload_start(self, data: dict) -> None:
        reason = data.get("reason") or "reload solicitado pela extensão"

        reason_lower = str(reason or "").lower()
        is_sync_failure = (
            "sincronização" in reason_lower
            or "sincronizacao" in reason_lower
            or "sync" in reason_lower
            or "fora da fila" in reason_lower
            or "fora de fila" in reason_lower
        )

        recent_swipe_age = _seconds_since_last_swipe_confirmed()
        if is_sync_failure and recent_swipe_age < 12.0:
            logger.warning(
                "POST /reload-start ignorado: sync failure muito perto de swipe confirmado age=%.1fs reason=%r",
                recent_swipe_age,
                reason,
            )
            self._respond(
                200,
                {
                    "ok": True,
                    "ignored": "recent_swipe_confirmed",
                    "age": round(recent_swipe_age, 1),
                    "reason": reason,
                },
            )
            return

        recent_reload_age = _seconds_since_last_reload_start_accepted()
        min_reload_start_interval = float(
            (_load_config().get("swiper", {}) or {}).get("min_reload_start_interval_seconds", 45)
            or 45
        )

        if is_sync_failure and recent_reload_age < min_reload_start_interval:
            logger.warning(
                "POST /reload-start ignorado: reload-start em cooldown age=%.1fs min=%.1fs reason=%r",
                recent_reload_age,
                min_reload_start_interval,
                reason,
            )
            self._respond(
                200,
                {
                    "ok": True,
                    "ignored": "reload_start_cooldown",
                    "age": round(recent_reload_age, 1),
                    "min_interval": min_reload_start_interval,
                    "reason": reason,
                },
            )
            return

        _mark_reload_start_accepted()

        reload_result = request_tinder_reload(
            reason,
            source="server_reload_start_endpoint",
            notify_event="reload_start",
            notify_title="Tinder IA vai recarregar o Tinder",
            notify_message=str(reason or "")[:180],
            urgency="normal",
        )
        generation = reload_result["generation"]
        cleared = _clear_processing_queues(reason)
        logger.warning(
            "POST /reload-start recebido generation=%s reason=%r pending_batches=%s pending_swipes=%s",
            generation,
            reason,
            cleared["pending_batches_cleared"],
            cleared["pending_swipes_cleared"],
        )
        self._respond(200, {"ok": True, "generation": generation, **cleared})

    def _handle_queue_reset(self, data: dict) -> None:
        reason = data.get("reason") or "estado da tela invalidou a fila atual"
        cleared = _clear_processing_queues(reason)
        self._respond(200, {"ok": True, **cleared})

    def _handle_network_capture(self, data: dict) -> None:
        saved, path = _save_network_capture_payload(data)
        confirmed_swipes = _process_network_capture_events(data)
        logger.info("POST /network-capture salvo events=%s path=%s confirmed_swipes=%s", saved, path, confirmed_swipes)
        self._respond(200, {"ok": True, "saved": saved, "path": path, "confirmed_swipes": confirmed_swipes})

    def _handle_extension_log(self, data: dict) -> None:
        source = str(data.get("source") or "extension")[:40]
        event = str(data.get("event") or "event")[:80]
        message = str(data.get("message") or "")[:500]
        level = str(data.get("level") or "info").strip().lower()
        details = data.get("details") if isinstance(data.get("details"), dict) else {}
        log_message = "EXT %s %s: %s details=%s"
        args = (source, event, message, details)
        if level in {"error", "exception"}:
            logger.error(log_message, *args)
        elif level in {"warn", "warning"}:
            logger.warning(log_message, *args)
        else:
            logger.info(log_message, *args)
        self._respond(200, {"ok": True})

    def _handle_control(self) -> None:
        reload_now, reason, requested_at = state.peek_reload_request()
        nav_now, nav_url, nav_reason, nav_requested_at, nav_generation = state.peek_navigation_request()
        reload_generation = state.get_reload_generation()
        _log_control_pending_snapshot(
            reload_now=reload_now,
            reason=reason,
            requested_at=requested_at,
            reload_generation=reload_generation,
            nav_now=nav_now,
            nav_url=nav_url,
            nav_reason=nav_reason,
            nav_requested_at=nav_requested_at,
            nav_generation=nav_generation,
        )
        self._respond(
            200,
            {
                "reload": reload_now,
                "reason": reason,
                "requested_at": requested_at,
                "reload_generation": reload_generation,
                "navigate": nav_now,
                "navigate_url": nav_url,
                "navigate_reason": nav_reason,
                "navigate_requested_at": nav_requested_at,
                "navigate_generation": nav_generation,
                "capture_active": state.is_tinder_capture_active(),
                "startup_warmup": _startup_warmup_snapshot(),
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
    state.clear_reload_request()
    state.clear_navigation_request()
    state.clear_active_profiles()
    state.set_current("", "", 0)
    _bump_processing_generation("server_start")
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
    restart_cfg = _memory_restart_cfg()
    if restart_cfg.get("enabled", True):
        print(
            "  Auto-restart RAM   : "
            f"{restart_cfg.get('pressure_seconds', 45):.0f}s de pressão contínua"
        )

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

    warmup_cfg = _startup_warmup_cfg()
    if warmup_cfg.get("enabled", True):
        print(
            "  Warmup inicial    : "
            f"aguarda ate {warmup_cfg.get('max_wait_seconds', 90):.0f}s "
            f"(cooldown minimo {warmup_cfg.get('min_cooldown_seconds', 8):.0f}s)"
        )
    _start_startup_warmup(model_data)

    # Inicia thread do scheduler
    start_pause_hotkey_listener()
    threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler").start()
    threading.Thread(target=_profiles_loop, daemon=True, name="profiles-queue").start()
    _start_memory_restart_services()
    _start_reload_ack_watchdog()

    server = TinderHTTPServer(("localhost", PORT), TinderHandler)

    if browser_on:
        try:
            from browser_launcher import launch_in_background
            launch_in_background(on_ready=lambda: logger.info("Browser pronto e Tinder carregado"))
        except Exception:
            logger.exception("Falha ao iniciar browser automático")

    exit_reason = "normal"
    try:
        logger.info("ThreadingHTTPServer ouvindo em localhost:%s", PORT)
        server.serve_forever()
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        logger.info("Servidor encerrado por KeyboardInterrupt")
        print("\n\n  Servidor encerrado.\n")
    except BaseException as exc:
        exit_reason = f"{type(exc).__name__}"
        logger.exception("Servidor encerrando por excecao fatal: %s", type(exc).__name__)
        raise
    finally:
        try:
            server.server_close()
        except Exception:
            logger.debug("Falha ao fechar socket do servidor", exc_info=True)
        try:
            from photo_semantic_embeddings import release_model

            release_model(f"server_shutdown:{exit_reason}", include_worker=True)
        except Exception:
            logger.debug("Falha ao encerrar CLIP no shutdown do servidor", exc_info=True)
        logger.warning("Servidor finalizado: reason=%s", exit_reason)


if __name__ == "__main__":
    main()
