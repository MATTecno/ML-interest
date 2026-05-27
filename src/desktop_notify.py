"""Notificacoes desktop opcionais para eventos que pedem atencao."""

from __future__ import annotations

import shutil
import subprocess
import time
from threading import Lock

from config import load_config
from logging_config import get_logger

logger = get_logger(__name__)

_lock = Lock()
_last_sent: dict[str, float] = {}


def _cfg() -> dict:
    try:
        return (load_config().get("notifications", {}) or {})
    except Exception:
        logger.debug("Falha ao carregar config de notificacoes", exc_info=True)
        return {}


def notify(event: str, title: str, message: str = "", urgency: str = "normal") -> bool:
    """Envia notificacao Linux via notify-send, com cooldown por evento."""
    cfg = _cfg()
    if not bool(cfg.get("enabled", False)):
        return False

    events_cfg = cfg.get("events", {}) or {}
    if event and events_cfg.get(event, True) is False:
        return False

    cooldown = float(cfg.get("cooldown_seconds", 60) or 0)
    key = event or title
    now = time.time()
    with _lock:
        last = _last_sent.get(key, 0.0)
        if cooldown > 0 and now - last < cooldown:
            return False
        _last_sent[key] = now

    command = str(cfg.get("command") or "notify-send").strip() or "notify-send"
    binary = shutil.which(command)
    if not binary:
        logger.info("Notificacao ignorada: comando %r nao encontrado event=%s title=%r", command, event, title)
        return False

    app_name = str(cfg.get("app_name") or "Tinder IA").strip() or "Tinder IA"
    timeout_ms = int(float(cfg.get("timeout_ms", 8000) or 8000))
    transient = bool(cfg.get("transient", True))
    urgency_overrides = cfg.get("urgency_overrides", {}) or {}
    if event and event in urgency_overrides:
        urgency = str(urgency_overrides.get(event) or urgency)
    safe_urgency = urgency if urgency in {"low", "normal", "critical"} else "normal"
    args = [
        binary,
        "--app-name",
        app_name,
        "--urgency",
        safe_urgency,
        "--expire-time",
        str(timeout_ms),
        str(title or app_name),
    ]
    if transient:
        args[1:1] = ["--hint", "int:transient:1"]
    if message:
        args.append(str(message))

    try:
        subprocess.run(args, check=False, timeout=2.0)
        logger.info("Notificacao enviada: event=%s title=%r", event, title)
        return True
    except Exception:
        logger.debug("Falha ao enviar notificacao desktop", exc_info=True)
        return False
