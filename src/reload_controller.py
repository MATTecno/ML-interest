"""Centralized Tinder reload requests.

The browser extension is still responsible for executing the actual page reload.
This module only records one reload/navigation request in the shared server state
and emits one consistent log/notification for callers.
"""

from __future__ import annotations

from typing import Any

from config import load_config
from desktop_notify import notify as desktop_notify
from logging_config import get_logger
import state

logger = get_logger(__name__)


def canonical_recs_url() -> str:
    try:
        cfg = load_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        guard_cfg = browser_cfg.get("enforce_recs_url", {}) or {}
        return str(
            guard_cfg.get("canonical_url")
            or browser_cfg.get("tinder_url")
            or "https://tinder.com/app/recs"
        ).strip()
    except Exception:
        return "https://tinder.com/app/recs"


def request_tinder_reload(
    reason: str = "",
    *,
    source: str = "",
    navigate_to_recs: bool = True,
    target_url: str | None = None,
    notify_event: str = "",
    notify_title: str = "",
    notify_message: str | None = None,
    urgency: str = "normal",
) -> dict[str, Any]:
    """Request a Tinder reload through one shared path."""
    clean_reason = str(reason or "").strip() or "reload solicitado"
    clean_source = str(source or "").strip() or "unknown"
    generation = state.request_reload(clean_reason)

    navigation_generation = 0
    navigation_url = ""
    if navigate_to_recs:
        navigation_url = str(target_url or canonical_recs_url()).strip()
        navigation_generation = state.request_navigation(navigation_url, clean_reason)

    if notify_event and notify_title:
        try:
            desktop_notify(
                notify_event,
                notify_title,
                str(notify_message if notify_message is not None else clean_reason)[:180],
                urgency=urgency,
            )
        except Exception:
            logger.exception("Falha ao emitir notificacao de reload: source=%s", clean_source)

    logger.warning(
        "Reload centralizado solicitado: source=%s generation=%s navigate=%s nav_generation=%s target=%s reason=%r",
        clean_source,
        generation,
        bool(navigation_generation),
        navigation_generation,
        navigation_url or "(sem navegacao)",
        clean_reason,
    )
    return {
        "generation": generation,
        "reason": clean_reason,
        "source": clean_source,
        "navigation_generation": navigation_generation,
        "navigation_url": navigation_url,
    }
