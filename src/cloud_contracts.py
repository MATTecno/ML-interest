"""Contratos e sanitizacao para o modo cloud.

Este modulo fica propositalmente leve: ele nao depende de Flask/FastAPI nem de
modelos de ML. A ideia e garantir que a API cloud nunca persista headers,
cookies ou tokens vindos da extensao.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any


SENSITIVE_KEY_RE = re.compile(
    r"(authorization|cookie|token|secret|password|x-auth|bearer|session)",
    re.IGNORECASE,
)

TEXT_LIMITS = {
    "name": 80,
    "bio": 2000,
    "profile_id": 160,
    "tinder_id": 160,
    "session_id": 160,
    "device_id": 160,
}

ALLOWED_COMMANDS = {
    "open_tinder",
    "set_current",
    "start_autoswipe",
    "pause_autoswipe",
    "stop_autoswipe",
    "swipe_left",
    "swipe_right",
    "reload_tinder",
    "health_check",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    clean = re.sub(r"[^a-z0-9_-]", "", prefix.lower()) or "id"
    return f"{clean}_{uuid.uuid4().hex[:16]}"


def _clean_text(value: Any, max_len: int = 500) -> str:
    text = str(value or "").replace("\x00", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_len]


def _clean_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _clean_float_or_empty(value: Any) -> float | str:
    if value in ("", None):
        return ""
    try:
        return float(value)
    except Exception:
        return ""


def _clean_list(value: Any, max_items: int = 40, item_len: int = 80) -> list[str]:
    if isinstance(value, str):
        raw_items = re.split(r"[,;\n]", value)
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    out: list[str] = []
    for item in raw_items:
        clean = _clean_text(item, item_len)
        if clean and clean not in out:
            out.append(clean)
        if len(out) >= max_items:
            break
    return out


def _clean_dict(value: Any, max_items: int = 40) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, val in value.items():
        clean_key = _clean_text(key, 80)
        if not clean_key or SENSITIVE_KEY_RE.search(clean_key):
            continue
        out[clean_key] = _clean_text(val, 240)
        if len(out) >= max_items:
            break
    return out


def sanitize_headers(headers: Any) -> dict[str, str]:
    """Retorna apenas headers nao sensiveis, para diagnostico seguro."""
    if not isinstance(headers, dict):
        return {}
    safe: dict[str, str] = {}
    for key, value in headers.items():
        name = _clean_text(key, 120)
        if not name or SENSITIVE_KEY_RE.search(name):
            continue
        safe[name.lower()] = _clean_text(value, 240)
    return safe


def sanitize_photo_features(value: Any) -> dict[str, float | int | str]:
    """Mantem apenas escalares seguros de features; remove embeddings brutos."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, float | int | str] = {}
    for key, val in value.items():
        clean_key = _clean_text(key, 120)
        if not clean_key:
            continue
        if clean_key.startswith("_") and clean_key not in {
            "_dominant_race",
            "_faces_found",
            "_photos_analyzed",
            "_photos_failed",
            "_photos_timed_out",
            "_failure_reason",
        }:
            continue
        if isinstance(val, (int, float)):
            out[clean_key] = float(val)
        elif isinstance(val, str):
            out[clean_key] = _clean_text(val, 240)
    return out


def sanitize_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normaliza o perfil recebido da extensao ou de testes."""
    raw = payload.get("profile") if isinstance(payload.get("profile"), dict) else payload
    raw = raw if isinstance(raw, dict) else {}

    profile_id = (
        raw.get("profile_id")
        or raw.get("tinder_id")
        or raw.get("_tinder_id")
        or payload.get("profile_id")
        or new_id("profile")
    )
    tinder_id = raw.get("_tinder_id") or raw.get("tinder_id") or raw.get("id") or profile_id
    descriptors = raw.get("_descriptors") or raw.get("descriptors") or {}
    photo_features = (
        raw.get("_photo_features")
        or raw.get("photo_features")
        or payload.get("photo_features")
        or {}
    )

    profile = {
        "profile_id": _clean_text(profile_id, TEXT_LIMITS["profile_id"]),
        "_tinder_id": _clean_text(tinder_id, TEXT_LIMITS["tinder_id"]),
        "name": _clean_text(raw.get("name"), TEXT_LIMITS["name"]),
        "age": _clean_int(raw.get("age"), 0),
        "distance_km": _clean_float_or_empty(raw.get("distance_km") or raw.get("_distance_km")),
        "bio": _clean_text(raw.get("bio"), TEXT_LIMITS["bio"]),
        "interests": _clean_list(raw.get("interests")),
        "_descriptors": _clean_dict(descriptors),
        "_photo_features": sanitize_photo_features(photo_features),
    }
    return profile


def sanitize_visible_profile(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("visible_profile") if isinstance(payload.get("visible_profile"), dict) else payload
    raw = raw if isinstance(raw, dict) else {}
    return {
        "name": _clean_text(raw.get("name"), TEXT_LIMITS["name"]),
        "tinder_id": _clean_text(raw.get("tinder_id") or raw.get("_tinder_id"), TEXT_LIMITS["tinder_id"]),
        "age": _clean_int(raw.get("age"), 0),
    }


def sanitize_command(payload: dict[str, Any]) -> dict[str, Any]:
    command = _clean_text(payload.get("command"), 80)
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"comando nao permitido: {command}")

    device_id = _clean_text(payload.get("device_id") or "local-pc", TEXT_LIMITS["device_id"])
    session_id = _clean_text(payload.get("session_id"), TEXT_LIMITS["session_id"])
    profile = sanitize_profile_payload(payload.get("profile") or {}) if payload.get("profile") else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    visible = sanitize_visible_profile(payload.get("visible_profile") or {}) if payload.get("visible_profile") else {}
    return {
        "command_id": _clean_text(payload.get("command_id") or new_id("cmd"), 80),
        "command": command,
        "device_id": device_id,
        "session_id": session_id,
        "profile_id": _clean_text(payload.get("profile_id") or profile.get("profile_id"), TEXT_LIMITS["profile_id"]),
        "profile": profile,
        "result": result,
        "visible_profile": visible,
        "issued_at": utc_now_iso(),
    }

