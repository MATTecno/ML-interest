"""Guardas leves para evitar processar perfis quando o sistema está sem RAM."""

from __future__ import annotations

import os
import time

from config import load_config


def read_meminfo() -> tuple[int, int]:
    """Retorna (MemTotal, MemAvailable) em kB, ou zeros se indisponível."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            data = {line.split(":", 1)[0]: int(line.split()[1]) for line in f if ":" in line}
        total = data.get("MemTotal", 0)
        available = data.get(
            "MemAvailable",
            data.get("MemFree", 0) + data.get("Buffers", 0) + data.get("Cached", 0),
        )
        return total, available
    except Exception:
        return 0, 0


def read_swapinfo() -> tuple[int, int]:
    """Retorna (SwapTotal, SwapFree) em kB, ou zeros se indisponível."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            data = {line.split(":", 1)[0]: int(line.split()[1]) for line in f if ":" in line}
        return data.get("SwapTotal", 0), data.get("SwapFree", 0)
    except Exception:
        return 0, 0


def get_system_resource_snapshot() -> dict[str, float]:
    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:
        load1, load5, load15 = 0.0, 0.0, 0.0

    cpu_count = os.cpu_count() or 1
    cpu_pct = min(100.0, (load1 / cpu_count) * 100.0)
    total_kb, available_kb = read_meminfo()
    swap_total_kb, swap_free_kb = read_swapinfo()
    swap_used_kb = max(0, swap_total_kb - swap_free_kb)
    mem_used_kb = max(0, total_kb - available_kb)
    mem_used_pct = (mem_used_kb / total_kb * 100.0) if total_kb else 0.0
    swap_used_pct = (swap_used_kb / swap_total_kb * 100.0) if swap_total_kb else 0.0
    return {
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "cpu_count": float(cpu_count),
        "cpu_pct": cpu_pct,
        "mem_total_mb": total_kb / 1024.0,
        "mem_avail_mb": available_kb / 1024.0,
        "mem_used_mb": mem_used_kb / 1024.0,
        "mem_used_pct": mem_used_pct,
        "swap_total_mb": swap_total_kb / 1024.0,
        "swap_free_mb": swap_free_kb / 1024.0,
        "swap_used_mb": swap_used_kb / 1024.0,
        "swap_used_pct": swap_used_pct,
    }


def _get_swiper_limits(config: dict | None = None) -> tuple[float, float, float]:
    cfg = config if config is not None else load_config()
    swiper_cfg = (cfg.get("swiper", {}) if isinstance(cfg, dict) else {}) or {}
    try:
        max_used_pct = float(swiper_cfg.get("mem_pressure_threshold_percent", 92) or 0)
    except Exception:
        max_used_pct = 92.0
    try:
        min_available_mb = float(swiper_cfg.get("min_mem_available_mb", 0) or 0)
    except Exception:
        min_available_mb = 0.0
    try:
        max_swap_used_pct = float(swiper_cfg.get("max_swap_used_percent", 95) or 0)
    except Exception:
        max_swap_used_pct = 95.0
    return max_used_pct, min_available_mb, max_swap_used_pct


def is_memory_pressure(
    config: dict | None = None,
    snapshot: dict[str, float] | None = None,
) -> tuple[bool, str, dict[str, float]]:
    """Indica se novos trabalhos pesados devem ser pausados por falta de RAM."""
    snap = snapshot or get_system_resource_snapshot()
    max_used_pct, min_available_mb, max_swap_used_pct = _get_swiper_limits(config)
    reasons: list[str] = []

    used_pct = float(snap.get("mem_used_pct", 0.0) or 0.0)
    avail_mb = float(snap.get("mem_avail_mb", 0.0) or 0.0)
    swap_total_mb = float(snap.get("swap_total_mb", 0.0) or 0.0)
    swap_pct = float(snap.get("swap_used_pct", 0.0) or 0.0)

    if max_used_pct > 0 and used_pct >= max_used_pct:
        reasons.append(f"mem_used_pct={used_pct:.1f}>={max_used_pct:.1f}")
    if min_available_mb > 0 and avail_mb > 0 and avail_mb <= min_available_mb:
        reasons.append(f"mem_avail_mb={avail_mb:.0f}<={min_available_mb:.0f}")
    if swap_total_mb > 0 and max_swap_used_pct > 0 and swap_pct >= max_swap_used_pct:
        reasons.append(f"swap_used_pct={swap_pct:.1f}>={max_swap_used_pct:.1f}")

    return bool(reasons), "; ".join(reasons), snap


def memory_relief_reached(config: dict | None = None, snapshot: dict[str, float] | None = None) -> tuple[bool, str, dict[str, float]]:
    """Condição com histerese para retomar trabalho pesado sem ficar pausando em loop."""
    cfg = config if config is not None else load_config()
    snap = snapshot or get_system_resource_snapshot()
    swiper_cfg = (cfg.get("swiper", {}) if isinstance(cfg, dict) else {}) or {}
    max_used_pct, min_available_mb, pressure_max_swap_used_pct = _get_swiper_limits(cfg)
    resume_cfg = (swiper_cfg.get("memory_resume", {}) or {}) if isinstance(swiper_cfg, dict) else {}
    try:
        resume_used_pct = float(resume_cfg.get("resume_below_percent", max(0.0, max_used_pct - 4.0)) or 0)
    except Exception:
        resume_used_pct = max(0.0, max_used_pct - 4.0)
    try:
        resume_available_mb = float(
            resume_cfg.get("resume_min_available_mb", max(min_available_mb + 512.0, min_available_mb * 1.25))
            or 0
        )
    except Exception:
        resume_available_mb = max(min_available_mb + 512.0, min_available_mb * 1.25)
    try:
        max_swap_used_pct = float(
            resume_cfg.get(
                "max_swap_used_percent",
                max(0.0, pressure_max_swap_used_pct - 5.0) if pressure_max_swap_used_pct else 0,
            )
            or 0
        )
    except Exception:
        max_swap_used_pct = 95.0

    reasons: list[str] = []
    used_pct = float(snap.get("mem_used_pct", 0.0) or 0.0)
    avail_mb = float(snap.get("mem_avail_mb", 0.0) or 0.0)
    swap_pct = float(snap.get("swap_used_pct", 0.0) or 0.0)
    if resume_used_pct > 0 and used_pct > resume_used_pct:
        reasons.append(f"mem_used_pct={used_pct:.1f}>{resume_used_pct:.1f}")
    if resume_available_mb > 0 and avail_mb > 0 and avail_mb < resume_available_mb:
        reasons.append(f"mem_avail_mb={avail_mb:.0f}<{resume_available_mb:.0f}")
    if max_swap_used_pct > 0 and swap_pct >= max_swap_used_pct:
        reasons.append(f"swap_used_pct={swap_pct:.1f}>={max_swap_used_pct:.1f}")
    return not reasons, "; ".join(reasons), snap


def wait_for_memory_relief(
    config: dict | None = None,
    max_wait_seconds: float | None = None,
    check_interval_seconds: float | None = None,
) -> tuple[bool, str, dict[str, float], float]:
    """Aguarda a RAM aliviar. Retorna (cleared, reason, snapshot, waited_seconds)."""
    cfg = config if config is not None else load_config()
    photos_cfg = (cfg.get("photos", {}) if isinstance(cfg, dict) else {}) or {}
    if max_wait_seconds is None:
        try:
            max_wait_seconds = float(photos_cfg.get("memory_pressure_max_wait_seconds", 90) or 0)
        except Exception:
            max_wait_seconds = 90.0
    if check_interval_seconds is None:
        try:
            check_interval_seconds = float(photos_cfg.get("memory_pressure_check_interval_seconds", 5) or 5)
        except Exception:
            check_interval_seconds = 5.0

    max_wait_seconds = max(0.0, float(max_wait_seconds or 0.0))
    check_interval_seconds = max(0.5, float(check_interval_seconds or 5.0))
    started = time.time()
    last_reason = ""
    last_snap: dict[str, float] = {}

    while True:
        pressure, pressure_reason, snap = is_memory_pressure(cfg)
        relief, relief_reason, snap = memory_relief_reached(cfg, snap)
        last_reason = pressure_reason or relief_reason
        last_snap = snap
        if not pressure and relief:
            return True, "", snap, time.time() - started
        waited = time.time() - started
        if waited >= max_wait_seconds:
            return False, last_reason, last_snap, waited
        time.sleep(min(check_interval_seconds, max_wait_seconds - waited))


def memory_pressure_photo_features(reason: str) -> dict:
    """Features neutras para quando a análise visual é pulada por proteção."""
    return {
        "_analysis_failed": True,
        "_analysis_skipped": True,
        "_failure_reason": f"memory_pressure:{reason}" if reason else "memory_pressure",
        "_photo_analysis_incomplete": True,
        "_photos_analyzed": 0,
        "_photos_failed": 0,
        "_photos_timed_out": 0,
        "photo_has_face": 1.0,
        "photo_faces_ratio": 0.0,
        "photo_failure_ratio": 0.0,
        "photo_woman_confidence": 0.5,
        "photo_gender_certainty": 0.0,
    }
