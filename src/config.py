"""Carregamento centralizado do config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from logging_config import get_logger


ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
logger = get_logger(__name__)


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.debug("Config carregado: %s", CONFIG_PATH)
        return cfg
    except Exception:
        logger.exception("Falha ao carregar config: %s", CONFIG_PATH)
        raise


def get_section(name: str, default: dict | None = None) -> dict:
    cfg = load_config()
    value = cfg.get(name, default if default is not None else {})
    return value or {}


def get_photos_config() -> dict:
    return get_section("photos")


def get_swiper_config() -> dict:
    return get_section("swiper")


def get_scheduler_config() -> dict:
    return get_section("scheduler")


def get_model_config() -> dict:
    return get_section("model")
