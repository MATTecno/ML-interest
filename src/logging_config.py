"""Configuracao centralizada de logs do Tinder-IA."""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
LOG_DIR = ROOT_DIR / "data" / "logs"
LOG_PATH = LOG_DIR / "tinder_ia.log"

_CONFIGURED = False


def setup_logging() -> Path:
    """
    Configura logging em arquivo rotativo.

    O terminal continua limpo; detalhes de erro, stack trace e pontos de
    progresso ficam em data/logs/tinder_ia.log.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return LOG_PATH

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level_name = os.environ.get("TINDER_IA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s"
    ))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    logging.captureWarnings(True)
    _install_exception_hooks()
    _CONFIGURED = True

    logging.getLogger(__name__).info("Logging inicializado em %s", LOG_PATH)
    return LOG_PATH


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _install_exception_hooks() -> None:
    previous_sys_hook = sys.excepthook
    previous_thread_hook = getattr(threading, "excepthook", None)

    def handle_sys_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            previous_sys_hook(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("unhandled").critical(
            "Excecao nao capturada no processo principal",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        previous_sys_hook(exc_type, exc_value, exc_traceback)

    def handle_thread_exception(args):
        if issubclass(args.exc_type, KeyboardInterrupt):
            if previous_thread_hook:
                previous_thread_hook(args)
            return
        logging.getLogger("unhandled.thread").critical(
            "Excecao nao capturada na thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        if previous_thread_hook:
            previous_thread_hook(args)

    sys.excepthook = handle_sys_exception
    threading.excepthook = handle_thread_exception
