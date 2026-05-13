"""Abre o Chrome com a extensão carregada e navega para o Tinder automaticamente."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from config import ROOT_DIR, load_config
from logging_config import get_logger

logger = get_logger(__name__)

EXTENSION_DIR = ROOT_DIR / "extension"


def _get_browser_cfg() -> dict:
    cfg = load_config()
    return cfg.get("browser", {})


def _launch(on_ready=None) -> None:
    from playwright.sync_api import sync_playwright

    bcfg = _get_browser_cfg()
    profile_dir = ROOT_DIR / bcfg.get("profile_dir", "data/chrome_profile")
    tinder_url = bcfg.get("tinder_url", "https://tinder.com/app/recs")
    channel = str(bcfg.get("channel", "chrome") or "").strip() or None
    wait_for_login = bool(bcfg.get("first_run_wait_for_login", True))
    profile_dir.mkdir(parents=True, exist_ok=True)
    first_run_marker = profile_dir / ".first_run_done"

    extension_path = str(EXTENSION_DIR.resolve())
    args = [
        f"--disable-extensions-except={extension_path}",
        f"--load-extension={extension_path}",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    first_run = not first_run_marker.exists()

    logger.info("Iniciando Chrome com extensão: profile=%s first_run=%s", profile_dir, first_run)

    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir.resolve()),
                headless=False,
                args=args,
                channel=channel,
            )
        except Exception:
            if not channel:
                raise
            logger.exception("Falha ao abrir Chrome channel=%s; tentando Chromium do Playwright", channel)
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir.resolve()),
                headless=False,
                args=args,
                channel=None,
            )

        page = context.pages[0] if context.pages else context.new_page()

        if first_run:
            print("\n  [Browser] Primeira execução — faça login no Tinder no Chrome que abriu.")
            print("  [Browser] Esse perfil fica salvo em data/chrome_profile para as próximas sessões.\n")
            page.goto("https://tinder.com", wait_until="domcontentloaded")
            if wait_for_login:
                try:
                    input("  Pressione Enter após fazer login no Tinder... ")
                except EOFError:
                    time.sleep(30)
            first_run_marker.touch()
            logger.info("Marcador de primeiro login criado: %s", first_run_marker)

        logger.info("Navegando para %s", tinder_url)
        try:
            page.goto(tinder_url, wait_until="domcontentloaded", timeout=30_000)
        except Exception:
            logger.warning("Timeout ao navegar para o Tinder; continuando assim mesmo")

        print(f"\n  [Browser] Chrome aberto e Tinder carregado: {tinder_url}\n")

        if on_ready:
            on_ready()

        # Mantém o contexto vivo até o processo encerrar
        try:
            while True:
                if not context.pages:
                    logger.info("Todas as abas fechadas; encerrando browser_launcher")
                    break
                time.sleep(5)
        except Exception:
            pass
        finally:
            try:
                context.close()
            except Exception:
                pass


def launch_in_background(on_ready=None) -> threading.Thread:
    """Inicia o Chrome em uma thread daemon. Retorna a thread."""
    t = threading.Thread(target=_launch, kwargs={"on_ready": on_ready}, daemon=True, name="browser-launcher")
    t.start()
    return t


def main() -> None:
    """Ponto de entrada para teste manual: python3 src/browser_launcher.py"""
    launch_in_background()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
