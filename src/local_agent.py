"""Agente local leve para o modo cloud.

Mantem uma conexao outbound por polling com a Cloud API. Ele nao faz ML pesado:
apenas abre/foca o Tinder, atualiza estado visivel e entrega decisoes ao
SwiperQueue local.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from logging_config import get_logger, setup_logging


setup_logging()
logger = get_logger(__name__)

_browser_thread: threading.Thread | None = None


def _cloud_cfg() -> dict[str, Any]:
    cfg = load_config().get("cloud", {}) or {}
    return {
        "api_url": os.environ.get("TINDER_IA_CLOUD_URL", cfg.get("api_url", "http://127.0.0.1:8080")),
        "device_id": os.environ.get("TINDER_IA_DEVICE_ID", cfg.get("device_id", "local-pc")),
        "device_token": os.environ.get("TINDER_IA_DEVICE_TOKEN", cfg.get("device_token", "")),
        "poll_interval_seconds": float(os.environ.get("TINDER_IA_AGENT_POLL_SECONDS", cfg.get("poll_interval_seconds", 1.5)) or 1.5),
    }


def _browser_cfg() -> dict[str, Any]:
    return load_config().get("browser", {}) or {}


class CloudClient:
    def __init__(self, api_url: str, device_id: str, token: str = "") -> None:
        self.api_url = api_url.rstrip("/")
        self.device_id = device_id
        self.token = token

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(self.api_url + path, data=body, headers=headers, method=method)
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise RuntimeError("resposta cloud invalida")
        return data

    def pending_commands(self) -> list[dict]:
        query = ""
        if self.token:
            query = "?" + urlencode({"token": self.token})
        data = self._request("GET", f"/api/devices/{self.device_id}/commands{query}")
        commands = data.get("commands") or []
        return commands if isinstance(commands, list) else []

    def ack(self, command_id: str, status: str, detail: str = "") -> None:
        payload = {"status": status, "detail": detail}
        self._request("POST", f"/api/commands/{command_id}/ack", payload)


def _open_tinder() -> str:
    """Abre o Tinder preferindo a sessao normal do Chrome ja logada."""
    bcfg = _browser_cfg()
    tinder_url = str(bcfg.get("tinder_url", "https://tinder.com/app/recs") or "https://tinder.com/app/recs")
    prefer_existing = bool(bcfg.get("prefer_existing_chrome", True))
    if prefer_existing:
        opened = _open_in_existing_browser(tinder_url)
        if opened:
            return opened

    global _browser_thread
    if _browser_thread is not None and _browser_thread.is_alive():
        return "browser already running"
    from browser_launcher import launch_in_background

    _browser_thread = launch_in_background()
    return "browser launch requested"


def _run_detached(cmd: list[str]) -> bool:
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        logger.debug("Falha ao executar comando de browser: %s", cmd, exc_info=True)
        return False


def _open_in_existing_browser(url: str) -> str:
    """
    Tenta usar o navegador padrao/sessao normal do usuario.

    Isso evita abrir o perfil isolado em `data/chrome_profile`, que pode nao
    estar logado no Tinder. Se todos os comandos falharem, o agente cai no
    fallback Playwright existente.
    """
    commands: list[list[str]] = []
    xdg_open = shutil.which("xdg-open")
    if xdg_open:
        commands.append([xdg_open, url])

    for browser in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
        path = shutil.which(browser)
        if path:
            commands.append([path, "--new-tab", url])

    for cmd in commands:
        if _run_detached(cmd):
            return f"opened existing browser via {Path(cmd[0]).name}"
    return ""


def _execute_command(command: dict[str, Any]) -> str:
    name = str(command.get("command") or "")
    if name == "health_check":
        return "ok"
    if name == "open_tinder":
        return _open_tinder()
    if name == "set_current":
        import state

        visible = command.get("visible_profile") or {}
        state.set_current(
            str(visible.get("name") or ""),
            str(visible.get("tinder_id") or ""),
            int(visible.get("age") or 0),
        )
        return "visible profile updated"
    if name == "start_autoswipe":
        import state
        from swiper import start_pause_hotkey_listener

        state.clear_swipe_stop_request()
        state.set_swipe_paused(False)
        start_pause_hotkey_listener()
        return "autoswipe armed"
    if name == "pause_autoswipe":
        import state

        state.set_swipe_paused(True)
        return "autoswipe paused"
    if name == "stop_autoswipe":
        from swiper import handle_hotkey_action

        handle_hotkey_action("stop", source="cloud")
        return "autoswipe stopped"
    if name in {"swipe_left", "swipe_right"}:
        import state
        from swiper import start_pause_hotkey_listener, swiper_queue

        profile = command.get("profile") or {}
        result = command.get("result") or {}
        decision = "CURTIR" if name == "swipe_right" else "NÃO CURTIR"
        if result:
            profile["_ml_result"] = result
            decision = str(result.get("decision") or decision)
        state.clear_swipe_stop_request()
        start_pause_hotkey_listener()
        swiper_queue.add([(profile, decision)])
        return f"queued {decision}"
    raise ValueError(f"comando desconhecido: {name}")


def poll_forever(client: CloudClient, interval: float) -> None:
    logger.warning("Local Agent conectado a %s como %s", client.api_url, client.device_id)
    while True:
        try:
            for command in client.pending_commands():
                command_id = str(command.get("command_id") or "")
                if not command_id:
                    continue
                try:
                    detail = _execute_command(command)
                    client.ack(command_id, "ok", detail)
                    logger.info("Comando cloud executado: %s %s", command.get("command"), detail)
                except Exception as exc:
                    logger.exception("Falha ao executar comando cloud: %s", command)
                    client.ack(command_id, "error", str(exc))
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            logger.warning("Cloud indisponivel para local agent: %s", exc)
        except Exception:
            logger.exception("Erro inesperado no local agent")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tinder IA Local Agent")
    parser.add_argument("--once", action="store_true", help="Processa comandos pendentes uma vez e sai")
    args = parser.parse_args()

    cfg = _cloud_cfg()
    client = CloudClient(cfg["api_url"], cfg["device_id"], cfg["device_token"])
    if args.once:
        for command in client.pending_commands():
            command_id = str(command.get("command_id") or "")
            try:
                detail = _execute_command(command)
                client.ack(command_id, "ok", detail)
            except Exception as exc:
                client.ack(command_id, "error", str(exc))
        return
    poll_forever(client, cfg["poll_interval_seconds"])


if __name__ == "__main__":
    main()
