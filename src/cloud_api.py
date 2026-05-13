"""Servidor cloud-ready para o Tinder-IA.

MVP sem dependencias web externas: usa http.server para facilitar deploy em VM
gratuita. Em producao, este modulo pode virar FastAPI sem mudar os contratos
principais.
"""

from __future__ import annotations

import base64
import html
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))

from cloud_contracts import (
    new_id,
    sanitize_command,
    sanitize_profile_payload,
    sanitize_visible_profile,
    utc_now_iso,
)
from config import ROOT_DIR
from logging_config import get_logger, setup_logging


setup_logging()
logger = get_logger(__name__)

DEFAULT_PORT = int(os.environ.get("TINDER_IA_CLOUD_PORT", "8080"))
RUNTIME_DIR = ROOT_DIR / "data" / "cloud_runtime"
PHOTOS_DIR = RUNTIME_DIR / "photos"
MAX_BODY_BYTES = int(os.environ.get("TINDER_IA_CLOUD_MAX_BODY_BYTES", str(12 * 1024 * 1024)))
MAX_PHOTO_BYTES = int(os.environ.get("TINDER_IA_CLOUD_MAX_PHOTO_BYTES", str(5 * 1024 * 1024)))
AUTH_TOKEN = os.environ.get("TINDER_IA_CLOUD_TOKEN", "").strip()
DEVICE_TOKEN = os.environ.get("TINDER_IA_DEVICE_TOKEN", AUTH_TOKEN).strip()


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Linha JSONL invalida ignorada: %s", path)
                continue
            if isinstance(item, dict):
                rows.append(item)
    if limit is not None and len(rows) > limit:
        return rows[-limit:]
    return rows


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("JSON invalido ignorado: %s", path)
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


class CloudStore:
    def __init__(self, root: Path = RUNTIME_DIR) -> None:
        self.root = root
        self.sessions_path = root / "sessions.jsonl"
        self.profiles_path = root / "profiles.jsonl"
        self.commands_path = root / "commands.jsonl"
        self.acks_path = root / "command_acks.json"
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = {
            "session_id": str(payload.get("session_id") or new_id("sess")),
            "device_id": str(payload.get("device_id") or "local-pc"),
            "status": str(payload.get("status") or "created"),
            "autoswipe_enabled": bool(payload.get("autoswipe_enabled", False)),
            "created_at": utc_now_iso(),
        }
        with self._lock:
            _append_jsonl(self.sessions_path, session)
        return session

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return _load_jsonl(self.sessions_path, limit=limit)

    def save_profile(self, record: dict[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.profiles_path, record)

    def list_profiles(self, limit: int = 50) -> list[dict[str, Any]]:
        return _load_jsonl(self.profiles_path, limit=limit)

    def enqueue_command(self, command: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            _append_jsonl(self.commands_path, command)
        return command

    def command_acks(self) -> dict[str, Any]:
        return _read_json(self.acks_path, {})

    def pending_commands(self, device_id: str, limit: int = 20) -> list[dict[str, Any]]:
        commands = _load_jsonl(self.commands_path)
        acks = self.command_acks()
        pending = [
            cmd for cmd in commands
            if str(cmd.get("device_id") or "") == device_id
            and str(cmd.get("command_id") or "") not in acks
        ]
        return pending[:limit]

    def ack_command(self, command_id: str, ack: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            acks = self.command_acks()
            acks[command_id] = {
                "command_id": command_id,
                "acked_at": utc_now_iso(),
                **ack,
            }
            _write_json(self.acks_path, acks)
            return acks[command_id]


STORE = CloudStore()


def _extract_photo_bytes(item: dict[str, Any]) -> tuple[bytes, str]:
    raw = item.get("content_base64") or item.get("data") or item.get("data_url") or ""
    mime = str(item.get("mime") or "image/jpeg").lower()
    if isinstance(raw, str) and raw.startswith("data:"):
        header, _, body = raw.partition(",")
        if ";base64" in header:
            mime = header[5:].split(";")[0] or mime
        raw = body
    if not isinstance(raw, str) or not raw:
        raise ValueError("foto sem base64")
    data = base64.b64decode(raw, validate=True)
    if len(data) > MAX_PHOTO_BYTES:
        raise ValueError("foto excede limite configurado")
    ext = ".jpg"
    if "png" in mime:
        ext = ".png"
    elif "webp" in mime:
        ext = ".webp"
    return data, ext


def _save_uploaded_photos(session_id: str, profile_id: str, photos: Any) -> list[Path]:
    if not isinstance(photos, list):
        return []
    saved: list[Path] = []
    safe_session = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")[:80] or "session"
    safe_profile = "".join(ch for ch in profile_id if ch.isalnum() or ch in "-_")[:80] or "profile"
    dest_dir = PHOTOS_DIR / safe_session / safe_profile
    dest_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(photos[:4], 1):
        if not isinstance(item, dict):
            continue
        try:
            data, ext = _extract_photo_bytes(item)
        except Exception as exc:
            logger.warning("Foto ignorada no upload: profile=%s error=%s", profile_id, exc)
            continue
        path = dest_dir / f"photo_{idx}{ext}"
        path.write_bytes(data)
        saved.append(path)
    return saved


def _analyze_uploaded_photos(paths: list[Path], age: int) -> dict[str, Any]:
    if not paths:
        return {}
    include_embedding = os.environ.get("TINDER_IA_CLOUD_INCLUDE_EMBEDDING", "0") == "1"
    for path in paths:
        try:
            from photo_features import analyze_local_photo

            features = analyze_local_photo(path, age, include_embedding=include_embedding)
            if features:
                return features
        except Exception:
            logger.exception("Falha na analise cloud da foto: %s", path)
    return {}


def _predict_profile(profile: dict[str, Any]) -> dict[str, Any]:
    try:
        import model as mdl
        from predictor import predict

        model_data = mdl.load_model()
        if model_data is None:
            try:
                mdl.train_model()
                model_data = mdl.load_model()
            except Exception:
                logger.exception("Cloud API nao conseguiu treinar modelo automaticamente")
        if model_data is None:
            raise RuntimeError("modelo_indisponivel")
        return predict(profile, model_data=model_data)
    except Exception as exc:
        logger.warning("Predicao cloud caiu em fallback: %s", exc)
        return {
            "decision": "NÃO CURTIR",
            "probability": 0.5,
            "raw_probability": 0.5,
            "model_type": "cloud_fallback",
            "n_samples": 0,
            "text_score": 0.5,
            "photo_score": 0.5,
            "in_review_band": True,
            "filter_reason": "modelo indisponivel na nuvem",
        }


def _command_for_decision(device_id: str, session_id: str, profile: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    decision = str(result.get("decision") or "").upper()
    command = "swipe_right" if decision == "CURTIR" else "swipe_left"
    return sanitize_command({
        "device_id": device_id,
        "session_id": session_id,
        "profile_id": profile.get("profile_id"),
        "command": command,
        "profile": profile,
        "result": result,
    })


class CloudHandler(BaseHTTPRequestHandler):
    server_version = "TinderIACloud/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("cloud_api %s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_body: str, status: int = 200) -> None:
        body = html_body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError("payload grande demais")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("payload deve ser objeto JSON")
        return data

    def _authorized(self, device: bool = False) -> bool:
        expected = DEVICE_TOKEN if device else AUTH_TOKEN
        if not expected:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {expected}":
            return True
        parsed = urlparse(self.path)
        token = parse_qs(parsed.query).get("token", [""])[0]
        return token == expected

    def _require_auth(self, device: bool = False) -> bool:
        if self._authorized(device=device):
            return True
        self._send_json({"ok": False, "error": "unauthorized"}, status=401)
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            self._send_json({
                "ok": True,
                "service": "tinder-ia-cloud",
                "time": utc_now_iso(),
                "auth_required": bool(AUTH_TOKEN),
            })
            return
        if path == "/":
            self._send_html(self._render_home())
            return
        if path == "/api/profiles":
            if not self._require_auth():
                return
            self._send_json({"ok": True, "profiles": STORE.list_profiles(limit=100)})
            return
        if path.startswith("/api/devices/") and path.endswith("/commands"):
            if not self._require_auth(device=True):
                return
            device_id = path.split("/")[3]
            self._send_json({"ok": True, "commands": STORE.pending_commands(device_id)})
            return
        self._send_json({"ok": False, "error": "not_found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if path == "/api/sessions":
            if not self._require_auth():
                return
            session = STORE.create_session(payload)
            self._send_json({"ok": True, "session": session})
            return

        if path == "/api/profiles":
            if not self._require_auth():
                return
            self._handle_profile(payload)
            return

        if path == "/api/commands":
            if not self._require_auth():
                return
            try:
                command = sanitize_command(payload)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            STORE.enqueue_command(command)
            self._send_json({"ok": True, "command": command})
            return

        if path.startswith("/api/commands/") and path.endswith("/ack"):
            if not self._require_auth(device=True):
                return
            command_id = path.split("/")[3]
            ack = STORE.ack_command(command_id, payload)
            self._send_json({"ok": True, "ack": ack})
            return

        self._send_json({"ok": False, "error": "not_found"}, status=404)

    def _handle_profile(self, payload: dict[str, Any]) -> None:
        session_id = str(payload.get("session_id") or new_id("sess"))
        device_id = str(payload.get("device_id") or "local-pc")
        profile = sanitize_profile_payload(payload)
        photos = _save_uploaded_photos(session_id, profile["profile_id"], payload.get("photos"))
        if photos and not profile.get("_photo_features"):
            profile["_photo_features"] = _analyze_uploaded_photos(photos, int(profile.get("age") or 0))

        visible = sanitize_visible_profile(payload) if payload.get("visible_profile") else {}
        if visible.get("name") or visible.get("tinder_id"):
            STORE.enqueue_command(sanitize_command({
                "device_id": device_id,
                "session_id": session_id,
                "command": "set_current",
                "visible_profile": visible,
            }))

        result = _predict_profile(profile)
        record = {
            "record_id": new_id("rec"),
            "session_id": session_id,
            "device_id": device_id,
            "profile": profile,
            "result": result,
            "photos_saved": [str(p.relative_to(ROOT_DIR)) for p in photos],
            "created_at": utc_now_iso(),
        }
        STORE.save_profile(record)

        command = None
        if bool(payload.get("autoswipe_enabled", False)):
            command = STORE.enqueue_command(_command_for_decision(device_id, session_id, profile, result))

        self._send_json({
            "ok": True,
            "profile": profile,
            "result": result,
            "command": command,
        })

    def _render_home(self) -> str:
        profiles = list(reversed(STORE.list_profiles(limit=25)))
        rows = []
        for item in profiles:
            profile = item.get("profile") or {}
            result = item.get("result") or {}
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('created_at', '')))}</td>"
                f"<td>{html.escape(str(profile.get('name', '')))}</td>"
                f"<td>{html.escape(str(profile.get('age', '')))}</td>"
                f"<td>{html.escape(str(result.get('decision', '')))}</td>"
                f"<td>{html.escape(str(result.get('probability', '')))}</td>"
                "</tr>"
            )
        return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tinder IA Cloud</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Tinder IA Cloud</h1>
  <p>API ativa. Health: <code>/health</code>. Perfis recentes: <code>/api/profiles</code>.</p>
  <table>
    <thead><tr><th>Quando</th><th>Nome</th><th>Idade</th><th>Decisao</th><th>Prob.</th></tr></thead>
    <tbody>{''.join(rows) or '<tr><td colspan="5">Nenhum perfil recebido ainda.</td></tr>'}</tbody>
  </table>
</body>
</html>"""


def run(host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), CloudHandler)
    logger.warning("Tinder IA Cloud API ouvindo em http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.warning("Tinder IA Cloud API encerrando")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()

