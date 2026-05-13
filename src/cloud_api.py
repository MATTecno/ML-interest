"""Servidor cloud-ready para o Tinder-IA.

MVP sem dependencias web externas: usa http.server para facilitar deploy em VM
gratuita. Em producao, este modulo pode virar FastAPI sem mudar os contratos
principais.
"""

from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

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
        self.reviews_path = root / "review_actions.jsonl"
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

    def save_review_action(self, action: dict[str, Any]) -> dict[str, Any]:
        record = {
            "action_id": new_id("rev"),
            "reviewed_at": utc_now_iso(),
            **action,
        }
        with self._lock:
            _append_jsonl(self.reviews_path, record)
        return record

    def list_review_actions(self, limit: int = 100) -> list[dict[str, Any]]:
        return _load_jsonl(self.reviews_path, limit=limit)


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


def _safe_cloud_photo_path(rel_path: str) -> Path | None:
    raw = str(rel_path or "").strip()
    if not raw or raw.startswith("/"):
        return None
    try:
        root = PHOTOS_DIR.resolve()
        candidate = (ROOT_DIR / raw).resolve()
    except Exception:
        return None
    if candidate == root or root not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _auth_query_from_path(path: str) -> str:
    parsed = urlparse(path)
    token = parse_qs(parsed.query).get("token", [""])[0]
    if token:
        return "?token=" + quote(token)
    return ""


def _clean_review_action(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {"like", "dislike", "skip"}
    action = str(payload.get("action") or "").strip().lower()
    if action not in allowed:
        raise ValueError(f"acao de review invalida: {action}")
    note = str(payload.get("note") or "").replace("\x00", "").strip()
    if len(note) > 500:
        note = note[:500]
    return {
        "record_id": str(payload.get("record_id") or "").strip()[:160],
        "profile_id": str(payload.get("profile_id") or "").strip()[:160],
        "session_id": str(payload.get("session_id") or "").strip()[:160],
        "device_id": str(payload.get("device_id") or "").strip()[:160],
        "action": action,
        "note": note,
    }


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

    def _send_file(self, path: Path) -> None:
        mime, _ = mimetypes.guess_type(path.name)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600")
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
        if path == "/review":
            if not self._require_auth():
                return
            self._send_html(self._render_review())
            return
        if path == "/photo":
            if not self._require_auth():
                return
            rel_path = parse_qs(parsed.query).get("path", [""])[0]
            photo_path = _safe_cloud_photo_path(rel_path)
            if photo_path is None:
                self._send_json({"ok": False, "error": "photo_not_found"}, status=404)
                return
            self._send_file(photo_path)
            return
        if path == "/api/profiles":
            if not self._require_auth():
                return
            self._send_json({"ok": True, "profiles": STORE.list_profiles(limit=100)})
            return
        if path == "/api/reviews":
            if not self._require_auth():
                return
            self._send_json({"ok": True, "reviews": STORE.list_review_actions(limit=100)})
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

        if path == "/api/reviews":
            if not self._require_auth():
                return
            try:
                action = _clean_review_action(payload)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            saved = STORE.save_review_action(action)
            self._send_json({"ok": True, "review": saved})
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
    a {{ color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Tinder IA Cloud</h1>
  <p>API ativa. Health: <code>/health</code>. Perfis recentes: <code>/api/profiles</code>. Painel: <a href="/review">/review</a>.</p>
  <table>
    <thead><tr><th>Quando</th><th>Nome</th><th>Idade</th><th>Decisao</th><th>Prob.</th></tr></thead>
    <tbody>{''.join(rows) or '<tr><td colspan="5">Nenhum perfil recebido ainda.</td></tr>'}</tbody>
  </table>
</body>
</html>"""

    def _render_review(self) -> str:
        auth_query = _auth_query_from_path(self.path)
        profiles = list(reversed(STORE.list_profiles(limit=60)))
        reviewed = {
            str(item.get("record_id") or ""): str(item.get("action") or "")
            for item in STORE.list_review_actions(limit=500)
        }
        cards = []
        for item in profiles:
            profile = item.get("profile") or {}
            result = item.get("result") or {}
            photos = [
                p for p in item.get("photos_saved") or []
                if _safe_cloud_photo_path(str(p)) is not None
            ]
            record_id = str(item.get("record_id") or "")
            profile_id = str(profile.get("profile_id") or "")
            review_action = reviewed.get(record_id, "")
            photo_html = "".join(
                f'<img src="/photo?path={quote(str(path))}{("&" + auth_query[1:]) if auth_query else ""}" loading="lazy" alt="">'
                for path in photos[:4]
            ) or '<div class="no-photo">sem foto salva</div>'
            probability = result.get("probability", "")
            try:
                probability = f"{float(probability) * 100:.0f}%"
            except Exception:
                probability = str(probability)
            descriptors = profile.get("_descriptors") or {}
            descriptor_html = "".join(
                f"<span>{html.escape(str(k))}: {html.escape(str(v))}</span>"
                for k, v in list(descriptors.items())[:8]
            )
            cards.append(f"""
      <article class="card" data-record-id="{html.escape(record_id)}">
        <div class="photos">{photo_html}</div>
        <div class="content">
          <div class="topline">
            <h2>{html.escape(str(profile.get('name', '')))} <small>{html.escape(str(profile.get('age', '')))}</small></h2>
            <span class="decision">{html.escape(str(result.get('decision', '')))} {html.escape(str(probability))}</span>
          </div>
          <p>{html.escape(str(profile.get('bio', '')))}</p>
          <div class="chips">{''.join(f'<span>{html.escape(str(x))}</span>' for x in profile.get('interests', [])[:12])}</div>
          <div class="chips muted">{descriptor_html}</div>
          <div class="meta">
            <span>{html.escape(str(item.get('created_at', '')))}</span>
            <span>{html.escape(str(result.get('model_type', '')))}</span>
            <span class="review-state">{'revisado: ' + html.escape(review_action) if review_action else 'pendente'}</span>
          </div>
          <div class="actions">
            <button data-action="like" data-record="{html.escape(record_id)}" data-profile="{html.escape(profile_id)}">Curtir</button>
            <button data-action="dislike" data-record="{html.escape(record_id)}" data-profile="{html.escape(profile_id)}">Nao curtir</button>
            <button data-action="skip" data-record="{html.escape(record_id)}" data-profile="{html.escape(profile_id)}">Pular</button>
          </div>
        </div>
      </article>""")

        return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tinder IA Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; color: #17202a; background: #f5f7f9; }}
    header {{ position: sticky; top: 0; z-index: 2; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 20px; border-bottom: 1px solid #dde3ea; background: rgba(255,255,255,.96); }}
    h1 {{ margin: 0; font-size: 20px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 18px; display: grid; gap: 14px; }}
    .card {{ display: grid; grid-template-columns: minmax(220px, 340px) 1fr; gap: 16px; padding: 12px; border: 1px solid #dde3ea; border-radius: 8px; background: #fff; }}
    .photos {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; align-content: start; }}
    .photos img, .no-photo {{ width: 100%; aspect-ratio: 3 / 4; object-fit: cover; border-radius: 6px; background: #edf1f5; }}
    .no-photo {{ display: grid; place-items: center; color: #67717c; }}
    .topline {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
    h2 {{ margin: 0; font-size: 22px; }}
    h2 small {{ font-weight: 500; color: #5f6b76; }}
    p {{ margin: 10px 0; line-height: 1.45; white-space: pre-wrap; }}
    .decision {{ padding: 4px 8px; border-radius: 999px; background: #eef3f7; white-space: nowrap; font-weight: 700; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .chips span {{ padding: 4px 7px; border-radius: 999px; background: #edf7f0; color: #23412e; font-size: 12px; }}
    .chips.muted span {{ background: #f1f3f5; color: #55616d; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; color: #687480; font-size: 12px; }}
    .actions {{ display: flex; gap: 8px; margin-top: 12px; }}
    button {{ border: 1px solid #c4ccd5; border-radius: 6px; background: #fff; padding: 8px 10px; cursor: pointer; font: inherit; }}
    button:hover {{ background: #f1f4f7; }}
    @media (max-width: 760px) {{
      .card {{ grid-template-columns: 1fr; }}
      header {{ align-items: start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Tinder IA Review</h1>
    <span>{len(profiles)} perfis recentes</span>
  </header>
  <main>{''.join(cards) or '<p>Nenhum perfil recebido ainda.</p>'}</main>
  <script>
    const authQuery = {json.dumps(auth_query)};
    document.addEventListener("click", async (event) => {{
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      const card = button.closest(".card");
      const payload = {{
        action: button.dataset.action,
        record_id: button.dataset.record,
        profile_id: button.dataset.profile,
      }};
      button.disabled = true;
      try {{
        const res = await fetch("/api/reviews" + authQuery, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        const data = await res.json();
        if (!res.ok || data.ok === false) throw new Error(data.error || "erro");
        const state = card.querySelector(".review-state");
        if (state) state.textContent = "revisado: " + payload.action;
      }} catch (err) {{
        alert("Falha ao salvar review: " + (err.message || err));
      }} finally {{
        button.disabled = false;
      }}
    }});
  </script>
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
