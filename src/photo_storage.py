"""Download, salvamento e retencao das fotos usadas para auditoria/treino."""

from __future__ import annotations

import csv
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from body_photo_rules import BODY_MEASUREMENT_MIN_STRENGTH, body_measurement_strength
from config import ROOT_DIR, get_photos_config
from logging_config import get_logger


PHOTOS_DIR = ROOT_DIR / "data" / "photos"
PROFILES_PATH = ROOT_DIR / "data" / "profiles.csv"
REVIEW_PATH = ROOT_DIR / "data" / "review_queue.csv"
logger = get_logger(__name__)
_RETENTION_LOCK = threading.Lock()


def safe_filename(text: str, max_len: int = 20) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len]


def _normalize_photo_key(text: str) -> str:
    base = unicodedata.normalize("NFD", (text or "").strip().lower())
    base = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
    base = base.replace("_", " ")
    return re.sub(r"\s+", " ", base).strip()


def _parse_photo_file_key(path: Path, label: int) -> tuple[str, int, int] | None:
    match = re.match(r"^[^_]+_(.+)_(\d+)(?:_(?:face|body))?\.jpg$", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    safe_name = match.group(1)
    age = int(match.group(2))
    return (_normalize_photo_key(safe_name), age, label)


def _profile_photo_keys(name: str, age: int, label: int) -> set[tuple[str, int, int]]:
    """
    Gera as chaves usadas para ligar CSV <-> arquivo salvo.

    O arquivo usa safe_filename(name), que pode truncar nomes compostos. Por isso
    registramos a chave completa e a chave do nome seguro/truncado.
    """
    keys = {(_normalize_photo_key(name), age, label)}
    safe_name = safe_filename(name)
    keys.add((_normalize_photo_key(safe_name), age, label))
    return {key for key in keys if key[0]}


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _sorted_existing_jpgs(dest_dir: Path) -> list[Path]:
    items: list[tuple[float, Path]] = []
    for path in dest_dir.glob("*.jpg"):
        mtime = _safe_mtime(path)
        if mtime is not None:
            items.append((mtime, path))
    return [path for _, path in sorted(items, key=lambda item: item[0])]


def _load_photo_training_counts() -> dict[tuple[str, int, int], dict[str, int]]:
    counts: dict[tuple[str, int, int], dict[str, int]] = defaultdict(
        lambda: {"consolidated": 0, "pending": 0}
    )

    if PROFILES_PATH.exists():
        with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("source", "real")).strip().lower() != "real":
                    continue

                try:
                    label = int(row.get("label", ""))
                    age = int(float(row.get("age", 0) or 0))
                except Exception:
                    continue

                saved_flag = str(row.get("photo_features_saved", "")).strip().lower()
                has_saved_features = saved_flag in {"1", "1.0", "true"}
                for key in _profile_photo_keys(row.get("name", ""), age, label):
                    if has_saved_features:
                        counts[key]["consolidated"] += 1
                    else:
                        counts[key]["pending"] += 1

    if REVIEW_PATH.exists():
        with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("review_status", "pending")).strip().lower() != "pending":
                    continue

                try:
                    label = int(row.get("label") or row.get("original_label", ""))
                    age = int(float(row.get("age", 0) or 0))
                except Exception:
                    continue

                for key in _profile_photo_keys(row.get("name", ""), age, label):
                    counts[key]["pending"] += 1

    return counts


def eligible_photo_eviction_candidates(dest_dir: Path, label: int) -> list[Path]:
    """
    Retorna fotos antigas que podem sair sem quebrar o treino.

    Politica:
      - nunca apagar fotos sem features salvas no CSV
      - proteger os arquivos mais novos quando houver linhas pendentes no CSV
      - dentre as consolidadas, apagar primeiro as mais antigas
      - fotos sem correspondencia no CSV sao auditoria, nao treino pendente;
        entram na limpeza quando passarem da janela de seguranca configurada
    """
    jpgs = _sorted_existing_jpgs(dest_dir)
    if not jpgs:
        return []

    counts = _load_photo_training_counts()
    cfg = get_photos_config()
    allow_legacy_audit_eviction = bool(cfg.get("allow_legacy_audit_eviction", True))
    audit_grace_seconds = max(
        0.0,
        float(cfg.get("audit_retention_grace_hours", 24)) * 60 * 60,
    )
    now = time.time()

    groups: dict[tuple[str, int, int] | str, list[Path]] = defaultdict(list)
    for jpg in jpgs:
        key = _parse_photo_file_key(jpg, label) or f"unparsed:{jpg.name}"
        groups[key].append(jpg)

    eligible: list[Path] = []
    for key, files in groups.items():
        ordered = [p for p in files if p.exists()]
        ordered.sort(key=lambda p: _safe_mtime(p) or 0.0)

        if not isinstance(key, tuple):
            if allow_legacy_audit_eviction:
                eligible.extend(
                    p for p in ordered
                    if (now - (_safe_mtime(p) or now)) >= audit_grace_seconds
                )
            continue

        info = counts.get(key, {"consolidated": 0, "pending": 0})
        pending = max(0, int(info.get("pending", 0)))
        consolidated = max(0, int(info.get("consolidated", 0)))

        if pending == 0 and consolidated == 0:
            if allow_legacy_audit_eviction:
                eligible.extend(
                    p for p in ordered
                    if (now - (_safe_mtime(p) or now)) >= audit_grace_seconds
                )
            continue

        protected_recent = set(ordered[-pending:]) if pending else set()
        remaining = [p for p in ordered if p not in protected_recent]
        eligible.extend(remaining[:consolidated])

    return sorted((p for p in eligible if p.exists()), key=lambda p: _safe_mtime(p) or 0.0)


def _download_bytes_with_retries(url: str, timeout_s: float, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Referer": "https://tinder.com/",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read()
        except (TimeoutError, ConnectionError, ConnectionResetError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            sleep_s = 0.35 * attempt
            logger.warning(
                "Download de foto falhou temporariamente; tentando novamente (%s/%s): %s",
                attempt,
                attempts,
                exc,
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"download_failed_after_retries: {last_error}")


def evict_oldest_consolidated(dest_dir: Path, max_files: int) -> None:
    with _RETENTION_LOCK:
        jpgs = _sorted_existing_jpgs(dest_dir)
        excess = len(jpgs) - max_files
        if excess <= 0:
            return

        label = 1 if dest_dir.name == "liked" else 0
        candidates = eligible_photo_eviction_candidates(dest_dir, label)
        logger.info(
            "Retencao de fotos: dir=%s files=%s max=%s excess=%s candidates=%s",
            dest_dir,
            len(jpgs),
            max_files,
            excess,
            len(candidates),
        )
        for jpg in candidates[:excess]:
            try:
                jpg.unlink(missing_ok=True)
                jpg.with_suffix(".txt").unlink(missing_ok=True)
                logger.info("Foto consolidada antiga removida: %s", jpg)
            except Exception:
                logger.exception("Falha ao remover foto antiga: %s", jpg)


def download_photo_async(
    url: str,
    tinder_id: str,
    name: str,
    age: int,
    decision: str,
    reason: str = "",
    role: str = "",
) -> None:
    """Baixa a foto em background sem bloquear o processamento principal."""
    cfg = get_photos_config()
    if not cfg.get("enabled", True):
        logger.debug("Download de foto ignorado: photos.enabled=false")
        return

    if decision == "CURTIR" and not cfg.get("save_liked", True):
        logger.debug("Download de foto curtida ignorado por config")
        return
    if decision == "NÃO CURTIR" and not cfg.get("save_disliked", True):
        logger.debug("Download de foto recusada ignorado por config")
        return

    def _download() -> None:
        dest_dir = PHOTOS_DIR / ("liked" if decision == "CURTIR" else "disliked")
        try:
            logger.info(
                "Download de foto iniciado: name=%r age=%s decision=%s id=%r role=%r",
                name,
                age,
                decision,
                tinder_id,
                role or "main",
            )
            dest_dir.mkdir(parents=True, exist_ok=True)

            safe_name = safe_filename(name)
            safe_id = tinder_id[:8] if tinder_id else "unknown"
            base = f"{safe_id}_{safe_name}_{age}"
            safe_role = re.sub(r"[^a-z0-9_-]", "", (role or "").strip().lower())
            suffix = f"_{safe_role}" if safe_role in {"face", "body"} else ""
            dest = dest_dir / f"{base}{suffix}.jpg"

            if not dest.exists() and url:
                timeout_s = float(cfg.get("download_timeout_seconds", 8))
                try:
                    dest.write_bytes(_download_bytes_with_retries(url, timeout_s))
                except RuntimeError as exc:
                    logger.warning(
                        "Foto nao salva apos retries: name=%r age=%s decision=%s id=%r error=%s",
                        name,
                        age,
                        decision,
                        tinder_id,
                        exc,
                    )
                    return
                logger.info("Foto salva: %s", dest)
            elif dest.exists():
                logger.debug("Foto ja existia: %s", dest)
            elif not url:
                logger.warning("Download de foto sem URL: name=%r id=%r", name, tinder_id)

            if reason:
                reason_path = dest_dir / f"{base}{suffix}.txt"
                reason_path.write_text(reason, encoding="utf-8")
                logger.debug("Motivo da foto salvo: %s", reason_path)

        except Exception:
            logger.exception("Erro no download/salvamento de foto: name=%r age=%s decision=%s id=%r", name, age, decision, tinder_id)
        finally:
            try:
                max_files = cfg.get("max_per_folder", 50)
                evict_oldest_consolidated(dest_dir, max_files)
            except Exception:
                logger.exception("Erro na retencao apos download: dir=%s", dest_dir)

    threading.Thread(target=_download, daemon=True).start()


def download_profile_photos_async(profile: dict, decision: str, reason: str = "") -> None:
    """
    Salva a foto principal para review e, quando houver sinal corporal,
    salva tambem um arquivo *_body.jpg para treino/review corporal.

    A escolha vem da analise ja feita em photo_features.analyze_photos; portanto
    nao adiciona custo pesado ao swipe automatico.
    """
    photo = profile.get("_photo_features") or {}
    cfg = get_photos_config()
    save_pair = bool(cfg.get("save_face_body_pair", True))
    pairs = profile.get("_photo_url_pairs") or []

    def storage_url(url: str) -> str:
        clean = (url or "").strip()
        if not clean:
            return ""
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            analysis_url = (pair.get("analysis_url") or "").strip()
            save_url = (pair.get("save_url") or "").strip()
            if clean == analysis_url and save_url:
                return save_url
        return clean

    face_url = storage_url(photo.get("_best_face_photo_url") or "")
    body_url = storage_url(photo.get("_best_body_photo_url") or "")
    review_url = storage_url(photo.get("_review_photo_url") or "")
    body_strength = body_measurement_strength(photo) if body_url else 0.0
    if body_url and body_strength < BODY_MEASUREMENT_MIN_STRENGTH:
        logger.info(
            "Foto corporal ignorada por sinal fraco: name=%r age=%s body_strength=%.3f",
            profile.get("name", ""),
            profile.get("age", ""),
            body_strength,
        )
        body_url = ""
    fallback_url = (profile.get("_photo_url") or "").strip()

    main_url = review_url or face_url or body_url or fallback_url
    profile["_review_photo_url"] = main_url
    profile["_face_photo_url"] = face_url
    profile["_body_photo_url"] = body_url

    if not main_url:
        download_photo_async(
            "",
            profile.get("_tinder_id", ""),
            profile.get("name", ""),
            int(profile.get("age", 0) or 0),
            decision,
            reason,
        )
        return

    seen: set[tuple[str, str]] = set()

    def add_download(url: str, role: str, role_label: str) -> None:
        clean = (url or "").strip()
        safe_role = role or "main"
        seen_key = (safe_role, clean)
        if not clean or seen_key in seen:
            return
        seen.add(seen_key)
        role_reason = reason
        if role_label:
            role_reason = (reason + "\n\n" if reason else "") + f"Foto salva para: {role_label}"
        download_photo_async(
            clean,
            profile.get("_tinder_id", ""),
            profile.get("name", ""),
            int(profile.get("age", 0) or 0),
            decision,
            role_reason,
            role=role,
        )

    add_download(main_url, "", "review/principal")
    if not save_pair:
        return
    if face_url and face_url != main_url:
        add_download(face_url, "face", "analise de rosto")
    # A review UI e o treino corporal procuram explicitamente arquivos *_body.jpg.
    # Mesmo quando a melhor foto corporal tambem e a principal, salvamos o alias.
    if body_url:
        add_download(body_url, "body", "analise de corpo")
