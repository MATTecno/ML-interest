"""Embeddings visuais semanticos por foto usando CLIP/SentenceTransformers.

O objetivo deste modulo e complementar as features leves/DeepFace existentes
com um vetor visual mais rico. O vetor bruto fica em cache separado e apenas as
componentes PCA entram no CSV/modelo principal.
"""

from __future__ import annotations

import hashlib
import gc
import json
import math
import pickle
import queue
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from config import ROOT_DIR, get_photos_config
from logging_config import get_logger


logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
DEFAULT_N_COMPONENTS = 32
SEMANTIC_EMBEDDING_FEATURE_NAMES = [
    f"photo_clip_pc_{i:02d}" for i in range(1, DEFAULT_N_COMPONENTS + 1)
]
SEMANTIC_PCA_PATH = ROOT_DIR / "data" / "photo_semantic_pca.pkl"
DEFAULT_CACHE_PATH = ROOT_DIR / "data" / "photo_semantic_cache.json"
DEFAULT_SQLITE_CACHE_PATH = ROOT_DIR / "data" / "photo_semantic_cache.sqlite"

_CACHE_LOCK = threading.RLock()
_CACHE_DATA: dict | None = None
_SQLITE_IMPORT_SIGNATURES: set[str] = set()
_MODEL_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()
_MODEL = None
_MODEL_NAME = ""
_MODEL_DEVICE = ""
_MODEL_LOAD_FAILED = False
_STATS = {
    "hits": 0,
    "misses": 0,
    "computed": 0,
    "errors": 0,
    "profile_batches": 0,
    "profile_batches_with_clip": 0,
    "photos_requested": 0,
    "worker_requests": 0,
    "worker_images": 0,
    "worker_failures": 0,
    "worker_timeouts": 0,
}


def _stat_add(key: str, value: int = 1) -> None:
    with _CACHE_LOCK:
        _STATS[key] = int(_STATS.get(key, 0)) + int(value)


def _semantic_cfg() -> dict:
    cfg = get_photos_config().get("semantic_embedding", {}) or {}
    return cfg if isinstance(cfg, dict) else {"enabled": bool(cfg)}


def semantic_embedding_enabled() -> bool:
    return bool(_semantic_cfg().get("enabled", False))


def model_name(config: dict | None = None) -> str:
    cfg = config if config is not None else _semantic_cfg()
    quality_mode = str(cfg.get("quality_mode") or "balanced").strip().lower()
    if quality_mode in {"high", "quality", "max", "accurate"}:
        high_quality = str(cfg.get("high_quality_model_name") or "").strip()
        if high_quality:
            return high_quality
    return str(cfg.get("model_name") or DEFAULT_MODEL_NAME)


def pca_components(config: dict | None = None) -> int:
    cfg = config if config is not None else _semantic_cfg()
    try:
        value = int(cfg.get("pca_components", DEFAULT_N_COMPONENTS))
    except Exception:
        value = DEFAULT_N_COMPONENTS
    return max(1, min(DEFAULT_N_COMPONENTS, value))


def batch_size(config: dict | None = None) -> int:
    cfg = config if config is not None else _semantic_cfg()
    try:
        value = int(cfg.get("batch_size", 8) or 8)
    except Exception:
        value = 8
    return max(1, min(64, value))


def batch_timeout_seconds(config: dict | None = None) -> float:
    cfg = config if config is not None else _semantic_cfg()
    try:
        value = float(cfg.get("batch_timeout_seconds", cfg.get("timeout_seconds", 20)) or 20)
    except Exception:
        value = 20.0
    return max(0.0, value)


def worker_enabled(config: dict | None = None) -> bool:
    cfg = config if config is not None else _semantic_cfg()
    return bool(cfg.get("worker_enabled", False))


def worker_start_timeout_seconds(config: dict | None = None) -> float:
    cfg = config if config is not None else _semantic_cfg()
    try:
        value = float(cfg.get("worker_start_timeout_seconds", 30) or 30)
    except Exception:
        value = 30.0
    return max(0.1, value)


def worker_request_timeout_seconds(config: dict | None = None) -> float:
    cfg = config if config is not None else _semantic_cfg()
    try:
        value = float(cfg.get("worker_request_timeout_seconds", batch_timeout_seconds(cfg)) or 0)
    except Exception:
        value = batch_timeout_seconds(cfg)
    return max(0.1, value)


def _cache_path() -> Path:
    raw = str(_semantic_cfg().get("cache_path") or "data/photo_semantic_cache.json")
    path = Path(raw)
    return path if path.is_absolute() else ROOT_DIR / path


def _cache_backend(config: dict | None = None) -> str:
    cfg = config if config is not None else _semantic_cfg()
    raw = str(cfg.get("cache_backend") or "").strip().lower()
    if raw in {"sqlite", "sqlite3", "db"}:
        return "sqlite"
    if raw in {"json", "file"}:
        return "json"
    suffix = _cache_path().suffix.lower()
    return "sqlite" if suffix in {".sqlite", ".sqlite3", ".db"} else "json"


def _sqlite_cache_path(config: dict | None = None) -> Path:
    cfg = config if config is not None else _semantic_cfg()
    raw = str(cfg.get("cache_path") or "").strip()
    if raw:
        path = Path(raw)
        path = path if path.is_absolute() else ROOT_DIR / path
        if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            return path
    return DEFAULT_SQLITE_CACHE_PATH


def _json_cache_path(config: dict | None = None) -> Path:
    cfg = config if config is not None else _semantic_cfg()
    if _cache_backend(cfg) == "sqlite":
        raw = str(cfg.get("legacy_json_cache_path") or "data/photo_semantic_cache.json")
    else:
        raw = str(cfg.get("cache_path") or "data/photo_semantic_cache.json")
    path = Path(raw)
    return path if path.is_absolute() else ROOT_DIR / path


def resolve_device(config: dict | None = None) -> str:
    cfg = config if config is not None else _semantic_cfg()
    requested = str(cfg.get("device") or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def release_model(reason: str = "", include_worker: bool = False) -> None:
    """Libera o modelo CLIP e caches CUDA quando o swipe precisa baixar memoria."""
    if worker_enabled() and not include_worker:
        logger.info("release_model ignorado: worker CLIP persistente ativo reason=%s", reason)
        return
    if include_worker:
        shutdown_worker(reason)
    global _MODEL, _MODEL_NAME, _MODEL_DEVICE
    with _MODEL_LOCK:
        had_model = _MODEL is not None
        device = _MODEL_DEVICE
        _MODEL = None
        _MODEL_NAME = ""
        _MODEL_DEVICE = ""
    if had_model:
        logger.info("Modelo visual semantico liberado da memoria: device=%s reason=%s", device, reason)
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        logger.debug("Falha ao limpar cache CUDA do modelo semantico", exc_info=True)


def shutdown_worker(reason: str = "") -> None:
    try:
        from photo_semantic_worker import shutdown_worker as _shutdown_worker

        logger.info("Encerrando worker CLIP: reason=%s", reason)
        _shutdown_worker()
    except Exception:
        logger.debug("Falha ao encerrar worker CLIP", exc_info=True)


def prewarm_worker(reason: str = "server_start") -> dict | None:
    cfg = _semantic_cfg()
    if not semantic_embedding_enabled() or not worker_enabled(cfg):
        return None

    from photo_semantic_worker import warmup_clip

    logger.info(
        "Prewarm CLIP solicitado: reason=%s model=%s device=%s",
        reason,
        model_name(cfg),
        resolve_device(cfg),
    )
    response = warmup_clip(
        model_name(cfg),
        resolve_device(cfg),
        batch_size(cfg),
        worker_start_timeout_seconds(cfg),
        worker_request_timeout_seconds(cfg),
    )
    logger.info(
        "Prewarm CLIP finalizado: ok=%s device=%s errors=%s",
        bool(response.get("ok")),
        response.get("device"),
        response.get("errors") or {},
    )
    return response


def prewarm_worker_async(reason: str = "server_start") -> None:
    cfg = _semantic_cfg()
    if not semantic_embedding_enabled() or not worker_enabled(cfg):
        return

    def _run() -> None:
        try:
            prewarm_worker(reason)
        except Exception:
            _stat_add("worker_failures")
            logger.exception("Prewarm CLIP falhou")

    threading.Thread(target=_run, daemon=True, name="clip-prewarm").start()


def _release_after_batch(config: dict | None = None) -> bool:
    cfg = config if config is not None else _semantic_cfg()
    if worker_enabled(cfg):
        return False
    return bool(cfg.get("release_after_batch", True))


def cache_stats(reset: bool = False) -> dict:
    with _CACHE_LOCK:
        stats = dict(_STATS)
        if reset:
            for key in _STATS:
                _STATS[key] = 0
        return stats


def source_key_for_url(url: str) -> str:
    parsed = urlsplit(url or "")
    stable = parsed.path or url or ""
    return f"url:{stable}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_key_for_path(path: str | Path) -> str:
    p = Path(path)
    try:
        resolved = p.resolve()
        rel = resolved.relative_to(ROOT_DIR.resolve())
        label = str(rel)
    except Exception:
        resolved = p
        label = str(p)
    try:
        digest = _sha256_file(resolved)
    except Exception:
        digest = ""
    return f"path:{label}:{digest[:16]}"


def _image_sha256_bgr(img_bgr) -> str:
    try:
        arr = np.ascontiguousarray(img_bgr)
        return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        return ""


def _sqlite_connect() -> sqlite3.Connection:
    path = _sqlite_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            source_key TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            device TEXT,
            created_at TEXT,
            image_sha256 TEXT,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photo_semantic_embeddings_model ON embeddings(model_name)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    return conn


def _sqlite_meta_get(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT value FROM cache_meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else ""


def _sqlite_meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cache_meta(key, value) VALUES(?, ?)",
        (key, value),
    )


def _entry_to_sqlite_row(source_key: str, entry: dict) -> tuple | None:
    if not isinstance(entry, dict):
        return None
    model = str(entry.get("model_name") or "").strip()
    if not model:
        return None
    try:
        arr = np.asarray(entry.get("embedding"), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 16 or np.any(np.isnan(arr)):
        return None
    return (
        str(source_key or entry.get("source_key") or "").strip(),
        model,
        str(entry.get("device") or ""),
        str(entry.get("created_at") or ""),
        str(entry.get("image_sha256") or ""),
        int(arr.size),
        sqlite3.Binary(np.ascontiguousarray(arr).tobytes()),
    )


def _sqlite_row_to_entry(source_key: str, row: tuple) -> dict:
    model_name, device, created_at, image_sha256, dim, blob = row
    try:
        arr = np.frombuffer(blob, dtype=np.float32, count=int(dim)).astype(float)
    except Exception:
        arr = np.array([], dtype=float)
    return {
        "model_name": model_name,
        "device": device or "",
        "created_at": created_at or "",
        "source_key": source_key,
        "image_sha256": image_sha256 or "",
        "embedding": arr,
    }


def migrate_legacy_json_to_sqlite(force: bool = False) -> dict:
    """Importa o cache JSON antigo para SQLite sem apagar o arquivo original."""
    json_path = _json_cache_path()
    sqlite_path = _sqlite_cache_path()
    if not json_path.exists():
        return {
            "ok": True,
            "skipped": "legacy_json_missing",
            "json_path": str(json_path),
            "sqlite_path": str(sqlite_path),
            "imported": 0,
        }

    started = time.perf_counter()
    stat = json_path.stat()
    signature = {
        "path": str(json_path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }
    signature_key = json.dumps(signature, sort_keys=True)
    if not force and signature_key in _SQLITE_IMPORT_SIGNATURES:
        return {
            "ok": True,
            "skipped": "already_checked",
            "json_path": str(json_path),
            "sqlite_path": str(sqlite_path),
            "imported": 0,
        }
    meta_key = f"legacy_json_import::{signature['path']}"

    with _CACHE_LOCK:
        with _sqlite_connect() as conn:
            previous = _sqlite_meta_get(conn, meta_key)
            current = signature_key
            if previous == current and not force:
                _SQLITE_IMPORT_SIGNATURES.add(signature_key)
                count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                return {
                    "ok": True,
                    "skipped": "already_imported",
                    "json_path": str(json_path),
                    "sqlite_path": str(sqlite_path),
                    "sqlite_entries": int(count),
                    "imported": 0,
                }

            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.exception("Falha ao importar cache JSON semantico para SQLite: %s", json_path)
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "json_path": str(json_path),
                    "sqlite_path": str(sqlite_path),
                    "imported": 0,
                }

            if not isinstance(data, dict):
                data = {}
            imported = 0
            skipped = 0
            batch = []
            for source_key, entry in data.items():
                row = _entry_to_sqlite_row(str(source_key), entry)
                if row is None or not row[0]:
                    skipped += 1
                    continue
                batch.append(row)
                if len(batch) >= 500:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO embeddings
                        (source_key, model_name, device, created_at, image_sha256, dim, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    imported += len(batch)
                    batch.clear()
            if batch:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (source_key, model_name, device, created_at, image_sha256, dim, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                imported += len(batch)
            _sqlite_meta_set(conn, meta_key, current)
            conn.commit()
            sqlite_entries = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            _SQLITE_IMPORT_SIGNATURES.add(signature_key)

    logger.info(
        "Cache CLIP JSON importado para SQLite: imported=%s skipped=%s sqlite_entries=%s json=%s sqlite=%s elapsed=%.2fs",
        imported,
        skipped,
        sqlite_entries,
        json_path,
        sqlite_path,
        time.perf_counter() - started,
    )
    return {
        "ok": True,
        "json_path": str(json_path),
        "sqlite_path": str(sqlite_path),
        "imported": imported,
        "skipped": skipped,
        "sqlite_entries": int(sqlite_entries),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _cache_get_entry(source_key: str) -> dict | None:
    if _cache_backend() != "sqlite":
        cache = _load_cache()
        with _CACHE_LOCK:
            entry = cache.get(source_key)
        return entry if isinstance(entry, dict) else None

    migrate_legacy_json_to_sqlite(force=False)
    with _CACHE_LOCK:
        with _sqlite_connect() as conn:
            row = conn.execute(
                """
                SELECT model_name, device, created_at, image_sha256, dim, embedding
                FROM embeddings
                WHERE source_key = ?
                """,
                (source_key,),
            ).fetchone()
    return _sqlite_row_to_entry(source_key, row) if row else None


def _save_cache_updates(updates: dict[str, dict]) -> None:
    if not updates:
        return
    if _cache_backend() != "sqlite":
        with _CACHE_LOCK:
            cache = _load_cache()
            cache.update(updates)
            _save_cache(cache)
        return

    migrate_legacy_json_to_sqlite(force=False)
    rows = []
    for source_key, entry in updates.items():
        row = _entry_to_sqlite_row(source_key, entry)
        if row is not None and row[0]:
            rows.append(row)
    if not rows:
        return
    with _CACHE_LOCK:
        with _sqlite_connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (source_key, model_name, device, created_at, image_sha256, dim, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()


def _load_cache() -> dict:
    global _CACHE_DATA
    with _CACHE_LOCK:
        if _CACHE_DATA is not None:
            return _CACHE_DATA
        path = _json_cache_path()
        if not path.exists():
            _CACHE_DATA = {}
            return _CACHE_DATA
        try:
            with open(path, encoding="utf-8") as f:
                _CACHE_DATA = json.load(f)
        except Exception:
            logger.exception("Falha ao carregar cache semantico de foto: %s", path)
            _CACHE_DATA = {}
        return _CACHE_DATA


def _save_cache(cache: dict) -> None:
    path = _json_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, separators=(",", ":"))
        tmp.replace(path)
    except Exception:
        logger.exception("Falha ao salvar cache semantico de foto: %s", path)


def _embedding_from_entry(entry: dict, expected_model: str) -> np.ndarray | None:
    if not isinstance(entry, dict) or entry.get("model_name") != expected_model:
        return None
    emb = entry.get("embedding")
    if emb is None:
        return None
    try:
        arr = np.array(emb, dtype=float)
        if arr.ndim != 1 or arr.size < 16 or np.any(np.isnan(arr)):
            return None
        return arr
    except Exception:
        return None


def get_cached_embedding(source_key: str, expected_model: str | None = None) -> np.ndarray | None:
    expected = expected_model or model_name()
    entry = _cache_get_entry(source_key)
    embedding = _embedding_from_entry(entry, expected)
    if embedding is not None:
        _STATS["hits"] += 1
    else:
        _STATS["misses"] += 1
    return embedding


def _get_model():
    global _MODEL, _MODEL_NAME, _MODEL_DEVICE, _MODEL_LOAD_FAILED
    cfg = _semantic_cfg()
    name = model_name(cfg)
    device = resolve_device(cfg)
    if _MODEL_LOAD_FAILED:
        raise RuntimeError("modelo visual semantico indisponivel neste processo")
    with _MODEL_LOCK:
        if _MODEL_LOAD_FAILED:
            raise RuntimeError("modelo visual semantico indisponivel neste processo")
        if _MODEL is not None and _MODEL_NAME == name and _MODEL_DEVICE == device:
            return _MODEL, device
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        logger.info("Carregando modelo visual semantico: model=%s device=%s", name, device)
        try:
            try:
                _MODEL = SentenceTransformer(name, device=device, local_files_only=True)
                logger.info("Modelo visual semantico carregado do cache local: model=%s", name)
            except TypeError:
                logger.warning(
                    "SentenceTransformer sem local_files_only; carregando modelo visual pelo caminho padrao: model=%s",
                    name,
                )
                _MODEL = SentenceTransformer(name, device=device)
            except Exception:
                logger.warning(
                    "Modelo visual semantico nao encontrado no cache local; tentando carga padrao: model=%s device=%s",
                    name,
                    device,
                    exc_info=True,
                )
                _MODEL = SentenceTransformer(name, device=device)
        except Exception:
            _MODEL_LOAD_FAILED = True
            logger.exception("Falha ao carregar modelo visual semantico: model=%s device=%s", name, device)
            raise
        _MODEL_NAME = name
        _MODEL_DEVICE = device
        logger.info(
            "Modelo visual semantico carregado: model=%s device=%s elapsed=%.2fs",
            name,
            device,
            time.perf_counter() - started,
        )
        return _MODEL, device


def _as_valid_embedding(value) -> np.ndarray | None:
    try:
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size < 16 or np.any(np.isnan(arr)):
            return None
        return arr
    except Exception:
        return None


def _compute_embedding_bgr(img_bgr) -> tuple[np.ndarray | None, str]:
    if img_bgr is None:
        return None, resolve_device()
    try:
        import cv2
        from PIL import Image as PILImage

        model, device = _get_model()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        with _ENCODE_LOCK:
            embedding = model.encode(
                pil_img,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return _as_valid_embedding(embedding), device
    except Exception:
        logger.exception("Falha ao calcular embedding visual semantico")
        return None, resolve_device()


def _compute_embeddings_pil_batch(pil_images: list) -> tuple[list[np.ndarray | None], str]:
    if not pil_images:
        return [], resolve_device()
    model, device = _get_model()
    with _ENCODE_LOCK:
        embeddings = model.encode(
            pil_images,
            batch_size=batch_size(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    arr = np.array(embeddings, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    results = [_as_valid_embedding(row) for row in arr[: len(pil_images)]]
    if len(results) < len(pil_images):
        results.extend([None] * (len(pil_images) - len(results)))
    return results, device


def _bgr_to_worker_image(label: str, source_key: str, img_bgr) -> dict | None:
    try:
        arr = np.asarray(img_bgr)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        if width <= 0 or height <= 0:
            return None
        return {
            "label": str(label),
            "source_key": str(source_key or ""),
            "width": width,
            "height": height,
            "rgb_bytes": rgb.tobytes(),
        }
    except Exception:
        logger.debug("Falha ao preparar imagem para worker CLIP: %s", source_key, exc_info=True)
        return None


def _compute_bgr_batch_with_worker(
    items: list[tuple[str, str, object]],
    config: dict | None = None,
    requested_batch_size: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, np.ndarray | None], str, dict[str, str]]:
    cfg = config if config is not None else _semantic_cfg()
    payload = []
    results: dict[str, np.ndarray | None] = {}
    for label, source_key, img_bgr in items:
        label = str(label)
        results[label] = None
        worker_image = _bgr_to_worker_image(label, source_key, img_bgr)
        if worker_image is not None:
            payload.append(worker_image)

    if not payload:
        logger.info("Worker CLIP nao acionado: imagens validas=0 items=%s", len(items))
        return results, resolve_device(cfg), {}

    from photo_semantic_worker import encode_rgb_batch

    request_batch_size = int(requested_batch_size or batch_size(cfg))
    request_timeout = float(timeout_seconds if timeout_seconds is not None else worker_request_timeout_seconds(cfg))
    _stat_add("worker_requests")
    _stat_add("worker_images", len(payload))
    logger.info(
        "Worker CLIP solicitado: images=%s batch_size=%s model=%s device=%s timeout=%.1fs",
        len(payload),
        request_batch_size,
        model_name(cfg),
        resolve_device(cfg),
        request_timeout,
    )
    response = encode_rgb_batch(
        payload,
        model_name(cfg),
        resolve_device(cfg),
        request_batch_size,
        worker_start_timeout_seconds(cfg),
        request_timeout,
    )
    device = str(response.get("device") or resolve_device(cfg))
    errors = {
        str(key): str(value)
        for key, value in (response.get("errors") or {}).items()
    }
    if not response.get("ok", False):
        _stat_add("worker_failures")
        logger.warning("Worker CLIP retornou falha: images=%s errors=%s", len(payload), errors)
        raise RuntimeError("; ".join(errors.values()) or "worker CLIP retornou falha")
    for label, raw_embedding in (response.get("embeddings") or {}).items():
        results[str(label)] = _as_valid_embedding(raw_embedding)
    ok_count = len([value for value in results.values() if value is not None])
    logger.info(
        "Worker CLIP finalizado: images=%s embeddings=%s errors=%s device=%s elapsed=%.2fs",
        len(payload),
        ok_count,
        len(errors),
        device,
        float(response.get("elapsed_seconds") or 0.0),
    )
    return results, device, errors


def _run_with_timeout(timeout_s: float, name: str, worker_fn):
    if timeout_s <= 0:
        return worker_fn()

    result_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)

    def _thread_worker() -> None:
        try:
            result_queue.put(worker_fn(), block=False)
        except BaseException as exc:
            try:
                result_queue.put(exc, block=False)
            except Exception:
                pass

    worker = threading.Thread(target=_thread_worker, daemon=True, name=name)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"timeout apos {timeout_s:.1f}s")
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        raise RuntimeError("worker sem resultado")
    if isinstance(result, BaseException):
        raise result
    return result


def embedding_for_bgr(source_key: str, img_bgr) -> np.ndarray | None:
    """Retorna embedding semantico para uma imagem BGR, usando cache por source_key."""
    if not semantic_embedding_enabled() or not source_key:
        return None

    expected_model = model_name()
    cached = get_cached_embedding(source_key, expected_model)
    if cached is not None:
        logger.debug("Cache HIT embedding semantico key=%s", source_key)
        return cached

    cfg = _semantic_cfg()
    try:
        timeout_s = float(cfg.get("timeout_seconds", 20) or 20)
    except Exception:
        timeout_s = 20.0

    started = time.perf_counter()

    cfg = _semantic_cfg()
    if worker_enabled(cfg):
        try:
            embeddings, device, errors = _compute_bgr_batch_with_worker(
                [(source_key, source_key, img_bgr)],
                cfg,
                requested_batch_size=1,
                timeout_seconds=worker_request_timeout_seconds(cfg),
            )
            embedding = embeddings.get(source_key)
            if errors:
                logger.warning("Worker CLIP retornou erros key=%s errors=%s", source_key, errors)
        except Exception:
            _STATS["errors"] += 1
            logger.exception("Embedding visual semantico falhou no worker CLIP key=%s", source_key)
            return None
    else:
        def _worker() -> tuple[np.ndarray | None, str]:
            return _compute_embedding_bgr(img_bgr)

        if timeout_s <= 0:
            embedding, device = _worker()
        else:
            result_queue: "queue.Queue[tuple[np.ndarray | None, str] | BaseException]" = queue.Queue(maxsize=1)

            def _thread_worker() -> None:
                try:
                    result_queue.put(_worker(), block=False)
                except BaseException as exc:
                    try:
                        result_queue.put(exc, block=False)
                    except Exception:
                        pass

            worker = threading.Thread(
                target=_thread_worker,
                daemon=True,
                name=f"photo-semantic-{hashlib.md5(source_key.encode()).hexdigest()[:8]}",
            )
            worker.start()
            worker.join(timeout_s)
            if worker.is_alive():
                _STATS["errors"] += 1
                logger.warning(
                    "Timeout no embedding visual semantico: key=%s timeout=%.1fs",
                    source_key,
                    timeout_s,
                )
                return None
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                _STATS["errors"] += 1
                return None
            if isinstance(result, BaseException):
                _STATS["errors"] += 1
                logger.exception(
                    "Embedding visual semantico falhou em worker",
                    exc_info=(type(result), result, result.__traceback__),
                )
                return None
            embedding, device = result

    if embedding is None:
        _STATS["errors"] += 1
        return None

    entry = {
        "model_name": expected_model,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_key": source_key,
        "image_sha256": _image_sha256_bgr(img_bgr),
        "embedding": [float(x) for x in embedding],
    }
    _save_cache_updates({source_key: entry})
    _STATS["computed"] += 1
    logger.info(
        "Embedding visual semantico calculado: key=%s dim=%s device=%s elapsed=%.2fs",
        source_key,
        embedding.size,
        device,
        time.perf_counter() - started,
    )
    return embedding


def embedding_for_path(path: str | Path) -> np.ndarray | None:
    """Calcula/retorna embedding semantico de uma imagem local."""
    if not semantic_embedding_enabled():
        return None
    try:
        import cv2

        p = Path(path)
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            return None
        return embedding_for_bgr(source_key_for_path(p), img_bgr)
    except Exception:
        logger.debug("Falha ao calcular embedding semantico local: %s", path, exc_info=True)
        return None


def embeddings_for_paths(paths: list[str | Path]) -> dict[str, np.ndarray | None]:
    """Calcula embeddings locais em lote, preservando cache por arquivo."""
    results: dict[str, np.ndarray | None] = {}
    if not paths:
        return results
    if not semantic_embedding_enabled():
        return {str(Path(path)): None for path in paths}

    expected_model = model_name()
    unique_items: list[tuple[str, Path, str]] = []
    seen: set[str] = set()
    hit_count = 0
    miss_count = 0

    for raw_path in paths:
        p = Path(raw_path)
        label = str(p)
        if label in seen:
            continue
        seen.add(label)
        source_key = source_key_for_path(p)
        cached = get_cached_embedding(source_key, expected_model)
        if cached is not None:
            results[label] = cached
            hit_count += 1
            continue
        miss_count += 1
        results[label] = None
        unique_items.append((label, p, source_key))

    if not unique_items:
        logger.info(
            "Embedding visual semantico em lote: total=%s hits=%s misses=0 computed=0",
            len(seen),
            hit_count,
        )
        return results

    cfg = _semantic_cfg()
    bs = batch_size(cfg)
    timeout_s = batch_timeout_seconds(cfg)
    started_total = time.perf_counter()
    computed_count = 0
    error_count = 0
    last_device = resolve_device(cfg)

    try:
        from PIL import Image as PILImage
    except Exception:
        logger.exception("PIL indisponivel para embeddings semanticos em lote")
        _STATS["errors"] += len(unique_items)
        return results

    pending_cache_updates: dict[str, dict] = {}
    for offset in range(0, len(unique_items), bs):
        chunk = unique_items[offset : offset + bs]
        pil_images = []
        loaded_items: list[tuple[str, Path, str]] = []
        for label, path, source_key in chunk:
            try:
                with PILImage.open(path) as img:
                    pil_images.append(img.convert("RGB").copy())
                loaded_items.append((label, path, source_key))
            except Exception:
                error_count += 1
                logger.debug("Falha ao abrir imagem para embedding semantico: %s", path, exc_info=True)

        if not pil_images:
            continue

        batch_started = time.perf_counter()
        thread_name = f"photo-semantic-batch-{offset // bs + 1}"
        try:
            embeddings, device = _run_with_timeout(
                timeout_s,
                thread_name,
                lambda images=pil_images: _compute_embeddings_pil_batch(images),
            )
            last_device = device
        except TimeoutError:
            error_count += len(loaded_items)
            logger.warning(
                "Timeout no embedding visual semantico em lote: count=%s timeout=%.1fs",
                len(loaded_items),
                timeout_s,
            )
            continue
        except Exception:
            error_count += len(loaded_items)
            logger.exception("Falha ao calcular embeddings visuais semanticos em lote")
            continue

        cache_updates: dict[str, dict] = {}
        for (label, path, source_key), embedding in zip(loaded_items, embeddings):
            if embedding is None:
                error_count += 1
                continue
            results[label] = embedding
            computed_count += 1
            try:
                image_sha = _sha256_file(path)
            except Exception:
                image_sha = ""
            cache_updates[source_key] = {
                "model_name": expected_model,
                "device": device,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_key": source_key,
                "image_sha256": image_sha,
                "embedding": [float(x) for x in embedding],
            }

        if cache_updates:
            pending_cache_updates.update(cache_updates)

        logger.info(
            "Embedding visual semantico em lote: batch=%s-%s/%s computed=%s device=%s elapsed=%.2fs",
            offset + 1,
            min(offset + len(chunk), len(unique_items)),
            len(unique_items),
            len(cache_updates),
            last_device,
            time.perf_counter() - batch_started,
        )

    if pending_cache_updates:
        cache_save_started = time.perf_counter()
        _save_cache_updates(pending_cache_updates)
        _STATS["computed"] += len(pending_cache_updates)
        logger.info(
            "Cache CLIP salvo apos lote local: updates=%s elapsed=%.2fs",
            len(pending_cache_updates),
            time.perf_counter() - cache_save_started,
        )

    if error_count:
        _STATS["errors"] += error_count
    logger.info(
        "Embedding visual semantico em lote concluido: total=%s hits=%s misses=%s computed=%s errors=%s batch_size=%s device=%s elapsed=%.2fs",
        len(seen),
        hit_count,
        miss_count,
        computed_count,
        error_count,
        bs,
        last_device,
        time.perf_counter() - started_total,
    )
    return results


def embeddings_for_bgr_sources(items: list[tuple[str, str, object]]) -> dict[str, np.ndarray | None]:
    """Calcula embeddings BGR em lote para fontes remotas ja baixadas.

    Cada item e (label, source_key, img_bgr). O label volta no dict de retorno;
    source_key e usado para cache persistente.
    """
    results: dict[str, np.ndarray | None] = {}
    if not items:
        return results
    _stat_add("profile_batches")
    _stat_add("photos_requested", len(items))
    if not semantic_embedding_enabled():
        logger.info("CLIP desativado: profile_batch_photos=%s", len(items))
        return {str(label): None for label, _, _ in items}

    expected_model = model_name()
    unique_items: list[tuple[str, str, object]] = []
    seen_keys: set[str] = set()
    hit_count = 0
    miss_count = 0

    for label, source_key, img_bgr in items:
        label = str(label)
        source_key = str(source_key or "")
        results[label] = None
        if not source_key or img_bgr is None:
            continue
        cached = get_cached_embedding(source_key, expected_model)
        if cached is not None:
            results[label] = cached
            hit_count += 1
            continue
        miss_count += 1
        if source_key in seen_keys:
            continue
        seen_keys.add(source_key)
        unique_items.append((label, source_key, img_bgr))

    if not unique_items:
        logger.info(
            "CLIP profile batch via cache: total_photos=%s cache_hits=%s misses=0 computed=0 stats=%s",
            len(items),
            hit_count,
            cache_stats(),
        )
        if hit_count:
            _stat_add("profile_batches_with_clip")
        return results

    cfg = _semantic_cfg()
    bs = batch_size(cfg)
    try:
        from resource_guard import get_system_resource_snapshot, memory_relief_reached

        relief, _, snap = memory_relief_reached(snapshot=get_system_resource_snapshot())
        if not relief:
            bs = 1
            logger.warning(
                "Embedding visual remoto em lote usando batch_size=1 por memoria alta: mem=%.1f%% avail=%.1fMB swap=%.1f%%",
                snap.get("mem_used_pct", 0.0),
                snap.get("mem_avail_mb", 0.0),
                snap.get("swap_used_pct", 0.0),
            )
    except Exception:
        pass
    timeout_s = batch_timeout_seconds(cfg)
    started_total = time.perf_counter()
    computed_count = 0
    error_count = 0
    last_device = resolve_device(cfg)
    use_worker = worker_enabled(cfg)
    logger.info(
        "CLIP profile batch iniciado: total_photos=%s cache_hits=%s misses=%s unique_misses=%s mode=%s batch_size=%s",
        len(items),
        hit_count,
        miss_count,
        len(unique_items),
        "worker" if use_worker else "in_process",
        bs,
    )

    if not use_worker:
        try:
            import cv2
            from PIL import Image as PILImage
        except Exception:
            logger.exception("Dependencias indisponiveis para embeddings semanticos remotos em lote")
            _STATS["errors"] += len(unique_items)
            return results

    pending_cache_updates: dict[str, dict] = {}
    for offset in range(0, len(unique_items), bs):
        chunk = unique_items[offset : offset + bs]
        loaded_items: list[tuple[str, str, object]] = []
        embeddings: list[np.ndarray | None]
        device = last_device

        if use_worker:
            loaded_items = list(chunk)
            if not loaded_items:
                continue
            batch_started = time.perf_counter()
            try:
                worker_results, device, worker_errors = _compute_bgr_batch_with_worker(
                    loaded_items,
                    cfg,
                    requested_batch_size=bs,
                    timeout_seconds=worker_request_timeout_seconds(cfg),
                )
                embeddings = [worker_results.get(label) for label, _, _ in loaded_items]
                error_count += len(worker_errors)
                last_device = device
            except TimeoutError:
                _stat_add("worker_timeouts")
                error_count += len(loaded_items)
                logger.warning(
                    "Timeout no worker CLIP remoto em lote: count=%s timeout=%.1fs",
                    len(loaded_items),
                    worker_request_timeout_seconds(cfg),
                )
                continue
            except Exception:
                _stat_add("worker_failures")
                error_count += len(loaded_items)
                logger.exception("Falha no worker CLIP remoto em lote")
                continue
        else:
            pil_images = []
            for label, source_key, img_bgr in chunk:
                try:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    pil_images.append(PILImage.fromarray(img_rgb))
                    loaded_items.append((label, source_key, img_bgr))
                except Exception:
                    error_count += 1
                    logger.debug("Falha ao preparar imagem BGR para embedding semantico: %s", source_key, exc_info=True)

            if not pil_images:
                continue

            batch_started = time.perf_counter()
            try:
                embeddings, device = _run_with_timeout(
                    timeout_s,
                    f"photo-semantic-remote-batch-{offset // bs + 1}",
                    lambda images=pil_images: _compute_embeddings_pil_batch(images),
                )
                last_device = device
            except TimeoutError:
                error_count += len(loaded_items)
                logger.warning(
                    "Timeout no embedding visual semantico remoto em lote: count=%s timeout=%.1fs",
                    len(loaded_items),
                    timeout_s,
                )
                continue
            except Exception:
                error_count += len(loaded_items)
                logger.exception("Falha ao calcular embeddings visuais semanticos remotos em lote")
                continue

        cache_updates: dict[str, dict] = {}
        for (label, source_key, img_bgr), embedding in zip(loaded_items, embeddings):
            if embedding is None:
                error_count += 1
                continue
            results[label] = embedding
            computed_count += 1
            cache_updates[source_key] = {
                "model_name": expected_model,
                "device": device,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_key": source_key,
                "image_sha256": _image_sha256_bgr(img_bgr),
                "embedding": [float(x) for x in embedding],
            }

        if cache_updates:
            pending_cache_updates.update(cache_updates)

        logger.info(
            "Embedding visual semantico remoto em lote: batch=%s-%s/%s computed=%s device=%s elapsed=%.2fs",
            offset + 1,
            min(offset + len(chunk), len(unique_items)),
            len(unique_items),
            len(cache_updates),
            last_device,
            time.perf_counter() - batch_started,
        )

    if pending_cache_updates:
        cache_save_started = time.perf_counter()
        _save_cache_updates(pending_cache_updates)
        _STATS["computed"] += len(pending_cache_updates)
        logger.info(
            "Cache CLIP salvo apos lote: updates=%s elapsed=%.2fs",
            len(pending_cache_updates),
            time.perf_counter() - cache_save_started,
        )

    if error_count:
        _STATS["errors"] += error_count
    logger.info(
        "CLIP profile batch concluido: total_photos=%s cache_hits=%s misses=%s computed=%s errors=%s mode=%s batch_size=%s device=%s elapsed=%.2fs stats=%s",
        len(items),
        hit_count,
        miss_count,
        computed_count,
        error_count,
        "worker" if use_worker else "in_process",
        bs,
        last_device,
        time.perf_counter() - started_total,
        cache_stats(),
    )
    if hit_count or computed_count:
        _stat_add("profile_batches_with_clip")
    if _release_after_batch(cfg):
        release_model("remote_batch_done")
    return results


def aggregate_embeddings(embeddings: list[np.ndarray | None]) -> tuple[np.ndarray | None, dict]:
    valid = []
    for emb in embeddings:
        if emb is None:
            continue
        arr = np.array(emb, dtype=float).reshape(-1)
        if arr.size < 16 or np.any(np.isnan(arr)):
            continue
        valid.append(arr)

    if not valid:
        return None, {
            "photo_semantic_embedding_saved": 0.0,
            "photo_carousel_useful_count": 0.0,
            "photo_carousel_duplicate_score": 0.0,
            "photo_carousel_visual_diversity": 0.0,
        }

    mean_emb = np.mean(valid, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    duplicate_score = 0.0
    diversity = 0.0
    if len(valid) >= 2:
        sims = []
        for i in range(len(valid)):
            a = valid[i]
            norm_a = np.linalg.norm(a)
            for j in range(i + 1, len(valid)):
                b = valid[j]
                denom = norm_a * np.linalg.norm(b)
                if denom <= 0:
                    continue
                sims.append(float(np.dot(a, b) / denom))
        if sims:
            duplicate_score = max(0.0, min(1.0, (max(sims) + 1.0) / 2.0))
            diversity = max(0.0, min(1.0, 1.0 - duplicate_score))

    return mean_emb, {
        "photo_semantic_embedding_saved": 1.0,
        "photo_carousel_useful_count": float(len(valid)),
        "photo_carousel_duplicate_score": round(duplicate_score, 4),
        "photo_carousel_visual_diversity": round(diversity, 4),
    }


def load_embeddings_from_cache(config: dict | None = None) -> list[np.ndarray]:
    expected = model_name(config)
    if _cache_backend(config) == "sqlite":
        migrate_legacy_json_to_sqlite(force=False)
        embeddings = []
        with _CACHE_LOCK:
            with _sqlite_connect() as conn:
                rows = conn.execute(
                    """
                    SELECT source_key, model_name, device, created_at, image_sha256, dim, embedding
                    FROM embeddings
                    WHERE model_name = ?
                    """,
                    (expected,),
                ).fetchall()
        for source_key, model, device, created_at, image_sha256, dim, blob in rows:
            entry = _sqlite_row_to_entry(
                str(source_key),
                (model, device, created_at, image_sha256, dim, blob),
            )
            emb = _embedding_from_entry(entry, expected)
            if emb is not None:
                embeddings.append(emb)
        return embeddings

    embeddings = []
    for entry in _load_cache().values():
        emb = _embedding_from_entry(entry, expected)
        if emb is not None:
            embeddings.append(emb)
    return embeddings


def fit_and_save_pca(embeddings: list[np.ndarray], config: dict | None = None) -> object | None:
    n_components = pca_components(config)
    if len(embeddings) < n_components * 2:
        logger.info(
            "PCA semantico visual nao treinado: embeddings=%s minimo=%s",
            len(embeddings),
            n_components * 2,
        )
        return None
    try:
        from sklearn.decomposition import PCA

        X = np.stack([np.array(e, dtype=float) for e in embeddings])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)
        variance = float(pca.explained_variance_ratio_.sum())
        SEMANTIC_PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SEMANTIC_PCA_PATH, "wb") as f:
            pickle.dump(pca, f)
        logger.info(
            "PCA semantico visual treinado: n=%s componentes=%s variancia=%.1f%%",
            len(embeddings),
            n_components,
            variance * 100,
        )
        return pca
    except Exception:
        logger.exception("Falha ao treinar PCA semantico visual")
        return None


def load_pca() -> object | None:
    if not SEMANTIC_PCA_PATH.exists():
        return None
    try:
        with open(SEMANTIC_PCA_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        logger.exception("Falha ao carregar PCA semantico visual: %s", SEMANTIC_PCA_PATH)
        return None


def nan_features() -> dict[str, float]:
    return {name: float("nan") for name in SEMANTIC_EMBEDDING_FEATURE_NAMES}


def apply_to_embedding(embedding: np.ndarray | None, pca: object | None) -> dict[str, float]:
    features = nan_features()
    if pca is None or embedding is None:
        return features
    try:
        arr = np.array(embedding, dtype=float).reshape(1, -1)
        if arr.shape[1] < 16 or np.any(np.isnan(arr)):
            return features
        pcs = pca.transform(arr)[0]
        for idx, value in enumerate(pcs[: len(SEMANTIC_EMBEDDING_FEATURE_NAMES)]):
            features[SEMANTIC_EMBEDDING_FEATURE_NAMES[idx]] = float(value)
        return features
    except Exception:
        logger.debug("Falha ao aplicar PCA semantico visual", exc_info=True)
        return features


def enrich_photo_features(photo_features: dict, pca: object | None) -> None:
    embedding = photo_features.get("_semantic_embedding")
    photo_features.update(apply_to_embedding(embedding, pca))
    photo_features["photo_semantic_embedding_saved"] = 1.0 if embedding is not None else float(
        photo_features.get("photo_semantic_embedding_saved", 0.0) or 0.0
    )


def jsonable_embedding(embedding: np.ndarray | None) -> list[float] | None:
    if embedding is None:
        return None
    try:
        arr = np.array(embedding, dtype=float).reshape(-1)
        if arr.size < 16 or any(math.isnan(float(x)) for x in arr):
            return None
        return [float(x) for x in arr]
    except Exception:
        return None
