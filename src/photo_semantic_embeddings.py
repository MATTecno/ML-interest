"""Embeddings visuais semanticos por foto usando CLIP/SentenceTransformers.

O objetivo deste modulo e complementar as features leves/DeepFace existentes
com um vetor visual mais rico. O vetor bruto fica em cache separado e apenas as
componentes PCA entram no CSV/modelo principal.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import queue
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

_CACHE_LOCK = threading.RLock()
_CACHE_DATA: dict | None = None
_MODEL_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()
_MODEL = None
_MODEL_NAME = ""
_MODEL_DEVICE = ""
_MODEL_LOAD_FAILED = False
_STATS = {"hits": 0, "misses": 0, "computed": 0, "errors": 0}


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


def _cache_path() -> Path:
    raw = str(_semantic_cfg().get("cache_path") or "data/photo_semantic_cache.json")
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


def _load_cache() -> dict:
    global _CACHE_DATA
    with _CACHE_LOCK:
        if _CACHE_DATA is not None:
            return _CACHE_DATA
        path = _cache_path()
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
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
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
    cache = _load_cache()
    with _CACHE_LOCK:
        entry = cache.get(source_key)
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
    with _CACHE_LOCK:
        cache = _load_cache()
        cache[source_key] = entry
        _save_cache(cache)
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
            with _CACHE_LOCK:
                cache = _load_cache()
                cache.update(cache_updates)
                _save_cache(cache)
            _STATS["computed"] += len(cache_updates)

        logger.info(
            "Embedding visual semantico em lote: batch=%s-%s/%s computed=%s device=%s elapsed=%.2fs",
            offset + 1,
            min(offset + len(chunk), len(unique_items)),
            len(unique_items),
            len(cache_updates),
            last_device,
            time.perf_counter() - batch_started,
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
