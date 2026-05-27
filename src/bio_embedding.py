"""
Embeddings semânticos de bio e interesses via sentence-transformers + PCA.

Modelo: paraphrase-multilingual-MiniLM-L12-v2 (~90 MB, multilingual, CPU-friendly)
Saída: 8 componentes PCA de bio + 4 de interesses = 12 features TEXT_EMB_FEATURE_NAMES.

O modelo é carregado uma única vez (singleton). Durante o swipe, `text_to_features`
recebe bio e lista de interesses e retorna as features em ~5-20 ms.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import ROOT_DIR
from logging_config import get_logger

logger = get_logger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BIO_PCA_PATH = ROOT_DIR / "data" / "bio_embedding_pca.pkl"
TEXT_EMBEDDING_CACHE_PATH = ROOT_DIR / "data" / "bio_embedding_cache.json"

N_BIO_COMPONENTS = 8
N_INT_COMPONENTS = 4
EMBEDDING_DIM = 384

BIO_EMB_FEATURE_NAMES = [f"bio_emb_pc_{i:02d}" for i in range(1, N_BIO_COMPONENTS + 1)]
INT_EMB_FEATURE_NAMES = [f"interest_emb_pc_{i:02d}" for i in range(1, N_INT_COMPONENTS + 1)]
TEXT_EMB_FEATURE_NAMES = BIO_EMB_FEATURE_NAMES + INT_EMB_FEATURE_NAMES

_NAN_FEATURES: dict[str, float] = {name: float("nan") for name in TEXT_EMB_FEATURE_NAMES}

_model = None
_MODEL_LOCK = threading.Lock()
_CACHE_LOCK = threading.RLock()
_CACHE_DATA: dict | None = None
_CACHE_STATS = {"hits": 0, "misses": 0, "computed": 0, "errors": 0}


def _cache_key(text: str) -> str:
    digest = hashlib.sha256((MODEL_NAME + "\0" + str(text or "")).encode("utf-8")).hexdigest()
    return f"{MODEL_NAME}:{digest}"


def _load_cache() -> dict:
    global _CACHE_DATA
    with _CACHE_LOCK:
        if _CACHE_DATA is not None:
            return _CACHE_DATA
        if not TEXT_EMBEDDING_CACHE_PATH.exists():
            _CACHE_DATA = {}
            return _CACHE_DATA
        try:
            with open(TEXT_EMBEDDING_CACHE_PATH, encoding="utf-8") as f:
                _CACHE_DATA = json.load(f)
        except Exception:
            logger.exception("Falha ao carregar cache de embeddings de texto: %s", TEXT_EMBEDDING_CACHE_PATH)
            _CACHE_DATA = {}
        return _CACHE_DATA


def _save_cache(cache: dict) -> None:
    TEXT_EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TEXT_EMBEDDING_CACHE_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        tmp.replace(TEXT_EMBEDDING_CACHE_PATH)
    except Exception:
        logger.exception("Falha ao salvar cache de embeddings de texto: %s", TEXT_EMBEDDING_CACHE_PATH)


def _embedding_from_entry(entry: dict | None) -> np.ndarray | None:
    if not isinstance(entry, dict) or entry.get("model_name") != MODEL_NAME:
        return None
    try:
        arr = np.array(entry.get("embedding"), dtype=float)
        if arr.shape == (EMBEDDING_DIM,) and np.isfinite(arr).all():
            return arr
    except Exception:
        pass
    return None


def cache_stats(reset: bool = False) -> dict:
    with _CACHE_LOCK:
        stats = dict(_CACHE_STATS)
        if reset:
            for key in _CACHE_STATS:
                _CACHE_STATS[key] = 0
        return stats


def get_model(local_only: bool = False):
    """Carrega o modelo sentence-transformers (singleton, lazy)."""
    global _model
    if _model is not None:
        return _model

    with _MODEL_LOCK:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {"local_files_only": True} if local_only else {}
            _model = SentenceTransformer(MODEL_NAME, **kwargs)
            logger.info(
                "sentence-transformers carregado: model=%s local_only=%s",
                MODEL_NAME,
                local_only,
            )
        except TypeError:
            if local_only:
                logger.warning(
                    "sentence-transformers sem suporte a local_files_only; tentando carga padrao: model=%s",
                    MODEL_NAME,
                )
            try:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME)
                logger.info("sentence-transformers carregado: model=%s", MODEL_NAME)
            except Exception:
                if local_only:
                    logger.warning("Falha ao carregar sentence-transformers do cache local: model=%s", MODEL_NAME, exc_info=True)
                else:
                    logger.exception("Falha ao carregar sentence-transformers model=%s", MODEL_NAME)
        except Exception:
            if local_only:
                logger.warning("Falha ao carregar sentence-transformers do cache local: model=%s", MODEL_NAME, exc_info=True)
            else:
                logger.exception("Falha ao carregar sentence-transformers model=%s", MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray | None:
    """Retorna embedding 384-dim normalizado, ou None em caso de falha."""
    if not text or not text.strip():
        return None
    key = _cache_key(text)
    with _CACHE_LOCK:
        cached = _embedding_from_entry(_load_cache().get(key))
    if cached is not None:
        _CACHE_STATS["hits"] += 1
        return cached
    _CACHE_STATS["misses"] += 1

    model = get_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        arr = np.array(vec, dtype=float)
        if arr.shape == (EMBEDDING_DIM,) and np.isfinite(arr).all():
            with _CACHE_LOCK:
                cache = _load_cache()
                cache[key] = {
                    "model_name": MODEL_NAME,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "text_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
                    "embedding": [float(x) for x in arr],
                }
                _save_cache(cache)
            _CACHE_STATS["computed"] += 1
            return arr
        return None
    except Exception:
        _CACHE_STATS["errors"] += 1
        logger.debug("Falha ao embedar texto", exc_info=True)
        return None


def batch_embed_texts(texts: list[str], batch_size: int = 64) -> list[np.ndarray | None]:
    """Embeddings em lote com cache persistente por texto exato."""
    results: list[np.ndarray | None] = [None] * len(texts)
    missing: list[tuple[int, str, str]] = []
    seen_missing: set[str] = set()

    with _CACHE_LOCK:
        cache = _load_cache()
        for idx, raw_text in enumerate(texts):
            text = str(raw_text or "")
            if not text.strip():
                continue
            key = _cache_key(text)
            cached = _embedding_from_entry(cache.get(key))
            if cached is not None:
                results[idx] = cached
                _CACHE_STATS["hits"] += 1
                continue
            _CACHE_STATS["misses"] += 1
            if key not in seen_missing:
                seen_missing.add(key)
                missing.append((idx, text, key))

    if not missing:
        return results

    model = get_model()
    if model is None:
        return results

    texts_to_encode = [item[1] for item in missing]
    try:
        vecs = model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    except Exception:
        _CACHE_STATS["errors"] += len(texts_to_encode)
        logger.exception("Falha ao encodar embeddings de texto em lote")
        return results

    updates: dict[str, dict] = {}
    computed_by_key: dict[str, np.ndarray] = {}
    for (_, text, key), vec in zip(missing, vecs):
        arr = np.array(vec, dtype=float)
        if arr.shape != (EMBEDDING_DIM,) or not np.isfinite(arr).all():
            _CACHE_STATS["errors"] += 1
            continue
        computed_by_key[key] = arr
        updates[key] = {
            "model_name": MODEL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "embedding": [float(x) for x in arr],
        }

    if updates:
        with _CACHE_LOCK:
            cache = _load_cache()
            cache.update(updates)
            _save_cache(cache)
        _CACHE_STATS["computed"] += len(updates)

    for idx, raw_text in enumerate(texts):
        if results[idx] is not None:
            continue
        key = _cache_key(str(raw_text or ""))
        if key in computed_by_key:
            results[idx] = computed_by_key[key]

    return results


def fit_and_save_pca(
    bio_embeddings: list[np.ndarray],
    int_embeddings: list[np.ndarray],
) -> tuple[object, object] | tuple[None, None]:
    """Treina e salva PCAs de bio e interesses. Retorna (bio_pca, int_pca) ou (None, None)."""
    from sklearn.decomposition import PCA

    min_bio = N_BIO_COMPONENTS * 2
    min_int = N_INT_COMPONENTS * 2

    if len(bio_embeddings) < min_bio or len(int_embeddings) < min_int:
        logger.info(
            "PCA de bio não treinado: bio=%s (mín=%s) int=%s (mín=%s)",
            len(bio_embeddings),
            min_bio,
            len(int_embeddings),
            min_int,
        )
        return None, None

    X_bio = np.stack(bio_embeddings)
    bio_pca = PCA(n_components=N_BIO_COMPONENTS, random_state=42)
    bio_pca.fit(X_bio)

    X_int = np.stack(int_embeddings)
    int_pca = PCA(n_components=N_INT_COMPONENTS, random_state=42)
    int_pca.fit(X_int)

    BIO_PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BIO_PCA_PATH, "wb") as f:
        pickle.dump({"bio_pca": bio_pca, "int_pca": int_pca}, f)

    logger.info(
        "PCA de bio treinado: bio_n=%s int_n=%s bio_var=%.1f%% int_var=%.1f%%",
        len(bio_embeddings),
        len(int_embeddings),
        float(bio_pca.explained_variance_ratio_.sum()) * 100,
        float(int_pca.explained_variance_ratio_.sum()) * 100,
    )
    return bio_pca, int_pca


def load_pca() -> tuple[object, object] | None:
    """Carrega PCAs salvos. Retorna (bio_pca, int_pca) ou None se não existir."""
    if not BIO_PCA_PATH.exists():
        return None
    try:
        with open(BIO_PCA_PATH, "rb") as f:
            data = pickle.load(f)
        return data["bio_pca"], data["int_pca"]
    except Exception:
        logger.exception("Falha ao carregar bio PCA: %s", BIO_PCA_PATH)
        return None


def text_to_features(
    bio: str,
    interests: list[str] | str,
    bio_pca: object | None,
    int_pca: object | None,
) -> dict[str, float]:
    """
    Computa 12 features de embedding semântico para bio + interesses.
    Retorna dict com NaN se PCA não treinado ou embedding falhou.
    """
    if bio_pca is None or int_pca is None:
        return dict(_NAN_FEATURES)

    if isinstance(interests, list):
        int_text = " ".join(str(i) for i in interests if i)
    else:
        int_text = str(interests or "")

    bio_emb = embed(bio or "")
    int_emb = embed(int_text)

    result: dict[str, float] = {}

    if bio_emb is not None:
        try:
            bio_pcs = bio_pca.transform(bio_emb.reshape(1, -1))[0]
            for i, name in enumerate(BIO_EMB_FEATURE_NAMES):
                result[name] = float(bio_pcs[i])
        except Exception:
            logger.debug("Falha ao aplicar bio PCA", exc_info=True)
            for name in BIO_EMB_FEATURE_NAMES:
                result[name] = float("nan")
    else:
        for name in BIO_EMB_FEATURE_NAMES:
            result[name] = float("nan")

    if int_emb is not None:
        try:
            int_pcs = int_pca.transform(int_emb.reshape(1, -1))[0]
            for i, name in enumerate(INT_EMB_FEATURE_NAMES):
                result[name] = float(int_pcs[i])
        except Exception:
            logger.debug("Falha ao aplicar interest PCA", exc_info=True)
            for name in INT_EMB_FEATURE_NAMES:
                result[name] = float("nan")
    else:
        for name in INT_EMB_FEATURE_NAMES:
            result[name] = float("nan")

    return result


def batch_text_to_features(
    bios: list[str],
    interests_list: list,
    bio_pca: object | None,
    int_pca: object | None,
) -> list[dict[str, float]]:
    """
    Versão em lote de text_to_features: encoda todos os textos de uma vez (muito mais rápido
    que chamar text_to_features por perfil dentro de um loop durante o treino).
    """
    n = len(bios)
    nan_row = dict(_NAN_FEATURES)

    if bio_pca is None or int_pca is None:
        return [dict(nan_row) for _ in range(n)]

    int_texts = []
    for interests in interests_list:
        if isinstance(interests, list):
            int_texts.append(" ".join(str(i) for i in interests if i))
        else:
            int_texts.append(str(interests or ""))

    bio_strs = [b or "" for b in bios]

    bio_vecs = batch_embed_texts(bio_strs, batch_size=64)
    int_vecs = batch_embed_texts(int_texts, batch_size=64)

    results = []
    for i in range(n):
        result: dict[str, float] = {}
        bio_vec = np.array(bio_vecs[i], dtype=float) if bio_vecs[i] is not None else np.array([])
        int_vec = np.array(int_vecs[i], dtype=float) if int_vecs[i] is not None else np.array([])

        if bio_strs[i].strip() and bio_vec.shape == (EMBEDDING_DIM,) and np.isfinite(bio_vec).all():
            try:
                bio_pcs = bio_pca.transform(bio_vec.reshape(1, -1))[0]
                for j, name in enumerate(BIO_EMB_FEATURE_NAMES):
                    result[name] = float(bio_pcs[j])
            except Exception:
                for name in BIO_EMB_FEATURE_NAMES:
                    result[name] = float("nan")
        else:
            for name in BIO_EMB_FEATURE_NAMES:
                result[name] = float("nan")

        if int_texts[i].strip() and int_vec.shape == (EMBEDDING_DIM,) and np.isfinite(int_vec).all():
            try:
                int_pcs = int_pca.transform(int_vec.reshape(1, -1))[0]
                for j, name in enumerate(INT_EMB_FEATURE_NAMES):
                    result[name] = float(int_pcs[j])
            except Exception:
                for name in INT_EMB_FEATURE_NAMES:
                    result[name] = float("nan")
        else:
            for name in INT_EMB_FEATURE_NAMES:
                result[name] = float("nan")

        results.append(result)

    return results


def collect_embeddings_from_df(df) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extrai embeddings de bio e interesses de um DataFrame em lote (batch).
    Usado em train_model() para ajustar as PCAs.
    """
    import pandas as pd

    bios: list[str] = []
    int_texts: list[str] = []

    for _, row in df.iterrows():
        bio_raw = row.get("bio", "")
        bio = "" if pd.isna(bio_raw) else str(bio_raw).strip()
        bios.append(bio)

        interests_raw = row.get("interests", "")
        if pd.isna(interests_raw) or not interests_raw:
            int_text = ""
        else:
            int_text = " ".join(x.strip() for x in str(interests_raw).split(",") if x.strip())
        int_texts.append(int_text)

    bio_vecs = batch_embed_texts(bios, batch_size=64)
    int_vecs = batch_embed_texts(int_texts, batch_size=64)

    bio_embeddings = [
        np.array(v, dtype=float)
        for bio, v in zip(bios, bio_vecs)
        if bio and v is not None and np.isfinite(v).all()
    ]
    int_embeddings = [
        np.array(v, dtype=float)
        for int_text, v in zip(int_texts, int_vecs)
        if int_text and v is not None and np.isfinite(v).all()
    ]

    return bio_embeddings, int_embeddings
