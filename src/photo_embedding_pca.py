"""
PCA dos embeddings faciais Facenet512 para features de treino/inferência.

O DeepFace já computa embeddings 512-dim por foto; este módulo comprime em
N_COMPONENTS componentes principais e os exporta como photo_emb_pc_01..NN.
Custo extra em swipe: zero — o embedding já existe no cache.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from config import ROOT_DIR
from logging_config import get_logger

logger = get_logger(__name__)

N_COMPONENTS = 16
PCA_PATH = ROOT_DIR / "data" / "photo_embedding_pca.pkl"
EMBEDDING_FEATURE_NAMES = [f"photo_emb_pc_{i:02d}" for i in range(1, N_COMPONENTS + 1)]

_NAN_FEATURES: dict[str, float] = {name: float("nan") for name in EMBEDDING_FEATURE_NAMES}

_CACHE_PATH = ROOT_DIR / "data" / "photo_feature_cache.json"


def load_embeddings_from_cache() -> list[np.ndarray]:
    """Retorna todos os embeddings válidos do cache de features de foto."""
    if not _CACHE_PATH.exists():
        return []
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        logger.exception("Falha ao ler photo_feature_cache.json para PCA")
        return []

    embeddings = []
    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        emb = entry.get("_embedding")
        if emb is None:
            continue
        try:
            arr = np.array(emb, dtype=float)
            if arr.shape == (512,) and not np.any(np.isnan(arr)):
                embeddings.append(arr)
        except Exception:
            pass
    return embeddings


def fit_and_save_pca(embeddings: list[np.ndarray]) -> object | None:
    """Treina PCA nos embeddings e salva em PCA_PATH. Retorna PCA ou None."""
    if len(embeddings) < N_COMPONENTS * 2:
        logger.info(
            "PCA não treinado: embeddings disponíveis=%s mínimo=%s",
            len(embeddings),
            N_COMPONENTS * 2,
        )
        return None

    from sklearn.decomposition import PCA

    X = np.stack(embeddings)
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(X)
    variance = float(pca.explained_variance_ratio_.sum())

    PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)

    logger.info(
        "PCA de embeddings treinado: n=%s componentes=%s variância=%.1f%%",
        len(embeddings),
        N_COMPONENTS,
        variance * 100,
    )
    return pca


def load_pca() -> object | None:
    """Carrega PCA salvo. Retorna None se não existir."""
    if not PCA_PATH.exists():
        return None
    try:
        with open(PCA_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        logger.exception("Falha ao carregar PCA: %s", PCA_PATH)
        return None


def apply_to_embedding(embedding: np.ndarray | None, pca: object | None) -> dict:
    """Aplica PCA a um embedding 512-dim. Retorna NaN dict se pca/embedding inválido."""
    if pca is None or embedding is None:
        return dict(_NAN_FEATURES)
    try:
        arr = np.array(embedding, dtype=float).reshape(1, -1)
        if arr.shape[1] != 512 or np.any(np.isnan(arr)):
            return dict(_NAN_FEATURES)
        pcs = pca.transform(arr)[0]
        return {name: float(pcs[i]) for i, name in enumerate(EMBEDDING_FEATURE_NAMES)}
    except Exception:
        logger.debug("Falha ao aplicar PCA ao embedding", exc_info=True)
        return dict(_NAN_FEATURES)


def enrich_photo_features(photo_features: dict, pca: object | None) -> None:
    """Adiciona photo_emb_pc_* ao dict photo_features a partir de _embedding."""
    pc_feats = apply_to_embedding(photo_features.get("_embedding"), pca)
    photo_features.update(pc_feats)


def get_embedding_from_cache_by_url(photo_url: str) -> np.ndarray | None:
    """Busca embedding no cache pela URL (usa mesmo _cache_key de photo_features.py)."""
    if not photo_url or not _CACHE_PATH.exists():
        return None
    try:
        cache_key = urlsplit(photo_url).path or photo_url
        with open(_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        entry = cache.get(cache_key)
        if not isinstance(entry, dict):
            return None
        emb = entry.get("_embedding")
        if emb is None:
            return None
        arr = np.array(emb, dtype=float)
        return arr if arr.shape == (512,) else None
    except Exception:
        logger.debug("Falha ao buscar embedding do cache url=%s", photo_url, exc_info=True)
        return None
