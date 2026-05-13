"""
Embeddings semânticos de bio e interesses via sentence-transformers + PCA.

Modelo: paraphrase-multilingual-MiniLM-L12-v2 (~90 MB, multilingual, CPU-friendly)
Saída: 8 componentes PCA de bio + 4 de interesses = 12 features TEXT_EMB_FEATURE_NAMES.

O modelo é carregado uma única vez (singleton). Durante o swipe, `text_to_features`
recebe bio e lista de interesses e retorna as features em ~5-20 ms.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from config import ROOT_DIR
from logging_config import get_logger

logger = get_logger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BIO_PCA_PATH = ROOT_DIR / "data" / "bio_embedding_pca.pkl"

N_BIO_COMPONENTS = 8
N_INT_COMPONENTS = 4
EMBEDDING_DIM = 384

BIO_EMB_FEATURE_NAMES = [f"bio_emb_pc_{i:02d}" for i in range(1, N_BIO_COMPONENTS + 1)]
INT_EMB_FEATURE_NAMES = [f"interest_emb_pc_{i:02d}" for i in range(1, N_INT_COMPONENTS + 1)]
TEXT_EMB_FEATURE_NAMES = BIO_EMB_FEATURE_NAMES + INT_EMB_FEATURE_NAMES

_NAN_FEATURES: dict[str, float] = {name: float("nan") for name in TEXT_EMB_FEATURE_NAMES}

_model = None


def get_model():
    """Carrega o modelo sentence-transformers (singleton, lazy)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL_NAME)
            logger.info("sentence-transformers carregado: model=%s", MODEL_NAME)
        except Exception:
            logger.exception("Falha ao carregar sentence-transformers model=%s", MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray | None:
    """Retorna embedding 384-dim normalizado, ou None em caso de falha."""
    if not text or not text.strip():
        return None
    model = get_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        arr = np.array(vec, dtype=float)
        if arr.shape == (EMBEDDING_DIM,) and np.isfinite(arr).all():
            return arr
        return None
    except Exception:
        logger.debug("Falha ao embedar texto", exc_info=True)
        return None


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

    model = get_model()
    if model is None:
        return [dict(nan_row) for _ in range(n)]

    int_texts = []
    for interests in interests_list:
        if isinstance(interests, list):
            int_texts.append(" ".join(str(i) for i in interests if i))
        else:
            int_texts.append(str(interests or ""))

    bio_strs = [b or "" for b in bios]

    try:
        bio_vecs = model.encode(bio_strs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        int_vecs = model.encode(int_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    except Exception:
        logger.exception("Falha ao encodar embeddings em lote (batch_text_to_features)")
        return [dict(nan_row) for _ in range(n)]

    results = []
    for i in range(n):
        result: dict[str, float] = {}
        bio_vec = np.array(bio_vecs[i], dtype=float)
        int_vec = np.array(int_vecs[i], dtype=float)

        if bio_strs[i].strip() and np.isfinite(bio_vec).all():
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

        if int_texts[i].strip() and np.isfinite(int_vec).all():
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

    model = get_model()
    if model is None:
        return [], []

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

    try:
        bio_vecs = model.encode(bios, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        int_vecs = model.encode(int_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    except Exception:
        logger.exception("Falha ao encodar embeddings em lote")
        return [], []

    bio_embeddings = [
        np.array(v, dtype=float)
        for bio, v in zip(bios, bio_vecs)
        if bio and np.isfinite(v).all()
    ]
    int_embeddings = [
        np.array(v, dtype=float)
        for int_text, v in zip(int_texts, int_vecs)
        if int_text and np.isfinite(v).all()
    ]

    return bio_embeddings, int_embeddings
