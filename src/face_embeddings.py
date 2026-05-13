"""
Vetor de preferência visual baseado em embeddings de rosto.

Como funciona:
  - DeepFace transforma cada rosto em um vetor de 512 números (embedding)
  - Este módulo mantém a média dos embeddings das faces curtidas (e rejeitadas)
  - Para um novo perfil, computa similaridade cosseno entre o rosto e o vetor médio
  - Essa similaridade (0–1) vira a feature `photo_face_similarity` no modelo ML

O vetor cresce com o uso: quanto mais você usa o sistema, mais precisa fica a
representação do seu gosto visual.
"""

import pickle
import numpy as np
from pathlib import Path
from logging_config import get_logger

PREF_PATH = Path(__file__).parent.parent / "data" / "face_preference.pkl"
MIN_LIKED_FOR_SIGNAL = 5  # mínimo de faces curtidas para começar a usar o sinal
logger = get_logger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_prefs() -> dict:
    if not PREF_PATH.exists():
        logger.debug("Arquivo de preferencia visual ainda nao existe: %s", PREF_PATH)
        return {
            "liked_centroid": None,
            "liked_count": 0,
            "disliked_centroid": None,
            "disliked_count": 0,
        }
    try:
        with open(PREF_PATH, "rb") as f:
            prefs = pickle.load(f)
        logger.debug(
            "Preferencia visual carregada: liked=%s disliked=%s",
            prefs.get("liked_count", 0),
            prefs.get("disliked_count", 0),
        )
        return prefs
    except Exception:
        logger.exception("Falha ao carregar preferencia visual: %s", PREF_PATH)
        raise


def _save_prefs(prefs: dict) -> None:
    PREF_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(PREF_PATH, "wb") as f:
            pickle.dump(prefs, f)
        logger.debug(
            "Preferencia visual salva: liked=%s disliked=%s",
            prefs.get("liked_count", 0),
            prefs.get("disliked_count", 0),
        )
    except Exception:
        logger.exception("Falha ao salvar preferencia visual: %s", PREF_PATH)
        raise


def compute_similarity(embedding: np.ndarray | None) -> float:
    """
    Computa similaridade do rosto ao vetor de preferência visual.

    Retorna 0.5 (neutro) se ainda não há dados suficientes.
    Retorna valor entre 0 e 1 quando há pelo menos MIN_LIKED_FOR_SIGNAL faces curtidas:
      ~0.9 = muito parecido com rostos curtidos
      ~0.5 = neutro / sem opinião formada
      ~0.1 = bem diferente dos rostos curtidos
    """
    if embedding is None:
        return 0.5

    prefs = load_prefs()
    liked_centroid = prefs.get("liked_centroid")
    liked_count = prefs.get("liked_count", 0)

    if liked_centroid is None or liked_count < MIN_LIKED_FOR_SIGNAL:
        return 0.5

    liked_centroid = np.array(liked_centroid)
    if embedding.shape != liked_centroid.shape:
        logger.warning(
            "Preferencia visual ignorada por dimensao incompatível: embedding=%s centroid=%s",
            embedding.shape,
            liked_centroid.shape,
        )
        return 0.5

    sim = _cosine_sim(embedding, liked_centroid)
    return round((sim + 1.0) / 2.0, 4)  # mapeia -1..1 → 0..1


def update_preference(embedding: np.ndarray | None, decision: str) -> None:
    """
    Atualiza o vetor de preferência com o embedding do rosto avaliado.
    Deve ser chamado apenas para perfis avaliados pelo ML (não por filtros absolutos).
    """
    if embedding is None:
        return

    prefs = load_prefs()

    if decision == "CURTIR":
        centroid_key, count_key = "liked_centroid", "liked_count"
    else:
        centroid_key, count_key = "disliked_centroid", "disliked_count"

    centroid = prefs.get(centroid_key)
    count = prefs.get(count_key, 0)

    if centroid is None:
        new_centroid = embedding.astype(float)
    else:
        # Média online: nova_média = (média_atual × n + novo) / (n + 1)
        new_centroid = (np.array(centroid) * count + embedding) / (count + 1)

    prefs[centroid_key] = new_centroid
    prefs[count_key] = count + 1
    _save_prefs(prefs)
    logger.info("Preferencia visual atualizada: decision=%s count=%s", decision, prefs[count_key])


def stats() -> str:
    """Retorna um resumo do estado atual do vetor de preferência."""
    prefs = load_prefs()
    liked = prefs.get("liked_count", 0)
    disliked = prefs.get("disliked_count", 0)
    ready = liked >= MIN_LIKED_FOR_SIGNAL
    status = "ativo" if ready else f"aguardando ({liked}/{MIN_LIKED_FOR_SIGNAL} curtidas)"
    return f"Preferência visual: {liked} curtidas | {disliked} rejeitadas | {status}"
