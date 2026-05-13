"""
Backfill de features de embedding PCA para perfis antigos em profiles.csv.

Estratégia:
  - Para cada perfil em profiles.csv sem photo_emb_pc_* preenchido,
    procura a foto de rosto salva em data/photos/{liked,disliked}/
  - Roda DeepFace Facenet512 na foto local (sem precisar de URL ou download)
  - Aplica PCA e atualiza profiles.csv

Uso:
    python backfill_embeddings.py          # processa tudo
    python backfill_embeddings.py --dry    # mostra o que faria, sem salvar
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import ROOT_DIR
from dataset import CSV_FIELDNAMES, EMBEDDING_FEATURE_NAMES
from features import PHOTO_FEATURE_NAMES
from logging_config import get_logger
from photo_embedding_pca import load_pca, apply_to_embedding, N_COMPONENTS
from photo_storage import PHOTOS_DIR, safe_filename

logger = get_logger(__name__)

PROFILES_PATH = ROOT_DIR / "data" / "profiles.csv"
_DEEPFACE_LOCK = threading.Lock()
_BACKEND = "opencv"


def _normalize_for_match(text: str) -> str:
    """Normalização para comparar nome do arquivo com nome do perfil."""
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()[:20]


def _parse_photo_stem(stem: str) -> tuple[str, str] | None:
    """
    Extrai (safe_name, age) do stem do arquivo.
    Formato: {8_char_id}_{safe_name}_{age}
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    # first part = 8-char ID, last part = age, middle = name
    age = parts[-1]
    if not age.isdigit():
        return None
    safe_name = "_".join(parts[1:-1])
    return safe_name.lower(), age


def _build_photo_index() -> dict[tuple[str, str], Path]:
    """
    Mapeia (safe_name_lower, age_str) → caminho da foto de rosto.
    Se há duplicatas (mesmo nome+age), usa a mais recente.
    """
    index: dict[tuple[str, str], Path] = {}
    for folder in ("liked", "disliked"):
        folder_path = PHOTOS_DIR / folder
        if not folder_path.exists():
            continue
        for photo in folder_path.glob("*.jpg"):
            if "_body" in photo.name:
                continue
            parsed = _parse_photo_stem(photo.stem)
            if parsed is None:
                continue
            key = parsed
            existing = index.get(key)
            if existing is None or photo.stat().st_mtime > existing.stat().st_mtime:
                index[key] = photo
    return index


def _profile_key(row: dict) -> tuple[str, str]:
    name = safe_filename(str(row.get("name", "") or "")).lower()
    age = str(row.get("age", "") or "").strip()
    return name, age


def _embedding_from_file(photo_path: Path) -> np.ndarray | None:
    """Extrai embedding Facenet512 de um arquivo de imagem local."""
    try:
        import cv2
        img_bgr = cv2.imread(str(photo_path))
        if img_bgr is None:
            return None

        from deepface import DeepFace
        with _DEEPFACE_LOCK:
            result = DeepFace.represent(
                img_bgr,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend=_BACKEND,
            )
        if result:
            arr = np.array(result[0]["embedding"], dtype=float)
            if arr.shape == (512,) and not np.any(np.isnan(arr)):
                return arr
    except Exception:
        logger.debug("Falha ao extrair embedding de %s", photo_path, exc_info=True)
    return None


def _needs_backfill(row: dict) -> bool:
    """Retorna True se o perfil tem foto salva mas não tem features de embedding."""
    saved = str(row.get("photo_features_saved", "")).strip()
    if saved not in ("1", "1.0"):
        return False
    return any(str(row.get(feat, "")).strip() == "" for feat in EMBEDDING_FEATURE_NAMES)


def run_backfill(dry: bool = False, max_workers: int = 1) -> dict:
    pca = load_pca()
    if pca is None:
        print("PCA não encontrado. Retreine o modelo primeiro (python -c 'from model_training import train_model; train_model()').")
        return {"skipped": 0, "updated": 0, "failed": 0}

    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    targets = [(i, row) for i, row in enumerate(rows) if _needs_backfill(row)]
    print(f"Perfis sem embedding: {len(targets)} de {len(rows)} total")

    if not targets:
        print("Nenhum perfil precisa de backfill.")
        return {"skipped": 0, "updated": 0, "failed": 0}

    photo_index = _build_photo_index()
    print(f"Fotos indexadas em disco: {len(photo_index)}")

    matchable = sum(1 for _, row in targets if _profile_key(row) in photo_index)
    print(f"Perfis com foto correspondente no disco: {matchable}")

    if dry:
        print("[--dry] Nenhuma alteração será feita.")
        return {"skipped": len(targets) - matchable, "updated": 0, "failed": 0, "matchable": matchable}

    updated = 0
    failed = 0
    skipped = 0
    t0 = time.perf_counter()

    def _process(item):
        i, row = item
        key = _profile_key(row)
        photo_path = photo_index.get(key)
        if photo_path is None:
            return i, None, "no_photo"
        embedding = _embedding_from_file(photo_path)
        if embedding is None:
            return i, None, "no_embedding"
        pc_feats = apply_to_embedding(embedding, pca)
        return i, pc_feats, "ok"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, item): item for item in targets}
        done = 0
        for future in as_completed(futures):
            done += 1
            i, pc_feats, status = future.result()
            if status == "ok" and pc_feats:
                for feat, val in pc_feats.items():
                    import math
                    rows[i][feat] = "" if (isinstance(val, float) and math.isnan(val)) else round(val, 6)
                updated += 1
            elif status == "no_photo":
                skipped += 1
            else:
                failed += 1

            if done % 50 == 0 or done == len(targets):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed
                remaining = (len(targets) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(targets)} — atualizados={updated} sem_foto={skipped} falhas={failed} "
                      f"({rate:.1f}/s, ~{remaining:.0f}s restantes)")

    # Reescreve profiles.csv com os novos valores
    tmp = PROFILES_PATH.with_suffix(f".tmp.{int(time.time() * 1000)}")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})
    tmp.replace(PROFILES_PATH)

    elapsed = time.perf_counter() - t0
    print(f"\nBackfill concluído em {elapsed:.1f}s: atualizados={updated} sem_foto={skipped} falhas={failed}")
    return {"updated": updated, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill de embeddings PCA em profiles.csv")
    parser.add_argument("--dry", action="store_true", help="Mostra o que faria sem salvar")
    parser.add_argument("--workers", type=int, default=1, help="Threads paralelas (padrão: 1, máx recomendado: 2)")
    args = parser.parse_args()
    run_backfill(dry=args.dry, max_workers=args.workers)
