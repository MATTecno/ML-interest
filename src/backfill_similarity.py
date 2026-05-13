"""
Recalcula photo_face_similarity nos perfis que ficaram com valor 0.5 (neutro).

Isso aconteceu com perfis processados antes do vetor de preferência visual
ter dados suficientes. Agora que há 1000+ faces curtidas, recalculamos.

Uso:
  python3 src/backfill_similarity.py
  python3 src/backfill_similarity.py --no-retrain
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import model as mdl
from logging_config import get_logger, setup_logging
from photo_features import _analyze_bgr, _load_path_as_bgr


PHOTOS_DIR = Path(__file__).parent.parent / "data" / "photos"
PROFILES_PATH = Path(__file__).parent.parent / "data" / "profiles.csv"
logger = get_logger(__name__)

NEUTRAL_SIM = 0.5


def _normalize_name(text: str) -> str:
    base = unicodedata.normalize("NFD", (text or "").strip().lower())
    base = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
    base = base.replace("_", " ")
    return re.sub(r"\s+", " ", base).strip()


def _parse_photo_filename(path: Path) -> tuple[str, int] | None:
    match = re.match(r"^[^_]+_(.+)_(\d+)(?:_(?:face|body))?\.jpg$", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return _normalize_name(match.group(1)), int(match.group(2))


def _has_neutral_sim(row: dict) -> bool:
    saved = str(row.get("photo_features_saved", "")).strip()
    if saved not in {"1", "1.0", "true", "True"}:
        return False
    try:
        sim = float(row.get("photo_face_similarity", "") or "")
        return sim == NEUTRAL_SIM
    except (ValueError, TypeError):
        return False


def _load_rows() -> list[dict]:
    mdl._ensure_profiles_schema()
    if not PROFILES_PATH.exists():
        return []
    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(rows: list[dict]) -> None:
    with open(PROFILES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mdl.CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _target_index_map(rows: list[dict]) -> dict[tuple[str, int, int], list[int]]:
    """Mapeia (nome_norm, idade, label) → índices de linhas com sim=0.5."""
    mapping: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if str(row.get("source", "real")).strip().lower() != "real":
            continue
        if not _has_neutral_sim(row):
            continue
        try:
            age = int(float(row.get("age", 0) or 0))
            label = int(row.get("label", 0))
        except Exception:
            continue
        key = (_normalize_name(row.get("name", "")), age, label)
        mapping[key].append(idx)
    return mapping


def backfill_similarity(retrain: bool = True) -> dict:
    rows = _load_rows()
    if not rows:
        print("  Nenhuma linha em profiles.csv.")
        return {"updated": 0, "skipped": 0}

    targets = _target_index_map(rows)
    total_targets = sum(len(v) for v in targets.values())
    print(f"  Perfis com similaridade neutra (0.5): {total_targets}")

    updated = 0
    skipped = 0

    for subdir, label in (("liked", 1), ("disliked", 0)):
        for photo_path in sorted((PHOTOS_DIR / subdir).glob("*.jpg")):
            parsed = _parse_photo_filename(photo_path)
            if not parsed:
                continue

            norm_name, age = parsed
            key = (norm_name, age, label)
            indices = targets.get(key)
            if not indices:
                continue

            target_idx = indices[0]

            print(f"  [{updated+skipped+1}/{total_targets}] {photo_path.name} ...", end=" ", flush=True)
            img_bgr = _load_path_as_bgr(photo_path)
            feat = _analyze_bgr(img_bgr, include_embedding=True)
            new_sim = feat.get("photo_face_similarity", NEUTRAL_SIM)

            if new_sim == NEUTRAL_SIM:
                print(f"continuou 0.5 (sem rosto ou embedding indisponível)")
                skipped += 1
                indices.pop(0)
                continue

            rows[target_idx]["photo_face_similarity"] = round(float(new_sim), 6)
            print(f"atualizado: {new_sim:.4f}")
            updated += 1
            indices.pop(0)

    if updated:
        _write_rows(rows)
        print(f"\n  CSV atualizado: {updated} perfis com nova similaridade.")

    if retrain and updated:
        print("\n  Retreinando modelo com as novas features...")
        mdl.train_model()
        print("  Modelo atualizado.")

    return {"updated": updated, "skipped": skipped}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-retrain", action="store_true")
    args = parser.parse_args()

    print("\n  Tinder-IA — Backfill de similaridade visual")
    print("  " + "-" * 44)
    result = backfill_similarity(retrain=not args.no_retrain)
    print()
    print(f"  Atualizados : {result['updated']}")
    print(f"  Pulados     : {result['skipped']}  (foto sem rosto detectável)")
    print()


if __name__ == "__main__":
    main()
