"""
Enriquecimento offline de features visuais pesadas/pos-sessao.

Uso:
  python3 src/offline_enrichment.py
  python3 src/offline_enrichment.py --limit 20
  python3 src/offline_enrichment.py --no-retrain
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backfill_photo_features import backfill
from config import ROOT_DIR
from features import PHOTO_FEATURE_NAMES
from logging_config import setup_logging
from photo_features import analyze_local_photo
from review_queue import REVIEW_FIELDNAMES, REVIEW_PATH, _ensure_schema as _ensure_review_schema


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _has_neutral_similarity(row: dict) -> bool:
    """Similaridade 0.5 com rosto salvo costuma ser fallback antigo."""
    similarity = _as_float(row.get("photo_face_similarity", ""), -1.0)
    has_face = _as_float(row.get("photo_has_face", ""), 0.0)
    return similarity == 0.5 and has_face > 0


def _needs_photo_features(row: dict) -> bool:
    saved = str(row.get("photo_features_saved", "")).strip().lower()
    if saved not in {"1", "1.0", "true"}:
        return True
    if any(not str(row.get(col, "")).strip() for col in PHOTO_FEATURE_NAMES):
        return True
    return _has_neutral_similarity(row)


def _write_review_rows(rows: list[dict]) -> None:
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REVIEW_FIELDNAMES})


def enrich_review_queue(limit: int | None = None) -> dict:
    _ensure_review_schema()
    if not REVIEW_PATH.exists():
        return {"updated": 0, "skipped": 0, "photos_seen": 0}

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    updated = 0
    skipped = 0
    photos_seen = 0

    for row in rows:
        if limit is not None and photos_seen >= limit:
            break
        if not _needs_photo_features(row):
            skipped += 1
            continue

        photo_path = (row.get("photo_path") or "").strip()
        if not photo_path:
            skipped += 1
            continue

        path = ROOT_DIR / photo_path
        if not path.exists():
            skipped += 1
            continue

        try:
            age = int(float(row.get("age", 0) or 0))
        except Exception:
            age = 0

        photos_seen += 1
        print(f"  [review] Enriquecendo {path.name} ...")
        feat = analyze_local_photo(path, age)
        for col in PHOTO_FEATURE_NAMES:
            value = feat.get(col, "")
            row[col] = "" if value in ("", None) else round(float(value), 6)
        row["photo_features_saved"] = "1"
        updated += 1

    if updated:
        _write_review_rows(rows)

    return {"updated": updated, "skipped": skipped, "photos_seen": photos_seen}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-retrain", action="store_true")
    args = parser.parse_args()

    print("\n  Tinder-IA - Enriquecimento visual offline")
    print("  " + "-" * 46)
    print("  Atualiza rosto, embedding, sinais corporais e leitura visual leve das fotos salvas.")

    review_result = enrich_review_queue(limit=args.limit)
    remaining = None
    if args.limit is not None:
        remaining = max(0, args.limit - review_result["photos_seen"])

    result = backfill(limit=remaining, retrain=not args.no_retrain)
    print()
    print(f"  Review atualizados : {review_result['updated']}")
    print(f"  Perfis atualizados : {result['updated']}")
    print(f"  Ignorados          : {review_result['skipped'] + result['skipped']}")
    print(f"  Processados        : {review_result['photos_seen'] + result['photos_seen']}")
    print()


if __name__ == "__main__":
    main()
