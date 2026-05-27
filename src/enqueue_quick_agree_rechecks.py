"""Enqueue old plain quick-agree rows for manual re-review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dataset import PROFILES_PATH, profile_row_signature
from review_queue import (
    REVIEW_FIELDNAMES,
    REVIEW_PATH,
    _history_review_row,
    _training_photo_path,
)


STRUCTURED_KEYS = {
    "photo_score_adjustment",
    "photo_score_reason",
    "photo_reason",
    "photo_positive_details",
    "photo_negative_details",
    "body_frame_correction",
    "body_build_correction",
    "descriptor_detail",
    "descriptor_positive_details",
    "descriptor_negative_details",
    "descriptor_not_positive",
    "descriptor_not_negative",
    "selected_interests",
    "interest_not_positive",
    "interest_not_negative",
    "bio_detail",
    "bio_not_positive",
    "bio_not_negative",
}


def _details(raw: str) -> dict:
    try:
        value = json.loads(str(raw or "{}"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _has_value(value) -> bool:
    return value not in ("", None, [], {})


def _safe_label(row: dict) -> int | None:
    try:
        value = int(float(row.get("label", "")))
    except Exception:
        return None
    return value if value in {0, 1} else None


def _is_plain_quick_agree(row: dict) -> bool:
    if str(row.get("source", "") or "").strip().lower() != "real":
        return False
    if str(row.get("feedback_domain", "") or "").strip().lower() != "other":
        return False
    if _safe_label(row) is None:
        return False
    details = _details(row.get("feedback_details", ""))
    if not details.get("quick_agree"):
        return False
    return not any(_has_value(details.get(key)) for key in STRUCTURED_KEYS)


def _read_review_rows() -> list[dict]:
    if not REVIEW_PATH.exists():
        return []
    with REVIEW_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _existing_recheck_targets(rows: list[dict]) -> set[tuple[str, str]]:
    targets: set[tuple[str, str]] = set()
    for row in rows:
        mode = str(row.get("review_mode") or "").strip().lower()
        if mode != "quick_agree_recheck":
            continue
        target_index = str(row.get("target_profile_index") or "").strip()
        target_signature = str(row.get("target_profile_signature") or "").strip()
        if target_index or target_signature:
            targets.add((target_index, target_signature))
    return targets


def enqueue(limit: int = 0, apply: bool = False, include_no_photo: bool = False) -> dict:
    if not PROFILES_PATH.exists():
        return {"enqueued": 0, "scanned": 0, "eligible": 0, "skipped_existing": 0, "skipped_no_photo": 0}

    with PROFILES_PATH.open(encoding="utf-8", newline="") as f:
        profile_rows = list(csv.DictReader(f))

    existing_targets = _existing_recheck_targets(_read_review_rows())
    selected: list[dict] = []
    scanned = eligible = skipped_existing = skipped_no_photo = 0

    for idx, row in enumerate(profile_rows):
        scanned += 1
        if not _is_plain_quick_agree(row):
            continue
        eligible += 1
        signature = profile_row_signature(row)
        target = (str(idx), signature)
        if target in existing_targets:
            skipped_existing += 1
            continue
        label = _safe_label(row)
        photo_path = _training_photo_path(row, label or 0)
        if not photo_path and not include_no_photo:
            skipped_no_photo += 1
            continue
        review_row = _history_review_row(row, idx, signature, photo_path, 5.0)
        review_row["review_mode"] = "quick_agree_recheck"
        review_row["source"] = "history_review"
        review_row["feedback_reason"] = ""
        review_row["review_priority"] = max(float(review_row.get("review_priority") or 0.0), 0.92)
        selected.append(review_row)
        existing_targets.add(target)
        if limit > 0 and len(selected) >= limit:
            break

    if apply and selected:
        REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = REVIEW_PATH.exists()
        with REVIEW_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            for row in selected:
                writer.writerow({key: row.get(key, "") for key in REVIEW_FIELDNAMES})

    return {
        "applied": apply,
        "enqueued": len(selected) if apply else 0,
        "would_enqueue": len(selected),
        "scanned": scanned,
        "eligible": eligible,
        "skipped_existing": skipped_existing,
        "skipped_no_photo": skipped_no_photo,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="adiciona os cards no review_queue.csv")
    parser.add_argument("--include-no-photo", action="store_true", help="inclui cards sem foto local, usando apenas atributos salvos")
    parser.add_argument("--limit", type=int, default=0, help="limite de cards; 0 = todos")
    args = parser.parse_args()
    print(json.dumps(
        enqueue(limit=args.limit, apply=args.apply, include_no_photo=args.include_no_photo),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
