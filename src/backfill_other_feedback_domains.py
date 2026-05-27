"""Backfill conservative feedback domains for old "other" rows.

Only rows with explicit structured clues are changed. Plain quick agrees remain
"other" because the original reason is ambiguous.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from dataset import CSV_FIELDNAMES, PROFILES_PATH
from logging_config import get_logger
from review_queue import REVIEW_FIELDNAMES, REVIEW_PATH

logger = get_logger(__name__)


def _details(raw: str) -> dict:
    try:
        value = json.loads(str(raw or "{}"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _has_value(value) -> bool:
    return value not in ("", None, [], {})


def _add(domains: list[str], domain: str) -> None:
    if domain and domain not in domains:
        domains.append(domain)


def infer_domains(row: dict) -> list[str]:
    details = _details(row.get("feedback_details", ""))
    reason = str(row.get("feedback_reason") or "").lower()
    domains: list[str] = []

    if any(_has_value(details.get(key)) for key in (
        "photo_score_adjustment",
        "photo_score_reason",
        "photo_reason",
        "photo_positive_details",
        "photo_negative_details",
        "body_frame_correction",
        "body_build_correction",
    )) or "score_foto_" in reason:
        _add(domains, "photo")

    if any(_has_value(details.get(key)) for key in (
        "descriptor_detail",
        "descriptor_positive_details",
        "descriptor_negative_details",
        "descriptor_not_positive",
        "descriptor_not_negative",
    )):
        _add(domains, "descriptors")

    if any(_has_value(details.get(key)) for key in (
        "selected_interests",
        "interest_not_positive",
        "interest_not_negative",
    )):
        _add(domains, "interests")

    if any(_has_value(details.get(key)) for key in (
        "bio_detail",
        "bio_not_positive",
        "bio_not_negative",
    )):
        _add(domains, "bio")

    return domains


def _generic_reason(reason: str) -> bool:
    clean = str(reason or "").strip()
    return clean in {"", "other", "concordou_com_ia", "concordou_com_super_like_ia", "sem certeza"}


def _reason_suffix(details: dict, primary: str) -> str:
    if primary == "photo":
        if details.get("body_build_correction"):
            return f"correcao_corpo: {details.get('body_build_correction')}"
        if details.get("body_frame_correction"):
            return f"correcao_enquadramento: {details.get('body_frame_correction')}"
        if details.get("photo_score_adjustment"):
            direction = "subir" if details.get("photo_score_adjustment") == "higher" else "baixar"
            return f"score_foto_{direction}: {details.get('photo_score_reason') or 'photo_general'}"
        return "sinal_foto"
    if primary == "descriptors":
        return "sinal_descritores"
    if primary == "interests":
        return "sinal_interesses"
    if primary == "bio":
        return "sinal_bio"
    return ""


def update_row(row: dict) -> tuple[bool, str]:
    current = str(row.get("feedback_domain") or "").strip().lower()
    if current != "other":
        return False, ""

    domains = infer_domains(row)
    if not domains:
        return False, ""

    primary = domains[0]
    secondary = domains[1:]
    details = _details(row.get("feedback_details", ""))
    details["primary_domain"] = primary
    details["selected_domains"] = domains
    if secondary:
        details["secondary_domains"] = secondary
    else:
        details.pop("secondary_domains", None)

    row["feedback_domain"] = primary
    row["feedback_secondary"] = ",".join(secondary)
    reason = str(row.get("feedback_reason") or "").strip()
    suffix = _reason_suffix(details, primary)
    if suffix and _generic_reason(reason):
        row["feedback_reason"] = f"{reason or 'concordou_com_ia'}; {suffix}"
    row["feedback_details"] = json.dumps(details, ensure_ascii=False, sort_keys=True)
    return True, primary


def _read_rows(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def backfill_file(path: Path, fieldnames: list[str], apply: bool) -> dict:
    rows, existing_fieldnames = _read_rows(path)
    if not rows:
        return {"path": str(path), "rows": 0, "updated": 0, "by_domain": {}}

    by_domain: dict[str, int] = {}
    updated = 0
    for row in rows:
        changed, domain = update_row(row)
        if not changed:
            continue
        updated += 1
        by_domain[domain] = by_domain.get(domain, 0) + 1

    if apply and updated:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak_other_domain_{stamp}")
        shutil.copy2(path, backup)
        _write_rows(path, rows, fieldnames or existing_fieldnames)
        logger.info("Backfill other domains aplicado: path=%s updated=%s backup=%s", path, updated, backup)
    return {"path": str(path), "rows": len(rows), "updated": updated, "by_domain": by_domain}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="aplica as mudanças; sem isso roda dry-run")
    parser.add_argument("--profiles-only", action="store_true", help="atualiza apenas profiles.csv")
    args = parser.parse_args()

    results = [backfill_file(PROFILES_PATH, CSV_FIELDNAMES, args.apply)]
    if not args.profiles_only:
        results.append(backfill_file(REVIEW_PATH, REVIEW_FIELDNAMES, args.apply))

    print(json.dumps({"applied": args.apply, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
