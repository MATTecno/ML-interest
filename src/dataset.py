"""Leitura, migracao e escrita dos datasets do projeto."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from features import PHOTO_FEATURE_NAMES, EMBEDDING_FEATURE_NAMES, SEMANTIC_EMBEDDING_FEATURE_NAMES
from logging_config import get_logger
from photo_identity import PHOTO_IDENTITY_FIELDNAMES, photo_identity_values_from_profile
from preference_labels import (
    infer_correction_type,
    infer_preference_tier,
)


PROFILES_PATH = ROOT_DIR / "data" / "profiles.csv"
SYNTHETIC_PATH = ROOT_DIR / "data" / "synthetic.csv"
logger = get_logger(__name__)

CSV_FIELDNAMES = [
    "name",
    "age",
    "distance_km",
    "bio",
    "interests",
    "descriptors",
    *PHOTO_IDENTITY_FIELDNAMES,
    *PHOTO_FEATURE_NAMES,
    *EMBEDDING_FEATURE_NAMES,
    *SEMANTIC_EMBEDDING_FEATURE_NAMES,
    "photo_features_saved",
    "ai_decision",
    "final_decision",
    "manual_corrected",
    "feedback_domain",
    "feedback_reason",
    "feedback_intensity",
    "feedback_sentiment",
    "feedback_secondary",
    "visual_face_label",
    "visual_body_label",
    "visual_style_label",
    "visual_overall_label",
    "feedback_details",
    "swipe_probability",
    "preference_tier",
    "correction_type",
    "review_priority",
    "ranking_score",
    "label",
    "source",
]

VISUAL_LABEL_FIELDS = (
    "visual_face_label",
    "visual_body_label",
    "visual_style_label",
    "visual_overall_label",
)
VISUAL_LABEL_VALUES = {"positive", "neutral", "negative"}


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CSV_FIELDNAMES)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.debug("CSV nao encontrado: %s", path)
        return _empty_df()

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        logger.exception("Falha ao carregar CSV: %s", path)
        raise
    for col in CSV_FIELDNAMES:
        if col not in df.columns:
            df[col] = ""
    logger.debug("CSV carregado: %s rows=%s", path, len(df))
    return df[CSV_FIELDNAMES]


def _ensure_profiles_schema() -> None:
    """Migra data/profiles.csv para o schema mais novo quando necessario."""
    if not PROFILES_PATH.exists():
        return

    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if fieldnames == CSV_FIELDNAMES:
        return

    logger.info("Migrando schema do profiles.csv: old=%s new=%s", fieldnames, CSV_FIELDNAMES)
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            migrated = {key: row.get(key, "") for key in CSV_FIELDNAMES}
            migrated["preference_tier"] = migrated.get("preference_tier") or infer_preference_tier(migrated)
            migrated["correction_type"] = migrated.get("correction_type") or infer_correction_type(migrated)
            writer.writerow(migrated)


def profile_row_signature(row: dict) -> str:
    """Assinatura estável de uma linha do profiles.csv para updates de histórico."""
    payload = {
        key: "" if row.get(key) is None else str(row.get(key, ""))
        for key in CSV_FIELDNAMES
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _profile_to_csv_row(profile: dict, label: int | str | None) -> dict:
    photo = profile.get("_photo_features") or {}
    label_value = "" if label in ("", None, "maybe") else int(float(label))
    final_decision = profile.get("final_decision", "")
    photo_identity = photo_identity_values_from_profile(profile)
    visual_labels = _visual_labels_from_profile(profile)
    row = {
        "name": profile.get("name", ""),
        "age": profile.get("age", ""),
        "distance_km": profile.get("distance_km", profile.get("_distance_km", "")) or "",
        "bio": profile.get("bio", ""),
        "interests": ",".join(profile.get("interests", []))
        if isinstance(profile.get("interests"), list)
        else profile.get("interests", ""),
        "descriptors": json.dumps(
            profile.get("_descriptors", {}) or profile.get("descriptors", {}) or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        **photo_identity,
        "photo_features_saved": 1 if photo else 0,
        "ai_decision": profile.get("ai_decision", ""),
        "final_decision": profile.get("final_decision", ""),
        "manual_corrected": profile.get("manual_corrected", ""),
        "feedback_domain": profile.get("feedback_domain", ""),
        "feedback_reason": profile.get("feedback_reason", ""),
        "feedback_intensity": profile.get("feedback_intensity", ""),
        "feedback_sentiment": profile.get("feedback_sentiment", ""),
        "feedback_secondary": profile.get("feedback_secondary", ""),
        **visual_labels,
        "feedback_details": profile.get("feedback_details", ""),
        "swipe_probability": profile.get("swipe_probability", ""),
        "preference_tier": profile.get("preference_tier", ""),
        "correction_type": profile.get("correction_type", ""),
        "review_priority": profile.get("review_priority", ""),
        "ranking_score": profile.get("ranking_score", ""),
        "label": label_value,
        "source": "real",
    }
    row["preference_tier"] = row["preference_tier"] or infer_preference_tier(row, final_decision)
    row["correction_type"] = row["correction_type"] or infer_correction_type(
        row,
        ai_decision=profile.get("ai_decision", ""),
        final_decision=final_decision,
    )

    for feat_name in PHOTO_FEATURE_NAMES:
        raw = photo.get(feat_name, "")
        if raw in ("", None):
            row[feat_name] = ""
            continue
        try:
            row[feat_name] = round(float(raw), 6)
        except Exception:
            row[feat_name] = ""

    for feat_name in EMBEDDING_FEATURE_NAMES:
        val = photo.get(feat_name)
        if val is None or val == "":
            row[feat_name] = ""
            continue
        try:
            f = float(val)
            row[feat_name] = "" if f != f else round(f, 6)  # f != f detecta NaN
        except Exception:
            row[feat_name] = ""

    for feat_name in SEMANTIC_EMBEDDING_FEATURE_NAMES:
        val = photo.get(feat_name)
        if val is None or val == "":
            row[feat_name] = ""
            continue
        try:
            f = float(val)
            row[feat_name] = "" if f != f else round(f, 6)
        except Exception:
            row[feat_name] = ""

    return row


def _visual_labels_from_profile(profile: dict) -> dict[str, str]:
    labels = {}
    for field in VISUAL_LABEL_FIELDS:
        value = str(profile.get(field, "") or "").strip().lower()
        labels[field] = value if value in VISUAL_LABEL_VALUES else ""

    details_raw = profile.get("feedback_details", "")
    try:
        details = json.loads(details_raw or "{}")
    except Exception:
        details = {}
    if not isinstance(details, dict):
        return labels

    for field in VISUAL_LABEL_FIELDS:
        if labels[field]:
            continue
        value = str(details.get(field, "") or "").strip().lower()
        labels[field] = value if value in VISUAL_LABEL_VALUES else ""
    return labels


def _write_profile_rows(rows: list[dict]) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})


def _same_profile_identity(current: dict, profile: dict) -> bool:
    current_name = str(current.get("name", "")).strip().lower()
    profile_name = str(profile.get("name", "")).strip().lower()
    current_age = str(current.get("age", "")).strip()
    profile_age = str(profile.get("age", "")).strip()
    if not current_name or current_name != profile_name or current_age != profile_age:
        return False

    current_keys = {
        str(current.get("photo_url_key", "") or "").strip(),
        str(current.get("photo_fingerprint", "") or "").strip(),
        str(current.get("bio", "") or "").strip(),
        str(current.get("interests", "") or "").strip(),
    }
    profile_interests = profile.get("interests", "")
    if isinstance(profile_interests, list):
        profile_interests = ",".join(profile_interests)
    profile_keys = {
        str(profile.get("photo_url_key", "") or "").strip(),
        str(profile.get("photo_fingerprint", "") or "").strip(),
        str(profile.get("bio", "") or "").strip(),
        str(profile_interests or "").strip(),
    }
    current_keys.discard("")
    profile_keys.discard("")
    return not current_keys or not profile_keys or bool(current_keys & profile_keys)


def replace_real_profile_row(
    row_index: int,
    expected_signature: str,
    profile: dict,
    label: int | str | None,
) -> dict | None:
    """Substitui uma linha real existente e retorna o backup da linha anterior."""
    _ensure_profiles_schema()
    if not PROFILES_PATH.exists():
        return None

    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if row_index < 0 or row_index >= len(rows):
        logger.warning("replace_real_profile_row: indice invalido=%s", row_index)
        return None

    current = rows[row_index]
    if str(current.get("source", "")).strip().lower() != "real":
        logger.warning("replace_real_profile_row: linha nao-real indice=%s", row_index)
        return None

    if expected_signature and profile_row_signature(current) != expected_signature:
        if not _same_profile_identity(current, profile):
            logger.warning("replace_real_profile_row: assinatura divergente indice=%s", row_index)
            return None
        logger.info(
            "replace_real_profile_row: assinatura mudou, mas identidade bate; atualizando index=%s name=%r",
            row_index,
            current.get("name"),
        )

    backup = {key: current.get(key, "") for key in CSV_FIELDNAMES}
    rows[row_index] = _profile_to_csv_row(profile, label)
    _write_profile_rows(rows)
    logger.info(
        "Entrada historica atualizada em profiles.csv: index=%s name=%r age=%s label=%s",
        row_index,
        rows[row_index].get("name"),
        rows[row_index].get("age"),
        label,
    )
    return backup


def restore_real_profile_row(row_index: int, backup_row: dict) -> bool:
    """Restaura uma linha de profiles.csv a partir de backup salvo na review."""
    _ensure_profiles_schema()
    if not PROFILES_PATH.exists() or not backup_row:
        return False

    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if row_index < 0 or row_index >= len(rows):
        logger.warning("restore_real_profile_row: indice invalido=%s", row_index)
        return False

    rows[row_index] = {key: backup_row.get(key, "") for key in CSV_FIELDNAMES}
    _write_profile_rows(rows)
    logger.info("Entrada historica restaurada em profiles.csv: index=%s", row_index)
    return True


def load_all_data(include_untrainable: bool = False) -> pd.DataFrame:
    """Carrega dados reais + sinteticos em um unico DataFrame."""
    _ensure_profiles_schema()
    real = _load_csv(PROFILES_PATH)
    synthetic = _load_csv(SYNTHETIC_PATH)
    frames = [frame for frame in (real, synthetic) if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CSV_FIELDNAMES)
    if not include_untrainable:
        label_numeric = pd.to_numeric(combined["label"], errors="coerce")
        combined = combined[label_numeric.isin([0, 1])].copy()
        combined["label"] = label_numeric[label_numeric.isin([0, 1])].astype(int).values
    logger.info(
        "Dataset carregado: real=%s synthetic=%s combined=%s trainable_only=%s",
        len(real),
        len(synthetic),
        len(combined),
        not include_untrainable,
    )
    return combined


def save_labeled_profile(profile: dict, label: int | str | None) -> int:
    """
    Salva um perfil rotulado em data/profiles.csv.
    Retorna o total de perfis reais rotulados.
    """
    _ensure_profiles_schema()
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = PROFILES_PATH.exists()

    row = _profile_to_csv_row(profile, label)

    with open(PROFILES_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(
        "Perfil rotulado salvo: name=%r age=%r label=%s feedback_domain=%r photo_saved=%s",
        row.get("name"),
        row.get("age"),
        label,
        row.get("feedback_domain"),
        row.get("photo_features_saved"),
    )

    real_df = _load_csv(PROFILES_PATH)
    return len(real_df)


def remove_last_real_profile(name: str, age) -> bool:
    """Remove a entrada mais recente em profiles.csv com o nome+idade informados.
    Retorna True se encontrou e removeu."""
    if not PROFILES_PATH.exists():
        return False

    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    target_name = str(name).strip().lower()
    target_age = str(age).strip()
    idx_to_remove = None
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        if (
            str(row.get("name", "")).strip().lower() == target_name
            and str(row.get("age", "")).strip() == target_age
            and str(row.get("source", "")).strip().lower() == "real"
        ):
            idx_to_remove = i
            break

    if idx_to_remove is None:
        logger.warning("remove_last_real_profile: nao encontrado name=%r age=%s", name, age)
        return False

    rows.pop(idx_to_remove)
    with open(PROFILES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})

    logger.info("Entrada removida de profiles.csv: name=%r age=%s", name, age)
    return True


def count_real_profiles() -> int:
    _ensure_profiles_schema()
    df = _load_csv(PROFILES_PATH)
    if "source" not in df.columns:
        return len(df)
    return int((df["source"].astype(str).str.lower() == "real").sum())


def count_trainable_real_profiles() -> int:
    _ensure_profiles_schema()
    df = _load_csv(PROFILES_PATH)
    if len(df) == 0:
        return 0
    label_numeric = pd.to_numeric(df.get("label", ""), errors="coerce")
    source_real = df["source"].astype(str).str.lower() == "real" if "source" in df.columns else True
    return int((source_real & label_numeric.isin([0, 1])).sum())


def should_retrain(count_before: int, count_after: int, retrain_every: int) -> bool:
    """Decide se deve retreinar baseado na quantidade de novos exemplos."""
    if count_before == 0:
        return True
    return (count_after // retrain_every) > (count_before // retrain_every)
