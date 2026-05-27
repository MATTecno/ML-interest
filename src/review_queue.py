"""Fila de revisao pos-sessao para decisoes feitas no modo automatico."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from config import ROOT_DIR
from dataset import (
    CSV_FIELDNAMES,
    PROFILES_PATH,
    profile_row_signature,
    replace_real_profile_row,
    restore_real_profile_row,
    save_labeled_profile,
)
from feedback import normalize_feedback
from features import PHOTO_FEATURE_NAMES
from filters import apply_hard_filters, apply_photo_filters
from logging_config import get_logger
from photo_identity import (
    photo_fingerprint_from_features,
    photo_identity_values_from_profile,
    photo_url_identity_keys,
    split_photo_url_keys,
)
from photo_storage import PHOTOS_DIR, safe_filename
from preference_labels import (
    decision_to_label,
    infer_correction_type,
    infer_preference_tier,
    is_maybe_decision,
    is_super_like_decision,
    review_priority_from_probability,
)


REVIEW_PATH = ROOT_DIR / "data" / "review_queue.csv"
logger = get_logger(__name__)

REVIEW_META_FIELDNAMES = [
    "review_mode",
    "target_profile_index",
    "target_profile_signature",
    "target_profile_backup",
]

REVIEW_FIELDNAMES = [
    "review_id",
    "created_at",
    "review_status",
    "reviewed_at",
    "photo_path",
    "photo_url",
    "original_label",
    "ai_snapshot",
    *REVIEW_META_FIELDNAMES,
    *CSV_FIELDNAMES,
]


def _decision_to_label(decision: str) -> str:
    return decision_to_label(decision)


def _decision_from_label(label: str) -> str:
    value = str(label).strip().upper()
    if value in {"", "TALVEZ", "MAYBE"}:
        return "TALVEZ"
    return "CURTIR" if value in {"1", "1.0", "CURTIR", "SUPER_LIKE", "SUPER LIKE"} else "NÃO CURTIR"


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _is_history_review(row: dict) -> bool:
    mode = str(row.get("review_mode") or "").strip().lower()
    source = str(row.get("source") or "").strip().lower()
    return mode in {"history", "quick_agree_recheck"} or source == "history_review"


def _is_recheck_review(row: dict) -> bool:
    return str(row.get("review_mode") or "").strip().lower() == "quick_agree_recheck"


def _read_review_rows() -> list[dict]:
    _ensure_schema()
    if not REVIEW_PATH.exists():
        return []
    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_key_text(value: object) -> str:
    base = unicodedata.normalize("NFD", str(value or "").strip().lower())
    clean = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
    return " ".join(clean.split())


def _photo_path_id_prefix(photo_path: str) -> str:
    name = Path(str(photo_path or "")).name
    prefix = name.split("_", 1)[0].strip()
    if prefix and prefix.lower() != "unknown":
        return prefix
    return ""


def _photo_file_hash(photo_path: str) -> str:
    rel = str(photo_path or "").strip()
    if not rel:
        return ""
    try:
        path = (ROOT_DIR / rel).resolve()
        path.relative_to(ROOT_DIR)
    except Exception:
        return ""
    if not path.exists() or not path.is_file():
        return ""
    try:
        digest = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 128), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return ""


def _photo_keys_from_row(row: dict) -> set[tuple]:
    keys: set[tuple] = set()
    raw_keys: list[str] = []

    raw_keys.extend(split_photo_url_keys(row.get("photo_url_keys", "")))
    raw_keys.extend(split_photo_url_keys(row.get("photo_url_key", "")))
    raw_keys.extend(photo_url_identity_keys(row.get("photo_url", "")))

    for key in raw_keys:
        if key:
            keys.add(("photo_identity", key))

    fingerprint = str(row.get("photo_fingerprint") or "").strip()
    if not fingerprint:
        fingerprint = photo_fingerprint_from_features(_review_photo_for_filters(row))
    age = str(row.get("age", "") or "").strip()
    if fingerprint and age:
        keys.add(("photo_fingerprint_age", fingerprint, age))

    file_hash = _photo_file_hash(row.get("photo_path", ""))
    if file_hash:
        keys.add(("photo_file_hash", file_hash))

    return keys


def _photo_keys_from_profile(profile: dict) -> set[tuple]:
    keys: set[tuple] = set()
    identity = photo_identity_values_from_profile(profile)
    raw_keys: list[str] = []
    raw_keys.extend(split_photo_url_keys(identity.get("photo_url_keys", "")))
    raw_keys.extend(split_photo_url_keys(identity.get("photo_url_key", "")))

    for key in raw_keys:
        if key:
            keys.add(("photo_identity", key))

    age = str(profile.get("age", "") or "").strip()
    fingerprint = identity.get("photo_fingerprint", "")
    if fingerprint and age:
        keys.add(("photo_fingerprint_age", fingerprint, age))

    return keys


def _review_dedupe_key_from_row(row: dict) -> tuple | None:
    keys = _review_dedupe_keys_from_row(row)
    return next(iter(keys), None)


def _review_dedupe_keys_from_row(row: dict) -> set[tuple]:
    keys: set[tuple] = set()
    id_prefix = _photo_path_id_prefix(row.get("photo_path", ""))
    if id_prefix:
        keys.add(("id", id_prefix))

    name = _normalize_key_text(row.get("name", ""))
    age = str(row.get("age", "") or "").strip()
    if name and age:
        bio = _normalize_key_text(row.get("bio", ""))[:120]
        interests = _normalize_key_text(row.get("interests", ""))[:120]
        if bio or interests:
            keys.add(("name_age_text", name, age, bio, interests))
        keys.add(("name_age", name, age))

    photo_url = str(row.get("photo_url", "") or "").strip()
    if photo_url:
        keys.add(("photo_url", photo_url))
        try:
            parsed = urlsplit(photo_url.replace("&amp;", "&"))
            if parsed.netloc and parsed.path:
                keys.add(("photo_url_path", parsed.netloc.lower(), parsed.path))
        except Exception:
            pass
    keys.update(_photo_keys_from_row(row))
    return keys


def _strong_review_dedupe_keys_from_row(row: dict) -> set[tuple]:
    """Chaves confiáveis o bastante para ligar treino, fila normal e histórico."""
    return {
        key
        for key in _review_dedupe_keys_from_row(row)
        if key and key[0] != "name_age"
    }


def _review_dedupe_key_from_profile(profile: dict) -> tuple | None:
    keys = _review_dedupe_keys_from_profile(profile)
    return next(iter(keys), None)


def _review_dedupe_keys_from_profile(profile: dict) -> set[tuple]:
    keys: set[tuple] = set()
    tinder_id = str(profile.get("_tinder_id", "") or "").strip()
    if tinder_id:
        keys.add(("id", tinder_id[:8]))

    name = _normalize_key_text(profile.get("name", ""))
    age = str(profile.get("age", "") or "").strip()
    if name and age:
        bio = _normalize_key_text(profile.get("bio", ""))[:120]
        interests_raw = profile.get("interests", "")
        interests = ",".join(interests_raw) if isinstance(interests_raw, list) else str(interests_raw or "")
        norm_interests = _normalize_key_text(interests)[:120]
        if bio or norm_interests:
            keys.add(("name_age_text", name, age, _normalize_key_text(bio), norm_interests))
        keys.add(("name_age", name, age))

    photo_url = str(profile.get("_review_photo_url") or profile.get("_photo_url") or "").strip()
    if photo_url:
        keys.add(("photo_url", photo_url))
        try:
            parsed = urlsplit(photo_url.replace("&amp;", "&"))
            if parsed.netloc and parsed.path:
                keys.add(("photo_url_path", parsed.netloc.lower(), parsed.path))
        except Exception:
            pass
    keys.update(_photo_keys_from_profile(profile))
    return keys


def _find_existing_review(dedupe_key: tuple | None) -> dict | None:
    return _find_existing_review_by_keys({dedupe_key} if dedupe_key else set())


def _find_existing_review_by_keys(dedupe_keys: set[tuple]) -> dict | None:
    if not dedupe_keys or not REVIEW_PATH.exists():
        return None

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if _review_dedupe_keys_from_row(row) & dedupe_keys:
                return row
    return None


def _find_existing_trained_profile_by_keys(dedupe_keys: set[tuple]) -> dict | None:
    if not dedupe_keys or not PROFILES_PATH.exists():
        return None

    try:
        with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("source", "") or "").strip().lower() != "real":
                    continue
                if _review_dedupe_keys_from_row(row) & dedupe_keys:
                    return row
    except OSError:
        logger.debug("Falha ao checar profiles.csv para dedupe de review", exc_info=True)
    return None


def _review_profile_for_filters(row: dict) -> dict:
    try:
        descriptors = json.loads(row.get("descriptors") or "{}")
        if not isinstance(descriptors, dict):
            descriptors = {}
    except Exception:
        descriptors = {}

    interests_raw = row.get("interests", "") or ""
    return {
        "name": row.get("name", "") or "",
        "age": row.get("age", "") or "",
        "bio": row.get("bio", "") or "",
        "interests": [x.strip() for x in interests_raw.split(",") if x.strip()],
        "_descriptors": descriptors,
    }


def _review_photo_for_filters(row: dict) -> dict:
    photo = {}
    for feat_name in PHOTO_FEATURE_NAMES:
        raw = row.get(feat_name, "")
        if raw in ("", None):
            continue
        photo[feat_name] = _as_float(raw, 0.0)
    return photo


def absolute_filter_reason(row_or_profile: dict) -> str:
    """Retorna motivo de filtro absoluto, ou string vazia se deve ir para review."""
    explicit = (row_or_profile.get("_filter_reason") or "").strip()
    if row_or_profile.get("_skip_prompt") and explicit:
        return explicit

    profile = (
        row_or_profile
        if "_descriptors" in row_or_profile or isinstance(row_or_profile.get("interests"), list)
        else _review_profile_for_filters(row_or_profile)
    )
    rejected, reason = apply_hard_filters(profile)
    if rejected:
        return reason

    photo = row_or_profile.get("_photo_features") or _review_photo_for_filters(row_or_profile)
    photo_rejected, photo_reason = apply_photo_filters(photo)
    if photo_rejected:
        return photo_reason

    return ""


def _photo_path_for(profile: dict, decision: str) -> str:
    tinder_id = (profile.get("_tinder_id") or "").strip()
    safe_id = tinder_id[:8] if tinder_id else "unknown"
    safe_name = safe_filename(profile.get("name", ""))
    age = profile.get("age", "")
    folder = "liked" if _decision_to_label(decision) == "1" else "disliked"
    path = PHOTOS_DIR / folder / f"{safe_id}_{safe_name}_{age}.jpg"
    return str(path.relative_to(ROOT_DIR))


def _snapshot_float(data: dict, key: str, default: float = 0.5) -> float:
    try:
        return float(data.get(key, default))
    except Exception:
        return default


def _make_ai_snapshot(profile: dict, ai_decision: str) -> str:
    """Congela os destaques da IA no momento em que o swipe foi decidido."""
    result = profile.get("_ml_result") or {}
    if not result or result.get("model_type") == "filtro":
        return ""

    try:
        from explainer import build_reason_groups, build_visual_subscores

        groups = build_reason_groups(result, top_n=8)
        visual_subscores = build_visual_subscores(result)
        prob = _snapshot_float(result, "probability", 0.5)
        decision = result.get("decision") or ai_decision
        confidence = prob if decision == "CURTIR" else 1.0 - prob
        safety = result.get("probability_safety", {}) or {}
        weights = result.get("weights", {}) or {}
        photo_components = result.get("photo_components", {}) or {}
        features = result.get("features", {}) or {}
        superlike_prob = result.get("superlike_probability")
        photo_model_probability = (
            round(_snapshot_float(photo_components, "photo_model_probability", 0.5), 4)
            if "photo_model_probability" in photo_components
            else ""
        )
        snapshot = {
            "schema_version": 2,
            "decision": decision,
            "confidence": round(confidence, 4),
            "probability": round(prob, 4),
            "raw_probability": round(_snapshot_float(result, "raw_probability", prob), 4),
            "model_type": result.get("model_type", ""),
            "training_version": result.get("training_version", ""),
            "n_samples": result.get("n_samples", ""),
            "text_model_type": result.get("text_model_type", ""),
            "photo_model_type": result.get("photo_model_type", ""),
            "superlike_model_type": result.get("superlike_model_type", ""),
            "text_n_samples": result.get("text_n_samples", ""),
            "photo_n_samples": result.get("photo_n_samples", ""),
            "superlike_n_samples": result.get("superlike_n_samples", ""),
            "superlike_positive_samples": result.get("superlike_positive_samples", ""),
            "photo_score": round(_snapshot_float(result, "photo_score", 0.5), 4),
            "text_score": round(_snapshot_float(result, "text_score", 0.5), 4),
            "text_model_probability": round(_snapshot_float(result, "text_model_probability", 0.5), 4),
            "text_preference_score": round(_snapshot_float(result, "text_preference_score", 0.5), 4),
            "photo_model_probability": photo_model_probability,
            "photo_score_mode": result.get("photo_score_mode", ""),
            "distance": {
                "km": round(_snapshot_float(features, "distance_km", 0.0), 1)
                if _snapshot_float(features, "distance_missing", 1.0) <= 0 else "",
                "score": round(_snapshot_float(features, "distance_score", 0.5), 4),
                "in_range": bool(_snapshot_float(features, "distance_in_range", 0.0) > 0),
                "missing": bool(_snapshot_float(features, "distance_missing", 1.0) > 0),
            },
            "distance_adjustment": result.get("distance_adjustment", {}) or {},
            "visual_subscores": visual_subscores,
            "photo_components": {
                key: round(float(value), 4)
                for key, value in photo_components.items()
                if isinstance(value, (int, float))
            },
            "superlike_probability": round(float(superlike_prob), 4) if superlike_prob is not None else "",
            "weights": {
                "photo_weight": round(_snapshot_float(weights, "photo_weight", 0.5), 4),
                "text_weight": round(_snapshot_float(weights, "text_weight", 0.5), 4),
            },
            "probability_safety": {
                "applied": bool(safety.get("applied")),
                "reason": safety.get("reason", ""),
                "max_confidence": safety.get("max_confidence", ""),
                "raw_probability": safety.get("raw_probability", ""),
                "safe_probability": safety.get("safe_probability", ""),
            },
            "groups": {
                "pro_like": list(groups.get("pro_like", []))[:7],
                "pro_pass": list(groups.get("pro_pass", []))[:7],
                "photo": list(groups.get("photo", []))[:8],
                "body": list(groups.get("body", []))[:8],
                "text": list(groups.get("text", []))[:8],
                "top": list(groups.get("top", []))[:6],
                "uncertainty": list(groups.get("uncertainty", []))[:5],
            },
        }
        return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        logger.exception("Falha ao montar snapshot da IA: name=%r", profile.get("name"))
        return ""


def _ensure_schema() -> None:
    if not REVIEW_PATH.exists():
        return

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if fieldnames == REVIEW_FIELDNAMES:
        return

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            migrated = {key: row.get(key, "") for key in REVIEW_FIELDNAMES}
            migrated["preference_tier"] = migrated.get("preference_tier") or infer_preference_tier(migrated)
            migrated["correction_type"] = migrated.get("correction_type") or infer_correction_type(migrated)
            if not migrated.get("ranking_score"):
                try:
                    snap = json.loads(migrated.get("ai_snapshot") or "{}")
                    migrated["ranking_score"] = snap.get("probability", "")
                except Exception:
                    migrated["ranking_score"] = ""
            if not migrated.get("review_priority"):
                migrated["review_priority"] = review_priority_from_probability(
                    migrated.get("ranking_score", ""),
                    migrated.get("correction_type", ""),
                )
            writer.writerow(migrated)


def _profile_to_row(profile: dict, ai_decision: str, final_decision: str) -> dict:
    photo = profile.get("_photo_features") or {}
    descriptors = profile.get("_descriptors", {}) or profile.get("descriptors", {}) or {}
    label = _decision_to_label(final_decision)
    ml_result = profile.get("_ml_result") or {}
    ranking_score = profile.get("ranking_score") or ml_result.get("ranking_score") or ml_result.get("probability") or ""
    correction_type = profile.get("correction_type") or ml_result.get("correction_type") or infer_correction_type(
        {"label": label},
        ai_decision=ai_decision,
        final_decision=final_decision,
    )
    preference_tier = profile.get("preference_tier") or ml_result.get("preference_tier") or infer_preference_tier(
        {"label": label, "feedback_details": profile.get("feedback_details", "")},
        final_decision=final_decision,
    )
    review_priority = profile.get("review_priority") or ml_result.get("review_priority") or review_priority_from_probability(
        ranking_score,
        correction_type,
    )
    photo_identity = photo_identity_values_from_profile(profile)

    row = {
        "review_id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "review_status": "pending",
        "reviewed_at": "",
        "photo_path": _photo_path_for(profile, final_decision),
        "photo_url": profile.get("_review_photo_url") or profile.get("_photo_url", "") or "",
        "original_label": final_decision if is_super_like_decision(final_decision) or is_maybe_decision(final_decision) else label,
        "ai_snapshot": _make_ai_snapshot(profile, ai_decision),
        "review_mode": "auto",
        "target_profile_index": "",
        "target_profile_signature": "",
        "target_profile_backup": "",
        "name": profile.get("name", ""),
        "age": profile.get("age", ""),
        "distance_km": profile.get("distance_km", profile.get("_distance_km", "")) or "",
        "bio": profile.get("bio", ""),
        "interests": ",".join(profile.get("interests", []))
        if isinstance(profile.get("interests"), list)
        else profile.get("interests", ""),
        "descriptors": json.dumps(descriptors, ensure_ascii=False, sort_keys=True),
        **photo_identity,
        "photo_features_saved": 1 if photo else 0,
        "ai_decision": ai_decision,
        "final_decision": final_decision,
        "manual_corrected": 0,
        "feedback_domain": "",
        "feedback_reason": "",
        "feedback_intensity": "",
        "feedback_sentiment": "",
        "feedback_secondary": "",
        "feedback_details": "",
        "preference_tier": preference_tier,
        "correction_type": correction_type,
        "review_priority": review_priority,
        "ranking_score": ranking_score,
        "label": label,
        "source": "auto_review",
    }

    for feat_name in PHOTO_FEATURE_NAMES:
        raw = photo.get(feat_name, "")
        if raw in ("", None):
            row[feat_name] = ""
            continue
        try:
            row[feat_name] = round(float(raw), 6)
        except Exception:
            row[feat_name] = ""

    return row


def _safe_label(row: dict) -> int | None:
    try:
        value = int(float(row.get("label", "")))
    except Exception:
        return None
    return value if value in (0, 1) else None


def _photo_index_key(path: Path) -> tuple[str, str, str] | None:
    stem = path.stem
    if stem.endswith("_body"):
        stem = stem[:-5]
    try:
        left, age = stem.rsplit("_", 1)
        _, safe_name = left.split("_", 1)
    except ValueError:
        return None
    folder = path.parent.name
    if folder not in {"liked", "disliked"} or not safe_name or not age:
        return None
    return folder, safe_name, age


def _build_training_photo_index() -> dict[tuple[str, str, str], list[Path]]:
    index: dict[tuple[str, str, str], list[Path]] = {}
    for folder in ("liked", "disliked"):
        base = PHOTOS_DIR / folder
        if not base.exists():
            continue
        for path in base.glob("*.jpg"):
            key = _photo_index_key(path)
            if key:
                index.setdefault(key, []).append(path)
    for paths in index.values():
        paths.sort(key=lambda path: (path.name.endswith("_body.jpg"), -path.stat().st_mtime))
    return index


def _training_photo_path(row: dict, label: int, photo_index: dict[tuple[str, str, str], list[Path]] | None = None) -> str:
    """Tenta localizar a foto local de um perfil antigo salvo em profiles.csv."""
    safe_name = safe_filename(row.get("name", ""))
    age = str(row.get("age", "") or "").strip()
    if not safe_name or not age:
        return ""

    preferred = "liked" if label == 1 else "disliked"
    folders = [preferred, "disliked" if preferred == "liked" else "liked"]
    if photo_index is not None:
        for folder in folders:
            for path in photo_index.get((folder, safe_name, age), []):
                if path.exists() and path.is_file():
                    return str(path.relative_to(ROOT_DIR))
        return ""

    patterns = [f"*_{safe_name}_{age}.jpg", f"*_{safe_name}_{age}_body.jpg"]
    matches: list[Path] = []
    for folder in folders:
        base = PHOTOS_DIR / folder
        if not base.exists():
            continue
        for pattern in patterns:
            matches.extend(base.glob(pattern))

    if not matches:
        return ""
    matches = [path for path in matches if path.exists() and path.is_file()]
    matches.sort(key=lambda path: (path.name.endswith("_body.jpg"), -path.stat().st_mtime))
    return str(matches[0].relative_to(ROOT_DIR)) if matches else ""


def _history_review_score(row: dict) -> float:
    """Prioriza histórico com pouco feedback humano ou features recentes faltantes."""
    score = 0.0
    if not str(row.get("feedback_domain", "")).strip():
        score += 2.0
    if not str(row.get("feedback_reason", "")).strip():
        score += 2.0
    details = str(row.get("feedback_details", "") or "").strip()
    if details in ("", "{}"):
        score += 1.5
    if str(row.get("manual_corrected", "")).strip() in ("", "0", "0.0"):
        score += 0.5

    prob_raw = row.get("swipe_probability") or row.get("ranking_score") or ""
    try:
        prob = max(0.0, min(1.0, float(prob_raw)))
        score += max(0.0, 1.0 - abs(prob - 0.5) * 2.0) * 1.5
    except Exception:
        score += 0.8

    if not str(row.get("photo_pose_torso_visibility", "")).strip():
        score += 0.8
    if not str(row.get("photo_clip_pc_01", "")).strip():
        score += 0.8
    if not str(row.get("photo_emb_pc_01", "")).strip():
        score += 0.6
    if not any(str(row.get(key, "")).strip() for key in (
        "photo_body_width_bucket_narrow",
        "photo_body_width_bucket_medium",
        "photo_body_width_bucket_wide",
    )):
        score += 0.5
    return round(score, 4)


def _history_review_row(source_row: dict, row_index: int, signature: str, photo_path: str, score: float) -> dict:
    label = _safe_label(source_row) or 0
    final_decision = source_row.get("final_decision") or _decision_from_label(str(label))
    original_label = final_decision or _decision_from_label(str(label))
    ranking_score = source_row.get("ranking_score") or source_row.get("swipe_probability") or ""
    review_priority = max(
        _as_float(source_row.get("review_priority", ""), 0.0),
        min(0.99, 0.45 + score / 10.0),
    )

    row = {
        key: source_row.get(key, "")
        for key in CSV_FIELDNAMES
    }
    row.update({
        "review_id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "review_status": "pending",
        "reviewed_at": "",
        "photo_path": photo_path,
        "photo_url": "",
        "original_label": original_label,
        "ai_snapshot": "",
        "review_mode": "history",
        "target_profile_index": str(row_index),
        "target_profile_signature": signature,
        "target_profile_backup": "",
        "ai_decision": source_row.get("ai_decision") or final_decision,
        "final_decision": final_decision,
        "manual_corrected": "",
        "feedback_domain": "",
        "feedback_reason": "",
        "feedback_intensity": "",
        "feedback_sentiment": "",
        "feedback_secondary": "",
        "feedback_details": "",
        "review_priority": round(review_priority, 4),
        "ranking_score": ranking_score,
        "label": str(label),
        "source": "history_review",
    })
    return row


def _history_review_context() -> tuple[
    set[tuple[str, str]],
    set[str],
    set[tuple[str, str]],
    set[str],
    set[tuple],
    set[tuple],
    int,
]:
    existing_rows = _read_review_rows()
    pending_history_targets: set[tuple[str, str]] = set()
    pending_history_indexes: set[str] = set()
    handled_history_targets: set[tuple[str, str]] = set()
    handled_history_indexes: set[str] = set()
    pending_keys: set[tuple] = set()
    handled_review_keys: set[tuple] = set()
    for row in existing_rows:
        status = row.get("review_status", "pending")
        if _is_history_review(row):
            target_index = row.get("target_profile_index", "")
            target_signature = row.get("target_profile_signature", "")
            if target_index:
                handled_history_indexes.add(target_index)
            if target_index or target_signature:
                handled_history_targets.add((target_index, target_signature))
            if status == "pending":
                if target_index:
                    pending_history_indexes.add(target_index)
                if target_index or target_signature:
                    pending_history_targets.add((target_index, target_signature))
            continue
        if status == "pending":
            pending_keys.update(_review_dedupe_keys_from_row(row))
        else:
            handled_review_keys.update(_strong_review_dedupe_keys_from_row(row))
    return (
        pending_history_targets,
        pending_history_indexes,
        handled_history_targets,
        handled_history_indexes,
        pending_keys,
        handled_review_keys,
        len(pending_history_indexes),
    )


def _load_profile_rows_for_history() -> list[dict]:
    if not PROFILES_PATH.exists():
        return []
    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        return [
            {key: raw.get(key, "") for key in CSV_FIELDNAMES}
            for raw in csv.DictReader(f)
        ]


def history_review_candidate_stats(min_score: float = 2.0) -> dict:
    """Conta perfis antigos ainda úteis para revisão histórica."""
    _ensure_schema()
    if not PROFILES_PATH.exists():
        return {
            "scanned": 0,
            "eligible": 0,
            "actionable": 0,
            "pending_history": 0,
            "skipped_no_photo": 0,
            "skipped_existing": 0,
        }

    (
        pending_history_targets,
        pending_history_indexes,
        handled_history_targets,
        handled_history_indexes,
        pending_keys,
        handled_review_keys,
        pending_history,
    ) = _history_review_context()
    scanned = 0
    eligible = 0
    actionable = 0
    skipped_no_photo = 0
    skipped_existing = 0
    photo_index = _build_training_photo_index()
    for idx, row in enumerate(_load_profile_rows_for_history()):
        scanned += 1
        if str(row.get("source", "real") or "real").strip().lower() != "real":
            continue
        label = _safe_label(row)
        if label is None:
            continue
        score = _history_review_score(row)
        if score < min_score:
            continue
        eligible += 1
        signature = profile_row_signature(row)
        target_index = str(idx)
        target_key = (target_index, signature)
        if (
            target_index in handled_history_indexes
            or target_key in handled_history_targets
            or target_index in pending_history_indexes
            or target_key in pending_history_targets
        ):
            skipped_existing += 1
            continue
        photo_path = _training_photo_path(row, label, photo_index)
        if not photo_path:
            skipped_no_photo += 1
            continue
        review_row = _history_review_row(row, idx, signature, photo_path, score)
        if _strong_review_dedupe_keys_from_row(review_row) & handled_review_keys:
            skipped_existing += 1
            continue
        if _review_dedupe_keys_from_row(review_row) & pending_keys:
            skipped_existing += 1
            continue
        actionable += 1

    return {
        "scanned": scanned,
        "eligible": eligible,
        "actionable": actionable,
        "pending_history": pending_history,
        "skipped_no_photo": skipped_no_photo,
        "skipped_existing": skipped_existing,
    }


def enqueue_history_review_candidates(limit: int = 40, min_score: float = 2.0) -> dict:
    """Cria reviews de histórico que atualizam linhas existentes do profiles.csv."""
    _ensure_schema()
    if not PROFILES_PATH.exists():
        return {
            "enqueued": 0,
            "scanned": 0,
            "eligible": 0,
            "actionable": 0,
            "pending_history": 0,
            "skipped_no_photo": 0,
            "skipped_existing": 0,
        }

    (
        pending_history_targets,
        pending_history_indexes,
        handled_history_targets,
        handled_history_indexes,
        pending_keys,
        handled_review_keys,
        pending_history,
    ) = _history_review_context()
    candidates: list[tuple[float, dict]] = []
    scanned = 0
    eligible = 0
    skipped_no_photo = 0
    skipped_existing = 0
    photo_index = _build_training_photo_index()
    for idx, row in enumerate(_load_profile_rows_for_history()):
        scanned += 1
        if str(row.get("source", "real") or "real").strip().lower() != "real":
            continue
        label = _safe_label(row)
        if label is None:
            continue
        score = _history_review_score(row)
        if score < min_score:
            continue
        eligible += 1
        signature = profile_row_signature(row)
        target_index = str(idx)
        target_key = (target_index, signature)
        if (
            target_index in handled_history_indexes
            or target_key in handled_history_targets
            or target_index in pending_history_indexes
            or target_key in pending_history_targets
        ):
            skipped_existing += 1
            continue
        photo_path = _training_photo_path(row, label, photo_index)
        if not photo_path:
            skipped_no_photo += 1
            continue
        review_row = _history_review_row(row, idx, signature, photo_path, score)
        if _strong_review_dedupe_keys_from_row(review_row) & handled_review_keys:
            skipped_existing += 1
            continue
        if _review_dedupe_keys_from_row(review_row) & pending_keys:
            skipped_existing += 1
            continue
        candidates.append((score, review_row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [row for _, row in candidates[:max(0, int(limit))]]
    if selected:
        REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = REVIEW_PATH.exists()
        with open(REVIEW_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            for row in selected:
                writer.writerow({key: row.get(key, "") for key in REVIEW_FIELDNAMES})

    logger.info(
        "Reviews historicos enfileirados: enqueued=%s scanned=%s no_photo=%s existing=%s",
        len(selected),
        scanned,
        skipped_no_photo,
        skipped_existing,
    )
    return {
        "enqueued": len(selected),
        "scanned": scanned,
        "eligible": eligible,
        "actionable": len(candidates),
        "pending_history": pending_history,
        "skipped_no_photo": skipped_no_photo,
        "skipped_existing": skipped_existing,
    }


def enqueue_auto_decision(profile: dict, ai_decision: str, final_decision: str) -> str:
    """Guarda uma decisao automatica para revisao posterior."""
    filter_reason = absolute_filter_reason(profile)
    if filter_reason:
        logger.info(
            "Review ignorado por filtro absoluto: name=%r final=%s reason=%s",
            profile.get("name"),
            final_decision,
            filter_reason,
        )
        return ""

    _ensure_schema()
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = REVIEW_PATH.exists()
    row = _profile_to_row(profile, ai_decision, final_decision)
    dedupe_keys = _review_dedupe_keys_from_profile(profile) | _review_dedupe_keys_from_row(row)
    existing = _find_existing_review_by_keys(dedupe_keys)
    if existing:
        logger.info(
            "Review duplicado ignorado: existing_id=%s status=%s name=%r final=%s key=%s",
            existing.get("review_id", ""),
            existing.get("review_status", "pending"),
            row.get("name"),
            final_decision,
            sorted(dedupe_keys),
        )
        return existing.get("review_id", "")
    trained = _find_existing_trained_profile_by_keys(dedupe_keys)
    if trained:
        logger.info(
            "Review ignorado: perfil ja treinado/revisado name=%r age=%r keys=%s",
            row.get("name"),
            row.get("age"),
            sorted(dedupe_keys),
        )
        return ""

    with open(REVIEW_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(
        "Perfil enfileirado para revisao: id=%s name=%r final=%s",
        row["review_id"],
        row.get("name"),
        final_decision,
    )
    return row["review_id"]


def skip_absolute_filter_reviews() -> int:
    """Marca como skipped pendências antigas que são filtros absolutos."""
    _ensure_schema()
    if not REVIEW_PATH.exists():
        return 0

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    changed = 0
    now = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        if row.get("review_status", "pending") != "pending":
            continue
        if _is_history_review(row):
            continue
        reason = absolute_filter_reason(row)
        if not reason:
            continue
        row["review_status"] = "skipped"
        row["reviewed_at"] = now
        row["feedback_domain"] = "other"
        row["feedback_reason"] = f"filtro_absoluto: {reason}"
        row["feedback_intensity"] = "1"
        row["preference_tier"] = "filtered_pass"
        row["correction_type"] = "none"
        changed += 1

    if changed:
        _rewrite_reviews(rows)
        logger.info("Reviews de filtros absolutos ocultados: %s", changed)
    return changed


def skip_duplicate_pending_reviews() -> int:
    """Oculta pendências duplicadas, mantendo só a versão pendente mais recente."""
    _ensure_schema()
    if not REVIEW_PATH.exists():
        return 0

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    non_pending_keys: set[tuple] = set()
    pending_by_key: dict[tuple, list[int]] = {}
    for idx, row in enumerate(rows):
        if _is_history_review(row):
            continue
        keys = _review_dedupe_keys_from_row(row)
        if not keys:
            continue
        if row.get("review_status", "pending") == "pending":
            for key in keys:
                pending_by_key.setdefault(key, []).append(idx)
        else:
            non_pending_keys.update(keys)

    now = datetime.now().isoformat(timespec="seconds")
    changed = 0
    to_skip: set[int] = set()
    for key, indexes in pending_by_key.items():
        if not indexes:
            continue

        if key in non_pending_keys:
            keep_idx = None
        else:
            keep_idx = max(indexes, key=lambda i: rows[i].get("created_at", ""))

        for idx in indexes:
            if idx == keep_idx:
                continue
            to_skip.add(idx)

    for idx in sorted(to_skip):
        row = rows[idx]
        if row.get("review_status", "pending") == "pending":
            row["review_status"] = "skipped"
            row["reviewed_at"] = now
            row["feedback_domain"] = "other"
            row["feedback_reason"] = "duplicado: mesmo perfil ja existe na fila de review"
            row["feedback_intensity"] = "0"
            changed += 1

    if changed:
        _rewrite_reviews(rows)
        logger.info("Reviews duplicados ocultados: %s", changed)
    return changed


def skip_duplicate_history_reviews() -> int:
    """Oculta reviews históricos pendentes que já foram tratados antes."""
    _ensure_schema()
    if not REVIEW_PATH.exists():
        return 0

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    handled_indexes: set[str] = set()
    handled_targets: set[tuple[str, str]] = set()
    handled_review_keys: set[tuple] = set()
    pending_by_index: dict[str, list[int]] = {}
    pending_by_target: dict[tuple[str, str], list[int]] = {}

    for idx, row in enumerate(rows):
        if not _is_history_review(row):
            if row.get("review_status", "pending") != "pending":
                handled_review_keys.update(_strong_review_dedupe_keys_from_row(row))
            continue
        target_index = row.get("target_profile_index", "")
        target_signature = row.get("target_profile_signature", "")
        target_key = (target_index, target_signature)
        if row.get("review_status", "pending") == "pending":
            if target_index:
                pending_by_index.setdefault(target_index, []).append(idx)
            elif target_index or target_signature:
                pending_by_target.setdefault(target_key, []).append(idx)
            continue
        if target_index:
            handled_indexes.add(target_index)
        if target_index or target_signature:
            handled_targets.add(target_key)

    to_skip: dict[int, str] = {}
    for target_index, indexes in pending_by_index.items():
        if target_index in handled_indexes:
            for idx in indexes:
                to_skip[idx] = "duplicado: perfil historico ja foi revisado ou descartado"
            continue
        keep_idx = max(indexes, key=lambda i: rows[i].get("created_at", ""))
        for idx in indexes:
            if idx != keep_idx:
                to_skip[idx] = "duplicado: mesmo perfil historico ja esta pendente"

    for target_key, indexes in pending_by_target.items():
        if target_key in handled_targets:
            for idx in indexes:
                to_skip[idx] = "duplicado: perfil historico ja foi revisado ou descartado"
            continue
        keep_idx = max(indexes, key=lambda i: rows[i].get("created_at", ""))
        for idx in indexes:
            if idx != keep_idx:
                to_skip[idx] = "duplicado: mesmo perfil historico ja esta pendente"

    for idx, row in enumerate(rows):
        if not _is_history_review(row):
            continue
        if row.get("review_status", "pending") != "pending":
            continue
        if idx in to_skip:
            continue
        if not _is_recheck_review(row) and _strong_review_dedupe_keys_from_row(row) & handled_review_keys:
            to_skip[idx] = "duplicado: perfil ja foi revisado na fila normal"

    now = datetime.now().isoformat(timespec="seconds")
    for idx, reason in to_skip.items():
        row = rows[idx]
        if row.get("review_status", "pending") != "pending":
            continue
        row["review_status"] = "skipped"
        row["reviewed_at"] = now
        row["feedback_domain"] = "other"
        row["feedback_reason"] = reason
        row["feedback_intensity"] = "0"
        row["correction_type"] = "none"

    if to_skip:
        _rewrite_reviews(rows)
        logger.info("Reviews historicos duplicados ocultados: %s", len(to_skip))
    return len(to_skip)


def load_reviews(status: str | None = "pending") -> list[dict]:
    _ensure_schema()
    if not REVIEW_PATH.exists():
        return []

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if status:
        rows = [row for row in rows if row.get("review_status", "pending") == status]
    rows.sort(
        key=lambda row: (
            _as_float(row.get("review_priority", ""), 0.0),
            row.get("created_at", ""),
        ),
        reverse=True,
    )
    return rows


def _rewrite_reviews(rows: list[dict]) -> None:
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REVIEW_PATH.with_suffix(f".tmp.{int(time.time() * 1000)}")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REVIEW_FIELDNAMES})
    tmp.replace(REVIEW_PATH)


def apply_review(
    review_id: str,
    final_decision: str,
    feedback_domain: str = "",
    feedback_reason: str = "",
    feedback_intensity: str = "",
    feedback_secondary: str = "",
    feedback_details: str = "",
) -> bool:
    """
    Marca uma revisao como aplicada e salva uma linha real no profiles.csv.

    Retorna True quando encontrou e aplicou a revisao.
    """
    _ensure_schema()
    if not REVIEW_PATH.exists():
        return False

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    found = False
    success = False
    for row in rows:
        if row.get("review_id") != review_id:
            continue

        found = True
        status = row.get("review_status", "pending")
        if status in {"reviewed", "skipped"}:
            logger.info(
                "Revisao ja finalizada; POST repetido tratado como idempotente: id=%s status=%s name=%r",
                review_id,
                status,
                row.get("name"),
            )
            success = True
            break

        history_review = _is_history_review(row)
        label = _decision_to_label(final_decision)
        original_label = str(row.get("original_label", row.get("label", ""))).strip()
        corrected = "1" if original_label and _decision_to_label(original_label) != label else "0"

        profile = {key: row.get(key, "") for key in CSV_FIELDNAMES}
        profile["final_decision"] = final_decision

        photo = {}
        for feat_name in PHOTO_FEATURE_NAMES:
            raw = row.get(feat_name, "")
            if raw not in ("", None):
                photo[feat_name] = raw
        if photo:
            photo_url = row.get("photo_url", "")
            if photo_url:
                from photo_embedding_pca import (
                    get_embedding_from_cache_by_url,
                    enrich_photo_features,
                    load_pca,
                )
                pca = load_pca()
                if pca is not None:
                    embedding = get_embedding_from_cache_by_url(photo_url)
                    if embedding is not None:
                        photo["_embedding"] = embedding
                        enrich_photo_features(photo, pca)
            profile["_photo_features"] = photo

        try:
            profile["_descriptors"] = json.loads(row.get("descriptors") or "{}")
        except Exception:
            profile["_descriptors"] = {}

        interests_raw = row.get("interests", "")
        profile["interests"] = [x.strip() for x in interests_raw.split(",") if x.strip()]
        profile.update(
            normalize_feedback(
                profile,
                final_decision,
                feedback_domain=feedback_domain,
                feedback_reason=feedback_reason,
                feedback_intensity=feedback_intensity,
                ai_decision=_decision_from_label(original_label),
            )
        )
        profile["manual_corrected"] = corrected
        profile["feedback_secondary"] = feedback_secondary
        profile["feedback_details"] = feedback_details
        profile["preference_tier"] = infer_preference_tier(profile, final_decision)
        profile["correction_type"] = infer_correction_type(profile, _decision_from_label(original_label), final_decision)
        profile["ranking_score"] = row.get("ranking_score", profile.get("ranking_score", ""))
        profile["review_priority"] = row.get("review_priority", profile.get("review_priority", ""))
        try:
            snap = json.loads(row.get("ai_snapshot") or "{}")
            ai_prob = snap.get("probability")
            if ai_prob is not None:
                profile["swipe_probability"] = round(float(ai_prob), 4)
                if not profile["ranking_score"]:
                    profile["ranking_score"] = profile["swipe_probability"]
        except Exception:
            pass
        if history_review:
            try:
                target_index = int(row.get("target_profile_index", ""))
            except Exception:
                target_index = -1
            backup = replace_real_profile_row(
                target_index,
                row.get("target_profile_signature", ""),
                profile,
                label,
            )
            if backup is None:
                logger.warning(
                    "Revisao historica nao aplicada: id=%s name=%r target_index=%s",
                    review_id,
                    row.get("name"),
                    row.get("target_profile_index", ""),
                )
                break
            row["target_profile_backup"] = json.dumps(backup, ensure_ascii=False, sort_keys=True)
        else:
            trained_duplicate = _find_existing_trained_profile_by_keys(_review_dedupe_keys_from_row(row))
            if trained_duplicate:
                row["review_status"] = "skipped"
                row["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
                row["feedback_domain"] = "other"
                row["feedback_reason"] = "duplicado: perfil ja estava salvo no treino"
                row["feedback_intensity"] = "0"
                row["correction_type"] = "none"
                row["review_priority"] = profile.get("review_priority", "")
                row["ranking_score"] = profile.get("ranking_score", "")
                success = True
                logger.info("Revisao duplicada ignorada sem salvar no treino: id=%s name=%r", review_id, row.get("name"))
                break

            save_labeled_profile(profile, label)

        row["review_status"] = "reviewed"
        row["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
        row["final_decision"] = final_decision
        row["manual_corrected"] = corrected
        row["feedback_domain"] = profile["feedback_domain"]
        row["feedback_reason"] = profile["feedback_reason"]
        row["feedback_intensity"] = profile["feedback_intensity"]
        row["feedback_sentiment"] = profile["feedback_sentiment"]
        row["feedback_secondary"] = feedback_secondary
        row["feedback_details"] = feedback_details
        row["preference_tier"] = profile["preference_tier"]
        row["correction_type"] = profile["correction_type"]
        row["ranking_score"] = profile.get("ranking_score", "")
        row["review_priority"] = profile.get("review_priority", "")
        row["label"] = label
        success = True
        if history_review:
            logger.info("Revisao historica aplicada: id=%s name=%r final=%s", review_id, row.get("name"), final_decision)
        else:
            logger.info("Revisao aplicada: id=%s name=%r final=%s", review_id, row.get("name"), final_decision)
        break

    if found:
        _rewrite_reviews(rows)
    return success


def undo_review(review_id: str) -> bool:
    """Reverte uma revisão para pendente; só remove do dataset quando ela foi aplicada."""
    from dataset import remove_last_real_profile

    _ensure_schema()
    if not REVIEW_PATH.exists():
        return False

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    found = False
    for row in rows:
        if row.get("review_id") != review_id:
            continue
        status = row.get("review_status")
        if status not in ("reviewed", "skipped"):
            break

        if status == "reviewed":
            if _is_history_review(row):
                try:
                    backup = json.loads(row.get("target_profile_backup") or "{}")
                except Exception:
                    backup = {}
                try:
                    target_index = int(row.get("target_profile_index", ""))
                except Exception:
                    target_index = -1
                if backup:
                    restore_real_profile_row(target_index, backup)
            else:
                remove_last_real_profile(row.get("name", ""), row.get("age", ""))
        original_label = row.get("original_label", "")
        row["review_status"] = "pending"
        row["reviewed_at"] = ""
        row["final_decision"] = (
            "SUPER_LIKE"
            if str(original_label).strip().upper() in {"SUPER_LIKE", "SUPER LIKE"}
            else _decision_from_label(original_label)
        )
        row["manual_corrected"] = ""
        row["feedback_domain"] = ""
        row["feedback_reason"] = ""
        row["feedback_intensity"] = ""
        row["feedback_sentiment"] = ""
        row["feedback_secondary"] = ""
        row["feedback_details"] = ""
        row["target_profile_backup"] = ""
        row["label"] = _decision_to_label(original_label or row.get("label", ""))
        found = True
        logger.info("Revisão desfeita: id=%s name=%r", review_id, row.get("name"))
        break

    if found:
        _rewrite_reviews(rows)
    return found


def skip_all_pending() -> int:
    """Marca todos os pendentes como skipped em uma única escrita."""
    _ensure_schema()
    if not REVIEW_PATH.exists():
        return 0

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    now = datetime.now().isoformat(timespec="seconds")
    count = 0
    for row in rows:
        if row.get("review_status", "pending") == "pending":
            row["review_status"] = "skipped"
            row["reviewed_at"] = now
            count += 1

    if count:
        _rewrite_reviews(rows)
    return count


def skip_review(review_id: str) -> bool:
    _ensure_schema()
    if not REVIEW_PATH.exists():
        return False

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    found = False
    for row in rows:
        if row.get("review_id") == review_id:
            row["review_status"] = "skipped"
            row["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            found = True
            break
    if found:
        _rewrite_reviews(rows)
    return found
