"""Treino e serializacao dos modelos de texto/foto."""

from __future__ import annotations

import pickle
import json
import hashlib
import warnings
import threading
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ROOT_DIR
from dataset import load_all_data, _load_csv, PROFILES_PATH
from feedback import normalize_photo_reason
from features import (
    extract_features,
    FEATURE_LABELS,
    FEATURE_NAMES,
    TEXT_FEATURE_NAMES,
    MODEL_PHOTO_FEATURE_NAMES,
    MODEL_META_FEATURE_NAMES,
    EMBEDDING_FEATURE_NAMES,
    SEMANTIC_EMBEDDING_FEATURE_NAMES,
    load_config,
)

PHOTO_ALL_FEATURE_NAMES = MODEL_PHOTO_FEATURE_NAMES + EMBEDDING_FEATURE_NAMES + SEMANTIC_EMBEDDING_FEATURE_NAMES
SUPERLIKE_FEATURE_NAMES = TEXT_FEATURE_NAMES + PHOTO_ALL_FEATURE_NAMES
from logging_config import get_logger
from photo_deep_feedback import log_dataset_stats_for_training


MODEL_PATH = ROOT_DIR / "models" / "classifier.pkl"
FEATURE_CACHE_PATH = ROOT_DIR / "data" / "training_feature_cache.json"
TRAINING_VERSION = 21
logger = get_logger(__name__)
_FEATURE_CACHE_SCHEMA_VERSION = 1
_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE_DATA: dict | None = None
_MODEL_CACHE_MTIME_NS = 0

_META_FEATURE_NAMES = MODEL_META_FEATURE_NAMES
_MIN_META_SAMPLES = 20
_LGBM_PARAMS = {
    "n_estimators": 160,
    "learning_rate": 0.04,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 12,
    "min_split_gain": 0.01,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}
_LGBM_GPU_DISABLED = False


def _json_default(value):
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _stable_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pca_signature(pca) -> str:
    if pca is None:
        return "none"
    parts = {
        "class": type(pca).__name__,
        "components": getattr(pca, "components_", None),
        "mean": getattr(pca, "mean_", None),
        "explained": getattr(pca, "explained_variance_ratio_", None),
    }
    return _stable_hash(parts)


def _feature_cache_context(config: dict, feature_names: list[str], bio_pca=None, int_pca=None) -> str:
    return _stable_hash({
        "schema": _FEATURE_CACHE_SCHEMA_VERSION,
        "training_version": TRAINING_VERSION,
        "config": config,
        "feature_names": feature_names,
        "bio_pca": _pca_signature(bio_pca),
        "int_pca": _pca_signature(int_pca),
    })


def _load_feature_cache() -> dict:
    if not FEATURE_CACHE_PATH.exists():
        return {"schema": _FEATURE_CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        with open(FEATURE_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema") != _FEATURE_CACHE_SCHEMA_VERSION:
            return {"schema": _FEATURE_CACHE_SCHEMA_VERSION, "entries": {}}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            return {"schema": _FEATURE_CACHE_SCHEMA_VERSION, "entries": {}}
        return data
    except Exception:
        logger.debug("Falha ao carregar cache de features de treino", exc_info=True)
        return {"schema": _FEATURE_CACHE_SCHEMA_VERSION, "entries": {}}


def _save_feature_cache(cache: dict) -> None:
    FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = cache.get("entries", {})
    if isinstance(entries, dict) and len(entries) > 50000:
        ordered = list(entries.items())[-50000:]
        cache["entries"] = dict(ordered)
    tmp = FEATURE_CACHE_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        tmp.replace(FEATURE_CACHE_PATH)
    except Exception:
        logger.debug("Falha ao salvar cache de features de treino", exc_info=True)


def _row_cache_payload(row: pd.Series) -> dict:
    payload = {}
    for key, value in row.to_dict().items():
        try:
            if pd.isna(value):
                payload[key] = ""
            else:
                payload[key] = value
        except Exception:
            payload[key] = value
    return payload


def _feature_cache_key(row: pd.Series, context_hash: str) -> str:
    return _stable_hash({"context": context_hash, "row": _row_cache_payload(row)})


def _cached_feature_values(entries: dict, key: str, feature_names: list[str]) -> list[float] | None:
    entry = entries.get(key)
    if not isinstance(entry, dict) or entry.get("feature_names") != feature_names:
        return None
    values = entry.get("values")
    if not isinstance(values, list) or len(values) != len(feature_names):
        return None
    try:
        return [float(v) for v in values]
    except Exception:
        return None


def _lgbm_gpu_params(config: dict | None = None, force_cpu: bool = False) -> dict:
    global _LGBM_GPU_DISABLED
    if force_cpu or _LGBM_GPU_DISABLED:
        return {}
    cfg = ((config or load_config()).get("model", {}) or {}).get("gpu", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return {}
    params = {"device_type": str(cfg.get("device_type") or "gpu")}
    if cfg.get("platform_id") not in ("", None):
        params["gpu_platform_id"] = int(cfg.get("platform_id"))
    if cfg.get("device_id") not in ("", None):
        params["gpu_device_id"] = int(cfg.get("device_id"))
    return params


def _make_lgbm_classifier(config: dict | None = None, force_cpu: bool = False) -> LGBMClassifier:
    return LGBMClassifier(**{**_LGBM_PARAMS, **_lgbm_gpu_params(config, force_cpu=force_cpu)})


def _is_lgbm_gpu_configured(config: dict | None = None) -> bool:
    return bool(_lgbm_gpu_params(config))


def _disable_lgbm_gpu_after_error(error: Exception) -> None:
    global _LGBM_GPU_DISABLED
    _LGBM_GPU_DISABLED = True
    logger.warning("LightGBM GPU desativado para este processo; retentando em CPU. Erro: %s", error)


def _linear_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])


def _build_X(
    df: pd.DataFrame,
    config: dict,
    feature_names: list[str],
    bio_pca=None,
    int_pca=None,
) -> np.ndarray:
    context_hash = _feature_cache_context(config, feature_names, bio_pca, int_pca)
    cache = _load_feature_cache()
    entries = cache.setdefault("entries", {})
    row_values: list[list[float] | None] = []
    missing: list[tuple[int, pd.Series]] = []
    cache_hits = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        key = _feature_cache_key(row, context_hash)
        cached = _cached_feature_values(entries, key, feature_names)
        if cached is not None:
            row_values.append(cached)
            cache_hits += 1
        else:
            row_values.append(None)
            missing.append((idx, row))

    text_feats_by_idx: dict[int, dict] = {}
    if missing and bio_pca is not None and int_pca is not None:
        from bio_embedding import batch_text_to_features as _batch_text_to_features

        bios_missing, ints_missing, idxs = [], [], []
        for idx, row in missing:
            idxs.append(idx)
            bios_missing.append("" if pd.isna(row.get("bio", "")) else str(row.get("bio", "")))
            interests_raw = row.get("interests", "")
            if pd.isna(interests_raw) or not interests_raw:
                ints_missing.append([])
            else:
                ints_missing.append([x.strip() for x in str(interests_raw).split(",") if x.strip()])
        for idx, feats in zip(idxs, _batch_text_to_features(bios_missing, ints_missing, bio_pca, int_pca)):
            text_feats_by_idx[idx] = feats

    cache_updates = 0
    for idx, row in missing:
        interests_raw = row.get("interests", "")
        if pd.isna(interests_raw) or not interests_raw:
            interests = []
        else:
            interests = [x.strip() for x in str(interests_raw).split(",") if x.strip()]

        age_raw = row.get("age", 0)
        age = 0 if pd.isna(age_raw) else age_raw

        profile = row.to_dict()
        profile["name"] = "" if pd.isna(row.get("name", "")) else str(row.get("name", ""))
        profile["age"] = age
        profile["bio"] = "" if pd.isna(row.get("bio", "")) else str(row.get("bio", ""))
        profile["interests"] = interests
        profile["descriptors"] = "" if pd.isna(row.get("descriptors", "")) else str(row.get("descriptors", ""))
        _apply_body_corrections(profile, _parse_feedback_details(row))

        if idx in text_feats_by_idx:
            profile["_text_features"] = text_feats_by_idx[idx]

        feats = extract_features(profile, config)
        values = [float(feats[f]) for f in feature_names]
        row_values[idx] = values
        entries[_feature_cache_key(row, context_hash)] = {
            "feature_names": list(feature_names),
            "values": values,
        }
        cache_updates += 1

    if cache_updates:
        _save_feature_cache(cache)
    logger.info(
        "Feature matrix pronta: rows=%s features=%s cache_hits=%s cache_misses=%s",
        len(df),
        len(feature_names),
        cache_hits,
        len(missing),
    )

    return np.array([values or [float("nan")] * len(feature_names) for values in row_values], dtype=float)


def _feedback_weight_config(config: dict) -> dict:
    defaults = {
        "enabled": True,
        "manual_correction_multiplier": 1.35,
        "pass_correction_multiplier": 2.75,
        "max_sample_weight": 8.0,
        "active_learning_boost": 0.5,
        "photo_deep_weight": 1.6,
        "intensity_multipliers": {
            "1": 0.85,
            "2": 1.0,
            "3": 1.35,
        },
        "synthetic_weight": 0.35,
        "photo_reason_multipliers": {
            "photo_face": 1.25,
            "photo_gender": 1.0,
            "photo_general": 1.0,
            "photo_body": 1.35,
            "photo_style": 0.75,
            "photo_context": 0.55,
        },
        "photo_model": {
            "photo": 2.6,
            "bio": 0.25,
            "interests": 0.25,
            "descriptors": 0.35,
            "other": 0.45,
            "empty": 0.35,
        },
        "text_model": {
            "photo": 0.25,
            "bio": 2.4,
            "interests": 2.4,
            "descriptors": 2.0,
            "other": 0.45,
            "empty": 0.35,
        },
    }
    user_cfg = config.get("model", {}).get("feedback_weights", {}) or {}
    merged = {**defaults, **user_cfg}
    merged["photo_model"] = {**defaults["photo_model"], **user_cfg.get("photo_model", {})}
    merged["text_model"] = {**defaults["text_model"], **user_cfg.get("text_model", {})}
    merged["intensity_multipliers"] = {
        **defaults["intensity_multipliers"],
        **user_cfg.get("intensity_multipliers", {}),
    }
    merged["photo_reason_multipliers"] = {
        **defaults["photo_reason_multipliers"],
        **user_cfg.get("photo_reason_multipliers", {}),
    }
    return merged


_PROXY_FEATURES: set[str] = set()  # proxies removidos de TEXT_FEATURE_NAMES

_BODY_FRAME_OVERRIDES: dict[str, dict[str, float]] = {
    "corpo_inteiro": {"photo_body_full_length": 1.0, "photo_body_upper_length": 0.0, "photo_body_closeup": 0.0, "photo_body_visible": 1.0},
    "meio_corpo":    {"photo_body_full_length": 0.0, "photo_body_upper_length": 1.0, "photo_body_closeup": 0.0, "photo_body_visible": 0.8},
    "close_up":      {"photo_body_full_length": 0.0, "photo_body_upper_length": 0.0, "photo_body_closeup": 1.0, "photo_body_visible": 0.1},
    "nao_visivel":   {"photo_body_full_length": 0.0, "photo_body_upper_length": 0.0, "photo_body_closeup": 0.0, "photo_body_visible": 0.0},
}

_BODY_BUILD_OVERRIDES: dict[str, dict[str, float]] = {
    # Encoding suave: buckets fracionados + ratio contínuo ensinam o gradiente ao LightGBM
    "estreita":      {"photo_body_width_bucket_narrow": 1.0, "photo_body_width_bucket_medium": 0.0, "photo_body_width_bucket_wide": 0.0, "photo_body_width_ratio": 0.15},
    "media_estreita":{"photo_body_width_bucket_narrow": 0.5, "photo_body_width_bucket_medium": 0.5, "photo_body_width_bucket_wide": 0.0, "photo_body_width_ratio": 0.33},
    "media":         {"photo_body_width_bucket_narrow": 0.0, "photo_body_width_bucket_medium": 1.0, "photo_body_width_bucket_wide": 0.0, "photo_body_width_ratio": 0.50},
    "media_ampla":   {"photo_body_width_bucket_narrow": 0.0, "photo_body_width_bucket_medium": 0.5, "photo_body_width_bucket_wide": 0.5, "photo_body_width_ratio": 0.67},
    "ampla":         {"photo_body_width_bucket_narrow": 0.0, "photo_body_width_bucket_medium": 0.0, "photo_body_width_bucket_wide": 1.0, "photo_body_width_ratio": 0.85},
    # "no_body": ML detectou sinal corporal mas não havia corpo real — zera tudo
    "no_body":       {"photo_body_width_bucket_narrow": 0.0, "photo_body_width_bucket_medium": 0.0, "photo_body_width_bucket_wide": 0.0, "photo_body_width_ratio": 0.0, "photo_body_visible": 0.0},
}


def _apply_body_corrections(profile: dict, details: dict) -> None:
    frame = str(details.get("body_frame_correction") or "").strip()
    build = str(details.get("body_build_correction") or "").strip()
    if frame in _BODY_FRAME_OVERRIDES:
        profile.update(_BODY_FRAME_OVERRIDES[frame])
    if build in _BODY_BUILD_OVERRIDES:
        profile.update(_BODY_BUILD_OVERRIDES[build])


def _parse_feedback_details(row: pd.Series) -> dict:
    raw = row.get("feedback_details", "")
    if raw is None or pd.isna(raw) or raw == "":
        return {}
    try:
        value = json.loads(str(raw))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _clean_numeric_token(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def _truthy_token(value) -> bool:
    return _clean_numeric_token(value).lower() in {"1", "true", "yes", "sim"}


def _photo_score_adjustment(row: pd.Series) -> str:
    details = _parse_feedback_details(row)
    value = str(details.get("photo_score_adjustment", "") or "").strip().lower()
    aliases = {
        "higher": "higher",
        "up": "higher",
        "positive": "higher",
        "good": "higher",
        "lower": "lower",
        "down": "lower",
        "negative": "lower",
        "bad": "lower",
    }
    return aliases.get(value, "")


def _photo_adjustment_reason(row: pd.Series) -> str:
    details = _parse_feedback_details(row)
    return str(
        details.get("photo_score_reason")
        or details.get("photo_reason")
        or row.get("feedback_reason", "")
        or "photo_general"
    )


def _photo_adjustment_intensity(row: pd.Series) -> str:
    details = _parse_feedback_details(row)
    value = _clean_numeric_token(details.get("photo_score_intensity") or row.get("feedback_intensity", ""))
    return value if value in {"1", "2", "3"} else "2"


def _target_action(row: pd.Series) -> str:
    details = _parse_feedback_details(row)
    return str(details.get("target_action") or "").strip().lower()


def _superlike_labels(df: pd.DataFrame) -> np.ndarray:
    labels = []
    for _, row in df.iterrows():
        labels.append(1 if _target_action(row) == "super_like" else 0)
    return np.array(labels, dtype=int)


_PHOTO_FINE_REASON_BY_DETAIL = {
    "face_beauty": "photo_face",
    "face_expression": "photo_face",
    "gender_presentation": "photo_gender",
    "body_shape": "photo_body",
    "body_fitness": "photo_body",
    "body_visibility": "photo_body",
    "context_lifestyle": "photo_context",
    "style_outfit": "photo_style",
    "black_photo": "photo_style",
    "photo_quality": "photo_style",
}


def _detail_list(value) -> list[str]:
    if value in ("", None):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in text.split(",") if x.strip()]
    return [str(value).strip()]


def _feedback_domains(row: pd.Series, details: dict | None = None) -> set[str]:
    allowed = {"photo", "interests", "bio", "descriptors", "other"}
    domains: set[str] = set()
    details = details if details is not None else _parse_feedback_details(row)

    def add(value) -> None:
        domain = str(value or "").strip().lower()
        if domain in allowed:
            domains.add(domain)

    add(row.get("feedback_domain", ""))
    for item in _detail_list(details.get("selected_domains")):
        add(item)
    for item in _detail_list(details.get("secondary_domains")):
        add(item)
    for item in _detail_list(row.get("feedback_secondary", "")):
        add(item)
    return domains


def _effective_feedback_domain(primary: str, domains: set[str], model_domain: str) -> str:
    primary = str(primary or "").strip().lower()
    if model_domain == "photo":
        return "photo" if "photo" in domains else primary
    text_domains = ("interests", "bio", "descriptors")
    if primary in text_domains and primary in domains:
        return primary
    for candidate in text_domains:
        if candidate in domains:
            return candidate
    return primary


def _photo_detail_reason(row: pd.Series, target_label: int) -> str:
    """
    Agrupa detalhes finos de foto em motivos canônicos do modelo visual.

    Só reforçamos detalhes alinhados ao alvo visual: pontos positivos quando a
    foto deve subir/curtir, pontos negativos quando a foto deve baixar/passar.
    Assim, um perfil recusado por texto mas com "rosto bom" não ensina ao
    modelo visual que aquele rosto era ruim.
    """
    details = _parse_feedback_details(row)
    source_key = "photo_positive_details" if int(target_label) == 1 else "photo_negative_details"
    selected = _detail_list(details.get(source_key))
    if not selected:
        return ""

    counts: dict[str, int] = {}
    for item in selected:
        reason = _PHOTO_FINE_REASON_BY_DETAIL.get(str(item).strip())
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return ""

    priority = {
        "photo_gender": 0,
        "photo_face": 1,
        "photo_body": 2,
        "photo_style": 3,
        "photo_context": 4,
        "photo_general": 5,
    }
    return sorted(counts.items(), key=lambda item: (-item[1], priority.get(item[0], 99)))[0][0]


def _photo_target_label(row: pd.Series, photo_adjustment: str = "") -> int:
    if photo_adjustment == "higher":
        return 1
    if photo_adjustment == "lower":
        return 0
    try:
        return int(float(row.get("label", 0) or 0))
    except Exception:
        return 0


def _training_labels(df: pd.DataFrame, domain: str) -> np.ndarray:
    labels = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values
    if domain != "photo":
        return labels

    adjusted = labels.copy()
    for idx, (_, row) in enumerate(df.iterrows()):
        adjustment = _photo_score_adjustment(row)
        if adjustment == "higher":
            adjusted[idx] = 1
        elif adjustment == "lower":
            adjusted[idx] = 0
    return adjusted


def _sample_weights(df: pd.DataFrame, domain: str, config: dict) -> np.ndarray:
    """
    Pesa exemplos conforme o motivo informado no prompt interativo.

    Se o usuario recusou/curtiu por foto, esse exemplo deve ensinar mais o
    modelo visual e menos o textual. Se foi por bio/interesses/descritores,
    fazemos o inverso. Linhas antigas/sem motivo e sinteticas ficam com peso
    baixo para nao contaminar sinais fortes.
    """
    weight_cfg = _feedback_weight_config(config)
    if not weight_cfg.get("enabled", True):
        return np.ones(len(df), dtype=float)

    weights = []
    model_key = "photo_model" if domain == "photo" else "text_model"
    domain_weights = weight_cfg[model_key]
    manual_multiplier = float(weight_cfg.get("manual_correction_multiplier", 1.25))
    pass_correction_multiplier = float(weight_cfg.get("pass_correction_multiplier", 2.25))
    max_sample_weight = float(weight_cfg.get("max_sample_weight", 8.0) or 0.0)
    intensity_multipliers = weight_cfg.get("intensity_multipliers", {})
    synthetic_weight = float(weight_cfg.get("synthetic_weight", 0.35))
    photo_reason_multipliers = weight_cfg.get("photo_reason_multipliers", {})

    for _, row in df.iterrows():
        source = str(row.get("source", "") or "").strip().lower()
        primary_feedback_domain = str(row.get("feedback_domain", "") or "").strip().lower()
        details = _parse_feedback_details(row) if source in {"real", "photo_deep"} else {}
        feedback_domains = _feedback_domains(row, details) if source in {"real", "photo_deep"} else set()
        feedback_domain = _effective_feedback_domain(primary_feedback_domain, feedback_domains, domain)
        feedback_reason = str(row.get("feedback_reason", "") or "").strip().lower()
        feedback_intensity = _clean_numeric_token(row.get("feedback_intensity", ""))
        corrected = _truthy_token(row.get("manual_corrected", ""))
        photo_adjustment = _photo_score_adjustment(row) if source == "real" else ""
        final_label = _clean_numeric_token(row.get("label", ""))
        ai_decision = str(row.get("ai_decision", row.get("original_label", "")) or "").strip().upper()

        if source == "synthetic":
            weight = synthetic_weight
        elif source == "photo_deep":
            if domain == "photo":
                weight = float(weight_cfg.get("photo_deep_weight", 1.6))
            else:
                weight = float(domain_weights.get("empty", 0.35))
        elif source == "real" and feedback_domain:
            weight = float(domain_weights.get(feedback_domain, domain_weights.get("other", 1.0)))
        elif source == "real":
            weight = float(domain_weights.get("empty", 1.0))
        else:
            weight = float(domain_weights.get("other", 0.45))

        if source == "real" and feedback_domain:
            if not (domain == "photo" and photo_adjustment):
                weight *= float(intensity_multipliers.get(feedback_intensity, 1.0))
            if domain == "photo" and feedback_domain == "photo" and not photo_adjustment:
                photo_reason = normalize_photo_reason(str(details.get("photo_reason") or feedback_reason))
                weight *= float(photo_reason_multipliers.get(photo_reason, 1.0))

        if domain == "photo" and photo_adjustment:
            weight = max(weight, float(domain_weights.get("photo", 2.6)))
            weight *= float(intensity_multipliers.get(_photo_adjustment_intensity(row), 1.0))
            photo_reason = normalize_photo_reason(_photo_adjustment_reason(row))
            weight *= float(photo_reason_multipliers.get(photo_reason, 1.0))

        if domain == "photo" and source == "real":
            target_label = _photo_target_label(row, photo_adjustment)
            fine_reason = _photo_detail_reason(row, target_label)
            if fine_reason:
                fine_intensity = (
                    _photo_adjustment_intensity(row)
                    if photo_adjustment
                    else (feedback_intensity if feedback_intensity in {"1", "2", "3"} else "2")
                )
                fine_weight = float(domain_weights.get("photo", 2.6))
                fine_weight *= float(intensity_multipliers.get(fine_intensity, 1.0))
                fine_weight *= float(photo_reason_multipliers.get(fine_reason, 1.0))
                weight = max(weight, fine_weight)

        # Correções explícitas de corpo voltam a treinar preferência visual.
        if domain == "photo" and source == "real":
            if details.get("body_frame_correction") or details.get("body_build_correction"):
                body_weight = float(domain_weights.get("photo", 2.6))
                body_weight *= float(photo_reason_multipliers.get("photo_body", 1.0))
                weight = max(weight, body_weight)

        # Correcoes manuais sao sinal forte porque a IA errou.
        if corrected:
            weight *= manual_multiplier
            if final_label in {"0", "0.0"} and ai_decision in {"CURTIR", "SUPER_LIKE", "SUPER LIKE", "1", "1.0"}:
                weight *= pass_correction_multiplier

        # Active learning: boost proporcional à incerteza do modelo na decisão original.
        # Amostras onde o modelo estava próximo de 0.5 são mais informativas.
        swipe_prob_raw = row.get("swipe_probability", "")
        if swipe_prob_raw not in ("", None):
            try:
                sp = float(swipe_prob_raw)
                if 0.0 <= sp <= 1.0:
                    uncertainty = 1.0 - abs(sp - 0.5) * 2.0  # 1.0=incerto, 0.0=certo
                    al_boost = float(weight_cfg.get("active_learning_boost", 0.5))
                    weight *= 1.0 + al_boost * uncertainty
            except (ValueError, TypeError):
                pass

        if max_sample_weight > 0:
            weight = min(weight, max_sample_weight)
        weights.append(weight)
    return np.array(weights, dtype=float)


def _fit_pipeline(
    df: pd.DataFrame,
    config: dict,
    feature_names: list[str],
    min_rf: int,
    domain: str,
    bio_pca=None,
    int_pca=None,
) -> tuple[Pipeline, str]:
    X = _build_X(df, config, feature_names, bio_pca, int_pca)
    y = _training_labels(df, domain)
    sample_weight = _sample_weights(df, domain, config)
    logger.info(
        "Treinando pipeline: domain=%s rows=%s features=%s min_rf=%s classes=%s",
        domain,
        len(df),
        len(feature_names),
        min_rf,
        sorted(set(y)),
    )

    if len(set(y)) < 2:
        raise ValueError("O treino precisa de pelo menos duas classes diferentes.")

    uses_lgbm = len(df) >= min_rf
    if uses_lgbm:
        classifier = _make_lgbm_classifier(config)
        model_type = "LightGBM"
    else:
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        model_type = "LogisticRegression"

    if model_type == "LogisticRegression":
        pipeline = _linear_pipeline(classifier)
    else:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ])
    try:
        pipeline.fit(X, y, clf__sample_weight=sample_weight)
    except Exception as exc:
        if uses_lgbm and _is_lgbm_gpu_configured(config):
            _disable_lgbm_gpu_after_error(exc)
            classifier = _make_lgbm_classifier(config, force_cpu=True)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", classifier),
            ])
            pipeline.fit(X, y, clf__sample_weight=sample_weight)
            model_type = "LightGBM-CPU-fallback"
        else:
            raise
    logger.info("Pipeline treinado: domain=%s model=%s rows=%s", domain, model_type, len(df))
    return pipeline, model_type


def _top_features(pipeline: Pipeline, feature_names: list[str], top_n: int = 3) -> list[tuple[str, float]]:
    clf = pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return []
    raw = clf.feature_importances_.astype(float)
    total = raw.sum()
    normalized = raw / total if total > 0 else raw
    return sorted(zip(feature_names, normalized), key=lambda x: x[1], reverse=True)[:top_n]


def _compute_retrain_report(
    old_model_data: dict | None,
    text_pipeline: Pipeline,
    photo_pipeline: Pipeline | None,
    meta_pipeline: Pipeline | None,
    df_real: pd.DataFrame,
    df_all: pd.DataFrame,
    config: dict,
    photo_n_samples: int,
    bio_pca=None,
    int_pca=None,
) -> float | None:
    """Imprime relatório de qualidade pós-retreino e retorna a acurácia calculada."""
    old_accuracy = old_model_data.get("real_accuracy") if old_model_data else None
    new_accuracy: float | None = None
    n_real = len(df_real)
    lines = ["", "  ── Retreino concluído ──────────────────────────────────"]

    # ── Dataset ──────────────────────────────────────────────────────────────
    n_synth = int((df_all["source"].astype(str).str.lower() == "synthetic").sum())
    n_liked = int((df_real["label"] == 1).sum()) if n_real > 0 else 0
    n_passed = n_real - n_liked
    pct_liked = n_liked / n_real * 100 if n_real > 0 else 0

    n_body = 0
    if "photo_body_visible" in df_real.columns:
        n_body = int((pd.to_numeric(df_real["photo_body_visible"], errors="coerce").fillna(0) > 0).sum())

    n_reviewed = n_photo_corrected = n_body_corrected = n_superlike = 0
    if "review_status" in df_real.columns:
        n_reviewed = int((df_real["review_status"].astype(str) == "reviewed").sum())
    if "feedback_details" in df_real.columns:
        for raw in df_real["feedback_details"].dropna():
            try:
                d = json.loads(str(raw))
                if isinstance(d, dict):
                    if d.get("body_frame_correction") or d.get("body_build_correction"):
                        n_body_corrected += 1
                    if d.get("photo_score_adjustment"):
                        n_photo_corrected += 1
                    if d.get("target_action") == "super_like":
                        n_superlike += 1
            except Exception:
                pass

    # ── Modelos ───────────────────────────────────────────────────────────────
    text_type = type(text_pipeline.named_steps.get("clf")).__name__ if text_pipeline else "?"
    photo_type = type(photo_pipeline.named_steps.get("clf")).__name__ if photo_pipeline else "heurístico"
    meta_str = "meta-modelo ativo (cross-val)" if meta_pipeline else "heurístico (poucos dados)"

    lines.append(f"  Modelos")
    lines.append(f"    Texto:    {text_type} · {n_real} reais  ({n_liked} curtidos {pct_liked:.0f}% · {n_passed} passados {100 - pct_liked:.0f}%)")
    lines.append(f"    Foto:     {photo_type} · {photo_n_samples} c/ foto  ·  {n_body} c/ corpo detectado")
    lines.append(f"    Ensemble: {meta_str}")
    if n_synth > 0:
        lines.append(f"    Sintéticos (peso reduzido): {n_synth}")

    # ── Feedback manual ───────────────────────────────────────────────────────
    if n_reviewed or n_photo_corrected or n_body_corrected or n_superlike:
        parts = []
        if n_reviewed:
            parts.append(f"{n_reviewed} revisados")
        if n_photo_corrected:
            parts.append(f"{n_photo_corrected} correções de foto")
        if n_body_corrected:
            parts.append(f"{n_body_corrected} correções de corpo")
        if n_superlike:
            parts.append(f"{n_superlike} super likes rotulados")
        lines.append(f"  Feedback: {' · '.join(parts)}")

    # ── Features com direção ──────────────────────────────────────────────────
    if n_real >= 10 and df_real["label"].nunique() >= 2:
        X = _build_X(df_real, config, TEXT_FEATURE_NAMES, bio_pca, int_pca)
        y = df_real["label"].values
        preds = text_pipeline.predict(X)
        new_accuracy = float((preds == y).mean())

        clf = text_pipeline.named_steps.get("clf")
        if clf is not None and hasattr(clf, "feature_importances_"):
            raw_imp = clf.feature_importances_.astype(float)
            total = raw_imp.sum()
            imp = raw_imp / total if total > 0 else raw_imp
            X_like = X[y == 1]
            X_pass = X[y == 0]
            direction = (X_like.mean(axis=0) - X_pass.mean(axis=0)) if (len(X_like) > 0 and len(X_pass) > 0) else np.zeros(len(TEXT_FEATURE_NAMES))
            ranked = sorted(enumerate(TEXT_FEATURE_NAMES), key=lambda t: imp[t[0]], reverse=True)
            top_like = [(name, imp[i]) for i, name in ranked if direction[i] > 0][:3]
            top_pass = [(name, imp[i]) for i, name in ranked if direction[i] < 0][:3]
            if top_like:
                lines.append("  Pesa para CURTIR")
                for fname, fimp in top_like:
                    proxy = "  ⚠ proxy indireto" if fname in _PROXY_FEATURES else ""
                    lines.append(f"    + {FEATURE_LABELS.get(fname, fname)}  ({fimp * 100:.1f}%){proxy}")
            if top_pass:
                lines.append("  Pesa para PASSAR")
                for fname, fimp in top_pass:
                    proxy = "  ⚠ proxy indireto" if fname in _PROXY_FEATURES else ""
                    lines.append(f"    − {FEATURE_LABELS.get(fname, fname)}  ({fimp * 100:.1f}%){proxy}")

        acc_str = f"{new_accuracy * 100:.1f}%"
        if old_accuracy is not None:
            delta = new_accuracy - old_accuracy
            sign = "+" if delta >= 0 else ""
            acc_str += f"  ({sign}{delta * 100:.1f}% vs anterior)"
        lines.append(f"  Acurácia no treino: {acc_str}  ⚠  medida no próprio dataset")
    else:
        lines.append(f"  Acurácia não calculada: perfis insuficientes ({n_real})")

    lines += ["  ────────────────────────────────────────────────────────", ""]
    print("\n".join(lines))
    return new_accuracy


def _lgbm_template(config: dict | None = None, force_cpu: bool = False) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", _make_lgbm_classifier(config, force_cpu=force_cpu)),
    ])


def _meta_feature_row(text_prob: float, photo_prob: float, features: dict) -> list[float]:
    values = {
        "text_model_prob": float(text_prob),
        "photo_model_prob": float(photo_prob),
    }
    for name in _META_FEATURE_NAMES:
        if name in values:
            continue
        default = 0.5 if name in {"photo_face_similarity", "photo_faces_ratio", "photo_image_sharpness"} else 0.0
        values[name] = float(features.get(name, default) or default)
    return [values[name] for name in _META_FEATURE_NAMES]


def _fit_meta_pipeline(
    df: pd.DataFrame,
    config: dict,
    text_pipeline: Pipeline,
    photo_pipeline: Pipeline | None,
    bio_pca=None,
    int_pca=None,
) -> Pipeline | None:
    """
    Treina o meta-modelo usando predições fora-da-amostra (cross-validation).
    Cada perfil é avaliado por um modelo que nunca o viu, eliminando o vazamento
    de dados que causa probabilidades infladas (ex: 99% para perfis mediocres).
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    real_df = df[df["source"].astype(str).str.lower() == "real"].copy()
    real_df["label"] = pd.to_numeric(real_df["label"], errors="coerce")
    real_df = real_df.dropna(subset=["label"])
    real_df["label"] = real_df["label"].astype(int)
    real_df = real_df.reset_index(drop=True)

    if len(real_df) < _MIN_META_SAMPLES or real_df["label"].nunique() < 2:
        logger.info("Meta-modelo nao treinado: real_samples=%s min=%s", len(real_df), _MIN_META_SAMPLES)
        return None

    # Pré-computa embeddings em batch para o meta-modelo (evita N encodes individuais)
    meta_text_feats_cache: list[dict] | None = None
    if bio_pca is not None and int_pca is not None:
        from bio_embedding import batch_text_to_features as _batch_text_to_features
        bios_meta, ints_meta = [], []
        for _, row in real_df.iterrows():
            bios_meta.append("" if pd.isna(row.get("bio", "")) else str(row.get("bio", "")))
            ir = row.get("interests", "")
            if pd.isna(ir) or not ir:
                ints_meta.append([])
            else:
                ints_meta.append([x.strip() for x in str(ir).split(",") if x.strip()])
        meta_text_feats_cache = _batch_text_to_features(bios_meta, ints_meta, bio_pca, int_pca)

    all_feats = []
    for idx, (_, row) in enumerate(real_df.iterrows()):
        interests_raw = row.get("interests", "")
        interests = [] if (pd.isna(interests_raw) or not interests_raw) else [
            x.strip() for x in str(interests_raw).split(",") if x.strip()
        ]
        bio = "" if pd.isna(row.get("bio", "")) else str(row.get("bio", ""))
        profile = row.to_dict()
        profile["name"] = "" if pd.isna(row.get("name", "")) else str(row.get("name", ""))
        profile["age"] = 0 if pd.isna(row.get("age", 0)) else row.get("age", 0)
        profile["bio"] = bio
        profile["interests"] = interests
        profile["descriptors"] = "" if pd.isna(row.get("descriptors", "")) else str(row.get("descriptors", ""))
        if meta_text_feats_cache is not None:
            profile["_text_features"] = meta_text_feats_cache[idx]
        feats = extract_features(profile, config)
        all_feats.append(feats)

    y_real = real_df["label"].values
    class_counts = np.bincount(y_real)
    min_class_count = int(class_counts[class_counts > 0].min())
    if min_class_count < 2:
        logger.info("Meta-modelo nao treinado: classe minoritaria com %s exemplo", min_class_count)
        return None

    n_folds = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # OOF predictions do modelo de texto (sem vazamento)
    X_text_all = np.array([[f[feat] for feat in TEXT_FEATURE_NAMES] for f in all_feats], dtype=float)
    try:
        text_oof = cross_val_predict(_lgbm_template(config), X_text_all, y_real, cv=cv, method="predict_proba")[:, 1]
    except Exception as exc:
        if _is_lgbm_gpu_configured(config):
            _disable_lgbm_gpu_after_error(exc)
            text_oof = cross_val_predict(_lgbm_template(config, force_cpu=True), X_text_all, y_real, cv=cv, method="predict_proba")[:, 1]
        else:
            raise

    # OOF predictions do modelo de foto (só perfis com rosto detectado)
    photo_oof = np.full(len(real_df), 0.5)
    has_face_mask = np.array([float(f.get("photo_has_face", 0.0)) > 0 for f in all_feats])
    photo_indices = np.where(has_face_mask)[0]
    if photo_pipeline is not None and len(photo_indices) >= _MIN_META_SAMPLES:
        y_photo = _training_labels(real_df.iloc[photo_indices].copy(), "photo")
        if np.unique(y_photo).size >= 2:
            X_photo_sub = np.array([[all_feats[i][feat] for feat in PHOTO_ALL_FEATURE_NAMES] for i in photo_indices], dtype=float)
            photo_class_counts = np.bincount(y_photo)
            min_photo_class_count = int(photo_class_counts[photo_class_counts > 0].min())
            if min_photo_class_count >= 2:
                n_folds_photo = max(2, min(5, min_photo_class_count))
                cv_photo = StratifiedKFold(n_splits=n_folds_photo, shuffle=True, random_state=42)
                try:
                    photo_oof_sub = cross_val_predict(_lgbm_template(config), X_photo_sub, y_photo, cv=cv_photo, method="predict_proba")[:, 1]
                except Exception as exc:
                    if _is_lgbm_gpu_configured(config):
                        _disable_lgbm_gpu_after_error(exc)
                        photo_oof_sub = cross_val_predict(
                            _lgbm_template(config, force_cpu=True),
                            X_photo_sub,
                            y_photo,
                            cv=cv_photo,
                            method="predict_proba",
                        )[:, 1]
                    else:
                        raise
                photo_oof[photo_indices] = photo_oof_sub

    X_meta = np.array([
        _meta_feature_row(text_oof[idx], photo_oof[idx], feats)
        for idx, feats in enumerate(all_feats)
    ], dtype=float)

    from sklearn.calibration import CalibratedClassifierCV

    base_meta = _linear_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.12, random_state=42)
    )

    # Platt scaling via cross-val: cada fold treina o base e calibra no holdout,
    # eliminando o viés de calibrar no mesmo dado de treino
    n_folds_cal = max(2, min(5, min_class_count))
    meta_pipeline = CalibratedClassifierCV(base_meta, method="sigmoid", cv=n_folds_cal)
    meta_pipeline.fit(X_meta, y_real)
    logger.info(
        "Meta-modelo treinado com OOF + calibração Platt: real_samples=%s folds=%s cal_folds=%s",
        len(real_df), n_folds, n_folds_cal,
    )
    return meta_pipeline


def _fit_superlike_pipeline(
    df: pd.DataFrame,
    config: dict,
    min_rf: int,
    bio_pca=None,
    int_pca=None,
) -> tuple[Pipeline | None, str, int, int]:
    """
    Treina um modelo separado para SUPER LIKE.

    Escopo proposital: só compara perfis reais que terminaram como CURTIR.
    Assim o modelo aprende "curtir normal vs curtir forte", sem misturar com
    perfis que você recusaria.
    """
    work = df.copy()
    work["label"] = pd.to_numeric(work.get("label", 0), errors="coerce").fillna(0).astype(int)
    work = work[
        (work["source"].astype(str).str.lower() == "real")
        & (work["label"] == 1)
    ].copy()
    work = work.reset_index(drop=True)

    if len(work) < 20:
        logger.info("Modelo superlike nao treinado: likes reais insuficientes=%s", len(work))
        return None, "indisponível", len(work), 0

    y = _superlike_labels(work)
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive < 5 or negative < 5:
        logger.info(
            "Modelo superlike nao treinado: positivos=%s negativos=%s",
            positive,
            negative,
        )
        return None, "indisponível", len(work), positive

    X = _build_X(work, config, SUPERLIKE_FEATURE_NAMES, bio_pca, int_pca)
    sample_weight = np.ones(len(work), dtype=float)
    for idx, (_, row) in enumerate(work.iterrows()):
        details = _parse_feedback_details(row)
        intensity = str(row.get("feedback_intensity", "") or "").strip()
        if intensity == "3":
            sample_weight[idx] *= 1.35
        elif intensity == "1":
            sample_weight[idx] *= 0.85
        if details.get("photo_positive_details") or details.get("photo_score_adjustment") == "higher":
            sample_weight[idx] *= 1.15
        if y[idx] == 1:
            sample_weight[idx] *= max(1.0, negative / max(positive, 1) * 0.7)

    if len(work) >= min_rf:
        classifier = _make_lgbm_classifier(config)
        model_type = "LightGBM"
    else:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model_type = "LogisticRegression"

    if model_type == "LogisticRegression":
        pipeline = _linear_pipeline(classifier)
    else:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ])
    try:
        pipeline.fit(X, y, clf__sample_weight=sample_weight)
    except Exception as exc:
        if model_type == "LightGBM" and _is_lgbm_gpu_configured(config):
            _disable_lgbm_gpu_after_error(exc)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", _make_lgbm_classifier(config, force_cpu=True)),
            ])
            pipeline.fit(X, y, clf__sample_weight=sample_weight)
            model_type = "LightGBM-CPU-fallback"
        else:
            raise
    logger.info(
        "Modelo superlike treinado: model=%s samples=%s positivos=%s negativos=%s",
        model_type,
        len(work),
        positive,
        negative,
    )
    return pipeline, model_type, len(work), positive


def _photo_deep_feedback_df(config: dict, pca=None, semantic_pca=None) -> pd.DataFrame:
    """Transforma rotulos da aba visual profunda em linhas extras do modelo de foto."""
    weight_cfg = _feedback_weight_config(config)
    if float(weight_cfg.get("photo_deep_weight", 0.0) or 0.0) <= 0:
        return pd.DataFrame()

    try:
        from photo_deep_feedback import _label_from_deep_record, _latest_labeled_records, _safe_photo_path
        from photo_features import analyze_local_photo
        from photo_embedding_pca import enrich_photo_features as enrich_face_pca
        from photo_semantic_embeddings import enrich_photo_features as enrich_semantic_pca
    except Exception:
        logger.debug("Feedback visual profundo indisponivel para treino principal", exc_info=True)
        return pd.DataFrame()

    records = _latest_labeled_records()
    if len(records) < 8:
        logger.info("Feedback visual profundo ignorado no modelo principal: samples=%s min=8", len(records))
        return pd.DataFrame()

    rows = []
    for rec in records:
        label = _label_from_deep_record(rec)
        if label is None:
            continue
        path_rel = str(rec.get("photo_path") or "")
        path = _safe_photo_path(path_rel)
        if path is None:
            continue
        try:
            feats = analyze_local_photo(path, include_embedding=True)
            if pca is not None:
                enrich_face_pca(feats, pca)
            if semantic_pca is not None:
                enrich_semantic_pca(feats, semantic_pca)
        except Exception:
            logger.debug("Feedback visual profundo ignorado por falha em %s", path_rel, exc_info=True)
            continue

        row = {
            "name": Path(path_rel).stem,
            "age": "",
            "bio": "",
            "interests": "",
            "descriptors": "{}",
            "photo_features_saved": 1,
            "source": "photo_deep",
            "label": int(label),
            "feedback_domain": "photo",
            "feedback_reason": "photo_deep",
            "feedback_intensity": "2",
            "manual_corrected": "1",
            "ai_decision": "CURTIR" if int(label) == 0 else "NÃO CURTIR",
            "feedback_details": json.dumps(
                {
                    "photo_positive_details": rec.get("positive_tags") or [],
                    "photo_negative_details": rec.get("negative_tags") or [],
                    "source": "photo_deep_feedback",
                    "photo_path": path_rel,
                },
                ensure_ascii=False,
            ),
        }
        for feat_name in PHOTO_ALL_FEATURE_NAMES:
            value = feats.get(feat_name, "")
            if value in ("", None):
                row[feat_name] = ""
                continue
            try:
                value_f = float(value)
                row[feat_name] = "" if np.isnan(value_f) else value_f
            except Exception:
                row[feat_name] = ""
        rows.append(row)

    logger.info("Feedback visual profundo adicionado ao modelo de foto: samples=%s", len(rows))
    return pd.DataFrame(rows)


def train_model(df: pd.DataFrame | None = None) -> Pipeline:
    """Treina e salva o ensemble atual. Retorna o pipeline de texto treinado."""
    config = load_config()
    model_cfg = config.get("model", {})
    min_rf = int(model_cfg.get("min_samples_for_rf", 30))
    min_photo_samples = int(model_cfg.get("min_photo_samples_for_model", 8))

    if df is None:
        df = load_all_data()

    label_numeric = pd.to_numeric(df.get("label", ""), errors="coerce")
    df = df[label_numeric.isin([0, 1])].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)

    if len(df) == 0:
        logger.error("Treino solicitado sem dados")
        raise ValueError("Nenhum dado disponível para treinar o modelo.")

    # Carrega modelo antigo antes de sobrescrever, para comparar acurácia depois
    old_model_data = load_model()

    logger.info("train_model iniciado: rows=%s", len(df))

    # Treina PCA de embeddings semânticos de bio/interesses antes de qualquer pipeline
    from bio_embedding import collect_embeddings_from_df, fit_and_save_pca as _fit_bio_pca
    bio_embs, int_embs = collect_embeddings_from_df(df)
    bio_pca, int_pca = _fit_bio_pca(bio_embs, int_embs)

    from photo_embedding_pca import load_embeddings_from_cache, fit_and_save_pca
    pca = fit_and_save_pca(load_embeddings_from_cache())

    from photo_semantic_embeddings import (
        cache_stats as semantic_cache_stats,
        fit_and_save_pca as fit_and_save_semantic_pca,
        load_embeddings_from_cache as load_semantic_embeddings_from_cache,
        model_name as semantic_model_name,
        resolve_device as resolve_semantic_device,
    )
    semantic_pca = fit_and_save_semantic_pca(load_semantic_embeddings_from_cache())
    semantic_stats = semantic_cache_stats(reset=False)
    logger.info(
        "Embedding visual semantico no treino: model=%s device=%s cache=%s",
        semantic_model_name(),
        resolve_semantic_device(),
        semantic_stats,
    )

    text_pipeline, text_model_type = _fit_pipeline(
        df,
        config,
        TEXT_FEATURE_NAMES,
        min_rf,
        domain="text",
        bio_pca=bio_pca,
        int_pca=int_pca,
    )

    photo_saved = pd.to_numeric(df.get("photo_features_saved", 0), errors="coerce").fillna(0)
    photo_df = df[photo_saved > 0].copy()
    photo_deep_df = _photo_deep_feedback_df(config, pca=pca, semantic_pca=semantic_pca)
    if not photo_deep_df.empty:
        photo_df = pd.concat([photo_df, photo_deep_df], ignore_index=True, sort=False)
    photo_pipeline = None
    photo_model_type = "Heurístico"
    photo_n_samples = len(photo_df)

    if photo_n_samples >= min_photo_samples and photo_df["label"].nunique() >= 2:
        photo_pipeline, photo_model_type = _fit_pipeline(
            photo_df,
            config,
            PHOTO_ALL_FEATURE_NAMES,
            min_rf,
            domain="photo",
        )
    else:
        logger.info(
            "Modelo de foto nao treinado: photo_samples=%s min=%s classes=%s",
            photo_n_samples,
            min_photo_samples,
            photo_df["label"].nunique() if len(photo_df) else 0,
        )

    meta_pipeline = _fit_meta_pipeline(df, config, text_pipeline, photo_pipeline, bio_pca, int_pca)
    superlike_pipeline, superlike_model_type, superlike_n_samples, superlike_positive_samples = _fit_superlike_pipeline(
        df,
        config,
        min_rf,
        bio_pca,
        int_pca,
    )
    meta_suffix = " + meta-modelo" if meta_pipeline is not None else ""
    superlike_suffix = " + superlike" if superlike_pipeline is not None else ""
    model_type = (
        f"Ensemble ({text_model_type} texto + {photo_model_type} foto{meta_suffix}{superlike_suffix})"
        if photo_pipeline is not None
        else f"Ensemble ({text_model_type} texto + foto heurística{meta_suffix}{superlike_suffix})"
    )

    # Relatório de qualidade comparando com o modelo anterior
    df_real = _load_csv(PROFILES_PATH)
    df_real = df_real[df_real["source"].astype(str).str.lower() == "real"].copy()
    df_real["label"] = pd.to_numeric(df_real["label"], errors="coerce")
    df_real = df_real.dropna(subset=["label"])
    df_real["label"] = df_real["label"].astype(int)
    real_accuracy = _compute_retrain_report(
        old_model_data, text_pipeline, photo_pipeline, meta_pipeline,
        df_real, df, config, photo_n_samples, bio_pca, int_pca,
    )
    try:
        from model_evaluation import evaluate_profiles

        evaluation = evaluate_profiles(config=config, write_reports=True)
    except Exception:
        logger.exception("Falha ao gerar avaliacao offline")
        evaluation = {"status": "error"}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_payload = {
            "training_version": TRAINING_VERSION,
            "text_pipeline": text_pipeline,
            "photo_pipeline": photo_pipeline,
            "meta_pipeline": meta_pipeline,
            "superlike_pipeline": superlike_pipeline,
            "text_model_type": text_model_type,
            "photo_model_type": photo_model_type,
            "superlike_model_type": superlike_model_type,
            "text_n_samples": len(df),
            "photo_n_samples": photo_n_samples,
            "photo_deep_n_samples": int(len(photo_deep_df)),
            "superlike_n_samples": superlike_n_samples,
            "superlike_positive_samples": superlike_positive_samples,
            "text_feature_names": TEXT_FEATURE_NAMES,
            "photo_feature_names": PHOTO_ALL_FEATURE_NAMES,
            "superlike_feature_names": SUPERLIKE_FEATURE_NAMES,
            "pca": pca,
            "semantic_pca": semantic_pca,
            "semantic_feature_names": SEMANTIC_EMBEDDING_FEATURE_NAMES,
            "semantic_embedding_model": semantic_model_name(),
            "semantic_embedding_device": resolve_semantic_device(),
            "semantic_cache_stats": semantic_stats,
            "bio_pca": (bio_pca, int_pca) if bio_pca is not None else None,
            "meta_feature_names": _META_FEATURE_NAMES,
            "feature_names": FEATURE_NAMES,
            "model_type": model_type,
            "n_samples": len(df),
            "real_accuracy": real_accuracy,
            "lgbm_gpu_configured": bool(((model_cfg.get("gpu", {}) or {}).get("enabled", False))),
            "lgbm_gpu_active": bool(_is_lgbm_gpu_configured(config)),
            "evaluation": evaluation,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_payload, f)
    try:
        mtime_ns = MODEL_PATH.stat().st_mtime_ns
    except Exception:
        mtime_ns = 0
    with _MODEL_CACHE_LOCK:
        global _MODEL_CACHE_DATA, _MODEL_CACHE_MTIME_NS
        _MODEL_CACHE_DATA = model_payload
        _MODEL_CACHE_MTIME_NS = mtime_ns
    logger.info(
        "Modelo salvo: path=%s type=%s total=%s photo_samples=%s",
        MODEL_PATH,
        model_type,
        len(df),
        photo_n_samples,
    )
    try:
        log_dataset_stats_for_training()
    except Exception:
        logger.debug("log_dataset_stats_for_training ignorado", exc_info=True)

    return text_pipeline


def load_model() -> dict | None:
    """Carrega o modelo salvo. Retorna None se nao existir ou se as features mudaram."""
    global _MODEL_CACHE_DATA, _MODEL_CACHE_MTIME_NS
    if not MODEL_PATH.exists():
        logger.info("Modelo salvo nao encontrado: %s", MODEL_PATH)
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE_DATA = None
            _MODEL_CACHE_MTIME_NS = 0
        return None

    try:
        mtime_ns = MODEL_PATH.stat().st_mtime_ns
    except Exception:
        mtime_ns = 0
    with _MODEL_CACHE_LOCK:
        if _MODEL_CACHE_DATA is not None and _MODEL_CACHE_MTIME_NS == mtime_ns:
            logger.debug(
                "Modelo retornado do cache: type=%s samples=%s",
                _MODEL_CACHE_DATA.get("model_type"),
                _MODEL_CACHE_DATA.get("n_samples"),
            )
            return _MODEL_CACHE_DATA

    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception:
        logger.exception("Falha ao carregar modelo salvo: %s", MODEL_PATH)
        raise

    expected_keys = {"text_pipeline", "text_feature_names", "photo_feature_names"}
    if not expected_keys.issubset(data.keys()):
        print("  [model] Estrutura do modelo alterada — modelo antigo descartado, retreinando...")
        logger.warning("Modelo descartado por estrutura antiga")
        MODEL_PATH.unlink(missing_ok=True)
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE_DATA = None
            _MODEL_CACHE_MTIME_NS = 0
        return None

    if (
        data.get("text_feature_names") != TEXT_FEATURE_NAMES
        or data.get("photo_feature_names") != PHOTO_ALL_FEATURE_NAMES
    ):
        print("  [model] Features alteradas — modelo antigo descartado, retreinando...")
        logger.warning("Modelo descartado por mudanca nas features")
        MODEL_PATH.unlink(missing_ok=True)
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE_DATA = None
            _MODEL_CACHE_MTIME_NS = 0
        return None

    if data.get("training_version") != TRAINING_VERSION:
        print("  [model] Treino atualizado — modelo antigo descartado, retreinando...")
        logger.warning("Modelo descartado por versao de treino: found=%r expected=%r", data.get("training_version"), TRAINING_VERSION)
        MODEL_PATH.unlink(missing_ok=True)
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE_DATA = None
            _MODEL_CACHE_MTIME_NS = 0
        return None

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE_DATA = data
        _MODEL_CACHE_MTIME_NS = mtime_ns
    logger.info("Modelo carregado: type=%s samples=%s", data.get("model_type"), data.get("n_samples"))
    return data
