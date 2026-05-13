"""Predicao e composicao dos scores de texto/foto."""

from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.pipeline import Pipeline

from decision_policy import apply_decision_policy
from features import (
    extract_features,
    TEXT_FEATURE_NAMES,
    MODEL_PHOTO_FEATURE_NAMES,
    MODEL_META_FEATURE_NAMES,
    EMBEDDING_FEATURE_NAMES,
    SEMANTIC_EMBEDDING_FEATURE_NAMES,
    load_config,
)
from logging_config import get_logger
from model_training import load_model
from text_preferences import score_profile_text

logger = get_logger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_mean(pairs: list[tuple[float, float]], neutral: float = 0.5) -> float:
    total_weight = sum(weight for _, weight in pairs if weight > 0)
    if total_weight <= 0:
        return neutral
    return sum(value * weight for value, weight in pairs if weight > 0) / total_weight


def _heuristic_photo_score(features: dict) -> tuple[float, dict]:
    has_face = float(features.get("photo_has_face", 0.0))
    similarity = float(features.get("photo_face_similarity", 0.5))
    faces_ratio = float(features.get("photo_faces_ratio", 1.0))
    body_visible = float(features.get("photo_body_visible", 0.0) or 0.0)
    body_full = float(features.get("photo_body_full_length", 0.0) or 0.0)
    body_upper = float(features.get("photo_body_upper_length", 0.0) or 0.0)
    body_closeup = float(features.get("photo_body_closeup", 0.0) or 0.0)
    body_quality = float(features.get("photo_body_signal_quality", 0.0) or 0.0)
    image_quality = _weighted_mean(
        [
            (float(features.get("photo_image_sharpness", 0.5)), 0.45),
            (float(features.get("photo_image_brightness", 0.5)), 0.25),
            (float(features.get("photo_image_contrast", 0.5)), 0.20),
            (float(features.get("photo_image_colorfulness", 0.5)), 0.10),
        ],
        neutral=0.5,
    )
    body_score = _weighted_mean(
        [
            (body_quality, 0.38),
            (body_visible, 0.26),
            (body_full, 0.14),
            (body_upper, 0.10),
            (1.0 - _clamp01(body_closeup), 0.06),
            (image_quality, 0.06),
        ],
        neutral=0.5,
    )

    if has_face <= 0:
        return 0.05, {
            "has_face": has_face,
            "face_similarity": similarity,
            "faces_ratio": faces_ratio,
            "body_score": round(body_score, 4),
            "body_visible": round(body_visible, 4),
            "body_quality": round(body_quality, 4),
            "image_quality": round(image_quality, 4),
        }

    score = _weighted_mean(
        [
            (similarity, 0.64),
            (body_score, 0.18),
            (faces_ratio, 0.08),
            (image_quality, 0.10),
        ]
    )
    return round(_clamp01(score), 4), {
        "has_face": has_face,
        "face_similarity": round(similarity, 4),
        "faces_ratio": round(faces_ratio, 4),
        "body_score": round(body_score, 4),
        "body_visible": round(body_visible, 4),
        "body_quality": round(body_quality, 4),
        "image_quality": round(image_quality, 4),
    }


def _photo_score(features: dict, model_data: dict) -> tuple[float, dict, str, float]:
    heuristic_score, components = _heuristic_photo_score(features)
    photo_pipeline = model_data.get("photo_pipeline")
    if photo_pipeline is None:
        return heuristic_score, components, "heurístico", heuristic_score

    photo_feat_names = model_data.get("photo_feature_names") or (
        MODEL_PHOTO_FEATURE_NAMES + EMBEDDING_FEATURE_NAMES + SEMANTIC_EMBEDDING_FEATURE_NAMES
    )
    X_photo = np.array([[features[f] for f in photo_feat_names]], dtype=float)
    photo_model_prob = float(photo_pipeline.predict_proba(X_photo)[0][1])
    score = _clamp01(photo_model_prob * 0.80 + heuristic_score * 0.20)
    components["photo_model_probability"] = round(photo_model_prob, 4)
    return round(score, 4), components, "supervisionado", photo_model_prob


def _text_score(features: dict, text_pref: dict, model_prob: float, config: dict) -> float:
    pref_weight = float(config.get("model", {}).get("text_preference_weight", 0.40))
    pref_weight = _clamp01(pref_weight)
    model_weight = 1.0 - pref_weight
    text_pref_score = float(text_pref.get("text_preference_score", 0.5))
    return round(
        _clamp01(model_prob * model_weight + text_pref_score * pref_weight),
        4,
    )


def _confidence_cap(
    text_score: float,
    photo_score: float,
    text_model_prob: float,
    photo_raw_prob: float,
    features: dict,
    config: dict,
) -> tuple[float, str]:
    """
    Define o teto de confiança exibida/usada para evitar saltos artificiais.

    LightGBM + meta-modelo podem produzir logits muito extremos quando entram
    poucos exemplos novos de uma vez. O cap não muda o lado da decisão, só evita
    transformar evidência moderada em "99% de certeza".
    """
    cfg = config.get("model", {}).get("probability_safety", {}) or {}
    default_cap = float(cfg.get("max_confidence", 0.92))
    strong_cap = float(cfg.get("strong_agreement_confidence", 0.95))
    disagreement_cap = float(cfg.get("disagreement_confidence", 0.78))
    weak_cap = float(cfg.get("weak_evidence_confidence", 0.84))

    text_side = text_score >= 0.5
    photo_side = photo_score >= 0.5
    text_strength = abs(float(text_score) - 0.5)
    photo_strength = abs(float(photo_score) - 0.5)
    raw_text_strength = abs(float(text_model_prob) - 0.5)
    raw_photo_strength = abs(float(photo_raw_prob) - 0.5)
    has_face = float(features.get("photo_has_face", 0.0) or 0.0) > 0
    faces_ratio = float(features.get("photo_faces_ratio", 0.0) or 0.0)
    body_quality = float(features.get("photo_body_signal_quality", 0.0) or 0.0)

    if text_side != photo_side:
        return _clamp01(disagreement_cap), "foto/texto discordam"

    if (
        text_strength >= 0.20
        and photo_strength >= 0.20
        and raw_text_strength >= 0.20
        and raw_photo_strength >= 0.20
        and (has_face or body_quality >= 0.35)
    ):
        return _clamp01(strong_cap), "evidência forte e concordante"

    if (
        text_strength < 0.08
        or photo_strength < 0.08
        or (has_face and 0 < faces_ratio < 0.50)
    ):
        return _clamp01(weak_cap), "evidência moderada/fraca"

    return _clamp01(default_cap), "cap padrão"


def _apply_probability_safety(
    raw_prob: float,
    text_score: float,
    photo_score: float,
    text_model_prob: float,
    photo_raw_prob: float,
    features: dict,
    config: dict,
) -> tuple[float, dict]:
    cfg = config.get("model", {}).get("probability_safety", {}) or {}
    if not bool(cfg.get("enabled", True)):
        return round(_clamp01(raw_prob), 4), {
            "enabled": False,
            "raw_probability": round(_clamp01(raw_prob), 4),
            "applied": False,
        }

    raw_prob = _clamp01(raw_prob)
    cap, reason = _confidence_cap(
        text_score=text_score,
        photo_score=photo_score,
        text_model_prob=text_model_prob,
        photo_raw_prob=photo_raw_prob,
        features=features,
        config=config,
    )
    cap = max(0.51, min(0.99, cap))

    if raw_prob >= 0.5:
        safe_prob = min(raw_prob, cap)
    else:
        safe_prob = max(raw_prob, 1.0 - cap)

    applied = abs(safe_prob - raw_prob) > 1e-6
    return round(safe_prob, 4), {
        "enabled": True,
        "raw_probability": round(raw_prob, 4),
        "safe_probability": round(safe_prob, 4),
        "max_confidence": round(cap, 4),
        "reason": reason,
        "applied": applied,
    }


def _apply_race_affinity(raw_prob: float, features: dict, config: dict) -> tuple[float, dict]:
    """Aplica afinidades raciais configuradas como ajuste pequeno e explícito."""
    race_cfg = config.get("preferences", {}).get("race_affinity", {}) or {}
    if not isinstance(race_cfg, dict) or not race_cfg:
        return raw_prob, {}

    mapping = {
        "black": "photo_race_black",
        "white": "photo_race_white",
        "asian": "photo_race_asian",
        "indian": "photo_race_indian",
        "middleeastern": "photo_race_middleeastern",
        "middle_eastern": "photo_race_middleeastern",
        "latina": "photo_race_latina",
        "latino": "photo_race_latina",
    }
    adjustment = 0.0
    signals: dict[str, float] = {}
    for key, feat_name in mapping.items():
        try:
            affinity = float(race_cfg.get(key, 0.0) or 0.0)
            signal = _clamp01(float(features.get(feat_name, 0.0) or 0.0))
        except Exception:
            continue
        if affinity == 0.0 or signal <= 0.0:
            continue
        adjustment += affinity * signal
        signals[key] = round(signal, 4)

    adjustment = max(-0.20, min(0.20, adjustment))
    adjusted = _clamp01(raw_prob + adjustment)
    return adjusted, {
        "applied": abs(adjusted - raw_prob) > 1e-6,
        "delta": round(adjusted - raw_prob, 4),
        "signals": signals,
    }


def _gender_compatibility_filter(features: dict, config: dict) -> dict:
    """Usa gênero inferido só como compatibilidade/filtro de baixa confiança."""
    cfg = config.get("model", {}).get("gender_compatibility", {}) or {}
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "forced_pass": False}

    target = str(cfg.get("target", "woman") or "woman").strip().lower()
    has_face = float(features.get("photo_has_face", 0.0) or 0.0) > 0
    woman_conf = _clamp01(float(features.get("photo_woman_confidence", 0.5) or 0.5))
    gender_certainty = _clamp01(
        float(features.get("photo_gender_certainty", abs(woman_conf - 0.5) * 2) or 0.0)
    )
    min_certainty = _clamp01(float(cfg.get("min_certainty_for_filter", 0.65) or 0.65))
    details = {
        "enabled": True,
        "forced_pass": False,
        "target": target,
        "woman_confidence": round(woman_conf, 4),
        "gender_certainty": round(gender_certainty, 4),
    }
    if not has_face or gender_certainty < min_certainty:
        return details

    if target in {"woman", "female", "mulher"}:
        min_woman = _clamp01(float(cfg.get("min_woman_confidence", 0.20) or 0.20))
        if woman_conf <= min_woman:
            details.update({
                "forced_pass": True,
                "reason": "gênero visual incompatível com o alvo configurado",
            })
    elif target in {"man", "male", "homem"}:
        max_woman = _clamp01(float(cfg.get("max_woman_confidence", 0.80) or 0.80))
        if woman_conf >= max_woman:
            details.update({
                "forced_pass": True,
                "reason": "gênero visual incompatível com o alvo configurado",
            })
    return details


def _apply_distance_preference(raw_prob: float, features: dict, config: dict) -> tuple[float, dict]:
    """Ajuste suave por distância configurada; não age quando distância veio ausente."""
    if float(features.get("distance_missing", 1.0) or 0.0) >= 1.0:
        return raw_prob, {}

    weight = float(config.get("model", {}).get("distance_adjustment_weight", 0.0) or 0.0)
    if weight <= 0:
        return raw_prob, {}

    try:
        distance_score = _clamp01(float(features.get("distance_score", 0.5)))
    except Exception:
        distance_score = 0.5
    delta = (distance_score - 0.5) * _clamp01(weight)
    adjusted = _clamp01(raw_prob + delta)
    if abs(delta) <= 0.001:
        return adjusted, {}
    return adjusted, {
        "distance_km": round(float(features.get("distance_km", 0.0) or 0.0), 1),
        "distance_score": round(distance_score, 4),
        "delta": round(delta, 4),
    }


def _feature_importances(pipeline: Pipeline | None, feature_names: list[str]) -> dict[str, float]:
    if pipeline is None:
        return {}
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return {}
    return dict(zip(feature_names, clf.feature_importances_))


def _meta_display_weights(meta_pipeline) -> tuple[float, float]:
    """Aproxima pesos foto/texto a partir dos coeficientes do meta-modelo."""
    pipeline = meta_pipeline
    # CalibratedClassifierCV não tem named_steps — acessa o estimador base
    if hasattr(meta_pipeline, "calibrated_classifiers_") and meta_pipeline.calibrated_classifiers_:
        pipeline = meta_pipeline.calibrated_classifiers_[0].estimator
    if not hasattr(pipeline, "named_steps"):
        return 0.5, 0.5
    clf = pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "coef_"):
        return 0.5, 0.5

    coefs = np.abs(clf.coef_[0])
    if len(coefs) < 2:
        return 0.5, 0.5

    text_weight = float(coefs[0])
    photo_weight = float(coefs[1:].sum())
    total = photo_weight + text_weight
    if total <= 0:
        return 0.5, 0.5
    return photo_weight / total, text_weight / total


def _meta_feature_row(text_prob: float, photo_prob: float, features: dict, names: list[str]) -> list[float]:
    values = {
        "text_model_prob": float(text_prob),
        "photo_model_prob": float(photo_prob),
    }
    for name in names:
        if name in values:
            continue
        default = 0.5 if name in {"photo_face_similarity", "photo_faces_ratio", "photo_image_sharpness"} else 0.0
        values[name] = float(features.get(name, default) or default)
    return [values[name] for name in names]


def predict(profile: dict, model_data: dict | None = None) -> dict:
    """
    Recebe um perfil e retorna decisao, probabilidade, scores e explicacoes.
    """
    if model_data is None:
        model_data = load_model()
    if model_data is None:
        logger.error("predict chamado sem modelo treinado")
        raise RuntimeError("Modelo não treinado. Rode train_model() primeiro.")

    logger.debug("Predicao iniciada: name=%r age=%r", profile.get("name"), profile.get("age"))
    config = load_config()

    pca = model_data.get("pca")
    if pca is not None:
        from photo_embedding_pca import enrich_photo_features
        photo_features = profile.get("_photo_features")
        if isinstance(photo_features, dict):
            enrich_photo_features(photo_features, pca)

    semantic_pca = model_data.get("semantic_pca")
    if semantic_pca is not None:
        from photo_semantic_embeddings import enrich_photo_features as enrich_semantic_photo_features
        photo_features = profile.get("_photo_features")
        if isinstance(photo_features, dict):
            enrich_semantic_photo_features(photo_features, semantic_pca)

    bio_pca_tuple = model_data.get("bio_pca")
    if bio_pca_tuple is not None:
        from bio_embedding import text_to_features as _bio_text_to_features
        bio_pca, int_pca = bio_pca_tuple
        bio = profile.get("bio", "") or ""
        interests = profile.get("interests", []) or []
        profile["_text_features"] = _bio_text_to_features(bio, interests, bio_pca, int_pca)

    features = extract_features(profile, config)

    text_pipeline = model_data["text_pipeline"]
    X_text = np.array([[features[f] for f in TEXT_FEATURE_NAMES]], dtype=float)
    text_model_prob = float(text_pipeline.predict_proba(X_text)[0][1])
    text_pref = score_profile_text(
        profile.get("bio", ""),
        profile.get("interests", []),
        profile.get("_descriptors") or profile.get("descriptors") or {},
    )

    photo_score, photo_components, photo_score_mode, photo_raw_prob = _photo_score(features, model_data)
    superlike_probability = None
    superlike_pipeline = model_data.get("superlike_pipeline")
    superlike_feature_names = model_data.get("superlike_feature_names") or []
    if superlike_pipeline is not None and superlike_feature_names:
        try:
            X_super = np.array([[features[f] for f in superlike_feature_names]], dtype=float)
            superlike_probability = round(_clamp01(float(superlike_pipeline.predict_proba(X_super)[0][1])), 4)
        except Exception:
            logger.exception("Falha ao calcular probabilidade de superlike: name=%r", profile.get("name"))
            superlike_probability = None

    meta_pipeline = model_data.get("meta_pipeline")
    if meta_pipeline is not None:
        meta_names = model_data.get("meta_feature_names") or MODEL_META_FEATURE_NAMES
        meta_X = np.array([_meta_feature_row(text_model_prob, photo_raw_prob, features, meta_names)], dtype=float)
        raw_prob = _clamp01(float(meta_pipeline.predict_proba(meta_X)[0][1]))
        text_score = _text_score(features, text_pref, text_model_prob, config)
        photo_weight, text_weight = _meta_display_weights(meta_pipeline)
    else:
        text_score = _text_score(features, text_pref, text_model_prob, config)
        model_cfg = config.get("model", {})
        photo_weight = float(model_cfg.get("photo_weight", 0.65))
        text_weight = float(model_cfg.get("text_weight", 0.35))
        total = photo_weight + text_weight
        if total <= 0:
            photo_weight, text_weight, total = 0.65, 0.35, 1.0
        photo_weight /= total
        text_weight /= total
        raw_prob = _clamp01(photo_score * photo_weight + text_score * text_weight)

    raw_prob, race_affinity_applied = _apply_race_affinity(raw_prob, features, config)
    raw_prob, distance_adjustment = _apply_distance_preference(raw_prob, features, config)

    prob, probability_safety = _apply_probability_safety(
        raw_prob=raw_prob,
        text_score=text_score,
        photo_score=photo_score,
        text_model_prob=text_model_prob,
        photo_raw_prob=photo_raw_prob,
        features=features,
        config=config,
    )

    gender_compatibility = _gender_compatibility_filter(features, config)
    policy = apply_decision_policy(
        prob,
        config=config,
        model_data=model_data,
        forced_pass=bool(gender_compatibility.get("forced_pass")),
        filter_reason=str(gender_compatibility.get("reason", "") or ""),
    )
    decision = policy["decision"]
    logger.info(
        "Predicao concluida: name=%r decision=%s prob=%.4f raw_prob=%.4f safety=%s text=%.4f photo=%.4f mode=%s meta=%s",
        profile.get("name"),
        decision,
        prob,
        float(probability_safety.get("raw_probability", prob)),
        probability_safety.get("reason", "disabled") if probability_safety.get("applied") else "none",
        text_score,
        photo_score,
        photo_score_mode,
        meta_pipeline is not None,
    )

    text_importances = _feature_importances(text_pipeline, TEXT_FEATURE_NAMES)
    photo_feat_names_for_imp = model_data.get("photo_feature_names") or (
        MODEL_PHOTO_FEATURE_NAMES + EMBEDDING_FEATURE_NAMES + SEMANTIC_EMBEDDING_FEATURE_NAMES
    )
    photo_importances = _feature_importances(model_data.get("photo_pipeline"), photo_feat_names_for_imp)
    importances = {}
    for feat_name, importance in text_importances.items():
        importances[feat_name] = importances.get(feat_name, 0.0) + importance * text_weight
    for feat_name, importance in photo_importances.items():
        importances[feat_name] = importances.get(feat_name, 0.0) + importance * photo_weight
    total_importance = sum(importances.values())
    if total_importance > 0:
        importances = {k: v / total_importance for k, v in importances.items()}

    return {
        "decision": decision,
        "probability": prob,
        "raw_probability": round(float(probability_safety.get("raw_probability", prob)), 4),
        "probability_safety": probability_safety,
        "features": features,
        "importances": importances,
        "model_type": model_data["model_type"],
        "n_samples": model_data["n_samples"],
        "training_version": model_data.get("training_version"),
        "text_model_type": model_data.get("text_model_type", "?"),
        "photo_model_type": model_data.get("photo_model_type", "Heurístico"),
        "text_n_samples": model_data.get("text_n_samples", model_data["n_samples"]),
        "photo_n_samples": model_data.get("photo_n_samples", 0),
        "text_model_probability": round(text_model_prob, 4),
        "text_score": text_score,
        "photo_score": photo_score,
        "photo_score_mode": photo_score_mode,
        "text_preference_score": round(float(text_pref.get("text_preference_score", 0.5)), 4),
        "text_preference": text_pref,
        "photo_components": photo_components,
        "race_affinity": race_affinity_applied,
        "gender_compatibility": gender_compatibility,
        "distance_adjustment": distance_adjustment,
        "decision_policy": policy,
        "like_threshold": policy["like_threshold"],
        "in_review_band": policy["in_review_band"],
        "ranking_score": policy["ranking_score"],
        "review_priority": policy["review_priority"],
        "preference_tier": policy["preference_tier"],
        "correction_type": policy["correction_type"],
        "superlike_probability": superlike_probability,
        "superlike_model_type": model_data.get("superlike_model_type", "indisponível"),
        "superlike_n_samples": model_data.get("superlike_n_samples", 0),
        "superlike_positive_samples": model_data.get("superlike_positive_samples", 0),
        "weights": {
            "photo_weight": round(photo_weight, 4),
            "text_weight": round(text_weight, 4),
        },
    }
