"""Política central de decisão, limiar calibrado e prioridade de review."""

from __future__ import annotations

from preference_labels import clamp01, review_priority_from_probability


DEFAULT_POLICY = {
    "mode": "recall_safe",
    "fallback_like_threshold": 0.48,
    "review_band": [0.40, 0.60],
    "threshold_optimizer": "maximize_recall_with_precision_floor",
    "precision_floor": 0.55,
    "target_like_recall": 0.88,
}


def policy_config(config: dict | None) -> dict:
    model_cfg = (config or {}).get("model", {}) or {}
    user_cfg = model_cfg.get("decision_policy", {}) or {}
    merged = {**DEFAULT_POLICY, **user_cfg}
    try:
        lo, hi = merged.get("review_band") or DEFAULT_POLICY["review_band"]
        merged["review_band"] = [float(lo), float(hi)]
    except Exception:
        merged["review_band"] = list(DEFAULT_POLICY["review_band"])
    merged["fallback_like_threshold"] = float(
        merged.get("fallback_like_threshold", DEFAULT_POLICY["fallback_like_threshold"])
        or DEFAULT_POLICY["fallback_like_threshold"]
    )
    return merged


def threshold_from_evaluation(model_data: dict | None, config: dict | None) -> float:
    cfg = policy_config(config)
    fallback = clamp01(cfg.get("fallback_like_threshold", 0.48), 0.48)
    evaluation = (model_data or {}).get("evaluation") or {}
    thresholds = evaluation.get("thresholds") or {}
    value = thresholds.get("like_threshold")
    try:
        threshold = float(value)
    except Exception:
        return fallback
    if 0.45 <= threshold <= 0.55:
        return threshold
    return fallback


def apply_decision_policy(
    probability: float,
    *,
    config: dict | None = None,
    model_data: dict | None = None,
    forced_pass: bool = False,
    filter_reason: str = "",
) -> dict:
    cfg = policy_config(config)
    prob = clamp01(probability, 0.5)
    threshold = threshold_from_evaluation(model_data, config)
    band_low, band_high = cfg["review_band"]
    in_review_band = band_low <= prob <= band_high

    if forced_pass:
        decision = "NÃO CURTIR"
        preference_tier = "filtered_pass"
        correction_type = "none"
    else:
        decision = "CURTIR" if prob >= threshold else "NÃO CURTIR"
        preference_tier = "like" if decision == "CURTIR" else "pass"
        correction_type = "uncertain" if in_review_band else "none"

    ranking_score = round(prob, 4)
    review_priority = review_priority_from_probability(prob, correction_type)
    if in_review_band:
        review_priority = max(review_priority, 0.85)

    return {
        "decision": decision,
        "like_threshold": round(threshold, 4),
        "policy_mode": cfg.get("mode", "recall_safe"),
        "review_band": [round(float(band_low), 4), round(float(band_high), 4)],
        "in_review_band": bool(in_review_band),
        "ranking_score": ranking_score,
        "review_priority": round(clamp01(review_priority, 0.5), 4),
        "preference_tier": preference_tier,
        "correction_type": correction_type,
        "filter_reason": filter_reason if forced_pass else "",
    }
