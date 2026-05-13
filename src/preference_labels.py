"""Helpers para rótulos ricos, correções e prioridade de revisão."""

from __future__ import annotations

import json


LIKE_DECISIONS = {"CURTIR", "SUPER_LIKE", "SUPER LIKE"}
PASS_DECISIONS = {"NÃO CURTIR", "NAO CURTIR", "PASSAR", "PASS"}
MAYBE_DECISIONS = {"TALVEZ", "MAYBE"}


def normalize_decision(decision: object) -> str:
    value = str(decision or "").strip().upper()
    if value in {"SUPERLIKE", "SUPER-CURTIR"}:
        return "SUPER_LIKE"
    if value in {"NAO CURTIR", "NÃO CURTIR", "PASSAR", "PASS"}:
        return "NÃO CURTIR"
    if value in {"TALVEZ", "MAYBE"}:
        return "TALVEZ"
    if value in {"CURTIR", "LIKE"}:
        return "CURTIR"
    return value


def is_like_decision(decision: object) -> bool:
    return normalize_decision(decision) in LIKE_DECISIONS


def is_super_like_decision(decision: object) -> bool:
    return normalize_decision(decision) in {"SUPER_LIKE", "SUPER LIKE"}


def is_maybe_decision(decision: object) -> bool:
    return normalize_decision(decision) in MAYBE_DECISIONS


def decision_to_label(decision: object) -> str:
    normalized = normalize_decision(decision)
    if not normalized:
        return ""
    if is_maybe_decision(decision):
        return ""
    return "1" if normalized in LIKE_DECISIONS else "0"


def _details(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def infer_preference_tier(row_or_profile: dict, final_decision: object = "") -> str:
    explicit = str(row_or_profile.get("preference_tier", "") or "").strip()
    if explicit in {"strong_like", "like", "maybe", "pass", "filtered_pass"}:
        return explicit

    details = _details(row_or_profile.get("feedback_details", ""))
    decision = normalize_decision(final_decision or row_or_profile.get("final_decision") or row_or_profile.get("original_label"))
    label = str(row_or_profile.get("label", "") or "").strip()
    filter_reason = str(row_or_profile.get("_filter_reason") or row_or_profile.get("feedback_reason") or "").lower()

    if is_maybe_decision(decision) or label.lower() == "maybe":
        return "maybe"
    if is_super_like_decision(decision) or details.get("target_action") == "super_like":
        return "strong_like"
    if "filtro_absoluto" in filter_reason or row_or_profile.get("_skip_prompt"):
        return "filtered_pass"
    if label in {"1", "1.0"} or decision == "CURTIR":
        return "like"
    if label in {"0", "0.0"} or decision == "NÃO CURTIR":
        return "pass"
    return ""


def infer_correction_type(
    row_or_profile: dict,
    ai_decision: object = "",
    final_decision: object = "",
) -> str:
    explicit = str(row_or_profile.get("correction_type", "") or "").strip()
    if explicit in {"none", "false_positive", "false_negative", "superlike_upgrade", "uncertain"}:
        return explicit

    tier = infer_preference_tier(row_or_profile, final_decision)
    if tier == "maybe":
        return "uncertain"

    ai = normalize_decision(ai_decision or row_or_profile.get("ai_decision") or row_or_profile.get("original_label"))
    final = normalize_decision(final_decision or row_or_profile.get("final_decision") or row_or_profile.get("label"))

    if is_super_like_decision(final) or tier == "strong_like":
        return "superlike_upgrade" if not is_super_like_decision(ai) else "none"
    if ai == "CURTIR" and final == "NÃO CURTIR":
        return "false_positive"
    if ai == "NÃO CURTIR" and final == "CURTIR":
        return "false_negative"
    return "none"


def clamp01(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def review_priority_from_probability(probability: object, correction_type: str = "") -> float:
    prob = clamp01(probability, 0.5)
    uncertainty = 1.0 - abs(prob - 0.5) * 2.0
    priority = 0.35 + 0.55 * uncertainty
    if correction_type in {"false_positive", "false_negative", "uncertain"}:
        priority = max(priority, 0.90)
    if correction_type == "superlike_upgrade":
        priority = max(priority, 0.75)
    return round(clamp01(priority, 0.5), 4)
