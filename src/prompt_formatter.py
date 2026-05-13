"""Formatacao do prompt interativo exibido no terminal."""

from __future__ import annotations

import os
import re
import unicodedata

from config import load_config
from features import FEATURE_LABELS


USE_COLOR = os.environ.get("NO_COLOR", "") == ""


class C:
    reset = "\033[0m" if USE_COLOR else ""
    bold = "\033[1m" if USE_COLOR else ""
    dim = "\033[2m" if USE_COLOR else ""
    green = "\033[32m" if USE_COLOR else ""
    red = "\033[31m" if USE_COLOR else ""
    yellow = "\033[33m" if USE_COLOR else ""
    cyan = "\033[36m" if USE_COLOR else ""


def _paint(text: str, color: str) -> str:
    return f"{color}{text}{C.reset}" if color else text


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFD", (text or "").strip().lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", value)


def _truncate(text: str, limit: int = 110) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def _score_badge(label: str, value: float, positive_high: bool = True) -> str:
    value = max(0.0, min(1.0, float(value)))
    pct = int(value * 100)
    if positive_high:
        color = C.green if value >= 0.65 else C.red if value <= 0.35 else C.yellow
    else:
        color = C.green if value <= 0.35 else C.red if value >= 0.65 else C.yellow
    return _paint(f"{label}={pct}%", color)


def _format_photo(photo: dict, ml: dict) -> str:
    if photo.get("_analysis_failed"):
        reason = photo.get("_failure_reason") or "indisponível"
        failed = photo.get("_photos_failed", 0)
        analyzed = photo.get("_photos_analyzed", 0)
        suffix = f" ({failed}/{analyzed} falharam)" if analyzed else ""
        return _paint(f"foto indisponível: {reason}{suffix}; usando neutro", C.yellow)

    if not photo.get("photo_has_face"):
        return _paint("sem rosto detectado", C.red)

    parts = [
        _score_badge("mulher", photo.get("photo_woman_confidence", 0.5)),
        _score_badge("similar", photo.get("photo_face_similarity", 0.5)),
        _score_badge("tom", photo.get("photo_skin_lightness", 0.5), positive_high=False),
    ]
    race = photo.get("_dominant_race")
    if race:
        parts.append(f"etnia={race}")

    try:
        smile = float(photo.get("photo_face_smile_score", 0.5) or 0.5)
        if abs(smile - 0.5) >= 0.10:
            parts.append(f"expressão={int(smile * 100)}%")
    except Exception:
        pass

    faces = photo.get("_faces_found")
    analyzed = photo.get("_photos_analyzed")
    if analyzed:
        face_color = C.green if faces else C.red
        parts.append(_paint(f"rostos={faces}/{analyzed}", face_color))

    body_quality = float(photo.get("photo_body_signal_quality", 0.0) or 0.0)
    if body_quality > 0.05:
        body_visible = float(photo.get("photo_body_visible", 0.0) or 0.0)
        body_width = float(photo.get("photo_body_width_ratio", 0.5) or 0.5)
        parts.append(_score_badge("corpo visível", body_visible))
        if body_quality >= 0.35:
            narrow_val = float(photo.get("photo_body_width_bucket_narrow", 0.0) or 0.0)
            medium_val = float(photo.get("photo_body_width_bucket_medium", 0.0) or 0.0)
            wide_val = float(photo.get("photo_body_width_bucket_wide", 0.0) or 0.0)
            if narrow_val > 0 and medium_val > 0 and wide_val == 0:
                parts.append("silhueta=média-estreita")
            elif medium_val > 0 and wide_val > 0 and narrow_val == 0:
                parts.append("silhueta=média-ampla")
            elif narrow_val > 0:
                parts.append("silhueta=estreita")
            elif medium_val > 0:
                parts.append("silhueta=média")
            elif wide_val > 0:
                parts.append("silhueta=ampla")
            else:
                parts.append(_score_badge("largura visual", body_width))
        else:
            parts.append(_score_badge("largura visual", body_width))

    if photo.get("_photo_analysis_incomplete"):
        failed = photo.get("_photos_failed", 0)
        timed_out = photo.get("_photos_timed_out", 0)
        parts.append(_paint(f"parcial: {failed} falha(s), {timed_out} timeout(s)", C.yellow))

    photo_score = ml.get("photo_score")
    if photo_score is not None:
        parts.append(_score_badge("score foto", photo_score))

    return "  ".join(parts)


def _format_interests(profile: dict, ml: dict, limit: int = 8) -> tuple[str, list[str]]:
    config = load_config()
    prefs = config.get("preferences", {})
    preferred = {_norm(x) for x in prefs.get("preferred_interests", [])}
    text_pref = ml.get("text_preference", {})
    learned_pos = {_norm(x) for x in text_pref.get("interest_pref_positive_signals", [])}
    learned_neg = {_norm(x) for x in text_pref.get("interest_pref_negative_signals", [])}

    rendered = []
    signals = []
    for raw in profile.get("interests") or []:
        key = _norm(raw)
        if key in learned_pos or key in preferred:
            rendered.append(_paint(f"+{raw}", C.green))
            signals.append(f"+ interesse compatível: {raw}")
        elif key in learned_neg:
            rendered.append(_paint(f"-{raw}", C.red))
            signals.append(f"- interesse costuma pesar contra: {raw}")
        else:
            rendered.append(raw)

    if not rendered:
        return "", signals

    suffix = "..." if len(rendered) > limit else ""
    return ", ".join(rendered[:limit]) + suffix, signals[:4]


def _format_bio_signals(profile: dict, ml: dict) -> list[str]:
    config = load_config()
    prefs = config.get("preferences", {})
    bio_norm = _norm(profile.get("bio") or "")
    text_pref = ml.get("text_preference", {})

    positives = []
    negatives = []

    for keyword in prefs.get("positive_bio_keywords", []):
        if _norm(keyword) and _norm(keyword) in bio_norm:
            positives.append(keyword)
    for keyword in prefs.get("negative_bio_keywords", []):
        if _norm(keyword) and _norm(keyword) in bio_norm:
            negatives.append(keyword)

    positives.extend(text_pref.get("bio_pref_positive_signals", []))
    negatives.extend(text_pref.get("bio_pref_negative_signals", []))

    lines = []
    if positives:
        lines.append(_paint("+ bio compatível: " + ", ".join(dict.fromkeys(positives).keys())[:80], C.green))
    if negatives:
        lines.append(_paint("- bio alerta: " + ", ".join(dict.fromkeys(negatives).keys())[:80], C.red))
    return lines


def _format_desc(profile: dict) -> str:
    desc = profile.get("_descriptors") or {}
    desc_keys = [
        "Pets", "Você fuma?", "Bebida", "Família", "Atividade física",
        "Formação", "Signo", "Linguagem do amor",
    ]
    parts = [f"{k}: {desc[k]}" for k in desc_keys if k in desc]
    return " | ".join(parts[:4])


def _format_top_factors(ml: dict, limit: int = 4) -> str:
    importances = ml.get("importances", {})
    features = ml.get("features", {})
    if not importances:
        return ""

    rendered = []
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:limit]:
        label = FEATURE_LABELS.get(feat, feat)
        val = features.get(feat, "?")
        if isinstance(val, float):
            val = f"{val:.2f}"
        rendered.append(f"{label}: {val} ({imp * 100:.0f}%)")
    return " / ".join(rendered)


def format_interactive_prompt(
    profile: dict,
    ai_decision: str,
    current_label: str,
    sync_ok: bool,
) -> str:
    """Monta o bloco do prompt interativo, com sinais destacados."""
    ml = profile.get("_ml_result", {})
    prob = ml.get("probability", 0.5)
    conf = int(prob * 100) if ai_decision == "CURTIR" else int((1 - prob) * 100)
    rec = "CURTIR" if ai_decision == "CURTIR" else "NÃO CURTIR"
    rec_color = C.green if ai_decision == "CURTIR" else C.red
    sync_color = C.green if sync_ok else C.red

    dist = profile.get("distance_km", profile.get("_distance_km"))
    dist_str = f"  |  {dist} km" if dist else ""
    interests_line, interest_signals = _format_interests(profile, ml)
    bio = _truncate(profile.get("bio") or "")
    desc = _format_desc(profile)
    factors = _format_top_factors(ml)

    text_score = ml.get("text_score", 0.5)
    photo_score = ml.get("photo_score", 0.5)
    text_pref_score = ml.get("text_preference_score", 0.5)
    superlike_score = ml.get("superlike_probability")
    safety = ml.get("probability_safety", {}) or {}
    confidence_note = " calibrado" if safety.get("applied") else ""
    score_line = (
        f"     Scores  : {_score_badge('foto', photo_score)}  "
        f"{_score_badge('texto', text_score)}  "
        f"{_score_badge('texto aprendido', text_pref_score)}"
    )
    if superlike_score is not None and ai_decision == "CURTIR":
        score_line += "  " + _score_badge("super like", superlike_score)

    lines = [
        "",
        f"  {'-' * 62}",
        f"  {C.bold}-> {profile.get('name','?')}, {profile.get('age','?')} anos{dist_str}{C.reset}",
        f"     Tela    : {_paint(current_label, sync_color)}  [{_paint('ok' if sync_ok else 'diferente', sync_color)}]",
        f"     Foto    : {_format_photo(profile.get('_photo_features', {}), ml)}",
        score_line,
    ]

    if bio:
        lines.append(f'     Bio     : "{bio}"')
    if interests_line:
        lines.append(f"     Inter.  : {interests_line}")
    if desc:
        lines.append(f"     Desc    : {desc}")

    signals = interest_signals + _format_bio_signals(profile, ml)
    if signals:
        lines.append("     Sinais  : " + " | ".join(signals[:4]))
    if factors:
        lines.append(f"     Motivos : {factors}")

    lines.extend([
        f"     IA      : {_paint(rec, rec_color)}  ({conf}%{confidence_note})",
        "",
        "  Escolha: [c] curtir  [p] passar  [Enter] aceitar IA  [n] inverter",
        "  Depois informe o motivo principal: [f] foto (+ detalhe)  [i] interesses  [b] bio  [d] descritores  [o/Enter] sem certeza",
        "  > ",
    ])
    return "\n".join(lines)
