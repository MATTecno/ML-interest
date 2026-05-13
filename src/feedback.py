"""Normalizacao dos sinais de feedback usados no treino."""

from __future__ import annotations

import json


DOMAIN_LABELS = {
    "photo": "foto",
    "interests": "interesses",
    "bio": "bio",
    "descriptors": "descritores",
    "other": "outro/sem certeza",
}

PHOTO_REASON_LABELS = {
    "photo_face": "rosto",
    "photo_gender": "gênero/apresentação visual",
    "photo_body": "corpo",
    "photo_context": "contexto/cenário",
    "photo_style": "estilo/pose",
    "photo_general": "foto geral/sem certeza",
}

PHOTO_KEY_TO_REASON = {
    "r": "photo_face",
    "c": "photo_body",
    "a": "photo_context",
    "e": "photo_style",
    "o": "photo_general",
    "": "photo_general",
}

KEY_TO_DOMAIN = {
    "f": "photo",
    "i": "interests",
    "b": "bio",
    "d": "descriptors",
    "o": "other",
    "": "other",
}


def normalize_photo_reason(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    aliases = {
        "photo_face": "photo_face",
        "face": "photo_face",
        "rosto": "photo_face",
        "cara": "photo_face",
        "photo_gender": "photo_gender",
        "gender": "photo_gender",
        "genero": "photo_gender",
        "gênero": "photo_gender",
        "feminilidade": "photo_gender",
        "masculino": "photo_gender",
        "masculina": "photo_gender",
        "ambigua": "photo_gender",
        "ambígua": "photo_gender",
        "photo_body": "photo_body",
        "body": "photo_body",
        "corpo": "photo_body",
        "shape": "photo_body",
        "photo_context": "photo_context",
        "context": "photo_context",
        "contexto": "photo_context",
        "cenario": "photo_context",
        "cenário": "photo_context",
        "fundo": "photo_context",
        "ambiente": "photo_context",
        "photo_style": "photo_style",
        "style": "photo_style",
        "estilo": "photo_style",
        "pose": "photo_style",
        "qualidade": "photo_style",
        "photo_general": "photo_general",
        "foto": "photo_general",
        "photo": "photo_general",
        "geral": "photo_general",
        "sem certeza": "photo_general",
    }
    if value in aliases:
        return aliases[value]
    if any(token in value for token in ["genero", "gênero", "feminilidade", "masculin", "ambigu", "homem"]):
        return "photo_gender"
    if any(token in value for token in ["rosto", "face", "cara"]):
        return "photo_face"
    if any(token in value for token in ["corpo", "body", "shape"]):
        return "photo_body"
    if any(token in value for token in ["context", "cenario", "cenário", "fundo", "ambiente", "lugar"]):
        return "photo_context"
    if any(token in value for token in ["estilo", "pose", "roupa", "qualidade", "style"]):
        return "photo_style"
    return "photo_general"


def normalize_domain(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    aliases = {
        "foto": "photo",
        "photo": "photo",
        "fotos": "photo",
        "interesse": "interests",
        "interesses": "interests",
        "interest": "interests",
        "interests": "interests",
        "bio": "bio",
        "descritor": "descriptors",
        "descritores": "descriptors",
        "descriptor": "descriptors",
        "descriptors": "descriptors",
        "outro": "other",
        "other": "other",
        "sem certeza": "other",
        "incerto": "other",
        "uncertain": "other",
        "mixed": "other",
    }
    return aliases.get(value, "other")


def normalize_intensity(raw: str | int | None, domain: str) -> str:
    value = str(raw or "").strip()
    if value in {"1", "2", "3"}:
        return value
    return "1" if normalize_domain(domain) == "other" else "2"


def _parse_descriptors(profile: dict) -> dict:
    raw = profile.get("_descriptors") or profile.get("descriptors") or {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def default_reason(profile: dict, domain: str) -> str:
    domain = normalize_domain(domain)
    if domain == "photo":
        return "photo_general"

    if domain == "interests":
        interests = profile.get("interests") or []
        if isinstance(interests, str):
            interests = [x.strip() for x in interests.split(",") if x.strip()]
        return ", ".join(str(x) for x in interests[:3]) or "interesses"

    if domain == "bio":
        bio = str(profile.get("bio") or "").strip().replace("\n", " ")
        return bio[:80] + ("..." if len(bio) > 80 else "") if bio else "bio vazia"

    if domain == "descriptors":
        descriptors = _parse_descriptors(profile)
        for key, value in descriptors.items():
            if key or value:
                return f"{key}: {value}"
        return "descritores"

    return "sem certeza"


def normalize_feedback(
    profile: dict,
    final_decision: str,
    feedback_domain: str | None = "",
    feedback_reason: str | None = "",
    feedback_intensity: str | int | None = "",
    ai_decision: str | None = None,
) -> dict:
    domain = normalize_domain(feedback_domain)
    intensity = normalize_intensity(feedback_intensity, domain)
    reason = (feedback_reason or "").strip() or default_reason(profile, domain)
    if domain == "photo":
        reason = normalize_photo_reason(reason)
    final_value = str(final_decision or "").strip().upper()
    if final_value in {"TALVEZ", "MAYBE"}:
        final = "TALVEZ"
    else:
        final = "CURTIR" if final_value in {"CURTIR", "SUPER_LIKE", "SUPER LIKE"} else "NÃO CURTIR"
    corrected = ""
    if ai_decision:
        corrected = 1 if final != ai_decision else 0

    data = {
        "feedback_domain": domain,
        "feedback_reason": reason,
        "feedback_intensity": intensity,
        "feedback_sentiment": "neutral" if final == "TALVEZ" else ("positive" if final == "CURTIR" else "negative"),
    }
    if corrected != "":
        data["manual_corrected"] = corrected
    return data
