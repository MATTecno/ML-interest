"""Conversao do JSON do Tinder para o formato interno de perfil."""

from __future__ import annotations

from datetime import date, datetime

from config import get_photos_config
from logging_config import get_logger

logger = get_logger(__name__)


def calc_age(birth_date_str: str) -> int:
    if not birth_date_str or not str(birth_date_str).strip():
        return 0
    try:
        birth = datetime.fromisoformat(str(birth_date_str).replace("Z", "+00:00"))
        today = date.today()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except Exception:
        logger.warning("Falha ao calcular idade: birth_date=%r", birth_date_str)
        return 0


def extract_descriptors(selected_descriptors: list) -> dict:
    result = {}
    for d in selected_descriptors:
        if not isinstance(d, dict):
            continue
        name = d.get("name") or d.get("section_name") or d.get("prompt") or d.get("id", "")
        name = str(name or "").strip()
        if not name:
            continue

        choices = d.get("choice_selections", [])
        if choices:
            values = [
                str(choice.get("name") or "").strip()
                for choice in choices
                if isinstance(choice, dict) and str(choice.get("name") or "").strip()
            ]
            if values:
                result[name] = ", ".join(values)
            continue

        measurable = d.get("measurable_selection")
        if isinstance(measurable, dict) and measurable.get("value") not in ("", None):
            unit = str(measurable.get("unit_of_measure") or "").strip()
            value = measurable.get("value")
            result[name] = f"{value} {unit}".strip()
    return result


def _value_text(value) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_value_text(item) for item in value]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        choices = value.get("choice_selections")
        if isinstance(choices, list):
            text = _value_text([
                choice.get("name") if isinstance(choice, dict) else choice
                for choice in choices
            ])
            if text:
                return text
        measurable = value.get("measurable_selection")
        if isinstance(measurable, dict) and measurable.get("value") not in ("", None):
            unit = str(measurable.get("unit_of_measure") or "").strip()
            return f"{measurable.get('value')} {unit}".strip()
        for key in (
            "name",
            "value",
            "display_value",
            "display_text",
            "body_text",
            "subtitle",
            "description",
            "text",
            "title_text",
        ):
            text = _value_text(value.get(key))
            if text:
                return text
    return ""


def _add_descriptor(result: dict, key: str, value: str) -> None:
    clean_key = str(key or "").strip()
    clean_value = str(value or "").strip()
    if not clean_key or not clean_value or clean_key == clean_value:
        return
    if len(clean_key) > 80 or len(clean_value) > 300:
        return
    if clean_key not in result:
        result[clean_key] = clean_value


def _extract_profile_detail_content(content, result: dict) -> None:
    """Extrai chips/textos estruturados como 'Informações básicas'."""
    def walk(node, section: str = "") -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, section)
            return
        if not isinstance(node, dict):
            return

        label = (
            node.get("name")
            or node.get("title")
            or node.get("title_text")
            or node.get("prompt")
            or node.get("section_name")
            or node.get("page_content_id")
            or section
        )
        value = ""
        for key in ("body_text", "value", "display_value", "display_text", "subtitle", "description", "text"):
            value = _value_text(node.get(key))
            if value:
                break
        if not value and isinstance(node.get("choice_selections"), list):
            value = _value_text(node.get("choice_selections"))
        if label and value:
            prefix = section if section and section != label else ""
            _add_descriptor(result, f"{prefix}: {label}" if prefix else str(label), value)

        next_section = str(node.get("section_name") or node.get("title") or node.get("page_content_id") or section or "").strip()
        for key in ("items", "contents", "content", "fields", "children", "sections", "page_content"):
            child = node.get(key)
            if isinstance(child, (list, dict)):
                walk(child, next_section)

    walk(content)


def extract_basic_info(user: dict) -> dict:
    """Campos extras do perfil que nem sempre aparecem em selected_descriptors."""
    result: dict = {}
    if not isinstance(user, dict):
        return result

    extra_fields = [
        ("Informações básicas: identidade de gênero", user.get("gender_identity") or user.get("gender_identities")),
        ("Informações básicas: all in gender", user.get("all_in_gender")),
        ("Informações básicas: orientação", user.get("sexual_orientations")),
        ("Informações básicas: pronomes", user.get("pronouns")),
    ]
    for key, value in extra_fields:
        text = _value_text(value)
        if text and text not in {"[]", "{}"}:
            _add_descriptor(result, key, text)

    if user.get("show_gender_on_profile") is True and user.get("gender") not in ("", None):
        _add_descriptor(result, "Informações básicas: gênero exibido", str(user.get("gender")))

    _extract_profile_detail_content(user.get("profile_detail_content"), result)
    return result


def pick_photo_url(photos: list, quality: str = "low") -> str | None:
    """Retorna a URL da primeira foto na resolucao configurada."""
    if not photos:
        return None

    first = photos[0]
    processed = first.get("processedFiles", [])
    if not processed:
        return first.get("url")

    quality_map = {"full": 0, "high": 1, "medium": -2, "low": -1}
    idx = quality_map.get(quality, -1)
    try:
        return processed[idx]["url"]
    except (IndexError, KeyError):
        logger.debug("Qualidade de foto indisponivel, usando fallback: quality=%s processed=%s", quality, len(processed))
        return processed[-1].get("url")


def parse_profile(result: dict) -> dict:
    """Converte um item de data.results[] para o formato do sistema."""
    user = result.get("user", {})
    exp = result.get("experiment_info", {})

    name = user.get("name", "")
    birth_date = user.get("birth_date", "")
    age = calc_age(birth_date) if birth_date else 0
    bio = user.get("bio", "") or ""

    selected_interests = exp.get("user_interests", {}).get("selected_interests", [])
    interests = [i.get("name", "") for i in selected_interests if i.get("name")]
    descriptors = extract_descriptors(user.get("selected_descriptors", []))
    descriptors.update(extract_basic_info(user))
    relationship_intent = user.get("relationship_intent") or {}
    if isinstance(relationship_intent, dict) and relationship_intent.get("body_text"):
        title = str(relationship_intent.get("title_text") or "Objetivo").strip() or "Objetivo"
        descriptors[title] = str(relationship_intent.get("body_text") or "").strip()

    distance_mi = result.get("distance_mi", None)
    distance_km = round(float(distance_mi) * 1.609) if distance_mi not in ("", None) else None

    cfg = get_photos_config()
    quality = cfg.get("quality", "low")
    analysis_quality = cfg.get("analysis_quality", quality)
    photos_raw = user.get("photos", [])
    photo_url = pick_photo_url(photos_raw, quality)

    photo_urls_analysis = [
        url for p in photos_raw
        if (url := pick_photo_url([p], analysis_quality))
    ]
    photo_url_pairs = []
    for p in photos_raw:
        analysis_url = pick_photo_url([p], analysis_quality)
        save_url = pick_photo_url([p], quality)
        if analysis_url or save_url:
            photo_url_pairs.append({
                "analysis_url": analysis_url or save_url,
                "save_url": save_url or analysis_url,
            })

    return {
        "name": name,
        "age": age,
        "distance_km": distance_km if distance_km is not None else "",
        "bio": bio,
        "interests": interests,
        "_descriptors": descriptors,
        "_distance_km": distance_km,
        "_tinder_id": user.get("_id", ""),
        "_content_hash": result.get("content_hash", ""),
        "_s_number": result.get("s_number", ""),
        "_online_now": bool(user.get("online_now", False)),
        "_recently_active": bool(user.get("recently_active", False)),
        "_photo_url": photo_url,
        "_photo_urls_analysis": photo_urls_analysis,
        "_photo_url_pairs": photo_url_pairs,
    }
