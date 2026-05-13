"""
Engenharia de features: transforma um perfil (dict) em vetor numérico (dict).
Cada função extrai um aspecto do perfil. O vetor final é o que o modelo ML recebe.
"""

import re
import json
import math
import unicodedata
from textblob import TextBlob

from config import load_config
from logging_config import get_logger
from bio_embedding import TEXT_EMB_FEATURE_NAMES
from photo_embedding_pca import EMBEDDING_FEATURE_NAMES
from photo_semantic_embeddings import SEMANTIC_EMBEDDING_FEATURE_NAMES
from text_preferences import signal_veto_tables

logger = get_logger(__name__)


def _normalize(text: str) -> str:
    """Lowercase + remove acentos para comparação de keywords."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def _count_keywords(text: str, keywords: list[str], ignored_keywords: set[str] | None = None) -> int:
    normalized = _normalize(text)
    ignored = {_normalize(x) for x in (ignored_keywords or set())}
    count = 0
    for kw in keywords:
        norm_kw = _normalize(kw)
        if norm_kw in ignored:
            continue
        if " " in norm_kw:
            if norm_kw in normalized:
                count += 1
        else:
            if re.search(r"\b" + re.escape(norm_kw) + r"\b", normalized):
                count += 1
    return count


def _as_float(value, default: float) -> float:
    try:
        if value in ("", None):
            return float(default)
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return float(default)
        return parsed
    except Exception:
        return float(default)


def _distance_settings(config: dict) -> tuple[float, float, float]:
    prefs = config.get("preferences", {}) or {}
    raw_range = prefs.get("distance_range_km") or [0, prefs.get("distance_preferred_max_km", 35)]
    try:
        min_km = float(raw_range[0]) if len(raw_range) > 0 else 0.0
    except Exception:
        min_km = 0.0
    try:
        preferred_max_km = float(raw_range[1]) if len(raw_range) > 1 else 35.0
    except Exception:
        preferred_max_km = 35.0
    if preferred_max_km < min_km:
        min_km, preferred_max_km = preferred_max_km, min_km
    try:
        soft_max_km = float(prefs.get("distance_soft_max_km", preferred_max_km * 1.7))
    except Exception:
        soft_max_km = preferred_max_km * 1.7
    soft_max_km = max(soft_max_km, preferred_max_km + 1.0)
    return min_km, preferred_max_km, soft_max_km


def _distance_features(profile: dict, config: dict) -> dict[str, float]:
    raw = profile.get("distance_km", profile.get("_distance_km", ""))
    try:
        distance_km = float(raw)
    except Exception:
        distance_km = float("nan")
    if raw in ("", None) or math.isnan(distance_km) or math.isinf(distance_km):
        return {
            "distance_km": 0.0,
            "distance_missing": 1.0,
            "distance_in_range": 0.0,
            "distance_over_preferred": 0.0,
            "distance_score": 0.5,
        }

    distance_km = max(0.0, distance_km)
    min_km, preferred_max_km, soft_max_km = _distance_settings(config)
    in_range = 1.0 if min_km <= distance_km <= preferred_max_km else 0.0
    over_preferred = max(0.0, distance_km - preferred_max_km) / max(soft_max_km - preferred_max_km, 1.0)
    over_preferred = max(0.0, min(1.0, over_preferred))

    if distance_km < min_km:
        score = max(0.0, min(1.0, distance_km / max(min_km, 1.0)))
    elif distance_km <= preferred_max_km:
        score = 1.0
    else:
        score = 1.0 - over_preferred

    return {
        "distance_km": round(distance_km, 4),
        "distance_missing": 0.0,
        "distance_in_range": in_range,
        "distance_over_preferred": round(over_preferred, 4),
        "distance_score": round(max(0.0, min(1.0, score)), 4),
    }


def _parse_descriptors(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw in ("", None):
        return {}

    text = str(raw).strip()
    if not text:
        return {}

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass

    result = {}
    for part in re.split(r"\s*\|\s*", text):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _descriptor_value(descriptors: dict, *names: str) -> str:
    wanted = {_normalize(name) for name in names}
    for key, value in descriptors.items():
        if _normalize(str(key)) in wanted:
            return str(value or "")
    return ""


def _descriptor_features(descriptors: dict, veto_tables: dict[str, set[str]] | None = None) -> dict:
    veto_tables = veto_tables or {}
    descriptor_not_positive = veto_tables.get("descriptor_not_positive", set())
    descriptor_not_negative = veto_tables.get("descriptor_not_negative", set())

    def _descriptor_vetoed(veto_set: set[str], *names: str) -> bool:
        if not veto_set:
            return False
        wanted = {_normalize(name) for name in names}
        for key, value in descriptors.items():
            key_norm = _normalize(str(key))
            if key_norm not in wanted:
                continue
            value_norm = _normalize(str(value or ""))
            detail_norm = _normalize(f"{key}: {value}")
            for item in veto_set:
                item_norm = _normalize(item)
                if item_norm in {key_norm, value_norm, detail_norm}:
                    return True
                if item_norm and (item_norm in detail_norm or detail_norm in item_norm):
                    return True
        return False

    family = _normalize(_descriptor_value(descriptors, "Família", "Familia"))
    smoke = _normalize(_descriptor_value(descriptors, "Você fuma?", "Voce fuma?", "Fumo"))
    drink = _normalize(_descriptor_value(descriptors, "Bebida"))
    activity = _normalize(_descriptor_value(descriptors, "Atividade física", "Atividade fisica"))
    pets = _normalize(_descriptor_value(descriptors, "Pets"))
    education = _normalize(_descriptor_value(descriptors, "Formação", "Formacao"))
    religion = _normalize(_descriptor_value(descriptors, "Religião", "Religiao"))
    relationship = _normalize(_descriptor_value(descriptors, "Tipo de relacionamento", "Relacionamento"))
    communication = _normalize(_descriptor_value(descriptors, "Estilo de comunicação", "Estilo de comunicacao"))
    sleep = _normalize(_descriptor_value(descriptors, "Hábitos de sono", "Habitos de sono"))

    family_pos_veto = _descriptor_vetoed(descriptor_not_positive, "Família", "Familia")
    family_neg_veto = _descriptor_vetoed(descriptor_not_negative, "Família", "Familia")
    smoke_neg_veto = _descriptor_vetoed(descriptor_not_negative, "Você fuma?", "Voce fuma?", "Fumo")
    drink_veto = _descriptor_vetoed(descriptor_not_positive, "Bebida") or _descriptor_vetoed(descriptor_not_negative, "Bebida")
    activity_pos_veto = _descriptor_vetoed(descriptor_not_positive, "Atividade física", "Atividade fisica")
    pets_pos_veto = _descriptor_vetoed(descriptor_not_positive, "Pets")
    education_pos_veto = _descriptor_vetoed(descriptor_not_positive, "Formação", "Formacao")
    religion_pos_veto = _descriptor_vetoed(descriptor_not_positive, "Religião", "Religiao")
    relationship_neg_veto = _descriptor_vetoed(descriptor_not_negative, "Tipo de relacionamento", "Relacionamento")
    communication_neg_veto = _descriptor_vetoed(descriptor_not_negative, "Estilo de comunicação", "Estilo de comunicacao")
    sleep_veto = _descriptor_vetoed(descriptor_not_positive, "Hábitos de sono", "Habitos de sono") or _descriptor_vetoed(descriptor_not_negative, "Hábitos de sono", "Habitos de sono")

    family_does_not_want = "nao quero" in family
    family_unsure = "ainda nao sei" in family
    family_wants = not family_does_not_want and "quero filhos" in family
    family_has_children_raw = (
        "tenho filho" in family
        or "tenho filhos" in family
        or ("ja tenho" in family and "filh" in family)
        or "meus filhos" in family
        or re.search(r"\b(mae|pai)\s+(solo|de|do|da)\b", family) is not None
    )
    family_has_children = family_has_children_raw and not family_wants and not family_does_not_want and not family_unsure

    desc_has_children = 0 if family_neg_veto else 1 if family_has_children else 0
    desc_wants_children = 0 if family_pos_veto else 1 if family_wants else 0
    desc_does_not_want_children = 0 if family_neg_veto else 1 if family_does_not_want else 0
    desc_unsure_children = 0 if family_neg_veto else 1 if family_unsure else 0
    desc_smokes = 0 if smoke_neg_veto else 1 if smoke and "nao fumo" not in smoke and "não fumo" not in smoke else 0
    desc_drinks = 0 if drink_veto else 1 if drink and "nao curto" not in drink and "não curto" not in drink and "parei" not in drink else 0
    desc_active = 0 if activity_pos_veto else 1 if any(x in activity for x in ["frequentemente", "todo dia"]) else 0
    desc_pet_positive = 0 if pets_pos_veto else 1 if any(x in pets for x in ["cachorro", "gato", "gosto", "amo"]) else 0
    desc_higher_education = 0 if education_pos_veto else 1 if any(x in education for x in ["superior", "faculdade", "pos", "pós"]) else 0
    desc_christian = 0 if religion_pos_veto else 1 if any(x in religion for x in ["crista", "cristã", "catolic", "evangelic"]) else 0
    desc_non_monogamy = 0 if relationship_neg_veto else 1 if "nao-monog" in relationship or "não-monog" in relationship else 0
    desc_bad_messaging = 0 if communication_neg_veto else 1 if any(x in communication for x in ["odeio", "demoro"]) else 0
    desc_night_person = 0 if sleep_veto else 1 if "noturna" in sleep else 0

    return {
        "desc_has_children": desc_has_children,
        "desc_wants_children": desc_wants_children,
        "desc_does_not_want_children": desc_does_not_want_children,
        "desc_unsure_children": desc_unsure_children,
        "desc_smokes": desc_smokes,
        "desc_drinks": desc_drinks,
        "desc_active": desc_active,
        "desc_pet_positive": desc_pet_positive,
        "desc_higher_education": desc_higher_education,
        "desc_christian": desc_christian,
        "desc_non_monogamy": desc_non_monogamy,
        "desc_bad_messaging": desc_bad_messaging,
        "desc_night_person": desc_night_person,
    }


def extract_features(profile: dict, config: dict | None = None) -> dict:
    """
    Recebe um perfil e retorna um dicionário de features numéricas.

    profile = {
        "name": str,
        "age": int,
        "bio": str,
        "interests": list[str]  # lista de strings
    }
    """
    if config is None:
        config = load_config()

    prefs = config["preferences"]
    # Um unico veto ja afeta a camada aprendida em text_preferences.py.
    # Para desligar features fixas do config, exigimos repeticao para evitar
    # apagar preferencias globais por um clique isolado na review.
    veto_tables = signal_veto_tables(min_count=2)
    cfg_suppressed = set(config.get("preferences", {}).get("desc_neg_suppressed") or [])
    if cfg_suppressed:
        veto_tables["descriptor_not_negative"] = veto_tables.get("descriptor_not_negative", set()) | cfg_suppressed
    age_min, age_max = prefs["age_range"]
    age_center = (age_min + age_max) / 2

    name: str = profile.get("name", "") or ""
    age: int = int(profile.get("age", 0) or 0)
    bio: str = profile.get("bio", "") or ""
    interests: list = profile.get("interests", []) or []
    descriptors = _parse_descriptors(profile.get("_descriptors") or profile.get("descriptors") or {})
    dist_features = _distance_features(profile, config)

    # Normaliza interesses para comparação
    interests_normalized = [_normalize(i) for i in interests]
    interest_not_positive = veto_tables.get("interest_not_positive", set())
    preferred_interests_normalized = [
        _normalize(i)
        for i in prefs.get("preferred_interests", [])
        if _normalize(i) not in interest_not_positive
    ]

    # --- Features de idade ---
    age_in_range = 1 if age_min <= age <= age_max else 0
    age_distance = abs(age - age_center) / max((age_max - age_min) / 2, 1)

    # --- Features de bio ---
    bio_length = len(bio)
    bio_word_count = len(bio.split()) if bio.strip() else 0
    bio_has_min_length = 1 if bio_length >= prefs.get("bio_min_length", 20) else 0
    bio_positive_kw = _count_keywords(
        bio,
        prefs.get("positive_bio_keywords", []),
        veto_tables.get("bio_not_positive", set()),
    )
    bio_negative_kw = _count_keywords(
        bio,
        prefs.get("negative_bio_keywords", []),
        veto_tables.get("bio_not_negative", set()),
    )

    # Sentimento da bio via TextBlob (funciona melhor com inglês, mas capta tom geral)
    bio_sentiment = 0.0
    if bio.strip():
        try:
            bio_sentiment = TextBlob(bio).sentiment.polarity
        except Exception:
            logger.debug("TextBlob falhou ao calcular sentimento da bio", exc_info=True)
            bio_sentiment = 0.0

    # --- Features de interesses ---
    interests_count = len(interests)
    interests_overlap = sum(
        1 for i in interests_normalized if i in preferred_interests_normalized
    )

    # --- Feature de nome ---
    name_length = len(name.strip())
    name_in_disliked = 1 if _normalize(name) in [_normalize(n) for n in prefs.get("disliked_names", [])] else 0
    desc_features = _descriptor_features(descriptors, veto_tables)

    _RACE_NEUTRAL = 1 / 6

    # --- Features de foto (opcionais — defaults neutros quando não disponíveis) ---
    photo = profile.get("_photo_features") or {}
    if not photo:
        photo = {k: v for k, v in profile.items() if str(k).startswith("photo_")}

    photo_has_face          = _as_float(photo.get("photo_has_face", 0), 0)
    photo_woman_confidence  = _as_float(photo.get("photo_woman_confidence", 0.5), 0.5)
    photo_skin_lightness    = _as_float(photo.get("photo_skin_lightness", 0.5), 0.5)
    photo_face_similarity   = _as_float(photo.get("photo_face_similarity", 0.5), 0.5)
    photo_faces_ratio       = _as_float(photo.get("photo_faces_ratio", 0.0), 0.0)
    photo_failure_ratio     = _as_float(photo.get("photo_failure_ratio", 0.0), 0.0)
    photo_gender_certainty  = _as_float(photo.get("photo_gender_certainty", abs(photo_woman_confidence - 0.5) * 2), abs(photo_woman_confidence - 0.5) * 2)
    photo_race_black        = _as_float(photo.get("photo_race_black",         _RACE_NEUTRAL), _RACE_NEUTRAL)
    photo_race_white        = _as_float(photo.get("photo_race_white",         _RACE_NEUTRAL), _RACE_NEUTRAL)
    photo_race_asian        = _as_float(photo.get("photo_race_asian",         _RACE_NEUTRAL), _RACE_NEUTRAL)
    photo_race_indian       = _as_float(photo.get("photo_race_indian",        _RACE_NEUTRAL), _RACE_NEUTRAL)
    photo_race_middleeastern= _as_float(photo.get("photo_race_middleeastern", _RACE_NEUTRAL), _RACE_NEUTRAL)
    photo_race_latina       = _as_float(photo.get("photo_race_latina",        _RACE_NEUTRAL), _RACE_NEUTRAL)
    photo_body_visible      = _as_float(photo.get("photo_body_visible",       0.0), 0.0)
    photo_body_full_length  = _as_float(photo.get("photo_body_full_length",   0.0), 0.0)
    photo_body_upper_length = _as_float(photo.get("photo_body_upper_length",  0.0), 0.0)
    photo_body_closeup      = _as_float(photo.get("photo_body_closeup",       0.0), 0.0)
    photo_body_width_ratio  = _as_float(photo.get("photo_body_width_ratio",   0.5), 0.5)
    photo_body_signal_quality = _as_float(photo.get("photo_body_signal_quality", 0.0), 0.0)
    photo_body_width_bucket_narrow = _as_float(photo.get("photo_body_width_bucket_narrow", 0.0), 0.0)
    photo_body_width_bucket_medium = _as_float(photo.get("photo_body_width_bucket_medium", 0.0), 0.0)
    photo_body_width_bucket_wide = _as_float(photo.get("photo_body_width_bucket_wide", 0.0), 0.0)
    photo_face_smile_score = _as_float(photo.get("photo_face_smile_score", 0.5), 0.5)
    photo_image_brightness = _as_float(photo.get("photo_image_brightness", 0.5), 0.5)
    photo_image_contrast = _as_float(photo.get("photo_image_contrast", 0.5), 0.5)
    photo_image_sharpness = _as_float(photo.get("photo_image_sharpness", 0.5), 0.5)
    photo_image_colorfulness = _as_float(photo.get("photo_image_colorfulness", 0.5), 0.5)
    photo_body_skin_ratio = _as_float(photo.get("photo_body_skin_ratio", 0.0), 0.0)
    photo_pose_shoulder_width = _as_float(photo.get("photo_pose_shoulder_width", 0.0), 0.0)
    photo_pose_hip_width = _as_float(photo.get("photo_pose_hip_width", 0.0), 0.0)
    photo_pose_shoulder_hip_ratio = _as_float(photo.get("photo_pose_shoulder_hip_ratio", 0.5), 0.5)
    photo_pose_torso_visibility = _as_float(photo.get("photo_pose_torso_visibility", 0.0), 0.0)
    photo_pose_torso_height = _as_float(photo.get("photo_pose_torso_height", 0.0), 0.0)
    photo_pose_upper_body_ratio = _as_float(photo.get("photo_pose_upper_body_ratio", 0.33), 0.33)
    photo_pose_leg_ratio = _as_float(photo.get("photo_pose_leg_ratio", 0.0), 0.0)
    photo_pose_body_coverage = _as_float(photo.get("photo_pose_body_coverage", 0.0), 0.0)
    photo_seg_shoulder_width = _as_float(photo.get("photo_seg_shoulder_width", 0.0), 0.0)
    photo_seg_waist_width = _as_float(photo.get("photo_seg_waist_width", 0.0), 0.0)
    photo_seg_hip_width = _as_float(photo.get("photo_seg_hip_width", 0.0), 0.0)
    photo_seg_shoulder_waist_ratio = _as_float(photo.get("photo_seg_shoulder_waist_ratio", 0.5), 0.5)
    photo_seg_body_coverage = _as_float(photo.get("photo_seg_body_coverage", 0.0), 0.0)
    photo_semantic_embedding_saved = _as_float(photo.get("photo_semantic_embedding_saved", 0.0), 0.0)
    photo_carousel_useful_count = _as_float(photo.get("photo_carousel_useful_count", 0.0), 0.0)
    photo_carousel_duplicate_score = _as_float(photo.get("photo_carousel_duplicate_score", 0.0), 0.0)
    photo_carousel_visual_diversity = _as_float(photo.get("photo_carousel_visual_diversity", 0.0), 0.0)

    # Embedding PCA — NaN quando não disponível (LightGBM trata como missing value)
    emb_features = {
        feat: _as_float(photo.get(feat), float("nan"))
        for feat in EMBEDDING_FEATURE_NAMES
    }

    semantic_emb_features = {
        feat: _as_float(photo.get(feat), float("nan"))
        for feat in SEMANTIC_EMBEDDING_FEATURE_NAMES
    }

    # Embeddings semânticos de bio/interesses — NaN quando PCA não treinado
    text_emb = profile.get("_text_features") or {}
    text_emb_features = {
        feat: _as_float(text_emb.get(feat), float("nan"))
        for feat in TEXT_EMB_FEATURE_NAMES
    }

    return {
        "age_in_range": age_in_range,
        "age_distance": age_distance,
        **dist_features,
        "bio_length": bio_length,
        "bio_word_count": bio_word_count,
        "bio_has_min_length": bio_has_min_length,
        "bio_positive_kw": bio_positive_kw,
        "bio_negative_kw": bio_negative_kw,
        "bio_sentiment": bio_sentiment,
        "interests_count": interests_count,
        "interests_overlap": interests_overlap,
        "name_length": name_length,
        "name_in_disliked": name_in_disliked,
        **desc_features,
        "photo_has_face": photo_has_face,
        "photo_woman_confidence": photo_woman_confidence,
        "photo_skin_lightness": photo_skin_lightness,
        "photo_face_similarity": photo_face_similarity,
        "photo_faces_ratio": photo_faces_ratio,
        "photo_failure_ratio": photo_failure_ratio,
        "photo_gender_certainty": photo_gender_certainty,
        "photo_race_black": photo_race_black,
        "photo_race_white": photo_race_white,
        "photo_race_asian": photo_race_asian,
        "photo_race_indian": photo_race_indian,
        "photo_race_middleeastern": photo_race_middleeastern,
        "photo_race_latina": photo_race_latina,
        "photo_body_visible": photo_body_visible,
        "photo_body_full_length": photo_body_full_length,
        "photo_body_upper_length": photo_body_upper_length,
        "photo_body_closeup": photo_body_closeup,
        "photo_body_width_ratio": photo_body_width_ratio,
        "photo_body_signal_quality": photo_body_signal_quality,
        "photo_body_width_bucket_narrow": photo_body_width_bucket_narrow,
        "photo_body_width_bucket_medium": photo_body_width_bucket_medium,
        "photo_body_width_bucket_wide": photo_body_width_bucket_wide,
        "photo_face_smile_score": photo_face_smile_score,
        "photo_image_brightness": photo_image_brightness,
        "photo_image_contrast": photo_image_contrast,
        "photo_image_sharpness": photo_image_sharpness,
        "photo_image_colorfulness": photo_image_colorfulness,
        "photo_body_skin_ratio": photo_body_skin_ratio,
        "photo_pose_shoulder_width": photo_pose_shoulder_width,
        "photo_pose_hip_width": photo_pose_hip_width,
        "photo_pose_shoulder_hip_ratio": photo_pose_shoulder_hip_ratio,
        "photo_pose_torso_visibility": photo_pose_torso_visibility,
        "photo_pose_torso_height": photo_pose_torso_height,
        "photo_pose_upper_body_ratio": photo_pose_upper_body_ratio,
        "photo_pose_leg_ratio": photo_pose_leg_ratio,
        "photo_pose_body_coverage": photo_pose_body_coverage,
        "photo_seg_shoulder_width": photo_seg_shoulder_width,
        "photo_seg_waist_width": photo_seg_waist_width,
        "photo_seg_hip_width": photo_seg_hip_width,
        "photo_seg_shoulder_waist_ratio": photo_seg_shoulder_waist_ratio,
        "photo_seg_body_coverage": photo_seg_body_coverage,
        "photo_semantic_embedding_saved": photo_semantic_embedding_saved,
        "photo_carousel_useful_count": photo_carousel_useful_count,
        "photo_carousel_duplicate_score": photo_carousel_duplicate_score,
        "photo_carousel_visual_diversity": photo_carousel_visual_diversity,
        **emb_features,
        **semantic_emb_features,
        **text_emb_features,
    }


FEATURE_NAMES = [
    "age_in_range",
    "age_distance",
    "distance_km",
    "distance_missing",
    "distance_in_range",
    "distance_over_preferred",
    "distance_score",
    "bio_length",
    "bio_word_count",
    "bio_has_min_length",
    "bio_positive_kw",
    "bio_negative_kw",
    "bio_sentiment",
    "interests_count",
    "interests_overlap",
    "name_length",
    "name_in_disliked",
    "desc_has_children",
    "desc_wants_children",
    "desc_does_not_want_children",
    "desc_unsure_children",
    "desc_smokes",
    "desc_drinks",
    "desc_active",
    "desc_pet_positive",
    "desc_higher_education",
    "desc_christian",
    "desc_non_monogamy",
    "desc_bad_messaging",
    "desc_night_person",
    "photo_has_face",
    "photo_woman_confidence",
    "photo_skin_lightness",
    "photo_face_similarity",
    "photo_race_black",
    "photo_race_white",
    "photo_race_asian",
    "photo_race_indian",
    "photo_race_middleeastern",
    "photo_race_latina",
    "photo_body_visible",
    "photo_body_full_length",
    "photo_body_upper_length",
    "photo_body_closeup",
    "photo_body_width_ratio",
    "photo_body_signal_quality",
    "photo_body_width_bucket_narrow",
    "photo_body_width_bucket_medium",
    "photo_body_width_bucket_wide",
    "photo_face_smile_score",
    "photo_image_brightness",
    "photo_image_contrast",
    "photo_image_sharpness",
    "photo_image_colorfulness",
    "photo_semantic_embedding_saved",
    "photo_carousel_useful_count",
    "photo_carousel_duplicate_score",
    "photo_carousel_visual_diversity",
    *EMBEDDING_FEATURE_NAMES,
    *SEMANTIC_EMBEDDING_FEATURE_NAMES,
    *TEXT_EMB_FEATURE_NAMES,
]

TEXT_FEATURE_NAMES = [
    "age_in_range",
    "age_distance",
    "distance_km",
    "distance_missing",
    "distance_in_range",
    "distance_over_preferred",
    "distance_score",
    "bio_has_min_length",
    "bio_positive_kw",
    "bio_negative_kw",
    "bio_sentiment",
    "interests_count",
    "interests_overlap",
    "name_in_disliked",
    "desc_has_children",
    "desc_wants_children",
    "desc_does_not_want_children",
    "desc_unsure_children",
    "desc_smokes",
    "desc_drinks",
    "desc_active",
    "desc_pet_positive",
    "desc_higher_education",
    "desc_christian",
    "desc_non_monogamy",
    "desc_bad_messaging",
    "desc_night_person",
    *TEXT_EMB_FEATURE_NAMES,
]

PHOTO_FEATURE_NAMES = [
    "photo_has_face",
    "photo_woman_confidence",
    "photo_skin_lightness",
    "photo_face_similarity",
    "photo_faces_ratio",
    "photo_failure_ratio",
    "photo_gender_certainty",
    "photo_race_black",
    "photo_race_white",
    "photo_race_asian",
    "photo_race_indian",
    "photo_race_middleeastern",
    "photo_race_latina",
    "photo_body_visible",
    "photo_body_full_length",
    "photo_body_upper_length",
    "photo_body_closeup",
    "photo_body_width_ratio",
    "photo_body_signal_quality",
    "photo_body_width_bucket_narrow",
    "photo_body_width_bucket_medium",
    "photo_body_width_bucket_wide",
    "photo_body_skin_ratio",
    "photo_face_smile_score",
    "photo_image_brightness",
    "photo_image_contrast",
    "photo_image_sharpness",
    "photo_image_colorfulness",
    "photo_pose_shoulder_width",
    "photo_pose_hip_width",
    "photo_pose_shoulder_hip_ratio",
    "photo_pose_torso_visibility",
    "photo_pose_torso_height",
    "photo_pose_upper_body_ratio",
    "photo_pose_leg_ratio",
    "photo_pose_body_coverage",
    "photo_seg_shoulder_width",
    "photo_seg_waist_width",
    "photo_seg_hip_width",
    "photo_seg_shoulder_waist_ratio",
    "photo_seg_body_coverage",
    "photo_semantic_embedding_saved",
    "photo_carousel_useful_count",
    "photo_carousel_duplicate_score",
    "photo_carousel_visual_diversity",
]

RACE_FEATURE_NAMES = [
    "photo_race_black",
    "photo_race_white",
    "photo_race_asian",
    "photo_race_indian",
    "photo_race_middleeastern",
    "photo_race_latina",
]

GENDER_PROXY_FEATURE_NAMES = [
    "photo_woman_confidence",
    "photo_gender_certainty",
    "photo_skin_lightness",
]

BODY_PREFERENCE_FEATURE_NAMES = [
    "photo_body_visible",
    "photo_body_full_length",
    "photo_body_upper_length",
    "photo_body_closeup",
    "photo_body_width_ratio",
    "photo_body_signal_quality",
    "photo_body_width_bucket_narrow",
    "photo_body_width_bucket_medium",
    "photo_body_width_bucket_wide",
    "photo_body_skin_ratio",
    "photo_pose_shoulder_width",
    "photo_pose_hip_width",
    "photo_pose_shoulder_hip_ratio",
    "photo_pose_torso_visibility",
    "photo_pose_torso_height",
    "photo_pose_upper_body_ratio",
    "photo_pose_leg_ratio",
    "photo_pose_body_coverage",
    "photo_seg_shoulder_width",
    "photo_seg_waist_width",
    "photo_seg_hip_width",
    "photo_seg_shoulder_waist_ratio",
    "photo_seg_body_coverage",
]

SENSITIVE_DECISION_FEATURE_NAMES = set(GENDER_PROXY_FEATURE_NAMES)

MODEL_PHOTO_FEATURE_NAMES = [
    name for name in PHOTO_FEATURE_NAMES
    if name not in SENSITIVE_DECISION_FEATURE_NAMES
]

MODEL_META_FEATURE_NAMES = [
    "text_model_prob",
    "photo_model_prob",
    "photo_has_face",
    "photo_face_similarity",
    "photo_faces_ratio",
    "photo_failure_ratio",
    "photo_image_sharpness",
    *RACE_FEATURE_NAMES,
    "photo_body_visible",
    "photo_body_full_length",
    "photo_body_upper_length",
    "photo_body_closeup",
    "photo_body_width_ratio",
    "photo_body_signal_quality",
    "photo_body_width_bucket_narrow",
    "photo_body_width_bucket_medium",
    "photo_body_width_bucket_wide",
    "photo_pose_torso_visibility",
    "photo_pose_body_coverage",
    "photo_seg_body_coverage",
    "photo_semantic_embedding_saved",
    "photo_carousel_useful_count",
    "photo_carousel_duplicate_score",
    "photo_carousel_visual_diversity",
]

FEATURE_LABELS = {
    "age_in_range": "Idade na faixa preferida",
    "age_distance": "Distância da idade ideal",
    "distance_km": "Distância do perfil (km)",
    "distance_missing": "Distância não informada",
    "distance_in_range": "Distância na faixa preferida",
    "distance_over_preferred": "Distância acima da preferência",
    "distance_score": "Score de distância",
    "bio_length": "Tamanho da bio (chars)",
    "bio_word_count": "Palavras na bio",
    "bio_has_min_length": "Bio tem tamanho mínimo",
    "bio_positive_kw": "Palavras positivas na bio",
    "bio_negative_kw": "Palavras negativas na bio",
    "bio_sentiment": "Sentimento da bio",
    "interests_count": "Qtd. de interesses",
    "interests_overlap": "Interesses em comum",
    "name_length": "Tamanho do nome",
    "name_in_disliked": "Nome na lista negativa",
    "desc_has_children": "Descritor: tem filhos",
    "desc_wants_children": "Descritor: quer filhos",
    "desc_does_not_want_children": "Descritor: não quer filhos",
    "desc_unsure_children": "Descritor: indecisa sobre filhos",
    "desc_smokes": "Descritor: fuma",
    "desc_drinks": "Descritor: bebe",
    "desc_active": "Descritor: ativa fisicamente",
    "desc_pet_positive": "Descritor: gosta de pets",
    "desc_higher_education": "Descritor: ensino superior",
    "desc_christian": "Descritor: cristã/religiosa",
    "desc_non_monogamy": "Descritor: não-monogamia",
    "desc_bad_messaging": "Descritor: comunicação ruim",
    "desc_night_person": "Descritor: pessoa noturna",
    "photo_has_face": "Rosto detectado na foto",
    "photo_woman_confidence": "Confiança: é mulher",
    "photo_skin_lightness": "Luminância da pele",
    "photo_face_similarity": "Similaridade ao gosto visual",
    "photo_faces_ratio": "Qualidade: proporção de rostos",
    "photo_failure_ratio": "Qualidade: falhas na análise",
    "photo_gender_certainty": "Qualidade: certeza de gênero",
    "photo_race_black": "Probabilidade: pessoa negra",
    "photo_race_white": "Probabilidade: pessoa branca",
    "photo_race_asian": "Probabilidade: pessoa asiática",
    "photo_race_indian": "Probabilidade: pessoa indiana",
    "photo_race_middleeastern": "Probabilidade: Oriente Médio",
    "photo_race_latina": "Probabilidade: pessoa latina",
    "photo_body_visible": "Corpo visível na foto",
    "photo_body_full_length": "Foto de corpo inteiro",
    "photo_body_upper_length": "Foto de meio corpo",
    "photo_body_closeup": "Foto muito fechada no rosto",
    "photo_body_width_ratio": "Composição corporal: largura visual",
    "photo_body_signal_quality": "Qualidade do sinal corporal",
    "photo_body_width_bucket_narrow": "Silhueta visual: estreita",
    "photo_body_width_bucket_medium": "Silhueta visual: média",
    "photo_body_width_bucket_wide": "Silhueta visual: ampla",
    "photo_face_smile_score": "Expressão/sorriso na foto",
    "photo_image_brightness": "Foto: iluminação/brilho",
    "photo_image_contrast": "Foto: contraste",
    "photo_image_sharpness": "Foto: nitidez",
    "photo_image_colorfulness": "Foto: cor/vivacidade",
    **{f"bio_emb_pc_{i:02d}": f"Bio semântica: componente {i}" for i in range(1, 9)},
    **{f"interest_emb_pc_{i:02d}": f"Interesses semânticos: componente {i}" for i in range(1, 5)},
    "photo_seg_shoulder_width": "Segmentação: largura visual dos ombros",
    "photo_seg_waist_width": "Segmentação: largura visual da cintura",
    "photo_seg_hip_width": "Segmentação: largura visual do quadril",
    "photo_seg_shoulder_waist_ratio": "Segmentação: proporção ombro/cintura",
    "photo_seg_body_coverage": "Segmentação: cobertura do corpo na foto",
    "photo_semantic_embedding_saved": "Embedding visual CLIP disponível",
    "photo_carousel_useful_count": "Carrossel: fotos úteis analisadas",
    "photo_carousel_duplicate_score": "Carrossel: fotos visualmente repetidas",
    "photo_carousel_visual_diversity": "Carrossel: diversidade visual das fotos",
    **{f"photo_clip_pc_{i:02d}": f"Foto CLIP: componente {i}" for i in range(1, 33)},
}
