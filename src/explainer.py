"""
Explicabilidade: mostra quais features mais influenciaram a decisão do modelo.
"""

from features import FEATURE_LABELS, SENSITIVE_DECISION_FEATURE_NAMES


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _pct(value: float) -> str:
    return f"{_clamp01(value) * 100:.0f}%"


def _score_label(value: float) -> str:
    value = _clamp01(value)
    if value >= 0.72:
        return "alto"
    if value >= 0.58:
        return "bom"
    if value >= 0.43:
        return "neutro"
    if value >= 0.28:
        return "baixo"
    return "muito baixo"


def _as_float(features: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default))
    except Exception:
        return default


def _append_unique(target: list[str], text: str) -> None:
    if text and text not in target:
        target.append(text)


def _body_width_label(value: float) -> str:
    value = _clamp01(value)
    if value >= 0.68:
        return "largura visual alta"
    if value >= 0.46:
        return "largura visual média"
    if value >= 0.28:
        return "largura visual baixa"
    return "largura visual muito baixa"


def _body_bucket_label(features: dict) -> str:
    narrow = _as_float(features, "photo_body_width_bucket_narrow", 0.0) > 0
    medium = _as_float(features, "photo_body_width_bucket_medium", 0.0) > 0
    wide = _as_float(features, "photo_body_width_bucket_wide", 0.0) > 0

    if narrow and medium and not wide:
        return "média-estreita"
    if medium and wide and not narrow:
        return "média-ampla"
    if narrow:
        return "estreita"
    if medium:
        return "média"
    if wide:
        return "ampla"
    return ""


def _smile_label(value: float) -> str:
    value = _clamp01(value)
    if value >= 0.70:
        return "expressão/sorriso bem presente"
    if value >= 0.52:
        return "expressão/sorriso presente"
    if value <= 0.22:
        return "expressão mais neutra/séria"
    return "expressão neutra"


def _brightness_label(value: float) -> str:
    value = _clamp01(value)
    if value <= 0.28:
        return "foto escura"
    if value >= 0.78:
        return "foto muito clara"
    if 0.40 <= value <= 0.68:
        return "iluminação equilibrada"
    return "iluminação intermediária"


def _sharpness_label(value: float) -> str:
    value = _clamp01(value)
    if value >= 0.62:
        return "foto nítida"
    if value <= 0.28:
        return "foto pouco nítida"
    return "nitidez média"


def _contrast_label(value: float) -> str:
    value = _clamp01(value)
    if value >= 0.65:
        return "contraste alto"
    if value <= 0.25:
        return "contraste baixo"
    return "contraste médio"


def _colorfulness_label(value: float) -> str:
    value = _clamp01(value)
    if value >= 0.62:
        return "cores/vivacidade altas"
    if value <= 0.22:
        return "cores pouco vivas"
    return "cores/vivacidade médias"


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _photo_brightness_quality(value: float) -> float:
    """Pontua iluminacao perto do meio, penalizando foto muito escura/clara."""
    value = _clamp01(value)
    return _clamp01(1.0 - abs(value - 0.52) / 0.52)


def _photo_looks_nearly_black(features: dict) -> bool:
    brightness = _as_float(features, "photo_image_brightness", 0.5)
    contrast = _as_float(features, "photo_image_contrast", 0.5)
    colorfulness = _as_float(features, "photo_image_colorfulness", 0.5)
    return brightness <= 0.055 or (brightness <= 0.09 and contrast <= 0.12 and colorfulness <= 0.12)


def _visual_subscore(key: str, label: str, score: float, note: str) -> dict:
    return {
        "key": key,
        "label": label,
        "score": round(_clamp01(score), 4),
        "note": note,
    }


def build_visual_subscores(result: dict) -> list[dict]:
    """Decompoe o score visual em sinais pequenos para a UI de review."""
    features = result.get("features", {}) or {}
    if not isinstance(features, dict):
        features = {}

    has_face = _as_float(features, "photo_has_face", 0.0)
    face_similarity = _as_float(features, "photo_face_similarity", 0.5)
    faces_ratio = _as_float(features, "photo_faces_ratio", 1.0 if has_face > 0 else 0.0)
    smile_score = _as_float(features, "photo_face_smile_score", 0.5)
    woman_conf = _as_float(features, "photo_woman_confidence", 0.5)
    gender_certainty = _as_float(features, "photo_gender_certainty", abs(woman_conf - 0.5) * 2)

    brightness = _as_float(features, "photo_image_brightness", 0.5)
    contrast = _as_float(features, "photo_image_contrast", 0.5)
    sharpness = _as_float(features, "photo_image_sharpness", 0.5)
    colorfulness = _as_float(features, "photo_image_colorfulness", 0.5)
    brightness_quality = _photo_brightness_quality(brightness)

    body_visible = _as_float(features, "photo_body_visible", 0.0)
    body_full = _as_float(features, "photo_body_full_length", 0.0)
    body_upper = _as_float(features, "photo_body_upper_length", 0.0)
    body_closeup = _as_float(features, "photo_body_closeup", 0.0)
    body_quality = _as_float(features, "photo_body_signal_quality", 0.0)
    pose_torso_visibility = _as_float(features, "photo_pose_torso_visibility", 0.0)
    pose_torso_height = _as_float(features, "photo_pose_torso_height", 0.0)
    pose_body_coverage = _as_float(features, "photo_pose_body_coverage", 0.0)
    seg_body_coverage = _as_float(features, "photo_seg_body_coverage", 0.0)

    if has_face <= 0:
        face_score = 0.0
        face_note = "sem rosto confiavel detectado; rosto quase nao pesou"
    else:
        face_score = _clamp01(
            face_similarity * 0.60
            + faces_ratio * 0.14
            + smile_score * 0.08
            + sharpness * 0.10
            + brightness_quality * 0.08
        )
        face_note = (
            f"parecido com historico {_pct(face_similarity)}; "
            f"rostos uteis {_pct(faces_ratio)}; expressao {_pct(smile_score)}"
        )

    presentation_score = _clamp01(woman_conf * 0.85 + gender_certainty * 0.15)
    if has_face <= 0:
        presentation_score = min(presentation_score, 0.42)
    presentation_note = (
        f"detector visual {_pct(woman_conf)}; "
        f"certeza {_pct(gender_certainty)}; nao representa identidade de genero"
    )

    body_score = _clamp01(
        body_quality * 0.40
        + body_visible * 0.28
        + body_full * 0.12
        + body_upper * 0.10
        + (1.0 - _clamp01(body_closeup)) * 0.05
        + brightness_quality * 0.05
    )
    if max(body_visible, body_full, body_upper, body_quality) <= 0.05:
        body_note = "pouco corpo visivel; este score mede leitura corporal disponivel"
    else:
        body_note = (
            f"visivel {_pct(body_visible)}; qualidade {_pct(body_quality)}; "
            f"inteiro {_pct(body_full)}; meio corpo {_pct(body_upper)}"
        )

    pose_score = _clamp01(
        pose_torso_visibility * 0.34
        + pose_body_coverage * 0.24
        + pose_torso_height * 0.14
        + max(body_full, body_upper) * 0.12
        + body_quality * 0.10
        + seg_body_coverage * 0.06
    )
    if max(pose_torso_visibility, pose_body_coverage, seg_body_coverage) <= 0.05:
        pose_note = "pose/composicao pouco legivel nesta foto"
    else:
        pose_note = (
            f"tronco {_pct(pose_torso_visibility)}; "
            f"cobertura {_pct(pose_body_coverage)}; segmentacao {_pct(seg_body_coverage)}"
        )

    quality_score = _clamp01(
        brightness_quality * 0.36
        + _clamp01(sharpness) * 0.28
        + _clamp01(contrast) * 0.20
        + _clamp01(colorfulness) * 0.16
    )
    if _photo_looks_nearly_black(features):
        quality_score = min(quality_score, 0.08)
        quality_note = "foto muito escura/preta; quase sem conteudo visual util"
    else:
        quality_note = (
            f"brilho {_pct(brightness)}; nitidez {_pct(sharpness)}; "
            f"contraste {_pct(contrast)}"
        )

    return [
        _visual_subscore("face", "rosto", face_score, face_note),
        _visual_subscore("body", "corpo", body_score, body_note),
        _visual_subscore("pose", "pose", pose_score, pose_note),
        _visual_subscore("quality", "qualidade", quality_score, quality_note),
    ]


def _build_reason_groups(result: dict, top_n: int = 5) -> dict[str, list[str]]:
    features = result.get("features", {}) or {}
    importances = result.get("importances", {}) or {}
    text_pref = result.get("text_preference", {}) or {}
    photo_components = result.get("photo_components", {}) or {}

    pro_like: list[str] = []
    pro_pass: list[str] = []
    uncertainty: list[str] = []
    photo_lines: list[str] = []
    body_lines: list[str] = []
    text_lines: list[str] = []
    top_lines: list[str] = []

    photo_score = float(result.get("photo_score", 0.5) or 0.5)
    text_score = float(result.get("text_score", 0.5) or 0.5)
    text_model_prob = float(result.get("text_model_probability", text_score) or text_score)
    text_pref_score = float(result.get("text_preference_score", 0.5) or 0.5)
    photo_model_prob = photo_components.get("photo_model_probability")

    has_face = _as_float(features, "photo_has_face", 0.0)
    woman_conf = _as_float(features, "photo_woman_confidence", 0.5)
    face_similarity = _as_float(features, "photo_face_similarity", 0.5)
    faces_ratio = _as_float(features, "photo_faces_ratio", 0.0)
    failure_ratio = _as_float(features, "photo_failure_ratio", 0.0)
    gender_certainty = _as_float(features, "photo_gender_certainty", abs(woman_conf - 0.5) * 2)
    smile_score = _as_float(features, "photo_face_smile_score", 0.5)
    image_brightness = _as_float(features, "photo_image_brightness", 0.5)
    image_contrast = _as_float(features, "photo_image_contrast", 0.5)
    image_sharpness = _as_float(features, "photo_image_sharpness", 0.5)
    image_colorfulness = _as_float(features, "photo_image_colorfulness", 0.5)

    if has_face <= 0:
        _append_unique(pro_pass, "nenhum rosto confiável foi detectado nas fotos analisadas")
        _append_unique(photo_lines, "sem rosto detectado; o score visual fica fraco/neutro")
    else:
        _append_unique(photo_lines, f"similaridade facial {_score_label(face_similarity)} ({_pct(face_similarity)})")
        if face_similarity >= 0.62:
            _append_unique(pro_like, f"rosto parecido com perfis que você costuma curtir ({_pct(face_similarity)})")
        elif face_similarity <= 0.42:
            _append_unique(pro_pass, f"rosto pouco parecido com seu histórico de curtidas ({_pct(face_similarity)})")

        if faces_ratio:
            _append_unique(photo_lines, f"rostos aproveitáveis em {_pct(faces_ratio)} das fotos analisadas")
        if 0 < faces_ratio < 0.5:
            _append_unique(uncertainty, "poucas fotos tiveram rosto detectável")
        if failure_ratio >= 0.25:
            _append_unique(uncertainty, f"parte da análise visual falhou ({_pct(failure_ratio)} das fotos)")
        _append_unique(photo_lines, f"{_smile_label(smile_score)} ({_pct(smile_score)})")

    _append_unique(photo_lines, f"{_sharpness_label(image_sharpness)} ({_pct(image_sharpness)})")
    _append_unique(photo_lines, f"{_brightness_label(image_brightness)} ({_pct(image_brightness)})")
    if image_contrast <= 0.25 or image_contrast >= 0.65:
        _append_unique(photo_lines, f"{_contrast_label(image_contrast)} ({_pct(image_contrast)})")
    if image_colorfulness <= 0.22 or image_colorfulness >= 0.62:
        _append_unique(photo_lines, f"{_colorfulness_label(image_colorfulness)} ({_pct(image_colorfulness)})")
    if image_sharpness <= 0.25 or image_brightness <= 0.22:
        _append_unique(uncertainty, "qualidade visual da foto pode ter atrapalhado a leitura")

    body_visible = _as_float(features, "photo_body_visible", 0.0)
    body_full = _as_float(features, "photo_body_full_length", 0.0)
    body_upper = _as_float(features, "photo_body_upper_length", 0.0)
    body_closeup = _as_float(features, "photo_body_closeup", 0.0)
    body_width = _as_float(features, "photo_body_width_ratio", 0.5)
    body_quality = _as_float(features, "photo_body_signal_quality", 0.0)
    pose_torso_visibility = _as_float(features, "photo_pose_torso_visibility", 0.0)
    pose_shoulder_width = _as_float(features, "photo_pose_shoulder_width", 0.0)
    pose_hip_width = _as_float(features, "photo_pose_hip_width", 0.0)
    pose_shoulder_hip_ratio = _as_float(features, "photo_pose_shoulder_hip_ratio", 0.5)
    pose_upper_body_ratio = _as_float(features, "photo_pose_upper_body_ratio", 0.33)
    pose_leg_ratio = _as_float(features, "photo_pose_leg_ratio", 0.0)
    pose_body_coverage = _as_float(features, "photo_pose_body_coverage", 0.0)
    seg_body_coverage = _as_float(features, "photo_seg_body_coverage", 0.0)
    seg_shoulder_width = _as_float(features, "photo_seg_shoulder_width", 0.0)
    seg_waist_width = _as_float(features, "photo_seg_waist_width", 0.0)
    seg_hip_width = _as_float(features, "photo_seg_hip_width", 0.0)
    seg_shoulder_waist_ratio = _as_float(features, "photo_seg_shoulder_waist_ratio", 0.5)

    if body_quality > 0.05:
        _append_unique(body_lines, f"corpo visível: {_pct(body_visible)}")
        if body_full >= 0.35:
            _append_unique(body_lines, f"sinal de corpo inteiro: {_pct(body_full)}")
        if body_upper >= 0.35:
            _append_unique(body_lines, f"sinal de meio corpo: {_pct(body_upper)}")
        if body_closeup >= 0.55:
            _append_unique(body_lines, f"foto bem fechada no rosto: {_pct(body_closeup)}")
        body_bucket = _body_bucket_label(features)
        if body_bucket:
            _append_unique(body_lines, f"silhueta visual {body_bucket} (sinal de composição)")
        else:
            _append_unique(body_lines, f"{_body_width_label(body_width)} ({_pct(body_width)})")
        _append_unique(body_lines, f"qualidade do sinal corporal: {_pct(body_quality)}")
    else:
        _append_unique(body_lines, "pouco corpo visível; o modelo priorizou rosto/texto")
        if has_face > 0:
            _append_unique(uncertainty, "sinal corporal fraco ou ausente nesta foto")

    if pose_torso_visibility >= 0.15:
        _append_unique(
            body_lines,
            f"pose corporal detectada: tronco {_pct(pose_torso_visibility)}; cobertura {_pct(pose_body_coverage)}",
        )
        if pose_shoulder_width > 0 or pose_hip_width > 0:
            _append_unique(
                body_lines,
                f"pose: ombro {_pct(pose_shoulder_width)} / quadril {_pct(pose_hip_width)} / proporção {_pct(pose_shoulder_hip_ratio)}",
            )
        if pose_upper_body_ratio > 0 or pose_leg_ratio > 0:
            _append_unique(
                body_lines,
                f"pose: parte superior {_pct(pose_upper_body_ratio)} / pernas {_pct(pose_leg_ratio)}",
            )
    elif body_visible > 0:
        _append_unique(uncertainty, "pose corporal não ficou forte o bastante para explicar muito")

    if seg_body_coverage >= 0.08:
        _append_unique(
            body_lines,
            f"segmentação corporal: cobertura {_pct(seg_body_coverage)}; ombro/cintura {_pct(seg_shoulder_waist_ratio)}",
        )
        if seg_shoulder_width > 0 or seg_waist_width > 0 or seg_hip_width > 0:
            _append_unique(
                body_lines,
                f"segmentação: ombro {_pct(seg_shoulder_width)} / cintura {_pct(seg_waist_width)} / quadril {_pct(seg_hip_width)}",
            )
    elif body_visible > 0 and body_quality > 0.05:
        _append_unique(uncertainty, "segmentação corporal com pouco sinal útil")

    if photo_model_prob is not None:
        photo_model_prob = float(photo_model_prob)
        _append_unique(photo_lines, f"modelo visual supervisionado estimou {_pct(photo_model_prob)} para curtir")

    if photo_score >= 0.62:
        _append_unique(pro_like, f"score visual favorável ({_pct(photo_score)})")
    elif photo_score <= 0.42:
        _append_unique(pro_pass, f"score visual desfavorável ({_pct(photo_score)})")

    age_in_range = _as_float(features, "age_in_range", 0.0)
    age_distance = _as_float(features, "age_distance", 0.0)
    distance_km = _as_float(features, "distance_km", 0.0)
    distance_missing = _as_float(features, "distance_missing", 1.0)
    distance_in_range = _as_float(features, "distance_in_range", 0.0)
    distance_score = _as_float(features, "distance_score", 0.5)
    distance_over_preferred = _as_float(features, "distance_over_preferred", 0.0)
    bio_length = _as_float(features, "bio_length", 0.0)
    bio_has_min = _as_float(features, "bio_has_min_length", 0.0)
    bio_positive = _as_float(features, "bio_positive_kw", 0.0)
    bio_negative = _as_float(features, "bio_negative_kw", 0.0)
    bio_sentiment = _as_float(features, "bio_sentiment", 0.0)
    interests_count = _as_float(features, "interests_count", 0.0)
    interests_overlap = _as_float(features, "interests_overlap", 0.0)

    if age_in_range:
        _append_unique(pro_like, "idade dentro da faixa configurada")
        _append_unique(text_lines, "idade dentro da faixa preferida")
    elif age_distance > 0:
        _append_unique(pro_pass, "idade fora ou distante da faixa preferida")
        _append_unique(text_lines, f"idade distante do centro da preferência ({age_distance:.2f})")

    if distance_missing <= 0:
        _append_unique(text_lines, f"distância do perfil: {distance_km:.0f} km")
        if distance_in_range > 0:
            _append_unique(pro_like, f"distância dentro da faixa configurada ({distance_km:.0f} km)")
        elif distance_score <= 0.35 or distance_over_preferred >= 0.65:
            _append_unique(pro_pass, f"distância acima da preferência ({distance_km:.0f} km)")
        elif distance_score < 0.50:
            _append_unique(uncertainty, f"distância um pouco fora da preferência ({distance_km:.0f} km)")
    else:
        _append_unique(uncertainty, "distância não informada pelo Tinder")

    if interests_overlap > 0:
        _append_unique(pro_like, f"{int(interests_overlap)} interesse(s) batem com preferências explícitas")
        _append_unique(text_lines, f"interesses em comum/preferidos: {int(interests_overlap)}")
    elif interests_count == 0:
        _append_unique(uncertainty, "perfil sem interesses listados")
        _append_unique(text_lines, "sem interesses listados")
    else:
        _append_unique(text_lines, f"{int(interests_count)} interesse(s), mas pouco overlap explícito")

    if bio_positive > 0:
        _append_unique(pro_like, f"bio contém {int(bio_positive)} palavra(s)/tema(s) positivo(s)")
        _append_unique(text_lines, f"palavras positivas na bio: {int(bio_positive)}")
    if bio_negative > 0:
        _append_unique(pro_pass, f"bio contém {int(bio_negative)} alerta(s) configurado(s)")
        _append_unique(text_lines, f"alertas na bio: {int(bio_negative)}")
    if bio_has_min:
        _append_unique(text_lines, f"bio com tamanho útil ({int(bio_length)} caracteres)")
    elif bio_length <= 0:
        _append_unique(uncertainty, "perfil sem bio")
        _append_unique(text_lines, "sem bio")
    else:
        _append_unique(text_lines, f"bio curta ({int(bio_length)} caracteres)")
    if abs(bio_sentiment) >= 0.20:
        direction = "positivo" if bio_sentiment > 0 else "negativo"
        _append_unique(text_lines, f"sentimento da bio levemente {direction} ({bio_sentiment:.2f})")

    learned_positive = (
        text_pref.get("interest_pref_positive_signals", [])
        + text_pref.get("bio_pref_positive_signals", [])
        + text_pref.get("descriptor_pref_positive_signals", [])
    )
    learned_negative = (
        text_pref.get("interest_pref_negative_signals", [])
        + text_pref.get("bio_pref_negative_signals", [])
        + text_pref.get("descriptor_pref_negative_signals", [])
    )
    if learned_positive:
        _append_unique(pro_like, "bate com sinais textuais que você já curtiu: " + ", ".join(learned_positive[:4]))
    if learned_negative:
        _append_unique(pro_pass, "lembra sinais textuais que você costuma passar: " + ", ".join(learned_negative[:4]))

    if text_score >= 0.62:
        _append_unique(pro_like, f"score textual favorável ({_pct(text_score)})")
    elif text_score <= 0.42:
        _append_unique(pro_pass, f"score textual desfavorável ({_pct(text_score)})")
    descriptor_pref_score = float(text_pref.get("descriptor_pref_score", 0.5) or 0.5)
    _append_unique(
        text_lines,
        f"modelo de texto: {_pct(text_model_prob)} | texto aprendido: {_pct(text_pref_score)} | descritores aprendidos: {_pct(descriptor_pref_score)}",
    )

    descriptor_flags = {
        "desc_has_children": ("tem filhos", "pass"),
        "desc_wants_children": ("quer filhos", "like"),
        "desc_does_not_want_children": ("não quer filhos", "pass"),
        "desc_unsure_children": ("indecisa sobre filhos", "pass"),
        "desc_smokes": ("fuma", "pass"),
        "desc_drinks": ("bebe", "neutral"),
        "desc_active": ("ativa fisicamente", "like"),
        "desc_pet_positive": ("gosta de pets", "like"),
        "desc_higher_education": ("ensino superior", "like"),
        "desc_christian": ("religiosa/cristã", "like"),
        "desc_non_monogamy": ("não-monogamia", "pass"),
        "desc_bad_messaging": ("comunicação ruim", "pass"),
        "desc_night_person": ("pessoa noturna", "neutral"),
    }
    desc_found = []
    for key, (label, polarity) in descriptor_flags.items():
        if _as_float(features, key, 0.0) <= 0:
            continue
        desc_found.append(label)
        if polarity == "like":
            _append_unique(pro_like, f"descritor favorável: {label}")
        elif polarity == "pass":
            _append_unique(pro_pass, f"descritor de alerta: {label}")
    if desc_found:
        _append_unique(text_lines, "descritores detectados: " + ", ".join(desc_found[:5]))

    if abs(photo_score - text_score) >= 0.28:
        stronger = "foto" if photo_score > text_score else "texto"
        weaker = "texto" if stronger == "foto" else "foto"
        _append_unique(uncertainty, f"{stronger} e {weaker} discordam bastante")

    if importances:
        safe_importances = [
            (feat_name, importance)
            for feat_name, importance in importances.items()
            if feat_name not in SENSITIVE_DECISION_FEATURE_NAMES
        ]
        for feat_name, importance in sorted(safe_importances, key=lambda x: x[1], reverse=True)[:top_n]:
            label = FEATURE_LABELS.get(feat_name, feat_name)
            value = _format_value(features.get(feat_name, "?"))
            top_lines.append(f"{label}: {value} (peso {importance * 100:.1f}%)")

    return {
        "pro_like": pro_like,
        "pro_pass": pro_pass,
        "photo": photo_lines,
        "body": body_lines,
        "text": text_lines,
        "uncertainty": uncertainty,
        "top": top_lines,
    }


def build_reason_groups(result: dict, top_n: int = 5) -> dict[str, list[str]]:
    """API pública para UIs reutilizarem os grupos de explicação."""
    return _build_reason_groups(result, top_n=top_n)


def _dominant_summary(result: dict, groups: dict[str, list[str]]) -> str:
    decision = result.get("decision", "?")
    photo_score = float(result.get("photo_score", 0.5) or 0.5)
    text_score = float(result.get("text_score", 0.5) or 0.5)
    weights = result.get("weights", {}) or {}
    photo_weight = float(weights.get("photo_weight", 0.5) or 0.5)
    text_weight = float(weights.get("text_weight", 0.5) or 0.5)

    if abs(photo_score - text_score) < 0.08:
        source = "foto e texto ficaram parecidos"
    elif photo_score > text_score:
        source = "as fotos puxaram mais a decisão"
    else:
        source = "o texto/perfil puxou mais a decisão"

    if decision == "CURTIR":
        reason = groups["pro_like"][0] if groups["pro_like"] else "o score combinado passou do limite de curtir"
    else:
        reason = groups["pro_pass"][0] if groups["pro_pass"] else "o score combinado ficou abaixo do limite de curtir"

    return (
        f"{source}; peso usado: foto {photo_weight * 100:.0f}% / texto {text_weight * 100:.0f}%. "
        f"Motivo principal: {reason}."
    )


def format_reason_summary(result: dict, max_lines: int = 8) -> str:
    """Resumo compacto dos motivos, útil para salvar junto com a foto/log."""
    groups = _build_reason_groups(result, top_n=4)
    decision = result.get("decision", "?")
    prob = float(result.get("probability", 0.0) or 0.0)
    confidence = prob if decision == "CURTIR" else (1 - prob)

    lines = [
        _dominant_summary(result, groups),
        f"Confiança calibrada: {confidence * 100:.0f}%.",
    ]
    if decision == "CURTIR":
        lines.extend(f"+ {item}" for item in groups["pro_like"][:3])
        lines.extend(f"- alerta: {item}" for item in groups["pro_pass"][:2])
    else:
        lines.extend(f"- {item}" for item in groups["pro_pass"][:3])
        lines.extend(f"+ contraponto: {item}" for item in groups["pro_like"][:2])
    lines.extend(f"? {item}" for item in groups["uncertainty"][:2])
    return "\n".join(lines[:max_lines])


def explain(result: dict, top_n: int = 5) -> str:
    """
    Formata a explicação de uma predição para exibição no terminal.

    result é o dict retornado por model.predict().
    """
    decision = result["decision"]
    prob = result["probability"]
    model_type = result["model_type"]
    n_samples = result["n_samples"]
    text_score = result.get("text_score", 0.5)
    photo_score = result.get("photo_score", 0.5)
    photo_score_mode = result.get("photo_score_mode", "heurístico")
    photo_n_samples = result.get("photo_n_samples", 0)
    superlike_prob = result.get("superlike_probability")
    superlike_n = result.get("superlike_n_samples", 0)
    superlike_pos = result.get("superlike_positive_samples", 0)
    weights = result.get("weights", {})
    groups = _build_reason_groups(result, top_n=top_n)
    safety = result.get("probability_safety", {}) or {}

    lines = []

    # Cabeçalho da decisão
    confidence = prob if decision == "CURTIR" else (1 - prob)
    if decision == "CURTIR":
        verdict = f"  CURTIR  ({prob * 100:.0f}% de confiança)"
        border = "=" * 46
        lines.append(border)
        lines.append(verdict)
        lines.append(border)
    else:
        verdict = f"  NÃO CURTIR  ({(1 - prob) * 100:.0f}% de confiança)"
        border = "-" * 46
        lines.append(border)
        lines.append(verdict)
        lines.append(border)

    if confidence < 0.65:
        lines.append(f"  ⚠  modelo incerto — seu feedback aqui é valioso para o treino")

    lines.append(f"  Modelo: {model_type} | {n_samples} exemplos de treino")
    lines.append(
        f"  Score final: foto={photo_score:.2f} x texto={text_score:.2f}  "
        f"(pesos {weights.get('photo_weight', 0.5)*100:.0f}%/{weights.get('text_weight', 0.5)*100:.0f}%)"
    )
    lines.append(f"  Foto: score {photo_score_mode} | {photo_n_samples} exemplos reais com foto")
    if superlike_prob is not None and decision == "CURTIR":
        lines.append(
            f"  Super like: {float(superlike_prob) * 100:.0f}% "
            f"({superlike_pos}/{superlike_n} curtidas fortes no treino)"
        )
    if safety.get("applied"):
        lines.append(
            "  Calibração: confiança ajustada de "
            f"{float(safety.get('raw_probability', prob)) * 100:.0f}% para {prob * 100:.0f}% "
            f"({safety.get('reason', 'segurança')})"
        )
    lines.append("")

    lines.append("  Resumo:")
    lines.append(f"    {_dominant_summary(result, groups)}")

    if groups["pro_like"]:
        lines.append("")
        lines.append("  Sinais pró-curtir:")
        for row in groups["pro_like"][:5]:
            lines.append(f"    [+] {row}")

    if groups["pro_pass"]:
        lines.append("")
        lines.append("  Sinais pró-passar:")
        for row in groups["pro_pass"][:5]:
            lines.append(f"    [-] {row}")

    if groups["photo"]:
        lines.append("")
        lines.append("  Fotos / leitura visual:")
        for row in groups["photo"][:8]:
            lines.append(f"    [foto] {row}")

    if groups["body"]:
        lines.append("")
        lines.append("  Corpo / composição da foto:")
        for row in groups["body"][:6]:
            lines.append(f"    [corpo] {row}")

    if groups["text"]:
        lines.append("")
        lines.append("  Texto / perfil:")
        for row in groups["text"][:7]:
            lines.append(f"    [texto] {row}")

    if groups["uncertainty"]:
        lines.append("")
        lines.append("  Incertezas:")
        for row in groups["uncertainty"][:4]:
            lines.append(f"    [?] {row}")

    if groups["top"]:
        lines.append("")
        lines.append("  Features mais influentes no modelo:")
        for row in groups["top"][:top_n]:
            lines.append(f"    [ml] {row}")

    lines.append("")
    return "\n".join(lines)


def format_stats(n_real: int, n_synthetic: int, model_type: str) -> str:
    lines = [
        "",
        "  Estatísticas do projeto",
        "  " + "-" * 30,
        f"  Perfis reais rotulados : {n_real}",
        f"  Perfis sintéticos      : {n_synthetic}",
        f"  Modelo atual           : {model_type}",
        f"  Total de treino        : {n_real + n_synthetic}",
        "",
    ]
    return "\n".join(lines)
