"""Identidade leve de foto/perfil para deduplicar reviews."""

from __future__ import annotations

import hashlib
from urllib.parse import urlsplit


PHOTO_IDENTITY_FIELDNAMES = [
    "photo_url_key",
    "photo_url_keys",
    "photo_fingerprint",
]


_FINGERPRINT_FEATURES = [
    "photo_has_face",
    "photo_woman_confidence",
    "photo_gender_certainty",
    "photo_skin_lightness",
    "photo_skin_saturation",
    "photo_race_asian",
    "photo_race_black",
    "photo_race_indian",
    "photo_race_latino",
    "photo_race_middle_eastern",
    "photo_race_white",
    "photo_image_brightness",
    "photo_image_contrast",
    "photo_image_sharpness",
    "photo_image_colorfulness",
    "photo_body_visible",
    "photo_body_full_length",
    "photo_body_upper_length",
    "photo_body_closeup",
    "photo_body_width_ratio",
    "photo_body_signal_quality",
    "photo_body_skin_ratio",
    "photo_pose_shoulder_width",
    "photo_pose_hip_width",
    "photo_pose_torso_visibility",
    "photo_pose_body_coverage",
    "photo_seg_body_coverage",
    "photo_seg_shoulder_width",
    "photo_seg_hip_width",
]


def _clean_url(value: object) -> str:
    return str(value or "").strip().replace("&amp;", "&")


def photo_url_identity_keys(url: object) -> list[str]:
    """Retorna chaves estáveis da foto, ignorando query/token assinado."""
    clean = _clean_url(url)
    if not clean:
        return []

    try:
        parsed = urlsplit(clean)
    except Exception:
        return [f"raw:{clean}"]

    if not parsed.scheme and not parsed.netloc:
        return [f"raw:{clean}"]

    host = parsed.netloc.lower()
    path = parsed.path or ""
    keys: list[str] = []
    if host and path:
        keys.append(f"path:{host}{path}")

    parts = [part for part in path.split("/") if part]
    if "u" in parts:
        idx = parts.index("u")
        owner = parts[idx + 1] if idx + 1 < len(parts) else ""
        photo_id = parts[idx + 2] if idx + 2 < len(parts) else ""
        if owner:
            keys.append(f"owner:{owner}")
        if owner and photo_id:
            keys.append(f"photo:{owner}/{photo_id}")

    return list(dict.fromkeys(keys))


def _iter_profile_photo_urls(profile: dict):
    for key in (
        "_photo_url",
        "_review_photo_url",
        "_face_photo_url",
        "_body_photo_url",
        "photo_url",
    ):
        value = profile.get(key)
        if value:
            yield value

    photo = profile.get("_photo_features") or {}
    if isinstance(photo, dict):
        for key in ("_review_photo_url", "_best_face_photo_url", "_best_body_photo_url"):
            value = photo.get(key)
            if value:
                yield value

    for value in profile.get("_photo_urls_analysis") or []:
        if value:
            yield value

    for pair in profile.get("_photo_url_pairs") or []:
        if not isinstance(pair, dict):
            continue
        for key in ("analysis_url", "save_url"):
            value = pair.get(key)
            if value:
                yield value


def photo_url_keys_from_profile(profile: dict) -> list[str]:
    keys: list[str] = []
    for url in _iter_profile_photo_urls(profile):
        keys.extend(photo_url_identity_keys(url))
    return list(dict.fromkeys(keys))


def split_photo_url_keys(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part for part in text.split("|") if part]


def photo_fingerprint_from_features(photo: dict | None) -> str:
    """Hash conservador de features visuais já calculadas."""
    if not isinstance(photo, dict) or not photo:
        return ""

    parts: list[str] = []
    for name in _FINGERPRINT_FEATURES:
        raw = photo.get(name)
        if raw in ("", None):
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if value != value:
            continue
        parts.append(f"{name}={value:.4f}")

    if len(parts) < 8:
        return ""
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def photo_identity_values_from_profile(profile: dict) -> dict[str, str]:
    photo = profile.get("_photo_features") or profile
    keys = photo_url_keys_from_profile(profile)
    fingerprint = (
        str(profile.get("photo_fingerprint") or "").strip()
        or photo_fingerprint_from_features(photo if isinstance(photo, dict) else {})
    )
    return {
        "photo_url_key": keys[0] if keys else str(profile.get("photo_url_key") or "").strip(),
        "photo_url_keys": "|".join(keys) if keys else str(profile.get("photo_url_keys") or "").strip(),
        "photo_fingerprint": fingerprint,
    }
