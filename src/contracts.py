"""Contratos leves para os dicionarios que circulam pelo pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class PhotoFeatures(TypedDict, total=False):
    photo_has_face: float
    photo_woman_confidence: float
    photo_skin_lightness: float
    photo_face_similarity: float
    photo_race_black: float
    photo_race_white: float
    photo_race_asian: float
    photo_race_indian: float
    photo_race_middleeastern: float
    photo_race_latina: float
    photo_body_visible: float
    photo_body_full_length: float
    photo_body_upper_length: float
    photo_body_closeup: float
    photo_body_width_ratio: float
    photo_body_signal_quality: float
    photo_body_width_bucket_narrow: float
    photo_body_width_bucket_medium: float
    photo_body_width_bucket_wide: float
    photo_face_smile_score: float
    photo_image_brightness: float
    photo_image_contrast: float
    photo_image_sharpness: float
    photo_image_colorfulness: float
    _dominant_race: str
    _embedding: Any
    _faces_found: int
    _photos_analyzed: int


class PredictionResult(TypedDict, total=False):
    decision: str
    probability: float
    features: dict[str, float]
    importances: dict[str, float]
    model_type: str
    n_samples: int
    text_score: float
    photo_score: float
    photo_score_mode: str
    filter_reason: str
    race_affinity: dict[str, Any]
    gender_compatibility: dict[str, Any]
    decision_policy: dict[str, Any]
    like_threshold: float
    in_review_band: bool
    ranking_score: float
    review_priority: float
    preference_tier: str
    correction_type: str


class Profile(TypedDict, total=False):
    name: str
    age: int
    distance_km: int | str | None
    bio: str
    interests: list[str]
    _descriptors: dict[str, str]
    _distance_km: int | None
    _tinder_id: str
    _photo_url: str | None
    _photo_urls_analysis: list[str]
    _photo_url_pairs: list[dict[str, str]]
    _review_photo_url: str
    _face_photo_url: str
    _body_photo_url: str
    _photo_features: PhotoFeatures
    _ml_result: PredictionResult
    _skip_prompt: bool
    _filter_reason: str
    preference_tier: str
    correction_type: str
    ranking_score: float
    review_priority: float


class VisibleProfile(TypedDict):
    name: str
    tinder_id: str
    age: int
    age_seconds: float
