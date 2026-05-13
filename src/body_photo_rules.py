"""Conservative rules for deciding whether a saved photo is usable for body review."""

from __future__ import annotations


BODY_MEASUREMENT_MIN_STRENGTH = 0.55


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        raw = row.get(key, default)
        if raw in ("", None):
            return default
        return float(raw)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def body_measurement_strength(row: dict) -> float:
    """
    Scores whether a photo has enough real body context for body training/review.

    Face-only close-ups can trigger false pose landmarks, so every signal also
    needs plausible framing from the lightweight body geometry.
    """
    body_visible = _clamp01(_as_float(row, "photo_body_visible"))
    body_quality = _clamp01(_as_float(row, "photo_body_signal_quality"))
    body_full = _clamp01(_as_float(row, "photo_body_full_length"))
    body_upper = _clamp01(_as_float(row, "photo_body_upper_length"))
    closeup = _clamp01(_as_float(row, "photo_body_closeup"))

    shoulder = _clamp01(_as_float(row, "photo_pose_shoulder_width"))
    hip = _clamp01(_as_float(row, "photo_pose_hip_width"))
    torso = _clamp01(_as_float(row, "photo_pose_torso_visibility"))
    coverage = _clamp01(_as_float(row, "photo_pose_body_coverage"))

    seg_cov = _clamp01(_as_float(row, "photo_seg_body_coverage"))
    seg_shoulder = _clamp01(_as_float(row, "photo_seg_shoulder_width"))
    seg_hip = _clamp01(_as_float(row, "photo_seg_hip_width"))
    seg_waist = _clamp01(_as_float(row, "photo_seg_waist_width"))

    bucket_ok = any(
        _as_float(row, key) > 0
        for key in (
            "photo_body_width_bucket_narrow",
            "photo_body_width_bucket_medium",
            "photo_body_width_bucket_wide",
        )
    )

    has_body_frame = (
        body_visible >= 0.30
        and closeup < 0.84
        and body_quality >= 0.18
        and (body_upper >= 0.20 or body_full >= 0.05 or body_visible >= 0.65)
    )
    has_strong_body_frame = (
        body_visible >= 0.50
        and closeup < 0.80
        and body_quality >= 0.40
        and (body_upper >= 0.30 or body_full >= 0.10 or body_visible >= 0.80)
    )

    pose_ok = (
        has_body_frame
        and torso >= 0.22
        and shoulder >= 0.045
        and hip >= 0.045
        and (coverage >= 0.12 or body_upper >= 0.30 or body_full >= 0.10)
    )
    seg_ok = (
        has_body_frame
        and seg_cov >= 0.10
        and (seg_shoulder >= 0.045 or seg_hip >= 0.045 or seg_waist >= 0.045)
    )
    canny_ok = (
        has_strong_body_frame
        and body_quality >= 0.42
        and (bucket_ok or body_upper >= 0.45 or body_full >= 0.20)
    )

    score = 0.0
    if pose_ok:
        score = max(
            score,
            min(1.0, 0.42 + torso * 0.24 + coverage * 0.14 + min(shoulder + hip, 1.0) * 0.10 + body_visible * 0.10),
        )
    if seg_ok:
        score = max(
            score,
            min(1.0, 0.34 + seg_cov * 0.30 + max(seg_shoulder, seg_hip, seg_waist) * 0.18 + body_visible * 0.12),
        )
    if canny_ok:
        score = max(
            score,
            min(0.74, 0.18 + body_visible * 0.22 + body_quality * 0.28 + max(body_upper, body_full) * 0.14),
        )
    return round(score, 4)
