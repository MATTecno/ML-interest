"""
Extrai features visuais de perfis usando DeepFace + Pillow + OpenCV.
Roda sincronamente antes da decisão ML.
"""

import io
import json
import os
import queue
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit
from pathlib import Path

from body_photo_rules import BODY_MEASUREMENT_MIN_STRENGTH, body_measurement_strength
from config import ROOT_DIR, get_photos_config
from logging_config import get_logger

# Suprime mensagens de log do TensorFlow/oneDNN antes de qualquer import deles
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

logger = get_logger(__name__)


CACHE_PATH = ROOT_DIR / "data" / "photo_feature_cache.json"
_CACHE_LOCK = threading.RLock()
_CACHE_DATA: dict | None = None
_EMOTION_ANALYSIS_AVAILABLE: bool | None = None
_EMOTION_WEIGHT_FILE = Path.home() / ".deepface" / "weights" / "facial_expression_model_weights.h5"


def _cache_key(url: str) -> str:
    """Usa o path da foto sem query assinada, que muda com o tempo."""
    parsed = urlsplit(url or "")
    return parsed.path or url


def _load_cache() -> dict:
    global _CACHE_DATA
    with _CACHE_LOCK:
        if _CACHE_DATA is not None:
            return _CACHE_DATA
        if not CACHE_PATH.exists():
            _CACHE_DATA = {}
            return _CACHE_DATA
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _CACHE_DATA = json.load(f)
        except Exception:
            logger.exception("Falha ao carregar cache de features de foto: %s", CACHE_PATH)
            _CACHE_DATA = {}
        return _CACHE_DATA


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        tmp.replace(CACHE_PATH)
    except Exception:
        logger.exception("Falha ao salvar cache de features de foto: %s", CACHE_PATH)


def _jsonable_features(features: dict) -> dict:
    """Converte features para JSON preservando embedding quando existir."""
    serializable = {}
    for key, value in features.items():
        if key == "_semantic_embedding":
            continue
        if key == "_embedding":
            if value is None:
                serializable[key] = None
            else:
                serializable[key] = [float(x) for x in value]
            continue
        try:
            if isinstance(value, (int, float, str, bool)) or value is None:
                serializable[key] = value
            else:
                serializable[key] = float(value)
        except Exception:
            serializable[key] = str(value)
    return serializable


def _features_from_cache(data: dict, source_key: str | None = None) -> dict:
    """Restaura features e recalcula similaridade com a preferência atual."""
    import numpy as np
    from face_embeddings import compute_similarity

    restored = _default_features()
    restored.update(data or {})
    embedding = restored.get("_embedding")
    if embedding is not None:
        embedding = np.array(embedding, dtype=float)
        restored["_embedding"] = embedding
        restored["photo_face_similarity"] = compute_similarity(embedding)
    if source_key:
        try:
            from photo_semantic_embeddings import (
                get_cached_embedding,
                semantic_embedding_enabled,
            )

            if semantic_embedding_enabled():
                semantic_embedding = get_cached_embedding(source_key)
                if semantic_embedding is not None:
                    restored["_semantic_embedding"] = semantic_embedding
                    restored["photo_semantic_embedding_saved"] = 1.0
        except Exception:
            logger.debug("Falha ao restaurar embedding semantico do cache", exc_info=True)
    return restored


def _get_cached_photo(url: str) -> dict | None:
    key = _cache_key(url)
    cache = _load_cache()
    data = cache.get(key)
    if not data:
        logger.debug("Cache MISS foto key=%s", key)
        return None
    missing = [feat for feat in _PHOTO_CACHE_REQUIRED_FEATURES if feat not in data]
    if missing:
        logger.debug("Cache antigo ignorado por features ausentes key=%s missing=%s", key, missing[:5])
        return None
    cfg = get_photos_config()
    if data.get("_analysis_failed"):
        logger.debug("Cache ignorado por falha anterior key=%s", key)
        return None
    if not bool(cfg.get("cache_no_face_results", False)) and not data.get("photo_has_face"):
        logger.debug("Cache sem rosto ignorado para permitir nova deteccao key=%s", key)
        return None
    try:
        source_key = ""
        try:
            from photo_semantic_embeddings import (
                get_cached_embedding,
                semantic_embedding_enabled,
                source_key_for_url,
            )

            source_key = source_key_for_url(url)
            if semantic_embedding_enabled() and get_cached_embedding(source_key) is None:
                logger.debug("Cache de foto ignorado: embedding semantico ausente key=%s", key)
                return None
        except Exception:
            logger.debug("Checagem de cache semantico ignorada key=%s", key, exc_info=True)
        logger.debug("Cache HIT foto key=%s", key)
        return _features_from_cache(data, source_key=source_key)
    except Exception:
        logger.exception("Falha ao restaurar features do cache key=%s", key)
        return None


def _put_cached_photo(url: str, features: dict) -> None:
    cfg = get_photos_config()
    if features.get("_analysis_failed"):
        logger.debug("Features de foto nao cacheadas por falha key=%s", _cache_key(url))
        return
    if not bool(cfg.get("cache_no_face_results", False)) and not features.get("photo_has_face"):
        logger.debug("Features de foto sem rosto nao cacheadas key=%s", _cache_key(url))
        return

    key = _cache_key(url)
    with _CACHE_LOCK:
        cache = _load_cache()
        cache[key] = _jsonable_features(features)
        _save_cache(cache)
    logger.debug("Features de foto salvas no cache key=%s", key)


def _download_as_bgr(url: str):
    """
    Baixa a foto e retorna como array BGR numpy.
    Usa Pillow para suportar JPEG, WebP, AVIF e PNG — independente do formato real.
    Retorna None se o download ou a leitura falhar.
    """
    try:
        import numpy as np
        from PIL import Image as PILImage

        cfg = get_photos_config()
        timeout_s = float(cfg.get("download_timeout_seconds", 8))
        logger.debug("Download de foto remoto iniciado: timeout=%.1fs key=%s", timeout_s, _cache_key(url))
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Referer": "https://tinder.com/",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            data = resp.read()

        if len(data) < 500:
            logger.warning("Download suspeito de foto: bytes=%s status=%s url_path=%s", len(data), status, _cache_key(url))
            return None

        pil_img = PILImage.open(io.BytesIO(data)).convert("RGB")
        logger.debug("Foto baixada: bytes=%s status=%s url_path=%s", len(data), status, _cache_key(url))
        return np.array(pil_img)[:, :, ::-1]  # RGB → BGR (padrão cv2/deepface)

    except urllib.error.HTTPError as e:
        logger.warning("HTTP ao baixar foto: code=%s reason=%s url_path=%s", e.code, e.reason, _cache_key(url), exc_info=True)
        return None
    except urllib.error.URLError as e:
        logger.warning("Falha de rede ao baixar foto: reason=%s url_path=%s", e.reason, _cache_key(url), exc_info=True)
        return None
    except Exception as e:
        logger.exception("Erro inesperado no download da foto url_path=%s", _cache_key(url))
        return None


def _load_path_as_bgr(path: str | Path):
    """
    Lê uma imagem local e retorna como array BGR numpy.
    Retorna None se a leitura falhar.
    """
    try:
        import numpy as np
        from PIL import Image as PILImage

        file_path = Path(path)
        if not file_path.exists():
            return None

        pil_img = PILImage.open(file_path).convert("RGB")
        return np.array(pil_img)[:, :, ::-1]
    except Exception as e:
        logger.exception("Erro ao ler foto local: %s", path)
        return None


def _skin_lightness(img_bgr, region: dict) -> float:
    """
    Luminância média da região do rosto no espaço LAB.
    Retorna 0.0 (muito escuro) a 1.0 (muito claro).
    """
    try:
        import cv2
        import numpy as np
        x = max(region.get("x", 0), 0)
        y = max(region.get("y", 0), 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
        if w <= 0 or h <= 0:
            return 0.5
        face_crop = img_bgr[y : y + h, x : x + w]
        if face_crop.size == 0:
            return 0.5
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        return float(lab[:, :, 0].mean()) / 255.0
    except Exception:
        return 0.5


_RACE_NEUTRAL = round(1 / 6, 4)


_BODY_FEATURE_DEFAULTS = {
    "photo_body_visible": 0.0,
    "photo_body_full_length": 0.0,
    "photo_body_upper_length": 0.0,
    "photo_body_closeup": 0.0,
    "photo_body_width_ratio": 0.5,
    "photo_body_signal_quality": 0.0,
    "photo_body_width_bucket_narrow": 0.0,
    "photo_body_width_bucket_medium": 0.0,
    "photo_body_width_bucket_wide": 0.0,
    "photo_body_skin_ratio": 0.0,
}

_SEMANTIC_FEATURE_DEFAULTS = {
    "photo_face_smile_score": 0.5,
    "photo_image_brightness": 0.5,
    "photo_image_contrast": 0.5,
    "photo_image_sharpness": 0.5,
    "photo_image_colorfulness": 0.5,
}

_POSE_FEATURE_DEFAULTS = {
    "photo_pose_shoulder_width": 0.0,
    "photo_pose_hip_width": 0.0,
    "photo_pose_shoulder_hip_ratio": 0.5,
    "photo_pose_torso_visibility": 0.0,
    "photo_pose_torso_height": 0.0,
    "photo_pose_upper_body_ratio": 0.33,  # ~valor mediano real; 0.5 enviesaria para "largo"
    "photo_pose_leg_ratio": 0.0,
    "photo_pose_body_coverage": 0.0,
}

# Segmentação por pixel (selfie_multiclass_256x256) — mede silhueta real do corpo.
# Só roda quando body_visible > 0.25 para evitar custo em fotos de rosto.
_SEG_FEATURE_DEFAULTS = {
    "photo_seg_shoulder_width": 0.0,
    "photo_seg_waist_width": 0.0,
    "photo_seg_hip_width": 0.0,
    "photo_seg_shoulder_waist_ratio": 0.5,
    "photo_seg_body_coverage": 0.0,
}

_CAROUSEL_FEATURE_DEFAULTS = {
    "photo_semantic_embedding_saved": 0.0,
    "photo_carousel_useful_count": 0.0,
    "photo_carousel_duplicate_score": 0.0,
    "photo_carousel_visual_diversity": 0.0,
}

_SEG_MODEL_PATH = ROOT_DIR / "data" / "models" / "selfie_multiclass_256x256.tflite"
_SEG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter"
    "/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
_SEG_SEGMENTER = None
_SEG_SEGMENTER_LOCK = threading.Lock()

_PHOTO_CACHE_REQUIRED_FEATURES = tuple(_BODY_FEATURE_DEFAULTS) + tuple(_SEMANTIC_FEATURE_DEFAULTS)


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _body_default_features() -> dict:
    return dict(_BODY_FEATURE_DEFAULTS)


def _semantic_default_features() -> dict:
    return dict(_SEMANTIC_FEATURE_DEFAULTS)


def _pose_default_features() -> dict:
    return dict(_POSE_FEATURE_DEFAULTS)


def _seg_default_features() -> dict:
    return dict(_SEG_FEATURE_DEFAULTS)


def _carousel_default_features() -> dict:
    return dict(_CAROUSEL_FEATURE_DEFAULTS)


def _ensure_seg_model() -> Path | None:
    """Baixa o modelo de segmentação se não existir. Retorna o path ou None em falha."""
    if _SEG_MODEL_PATH.exists():
        return _SEG_MODEL_PATH
    try:
        _SEG_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Baixando modelo de segmentação MediaPipe (~4 MB)…")
        urllib.request.urlretrieve(_SEG_MODEL_URL, _SEG_MODEL_PATH)
        logger.info("Modelo de segmentação salvo em %s", _SEG_MODEL_PATH)
        return _SEG_MODEL_PATH
    except Exception:
        logger.debug("Falha ao baixar modelo de segmentação", exc_info=True)
        return None


def _get_seg_segmenter():
    """Retorna (ou cria) instância cacheada do ImageSegmenter. None se indisponível."""
    global _SEG_SEGMENTER
    if _SEG_SEGMENTER is not None:
        return _SEG_SEGMENTER
    with _SEG_SEGMENTER_LOCK:
        if _SEG_SEGMENTER is not None:
            return _SEG_SEGMENTER
        try:
            import mediapipe as mp  # noqa: F401
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_seg_model()
            if model_path is None:
                return None
            options = mp_vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                output_category_mask=True,
            )
            _SEG_SEGMENTER = mp_vision.ImageSegmenter.create_from_options(options)
            return _SEG_SEGMENTER
        except Exception:
            logger.debug("Falha ao criar ImageSegmenter", exc_info=True)
            return None


def _estimate_segmentation_features(img_bgr) -> dict:
    """
    Mede larguras de ombro/cintura/quadril por pixel usando máscara de segmentação.
    Só deve ser chamada quando body_visible > 0.25 (corpo parcialmente visível).
    """
    defaults = _seg_default_features()
    if img_bgr is None or not _body_analysis_enabled():
        return defaults

    try:
        import cv2
        import numpy as np
        import mediapipe as mp

        segmenter = _get_seg_segmenter()
        if segmenter is None:
            return defaults

        img_h, img_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = segmenter.segment(mp_image)

        if not result.category_mask:
            return defaults

        # Classes: 0=bg, 1=hair, 2=body_skin, 3=face_skin, 4=clothes, 5=others
        mask_raw = result.category_mask.numpy_view()
        mask = cv2.resize(mask_raw.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Silhueta do corpo: pele + roupas (exclui fundo e cabelo)
        body_mask = ((mask >= 2) & (mask <= 5)).astype(np.uint8)
        body_coverage = float(body_mask.mean())

        if body_coverage < 0.04:
            return {**defaults, "photo_seg_body_coverage": round(body_coverage, 4)}

        # Bounding box vertical do corpo
        body_rows = np.where(body_mask.any(axis=1))[0]
        if len(body_rows) < 10:
            return {**defaults, "photo_seg_body_coverage": round(body_coverage, 4)}

        body_top = int(body_rows[0])
        body_bottom = int(body_rows[-1])
        body_height = max(body_bottom - body_top, 1)

        # Y-posições para ombro/cintura/quadril — usa pose landmarks se disponíveis
        landmarks = _pose_landmarks_from_bgr(img_bgr)
        if landmarks is not None:
            def _vis(lm):
                v = getattr(lm, "visibility", None)
                return float(v) if v is not None else 1.0
            ls, rs = landmarks[_L_SHOULDER], landmarks[_R_SHOULDER]
            lh, rh = landmarks[_L_HIP], landmarks[_R_HIP]
            s_vis = min(_vis(ls), _vis(rs))
            h_vis = min(_vis(lh), _vis(rh))
            shoulder_y = int(((ls.y + rs.y) / 2) * img_h) if s_vis >= 0.30 else int(body_top + body_height * 0.15)
            hip_y = int(((lh.y + rh.y) / 2) * img_h) if h_vis >= 0.30 else int(body_top + body_height * 0.65)
        else:
            shoulder_y = int(body_top + body_height * 0.15)
            hip_y = int(body_top + body_height * 0.65)
        waist_y = (shoulder_y + hip_y) // 2

        def _width_at_y(y_center: int, band: int = 8) -> float:
            y0 = max(0, y_center - band)
            y1 = min(img_h, y_center + band + 1)
            if y1 <= y0:
                return 0.0
            cols = np.where(body_mask[y0:y1, :].any(axis=0))[0]
            if len(cols) < 4:
                return 0.0
            return float(cols[-1] - cols[0]) / max(img_w, 1)

        band = max(6, int(img_h * 0.035))
        shoulder_w = _width_at_y(shoulder_y, band)
        waist_w = _width_at_y(waist_y, max(4, int(img_h * 0.025)))
        hip_w = _width_at_y(hip_y, band)

        if shoulder_w > 0.01 and waist_w > 0.01:
            shoulder_waist_ratio = _clamp01(shoulder_w / (shoulder_w + waist_w))
        else:
            shoulder_waist_ratio = 0.5

        return {
            "photo_seg_shoulder_width": round(_clamp01(shoulder_w), 4),
            "photo_seg_waist_width": round(_clamp01(waist_w), 4),
            "photo_seg_hip_width": round(_clamp01(hip_w), 4),
            "photo_seg_shoulder_waist_ratio": round(shoulder_waist_ratio, 4),
            "photo_seg_body_coverage": round(_clamp01(body_coverage), 4),
        }

    except ImportError:
        return defaults
    except Exception:
        logger.debug("Falha na segmentação de corpo", exc_info=True)
        return defaults


def _semantic_cfg() -> dict:
    cfg = get_photos_config().get("semantic_analysis", {}) or {}
    return cfg if isinstance(cfg, dict) else {"enabled": bool(cfg)}


def _semantic_analysis_enabled() -> bool:
    return bool(_semantic_cfg().get("enabled", True))


def _semantic_emotion_enabled() -> bool:
    if _EMOTION_ANALYSIS_AVAILABLE is False:
        return False
    cfg = _semantic_cfg()
    if not (bool(cfg.get("enabled", True)) and bool(cfg.get("emotion", True))):
        return False
    if _EMOTION_WEIGHT_FILE.exists():
        return True
    return bool(cfg.get("emotion_auto_download", False))


def _body_width_min_quality() -> float:
    return float(_semantic_cfg().get("body_width_min_quality", 0.35) or 0.35)


def _body_analysis_enabled() -> bool:
    cfg = get_photos_config().get("body_analysis", {}) or {}
    if isinstance(cfg, dict):
        return bool(cfg.get("enabled", True))
    return bool(cfg)


# mediapipe 0.10.x usa Tasks API com arquivo de modelo .task
_POSE_MODEL_PATH = ROOT_DIR / "data" / "models" / "pose_landmarker_lite.task"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
    "/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
# Índices dos landmarks (constantes — iguais em todas as versões do MediaPipe)
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_HIP, _R_HIP = 23, 24
_L_ANKLE, _R_ANKLE = 27, 28

_MEDIAPIPE_AVAILABLE: bool | None = None
_POSE_MIN_DIM = 512  # upscale para pelo menos esta dimensão antes de detectar
_POSE_LANDMARKER = None  # instância cacheada — evita crash de re-criação no 0.10.x
_POSE_LANDMARKER_LOCK = threading.Lock()


def _ensure_pose_model() -> Path | None:
    """Baixa o modelo de pose se não existir. Retorna o path ou None em falha."""
    if _POSE_MODEL_PATH.exists():
        return _POSE_MODEL_PATH
    try:
        _POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Baixando modelo de pose MediaPipe (~3 MB)…")
        urllib.request.urlretrieve(_POSE_MODEL_URL, _POSE_MODEL_PATH)
        logger.info("Modelo de pose salvo em %s", _POSE_MODEL_PATH)
        return _POSE_MODEL_PATH
    except Exception:
        logger.debug("Falha ao baixar modelo de pose", exc_info=True)
        return None


def _get_pose_landmarker():
    """
    Retorna (ou cria) uma instância cacheada do PoseLandmarker (Tasks API 0.10.x).
    Cria apenas uma vez por processo para evitar o crash de re-criação/destruição
    do MediaPipe 0.10.x. Retorna None se a API não estiver disponível.
    """
    global _POSE_LANDMARKER
    if _POSE_LANDMARKER is not None:
        return _POSE_LANDMARKER
    with _POSE_LANDMARKER_LOCK:
        if _POSE_LANDMARKER is not None:
            return _POSE_LANDMARKER
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_pose_model()
            if model_path is None:
                return None

            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.2,
                min_pose_presence_confidence=0.2,
                min_tracking_confidence=0.2,
            )
            _POSE_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(options)
            return _POSE_LANDMARKER
        except Exception:
            logger.debug("Falha ao criar PoseLandmarker", exc_info=True)
            return None


def _pose_landmarks_from_bgr(img_bgr):
    """
    Roda o detector de pose e retorna lista de 33 landmarks (ou None).
    Compatível com mediapipe 0.10.x (Tasks API) e 0.9.x (solutions API).
    Faz upscale automático se a imagem for muito pequena.
    """
    import cv2

    # Upscale se necessário para melhorar a detecção
    h, w = img_bgr.shape[:2]
    if max(h, w) < _POSE_MIN_DIM:
        scale = _POSE_MIN_DIM / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Tenta Tasks API (mediapipe 0.10.x) com instância cacheada ---
    try:
        import mediapipe as mp
        landmarker = _get_pose_landmarker()
        if landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)
            if result.pose_landmarks:
                return result.pose_landmarks[0]  # lista de 33 NormalizedLandmark
            return None
    except (ImportError, AttributeError):
        pass  # Tasks API não disponível, tenta legada
    except Exception:
        logger.debug("Falha na detecção Tasks API", exc_info=True)

    # --- Fallback: solutions API (mediapipe ≤ 0.9.x) ---
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.35,
            enable_segmentation=False,
        ) as pose:
            results = pose.process(img_rgb)
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark
    except Exception:
        return None


def _estimate_pose_features(img_bgr) -> dict:
    """
    Extrai landmarks de pose (MediaPipe) para medir largura de ombros, quadris
    e visibilidade do torso. Retorna defaults neutros se mediapipe não estiver
    instalado ou se nenhuma pose for detectada.
    """
    global _MEDIAPIPE_AVAILABLE
    defaults = _pose_default_features()

    if not _body_analysis_enabled():
        return defaults

    if _MEDIAPIPE_AVAILABLE is False:
        return defaults

    if img_bgr is None:
        return defaults

    try:
        import mediapipe  # noqa: F401 — confirma que mediapipe está instalado
        _MEDIAPIPE_AVAILABLE = True

        lm = _pose_landmarks_from_bgr(img_bgr)
        if lm is None:
            return defaults

        # Acesso por índice — funciona tanto com lista (Tasks) quanto com landmark (solutions)
        ls = lm[_L_SHOULDER]
        rs = lm[_R_SHOULDER]
        lh = lm[_L_HIP]
        rh = lm[_R_HIP]

        # Visibilidade: Tasks API usa .visibility (float 0-1)
        def _vis(landmark) -> float:
            v = getattr(landmark, "visibility", None)
            if v is None:
                return 1.0  # legado sem visibility → assume visível
            return float(v)

        shoulder_vis = min(_vis(ls), _vis(rs))
        hip_vis = min(_vis(lh), _vis(rh))

        torso_ids = [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP, _L_ELBOW, _R_ELBOW]
        torso_visibility = sum(_vis(lm[lid]) for lid in torso_ids) / len(torso_ids)

        shoulder_width = abs(rs.x - ls.x) if shoulder_vis >= 0.35 else 0.0
        hip_width = abs(rh.x - lh.x) if hip_vis >= 0.35 else 0.0

        # shoulder_hip_ratio: >0.5 = ombros mais largos, <0.5 = quadris mais largos
        if shoulder_width > 0.02 and hip_width > 0.02:
            shoulder_hip_ratio = _clamp01(shoulder_width / (shoulder_width + hip_width))
        else:
            shoulder_hip_ratio = 0.5

        # Torso height: distância vertical ombros→quadris (coords normalizadas 0-1)
        torso_height = 0.0
        upper_body_ratio = 0.33
        if shoulder_vis >= 0.35 and hip_vis >= 0.35:
            shoulder_mid_y = (ls.y + rs.y) / 2
            hip_mid_y = (lh.y + rh.y) / 2
            torso_height = _clamp01(abs(hip_mid_y - shoulder_mid_y))
            if torso_height > 0.02 and shoulder_width > 0:
                # ratio ombros/torso: ~0.33 típico; >0.5 = ombros largos p/ altura
                raw = shoulder_width / torso_height
                upper_body_ratio = _clamp01(raw / 3.0)  # [0,3] → [0,1]; ~1.0 = 0.33

        # Pernas visíveis: distância quadris→tornozelos
        la = lm[_L_ANKLE]
        ra = lm[_R_ANKLE]
        ankle_vis = min(_vis(la), _vis(ra))
        leg_ratio = 0.0
        if ankle_vis >= 0.35 and hip_vis >= 0.35:
            hip_mid_y = (lh.y + rh.y) / 2
            ankle_mid_y = (la.y + ra.y) / 2
            leg_ratio = _clamp01(abs(ankle_mid_y - hip_mid_y))

        # Cobertura total do corpo na imagem (ombros ao ponto mais baixo detectado)
        body_coverage = 0.0
        if shoulder_vis >= 0.35:
            top_y = min(ls.y, rs.y)
            if ankle_vis >= 0.35:
                bot_y = max(la.y, ra.y)
            elif hip_vis >= 0.35:
                bot_y = max(lh.y, rh.y)
            else:
                bot_y = top_y
            body_coverage = _clamp01(abs(bot_y - top_y))

        return {
            "photo_pose_shoulder_width": round(_clamp01(shoulder_width), 4),
            "photo_pose_hip_width": round(_clamp01(hip_width), 4),
            "photo_pose_shoulder_hip_ratio": round(shoulder_hip_ratio, 4),
            "photo_pose_torso_visibility": round(_clamp01(torso_visibility), 4),
            "photo_pose_torso_height": round(torso_height, 4),
            "photo_pose_upper_body_ratio": round(upper_body_ratio, 4),
            "photo_pose_leg_ratio": round(leg_ratio, 4),
            "photo_pose_body_coverage": round(body_coverage, 4),
        }

    except ImportError:
        _MEDIAPIPE_AVAILABLE = False
        logger.debug("mediapipe não instalado; pose features desativadas")
        return defaults
    except Exception:
        logger.debug("Falha na estimativa de pose", exc_info=True)
        return defaults


def _body_width_buckets(width_ratio: float, signal_quality: float) -> dict:
    buckets = {
        "photo_body_width_bucket_narrow": 0.0,
        "photo_body_width_bucket_medium": 0.0,
        "photo_body_width_bucket_wide": 0.0,
    }
    if not _semantic_analysis_enabled() or signal_quality < _body_width_min_quality():
        return buckets

    width_ratio = _clamp01(width_ratio)
    if width_ratio < 0.36:
        buckets["photo_body_width_bucket_narrow"] = 1.0
    elif width_ratio < 0.48:
        buckets["photo_body_width_bucket_narrow"] = 0.5
        buckets["photo_body_width_bucket_medium"] = 0.5
    elif width_ratio < 0.58:
        buckets["photo_body_width_bucket_medium"] = 1.0
    elif width_ratio < 0.66:
        buckets["photo_body_width_bucket_medium"] = 0.5
        buckets["photo_body_width_bucket_wide"] = 0.5
    else:
        buckets["photo_body_width_bucket_wide"] = 1.0
    return buckets


def _image_quality_features(img_bgr) -> dict:
    features = _semantic_default_features()
    if img_bgr is None or not _semantic_analysis_enabled():
        return features

    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean()) / 255.0
        contrast = _clamp01(float(gray.std()) / 80.0)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = _clamp01(np.log1p(lap_var) / np.log1p(1000.0))

        b, g, r = cv2.split(img_bgr.astype("float"))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        colorfulness = _clamp01(float(colorfulness) / 100.0)

        return {
            **features,
            "photo_image_brightness": round(_clamp01(brightness), 4),
            "photo_image_contrast": round(contrast, 4),
            "photo_image_sharpness": round(sharpness, 4),
            "photo_image_colorfulness": round(colorfulness, 4),
        }
    except Exception:
        logger.debug("Falha ao calcular qualidade leve da imagem", exc_info=True)
        return features


def _opencv_smile_score(img_bgr, region: dict | None) -> float:
    if img_bgr is None or not region or not _semantic_analysis_enabled():
        return _SEMANTIC_FEATURE_DEFAULTS["photo_face_smile_score"]

    try:
        import cv2

        x = max(int(region.get("x", 0) or 0), 0)
        y = max(int(region.get("y", 0) or 0), 0)
        w = int(region.get("w", 0) or 0)
        h = int(region.get("h", 0) or 0)
        if w <= 0 or h <= 0:
            return _SEMANTIC_FEATURE_DEFAULTS["photo_face_smile_score"]

        face_crop = img_bgr[y : y + h, x : x + w]
        if face_crop.size == 0:
            return _SEMANTIC_FEATURE_DEFAULTS["photo_face_smile_score"]

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
        smile_cascade = cv2.CascadeClassifier(cascade_path)
        if smile_cascade.empty():
            return _SEMANTIC_FEATURE_DEFAULTS["photo_face_smile_score"]

        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(max(12, int(w * 0.18)), max(8, int(h * 0.06))),
        )
        if len(smiles) == 0:
            return _SEMANTIC_FEATURE_DEFAULTS["photo_face_smile_score"]

        largest_area = max(float(sw * sh) for _, _, sw, sh in smiles)
        face_area = max(float(w * h), 1.0)
        return round(_clamp01(0.60 + (largest_area / face_area) * 3.0), 4)
    except Exception:
        logger.debug("Falha no fallback OpenCV de sorriso", exc_info=True)
        return _SEMANTIC_FEATURE_DEFAULTS["photo_face_smile_score"]


def _smile_score(analyzed: dict, img_bgr=None, region: dict | None = None) -> float:
    if not _semantic_analysis_enabled():
        return _SEMANTIC_FEATURE_DEFAULTS["photo_face_smile_score"]

    if not _semantic_emotion_enabled():
        return _opencv_smile_score(img_bgr, region)

    try:
        emotion = analyzed.get("emotion") or {}
        happy = emotion.get("happy", emotion.get("Happy", None))
        if happy is None:
            return _opencv_smile_score(img_bgr, region)
        return round(_clamp01(float(happy) / 100.0), 4)
    except Exception:
        return _opencv_smile_score(img_bgr, region)


def _estimate_body_features(img_bgr, face_region: dict | None) -> dict:
    """
    Extrai sinais leves de composicao corporal a partir da geometria da foto.

    Estes campos nao tentam medir peso real/BMI. Eles descrevem o quanto o corpo
    aparece na imagem e um proxy visual de largura quando ha regiao corporal
    visivel suficiente para o ML aprender preferencias sem travar o swipe.
    """
    features = _body_default_features()
    if img_bgr is None or not _body_analysis_enabled():
        return features

    try:
        import cv2
        import numpy as np

        img_h, img_w = img_bgr.shape[:2]
        region = face_region or {}
        x = max(int(region.get("x", 0) or 0), 0)
        y = max(int(region.get("y", 0) or 0), 0)
        face_w = int(region.get("w", 0) or 0)
        face_h = int(region.get("h", 0) or 0)

        if img_w <= 0 or img_h <= 0 or face_w <= 0 or face_h <= 0:
            return features

        face_bottom = min(img_h, y + face_h)
        face_center_x = max(0, min(img_w - 1, x + face_w // 2))
        face_h_ratio = face_h / max(img_h, 1)
        space_below = (img_h - face_bottom) / max(img_h, 1)

        body_visible = _clamp01((space_below - 0.18) / 0.45) * _clamp01((0.38 - face_h_ratio) / 0.28)
        body_full = _clamp01((space_below - 0.50) / 0.35) * _clamp01((0.16 - face_h_ratio) / 0.10)
        body_upper = (
            _clamp01((space_below - 0.25) / 0.35)
            * _clamp01((0.30 - face_h_ratio) / 0.20)
            * (1.0 - 0.35 * body_full)
        )
        closeup = max(
            _clamp01((face_h_ratio - 0.18) / 0.18),
            _clamp01((0.30 - space_below) / 0.30),
        )

        width_ratio = 0.5
        width_quality = 0.0
        if body_visible > 0.12:
            body_top = min(img_h - 1, int(y + face_h * 0.85))
            body_bottom = min(
                img_h,
                int(y + face_h * (6.0 if face_h_ratio < 0.13 else 4.8)),
            )
            half_width = int(max(face_w * 2.7, img_w * 0.18))
            left = max(0, face_center_x - half_width)
            right = min(img_w, face_center_x + half_width)

            if body_bottom - body_top >= max(24, int(face_h * 0.75)) and right > left:
                roi = img_bgr[body_top:body_bottom, left:right]
                if roi.size:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(gray, 45, 135)
                    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

                    col_activity = edges.mean(axis=0)
                    cx_roi = face_center_x - left
                    lo = max(0, int(cx_roi - face_w * 2.4))
                    hi = min(len(col_activity), int(cx_roi + face_w * 2.4))
                    band = col_activity[lo:hi]

                    if band.size >= 12:
                        threshold = max(
                            float(np.percentile(band, 65)),
                            float(band.mean() + band.std() * 0.25),
                            1.0,
                        )
                        active = np.where(band > threshold)[0] + lo
                        if active.size >= 4:
                            span = float(np.percentile(active, 90) - np.percentile(active, 10))
                            span_face_ratio = span / max(float(face_w), 1.0)
                            width_ratio = _clamp01((span_face_ratio - 1.4) / 3.0)
                            edge_density = float((edges > 0).mean())
                            active_ratio = active.size / max(float(band.size), 1.0)
                            width_quality = (
                                body_visible
                                * _clamp01(active_ratio * 4.0)
                                * _clamp01(edge_density * 12.0)
                            )

        signal_quality = max(width_quality, body_visible * 0.35, body_full * 0.50)
        signal_quality = _clamp01(signal_quality)

        # Proporção de pele visível na região do corpo (indica estilo de roupa)
        skin_ratio = 0.0
        if body_visible > 0.12:
            body_top = min(img_h - 1, int(y + face_h * 0.85))
            body_bottom = min(img_h, int(y + face_h * (6.0 if face_h_ratio < 0.13 else 4.8)))
            half_width = int(max(face_w * 2.7, img_w * 0.18))
            bx_left = max(0, face_center_x - half_width)
            bx_right = min(img_w, face_center_x + half_width)
            if body_bottom - body_top >= 16 and bx_right > bx_left:
                body_roi = img_bgr[body_top:body_bottom, bx_left:bx_right]
                if body_roi.size:
                    hsv = cv2.cvtColor(body_roi, cv2.COLOR_BGR2HSV)
                    skin_mask = cv2.inRange(
                        hsv,
                        np.array([0, 18, 40], dtype=np.uint8),
                        np.array([25, 255, 255], dtype=np.uint8),
                    )
                    skin_ratio = float((skin_mask > 0).mean())

        return {
            "photo_body_visible": round(_clamp01(body_visible), 4),
            "photo_body_full_length": round(_clamp01(body_full), 4),
            "photo_body_upper_length": round(_clamp01(body_upper), 4),
            "photo_body_closeup": round(_clamp01(closeup), 4),
            "photo_body_width_ratio": round(_clamp01(width_ratio), 4),
            "photo_body_signal_quality": round(signal_quality, 4),
            "photo_body_skin_ratio": round(_clamp01(skin_ratio), 4),
            **_body_width_buckets(width_ratio, signal_quality),
        }
    except Exception:
        logger.debug("Falha na estimativa leve de corpo", exc_info=True)
        return features


def _weighted_average(values: list[tuple[float, float]], neutral: float = 0.5) -> float:
    total = sum(weight for _, weight in values if weight > 0)
    if total <= 0:
        return neutral
    return sum(value * weight for value, weight in values if weight > 0) / total


def _default_features() -> dict:
    return {
        "photo_has_face": 0,
        "photo_woman_confidence": 0.5,
        "photo_skin_lightness": 0.5,
        "photo_face_similarity": 0.5,
        "photo_faces_ratio": 0.0,
        "photo_failure_ratio": 0.0,
        "photo_gender_certainty": 0.0,
        "photo_race_black": _RACE_NEUTRAL,
        "photo_race_white": _RACE_NEUTRAL,
        "photo_race_asian": _RACE_NEUTRAL,
        "photo_race_indian": _RACE_NEUTRAL,
        "photo_race_middleeastern": _RACE_NEUTRAL,
        "photo_race_latina": _RACE_NEUTRAL,
        **_body_default_features(),
        **_semantic_default_features(),
        **_pose_default_features(),
        **_seg_default_features(),
        **_carousel_default_features(),
        "_dominant_race": "",
        "_embedding": None,
        "_semantic_embedding": None,
        "_face_confidence": 0.0,
        "_analysis_failed": False,
        "_timed_out": False,
        "_download_failed": False,
    }


def _unavailable_features(reason: str) -> dict:
    """
    Retorna features neutras quando a foto nao pôde ser analisada.

    Importante: isso é diferente de "foto analisada e sem rosto". Quando a
    CDN/DeepFace falha, nao devemos transformar isso em dislike automatico.
    """
    return {
        **_default_features(),
        "photo_has_face": 1,
        "photo_woman_confidence": 0.5,
        "photo_skin_lightness": 0.5,
        "photo_face_similarity": 0.5,
        "photo_faces_ratio": 0.0,
        "photo_failure_ratio": 1.0,
        "photo_gender_certainty": 0.0,
        "_analysis_failed": True,
        "_failure_reason": reason,
    }


def _compute_embedding_bgr(img_bgr, backend: str):
    """Calcula embedding facial pesado. Deve ser chamado poucas vezes."""
    try:
        import numpy as np
        from deepface import DeepFace

        logger.debug("DeepFace.represent iniciado backend=%s", backend)
        repr_result = DeepFace.represent(
            img_bgr,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend=backend,
        )
        if repr_result and repr_result[0].get("face_confidence", 0) >= 0.5:
            return np.array(repr_result[0]["embedding"])
    except Exception:
        logger.exception("DeepFace.represent falhou backend=%s", backend)
    return None


def _compute_embedding_for_url(url: str):
    """
    Baixa a melhor foto e calcula apenas o embedding, com timeout próprio.
    Retorna None quando o embedding falha ou demora demais.
    """
    if not url:
        return None

    import time
    started_at = time.perf_counter()
    cfg = get_photos_config()
    timeout_s = float(cfg.get("embedding_timeout_seconds", 20))
    embedding_detector_backend = str(
        cfg.get("embedding_detector_backend", cfg.get("detector_backend", "opencv")) or "opencv"
    )

    def _worker():
        img_bgr = _download_as_bgr(url)
        if img_bgr is None:
            return None
        return _compute_embedding_bgr(img_bgr, embedding_detector_backend)

    if timeout_s <= 0:
        return _worker()

    result_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)

    def _thread_worker() -> None:
        try:
            result_queue.put(_worker(), block=False)
        except BaseException as exc:
            try:
                result_queue.put(exc, block=False)
            except Exception:
                pass

    worker = threading.Thread(
        target=_thread_worker,
        daemon=True,
        name=f"photo-embedding-{_cache_key(url)[-16:]}",
    )
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        logger.warning(
            "Timeout no embedding facial: key=%s timeout=%.1fs elapsed=%.3fs",
            _cache_key(url),
            timeout_s,
            time.perf_counter() - started_at,
        )
        return None

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        logger.warning("Embedding facial terminou sem resultado: key=%s", _cache_key(url))
        return None

    if isinstance(result, BaseException):
        logger.exception("Embedding facial falhou em worker", exc_info=(type(result), result, result.__traceback__))
        return None

    logger.info(
        "Embedding facial concluido key=%s has_embedding=%s elapsed=%.3fs",
        _cache_key(url),
        result is not None,
        time.perf_counter() - started_at,
    )
    return result


def _semantic_embedding_features_for_bgr(url: str, img_bgr) -> dict:
    """Calcula embedding CLIP separado do cache principal de foto."""
    features = {**_carousel_default_features(), "_semantic_embedding": None}
    try:
        from photo_semantic_embeddings import (
            embedding_for_bgr,
            semantic_embedding_enabled,
            source_key_for_url,
        )

        if not semantic_embedding_enabled():
            return features
        embedding = embedding_for_bgr(source_key_for_url(url), img_bgr)
        if embedding is None:
            return features
        features["_semantic_embedding"] = embedding
        features["photo_semantic_embedding_saved"] = 1.0
        features["photo_carousel_useful_count"] = 1.0
        return features
    except Exception:
        logger.debug("Embedding semantico visual indisponivel para foto", exc_info=True)
        return features


def _semantic_aggregate_from_results(results: list[dict]) -> tuple[object | None, dict]:
    try:
        from photo_semantic_embeddings import aggregate_embeddings

        embeddings = [
            item.get("_semantic_embedding")
            for item in results
            if not item.get("_analysis_failed")
        ]
        return aggregate_embeddings(embeddings)
    except Exception:
        logger.debug("Falha ao agregar embeddings semanticos do carrossel", exc_info=True)
        return None, _carousel_default_features()


def _analyze_bgr(img_bgr, include_embedding: bool | None = None) -> dict:
    """
    Analisa uma imagem já carregada em BGR e retorna features numéricas.
    Retorna defaults neutros se a análise falhar — nunca bloqueia o pipeline.

    Features retornadas (ML):
      photo_has_face           : 1 se rosto detectado, 0 se não
      photo_woman_confidence   : 0–1, confiança que é mulher
      photo_skin_lightness     : 0–1, luminância média da pele (0=escuro, 1=claro)
      photo_face_similarity    : 0–1, similaridade ao vetor de preferência visual
      photo_race_black         : 0–1, probabilidade de pessoa negra
      photo_race_white         : 0–1, probabilidade de pessoa branca
      photo_race_asian         : 0–1, probabilidade de pessoa asiática
      photo_race_indian        : 0–1, probabilidade de pessoa indiana
      photo_race_middleeastern : 0–1, probabilidade de Oriente Médio
      photo_race_latina        : 0–1, probabilidade de pessoa latina/hispânica
      photo_face_smile_score   : 0–1, expressão/sorriso via DeepFace emotion
      photo_image_*            : 0–1, brilho/contraste/nitidez/vivacidade
      photo_body_width_bucket_*: buckets neutros de silhueta visual quando confiável

    Campos internos (não são features ML):
      _dominant_race  : string, etnia dominante estimada
      _embedding      : np.ndarray (512 dims) — usado para atualizar preferência
    """
    if img_bgr is None:
        return _default_features()

    try:
        import numpy as np
        from deepface import DeepFace
        from face_embeddings import compute_similarity

        image_quality_features = _image_quality_features(img_bgr)
        cfg = get_photos_config()
        detector_backend = str(cfg.get("detector_backend", "opencv") or "opencv")
        detector_fallback_backend = str(cfg.get("detector_fallback_backend", "") or "").strip()
        embedding_strategy = str(cfg.get("embedding_strategy", "best_photo") or "best_photo")
        enable_embedding = bool(cfg.get("enable_face_embedding", False))
        if include_embedding is None:
            include_embedding = enable_embedding and embedding_strategy in {"all_photos", "per_photo"}
        embedding_detector_backend = str(
            cfg.get("embedding_detector_backend", detector_backend) or detector_backend
        )

        def _analyze_with_backend(backend: str) -> tuple[dict, float]:
            global _EMOTION_ANALYSIS_AVAILABLE
            logger.debug("DeepFace.analyze iniciado backend=%s", backend)

            def _run(actions: list[str]) -> tuple[dict, float]:
                result = DeepFace.analyze(
                    img_bgr,
                    actions=actions,
                    enforce_detection=False,
                    detector_backend=backend,
                    silent=True,
                )
                analyzed = result[0] if isinstance(result, list) else result
                return analyzed, float(analyzed.get("face_confidence", 0) or 0)

            actions = ["gender", "race"]
            if not _semantic_emotion_enabled():
                return _run(actions)

            try:
                analyzed, conf = _run(actions + ["emotion"])
                _EMOTION_ANALYSIS_AVAILABLE = True
                return analyzed, conf
            except Exception as exc:
                _EMOTION_ANALYSIS_AVAILABLE = False
                logger.warning(
                    "DeepFace emotion indisponivel; seguindo sem sorriso/expressao: %s",
                    exc,
                )
                logger.debug("Falha detalhada em DeepFace emotion", exc_info=True)
                return _run(actions)

        try:
            r, face_conf = _analyze_with_backend(detector_backend)
        except Exception as e:
            logger.exception("DeepFace.analyze falhou backend=%s", detector_backend)
            r, face_conf = {}, 0.0

        used_backend = detector_backend
        if (
            face_conf < 0.5
            and detector_fallback_backend
            and detector_fallback_backend != detector_backend
        ):
            try:
                logger.info(
                    "Rosto nao confirmado com backend=%s conf=%.3f; tentando fallback=%s",
                    detector_backend,
                    face_conf,
                    detector_fallback_backend,
                )
                fallback_r, fallback_conf = _analyze_with_backend(detector_fallback_backend)
                if fallback_conf > face_conf:
                    r, face_conf = fallback_r, fallback_conf
                    used_backend = detector_fallback_backend
            except Exception:
                logger.exception("DeepFace.analyze fallback falhou backend=%s", detector_fallback_backend)

        dominant_race = r.get("dominant_race", "?")

        if face_conf < 0.5:
            logger.debug(
                "Rosto rejeitado por baixa confianca: %.3f backend=%s",
                float(face_conf or 0),
                used_backend,
            )
            return {
                **_default_features(),
                **image_quality_features,
                "photo_has_face": 0,
                "photo_faces_ratio": 0.0,
                "photo_failure_ratio": 0.0,
                "photo_gender_certainty": 0.0,
            }

        woman_conf = r.get("gender", {}).get("Woman", 50.0) / 100.0

        race_raw = r.get("race", {})
        race_black          = race_raw.get("black",           100 * _RACE_NEUTRAL) / 100.0
        race_white          = race_raw.get("white",           100 * _RACE_NEUTRAL) / 100.0
        race_asian          = race_raw.get("asian",           100 * _RACE_NEUTRAL) / 100.0
        race_indian         = race_raw.get("indian",          100 * _RACE_NEUTRAL) / 100.0
        race_middleeastern  = race_raw.get("middle eastern",  100 * _RACE_NEUTRAL) / 100.0
        race_latina         = race_raw.get("latino hispanic", 100 * _RACE_NEUTRAL) / 100.0

        region = r.get("region", {})
        lightness = _skin_lightness(img_bgr, region) if region else 0.5
        body_features = _estimate_body_features(img_bgr, region)
        pose_features = _estimate_pose_features(img_bgr)
        smile_score = _smile_score(r, img_bgr, region)
        body_visible = float(body_features.get("photo_body_visible", 0.0))
        seg_features = (
            _estimate_segmentation_features(img_bgr)
            if body_visible > 0.25
            else _seg_default_features()
        )

        embedding_np = None
        if include_embedding:
            embedding_np = _compute_embedding_bgr(img_bgr, embedding_detector_backend)
        else:
            logger.debug("Embedding facial desativado; usando similaridade neutra")

        similarity = compute_similarity(embedding_np)
        logger.debug(
            "DeepFace concluido: backend=%s face_conf=%.3f woman=%.3f race=%s similarity=%.3f",
            used_backend,
            float(face_conf or 0),
            float(woman_conf or 0),
            dominant_race,
            float(similarity or 0),
        )

        return {
            "photo_has_face": 1,
            "photo_woman_confidence": round(woman_conf, 4),
            "photo_skin_lightness": round(lightness, 4),
            "photo_face_similarity": similarity,
            "photo_faces_ratio": 1.0,
            "photo_failure_ratio": 0.0,
            "photo_gender_certainty": round(abs(woman_conf - 0.5) * 2, 4),
            "photo_race_black": round(race_black, 4),
            "photo_race_white": round(race_white, 4),
            "photo_race_asian": round(race_asian, 4),
            "photo_race_indian": round(race_indian, 4),
            "photo_race_middleeastern": round(race_middleeastern, 4),
            "photo_race_latina": round(race_latina, 4),
            **body_features,
            **pose_features,
            **seg_features,
            **image_quality_features,
            "photo_face_smile_score": smile_score,
            "_dominant_race": dominant_race,
            "_embedding": embedding_np,
            "_face_confidence": round(float(face_conf or 0), 4),
        }

    except Exception as e:
        logger.exception("Erro geral na analise de foto")
        return _default_features()


def analyze_photo(url: str, profile_age: int = 0) -> dict:
    """
    Analisa a foto remota e retorna features numéricas para o modelo ML.
    """
    if not url:
        return _default_features()

    import time
    started_at = time.perf_counter()
    logger.info("Analise de foto iniciada key=%s", _cache_key(url))
    cached = _get_cached_photo(url)
    if cached is not None:
        logger.info("Analise de foto retornada do cache key=%s elapsed=%.3fs", _cache_key(url), time.perf_counter() - started_at)
        return cached

    cfg = get_photos_config()
    timeout_s = float(cfg.get("analysis_timeout_seconds", 25))

    if timeout_s <= 0:
        try:
            result = _analyze_photo_uncached(url)
            logger.info(
                "Analise de foto concluida key=%s has_face=%s woman=%.3f elapsed=%.3fs",
                _cache_key(url),
                result.get("photo_has_face"),
                float(result.get("photo_woman_confidence", 0.5)),
                time.perf_counter() - started_at,
            )
            return result
        except Exception:
            logger.exception("Analise de foto falhou sem timeout key=%s", _cache_key(url))
            return _unavailable_features("worker_exception")

    result_queue: "queue.Queue[dict | BaseException]" = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result_queue.put(_analyze_photo_uncached(url), block=False)
        except BaseException as exc:
            try:
                result_queue.put(exc, block=False)
            except Exception:
                pass

    worker = threading.Thread(
        target=_worker,
        daemon=True,
        name=f"photo-analysis-{_cache_key(url)[-16:]}",
    )
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        logger.warning(
            "Timeout na analise de foto: key=%s timeout=%.1fs elapsed=%.3fs",
            _cache_key(url),
            timeout_s,
            time.perf_counter() - started_at,
        )
        return {**_unavailable_features("timeout"), "_timed_out": True}

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        logger.warning("Analise de foto terminou sem resultado: key=%s", _cache_key(url))
        return _unavailable_features("empty_result")

    if isinstance(result, BaseException):
        logger.exception("Analise de foto falhou em worker", exc_info=(type(result), result, result.__traceback__))
        return _unavailable_features("worker_exception")

    logger.info(
        "Analise de foto concluida key=%s has_face=%s woman=%.3f elapsed=%.3fs",
        _cache_key(url),
        result.get("photo_has_face"),
        float(result.get("photo_woman_confidence", 0.5)),
        time.perf_counter() - started_at,
    )
    return result


def _analyze_photo_uncached(url: str) -> dict:
    """Baixa, analisa e grava cache. Deve ser chamado dentro do worker com timeout."""
    img_bgr = _download_as_bgr(url)
    if img_bgr is None:
        logger.warning("Analise de foto sem imagem key=%s", _cache_key(url))
        return {**_unavailable_features("download_failed"), "_download_failed": True}

    features = _analyze_bgr(img_bgr)
    features.update(_semantic_embedding_features_for_bgr(url, img_bgr))
    _put_cached_photo(url, features)
    return features


def analyze_pose_only(path: str | Path) -> dict:
    """
    Extrai apenas features de pose (MediaPipe) de uma foto local.
    Muito mais rápido que analyze_local_photo — não roda DeepFace.
    Útil para backfill de pose em fotos corporais já salvas.
    Retorna dict com photo_pose_* features (8 keys).
    """
    img_bgr = _load_path_as_bgr(path)
    if img_bgr is None:
        return _pose_default_features()
    return _estimate_pose_features(img_bgr)


def analyze_local_photo(path: str | Path, profile_age: int = 0, include_embedding: bool = True) -> dict:
    """
    Analisa uma foto já salva em disco.
    Usado para backfill do histórico em data/photos.
    """
    logger.info("Analise de foto local iniciada: %s", path)
    # Em backfills locais existe apenas uma imagem por chamada; forçamos o
    # embedding para tirar históricos antigos do fallback neutro de 0.5.
    img_bgr = _load_path_as_bgr(path)
    features = _analyze_bgr(img_bgr, include_embedding=include_embedding)
    try:
        from photo_semantic_embeddings import embedding_for_bgr, semantic_embedding_enabled, source_key_for_path

        if semantic_embedding_enabled():
            semantic_embedding = embedding_for_bgr(source_key_for_path(path), img_bgr)
            if semantic_embedding is not None:
                features["_semantic_embedding"] = semantic_embedding
                features["photo_semantic_embedding_saved"] = 1.0
                features["photo_carousel_useful_count"] = 1.0
    except Exception:
        logger.debug("Falha ao enriquecer foto local com embedding semantico: %s", path, exc_info=True)
    logger.info("Analise de foto local concluida: %s has_face=%s", path, features.get("photo_has_face"))
    return features


_RACE_KEYS = [
    "photo_race_black", "photo_race_white", "photo_race_asian",
    "photo_race_indian", "photo_race_middleeastern", "photo_race_latina",
]

_RACE_KEY_TO_LABEL = {
    "photo_race_black": "black",
    "photo_race_white": "white",
    "photo_race_asian": "asian",
    "photo_race_indian": "indian",
    "photo_race_middleeastern": "middle eastern",
    "photo_race_latina": "latina",
}


def _avg_result_feature(results: list[dict], key: str, default: float = 0.5) -> float:
    values = []
    for item in results:
        if item.get("_analysis_failed"):
            continue
        try:
            raw = item.get(key, default)
            if raw in ("", None):
                continue
            values.append(float(raw))
        except Exception:
            continue
    if not values:
        return default
    return sum(values) / len(values)


def _photo_brightness_quality(value: float) -> float:
    """Pontua iluminacao perto do meio, penalizando muito escura/clara."""
    value = _clamp01(value)
    return _clamp01(1.0 - abs(value - 0.52) / 0.52)


def _photo_looks_nearly_black(item: dict) -> bool:
    """True quando a imagem parece sem conteúdo útil por escuridão extrema."""
    if item.get("_analysis_failed"):
        return False
    brightness = _clamp01(float(item.get("photo_image_brightness", 0.5) or 0.5))
    contrast = _clamp01(float(item.get("photo_image_contrast", 0.5) or 0.5))
    colorfulness = _clamp01(float(item.get("photo_image_colorfulness", 0.5) or 0.5))
    return brightness <= 0.055 or (brightness <= 0.09 and contrast <= 0.12 and colorfulness <= 0.12)


def _content_photo_score(item: dict) -> float:
    """Score leve para fallback de review quando nao ha rosto/corpo bom."""
    if item.get("_analysis_failed") or _photo_looks_nearly_black(item):
        return 0.0
    brightness = _photo_brightness_quality(float(item.get("photo_image_brightness", 0.5) or 0.5))
    contrast = _clamp01(float(item.get("photo_image_contrast", 0.5) or 0.5))
    sharpness = _clamp01(float(item.get("photo_image_sharpness", 0.5) or 0.5))
    colorfulness = _clamp01(float(item.get("photo_image_colorfulness", 0.5) or 0.5))
    has_face = 1.0 if item.get("photo_has_face") else 0.0
    return round(
        _clamp01(
            brightness * 0.36
            + sharpness * 0.24
            + contrast * 0.18
            + colorfulness * 0.12
            + has_face * 0.10
        ),
        4,
    )


def _face_photo_score(item: dict) -> float:
    """Score leve para escolher a melhor foto de rosto salva no review."""
    if item.get("_analysis_failed") or not item.get("photo_has_face") or _photo_looks_nearly_black(item):
        return 0.0
    face_conf = _clamp01(float(item.get("_face_confidence", 0.0) or 0.0))
    woman_conf = _clamp01(float(item.get("photo_woman_confidence", 0.5) or 0.5))
    gender_certainty = _clamp01(float(item.get("photo_gender_certainty", 0.0) or 0.0))
    sharpness = _clamp01(float(item.get("photo_image_sharpness", 0.5) or 0.5))
    brightness = _photo_brightness_quality(float(item.get("photo_image_brightness", 0.5) or 0.5))
    return round(
        _clamp01(
            face_conf * 0.52
            + woman_conf * 0.20
            + gender_certainty * 0.10
            + sharpness * 0.12
            + brightness * 0.06
        ),
        4,
    )


def _body_photo_score(item: dict) -> float:
    """Score leve para escolher foto com melhor sinal de corpo/composicao."""
    if item.get("_analysis_failed") or _photo_looks_nearly_black(item):
        return 0.0
    measurement_strength = body_measurement_strength(item)
    quality = _clamp01(float(item.get("photo_body_signal_quality", 0.0) or 0.0))
    if measurement_strength < 0.50 or quality < 0.32:
        return 0.0
    visible = _clamp01(float(item.get("photo_body_visible", 0.0) or 0.0))
    full = _clamp01(float(item.get("photo_body_full_length", 0.0) or 0.0))
    upper = _clamp01(float(item.get("photo_body_upper_length", 0.0) or 0.0))
    closeup = _clamp01(float(item.get("photo_body_closeup", 0.0) or 0.0))
    sharpness = _clamp01(float(item.get("photo_image_sharpness", 0.5) or 0.5))
    brightness = _photo_brightness_quality(float(item.get("photo_image_brightness", 0.5) or 0.5))
    return round(
        _clamp01(
            quality * 0.42
            + visible * 0.30
            + full * 0.14
            + upper * 0.10
            + sharpness * 0.04
            + brightness * 0.03
            + measurement_strength * 0.12
            - closeup * 0.08
        ),
        4,
    )


def analyze_photos(urls: list[str], profile_age: int = 0, max_photos: int = 4) -> dict:
    """
    Analisa até max_photos fotos em paralelo e agrega os resultados.

    Estratégia de agregação:
      photo_has_face          : 1 se qualquer foto tiver rosto
      photo_woman_confidence  : máximo (usa a detecção mais confiante)
      photo_skin_lightness    : média (mais estável que uma foto só)
      photo_face_similarity   : média entre as fotos com rosto
      photo_race_*            : média das probabilidades por etnia
      _dominant_race          : etnia com maior probabilidade média
      _embedding              : média normalizada (representação mais robusta)
    """
    defaults = {
        "photo_has_face": 0,
        "photo_woman_confidence": 0.5,
        "photo_skin_lightness": 0.5,
        "photo_face_similarity": 0.5,
        "photo_faces_ratio": 0.0,
        "photo_failure_ratio": 0.0,
        "photo_gender_certainty": 0.0,
        **{k: _RACE_NEUTRAL for k in _RACE_KEYS},
        **_body_default_features(),
        **_semantic_default_features(),
        **_pose_default_features(),
        **_seg_default_features(),
        **_carousel_default_features(),
        "_dominant_race": "",
        "_embedding": None,
        "_semantic_embedding": None,
        "_faces_found": 0,
        "_photos_analyzed": 0,
        "_photos_failed": 0,
        "_photos_timed_out": 0,
        "_photo_analysis_incomplete": False,
        "_analysis_failed": False,
        "_best_face_photo_url": "",
        "_best_body_photo_url": "",
        "_review_photo_url": "",
        "_best_face_photo_score": 0.0,
        "_best_body_photo_score": 0.0,
        "_best_content_photo_url": "",
        "_best_content_photo_score": 0.0,
    }

    to_analyze = [u for u in urls if u][:max_photos]
    if not to_analyze:
        logger.info("analyze_photos sem URLs validas")
        return defaults

    import time
    started_at = time.perf_counter()
    cfg = get_photos_config()
    workers = max(1, min(len(to_analyze), int(cfg.get("analysis_parallel_workers", 4))))
    logger.info(
        "Analise de conjunto de fotos iniciada: count=%s max=%s workers=%s",
        len(to_analyze),
        max_photos,
        workers,
    )

    results = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="photo-set") as executor:
        future_to_url = {
            executor.submit(analyze_photo, url, profile_age): url
            for url in to_analyze
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                result["_source_url"] = url
                results.append(result)
            except Exception:
                logger.exception("Falha inesperada ao analisar foto do conjunto key=%s", _cache_key(url))
                result = _unavailable_features("future_exception")
                result["_source_url"] = url
                results.append(result)

    content_candidates = [r for r in results if _content_photo_score(r) > 0]
    best_content = max(content_candidates, key=_content_photo_score) if content_candidates else None
    content_score = _content_photo_score(best_content) if best_content else 0.0
    content_url = best_content.get("_source_url", "") if best_content else ""

    faces = [r for r in results if r.get("photo_has_face")]
    real_faces = [r for r in faces if not r.get("_analysis_failed")]
    failed_count = sum(1 for r in results if r.get("_analysis_failed"))
    timed_out_count = sum(1 for r in results if r.get("_timed_out"))
    download_failed_count = sum(1 for r in results if r.get("_download_failed"))
    semantic_embedding, semantic_carousel_features = _semantic_aggregate_from_results(results)

    if not real_faces:
        if failed_count:
            logger.warning(
                "Analise de conjunto indisponivel: failed=%s timed_out=%s download_failed=%s analyzed=%s elapsed=%.2fs",
                failed_count,
                timed_out_count,
                download_failed_count,
                len(to_analyze),
                time.perf_counter() - started_at,
            )
            return {
                **_unavailable_features("all_photos_failed"),
                "_photos_analyzed": len(to_analyze),
                "_photos_failed": failed_count,
                "_photos_timed_out": timed_out_count,
                "photo_faces_ratio": 0.0,
                "photo_failure_ratio": 1.0,
                "photo_gender_certainty": 0.0,
                **semantic_carousel_features,
                "_semantic_embedding": semantic_embedding,
                "_photo_analysis_incomplete": True,
            }

        logger.info(
            "Analise de conjunto sem rostos: analyzed=%s elapsed=%.2fs",
            len(to_analyze),
            time.perf_counter() - started_at,
        )
        return {
            **defaults,
            "_photos_analyzed": len(to_analyze),
            "photo_faces_ratio": 0.0,
            "photo_failure_ratio": 0.0,
            "photo_gender_certainty": 0.0,
            "photo_image_brightness": round(_avg_result_feature(results, "photo_image_brightness", 0.5), 4),
            "photo_image_contrast": round(_avg_result_feature(results, "photo_image_contrast", 0.5), 4),
            "photo_image_sharpness": round(_avg_result_feature(results, "photo_image_sharpness", 0.5), 4),
            "photo_image_colorfulness": round(_avg_result_feature(results, "photo_image_colorfulness", 0.5), 4),
            **semantic_carousel_features,
            "_semantic_embedding": semantic_embedding,
            "_review_photo_url": content_url or (to_analyze[0] if to_analyze else ""),
            "_best_content_photo_url": content_url,
            "_best_content_photo_score": round(content_score, 4),
        }

    cfg = get_photos_config()
    embedding_strategy = str(cfg.get("embedding_strategy", "best_photo") or "best_photo")
    enable_embedding = bool(cfg.get("enable_face_embedding", False))
    face_save_min_conf = float(cfg.get("face_save_min_confidence", 0.5) or 0.5)
    body_save_min_quality = float(cfg.get("body_save_min_quality", 0.35) or 0.35)
    body_save_min_strength = float(
        cfg.get("body_save_min_strength", BODY_MEASUREMENT_MIN_STRENGTH)
        or BODY_MEASUREMENT_MIN_STRENGTH
    )
    best_face = None
    if enable_embedding and embedding_strategy == "best_photo":
        from face_embeddings import compute_similarity

        best_face = max(
            real_faces,
            key=lambda f: float(f.get("_face_confidence", 0) or 0),
        )
        best_url = best_face.get("_source_url", "")
        if best_url and best_face.get("_embedding") is None:
            logger.info(
                "Calculando embedding apenas da melhor foto: key=%s face_conf=%.3f",
                _cache_key(best_url),
                float(best_face.get("_face_confidence", 0) or 0),
            )
            embedding = _compute_embedding_for_url(best_url)
            if embedding is not None:
                best_face["_embedding"] = embedding
                best_face["photo_face_similarity"] = compute_similarity(embedding)
                _put_cached_photo(best_url, best_face)

    best_face_for_save = max(real_faces, key=_face_photo_score)
    face_score = _face_photo_score(best_face_for_save)
    face_url = ""
    if face_score > 0 and float(best_face_for_save.get("_face_confidence", 0.0) or 0.0) >= face_save_min_conf:
        face_url = best_face_for_save.get("_source_url", "") or ""

    body_candidates = [
        f for f in real_faces
        if _body_photo_score(f) >= 0.25
        and float(f.get("photo_body_signal_quality", 0.0) or 0.0) >= body_save_min_quality
        and body_measurement_strength(f) >= body_save_min_strength
    ]
    best_body_for_save = max(body_candidates, key=_body_photo_score) if body_candidates else None
    body_score = _body_photo_score(best_body_for_save) if best_body_for_save else 0.0
    body_url = best_body_for_save.get("_source_url", "") if best_body_for_save else ""
    review_url = face_url or body_url or content_url or (to_analyze[0] if to_analyze else "")

    woman_conf = max(f["photo_woman_confidence"] for f in real_faces)
    lightness  = sum(f["photo_skin_lightness"]   for f in real_faces) / len(real_faces)
    if enable_embedding and embedding_strategy == "best_photo" and best_face is not None:
        similarity = float(best_face.get("photo_face_similarity", 0.5) or 0.5)
    else:
        similarity = sum(f["photo_face_similarity"] for f in real_faces) / len(real_faces)

    race_avgs = {k: sum(f[k] for f in real_faces) / len(real_faces) for k in _RACE_KEYS}
    dominant_key = max(race_avgs, key=lambda k: race_avgs[k])
    dominant_race = _RACE_KEY_TO_LABEL.get(dominant_key, "")
    body_visible = max(float(f.get("photo_body_visible", 0.0) or 0.0) for f in real_faces)
    body_full = max(float(f.get("photo_body_full_length", 0.0) or 0.0) for f in real_faces)
    body_upper = max(float(f.get("photo_body_upper_length", 0.0) or 0.0) for f in real_faces)
    body_closeup = sum(float(f.get("photo_body_closeup", 0.0) or 0.0) for f in real_faces) / len(real_faces)
    body_signal_quality = max(float(f.get("photo_body_signal_quality", 0.0) or 0.0) for f in real_faces)
    body_width_ratio = _weighted_average(
        [
            (
                float(f.get("photo_body_width_ratio", 0.5) or 0.5),
                float(f.get("photo_body_signal_quality", 0.0) or 0.0)
                * max(float(f.get("photo_body_visible", 0.0) or 0.0), 0.05),
            )
            for f in real_faces
        ],
        neutral=0.5,
    )
    body_width_buckets = _body_width_buckets(body_width_ratio, body_signal_quality)

    # Pose: escolhe o frame com maior visibilidade do torso
    pose_torso_max = max(
        (float(f.get("photo_pose_torso_visibility", 0.0) or 0.0) for f in real_faces),
        default=0.0,
    )
    best_pose = next(
        (f for f in real_faces
         if float(f.get("photo_pose_torso_visibility", 0.0) or 0.0) >= max(pose_torso_max * 0.9, 0.01)),
        None,
    )
    if best_pose is not None:
        pose_shoulder_width = float(best_pose.get("photo_pose_shoulder_width", 0.0) or 0.0)
        pose_hip_width = float(best_pose.get("photo_pose_hip_width", 0.0) or 0.0)
        pose_shoulder_hip_ratio = float(best_pose.get("photo_pose_shoulder_hip_ratio", 0.5) or 0.5)
        pose_torso_visibility = pose_torso_max
        pose_torso_height = float(best_pose.get("photo_pose_torso_height", 0.0) or 0.0)
        pose_upper_body_ratio = float(best_pose.get("photo_pose_upper_body_ratio", 0.5) or 0.5)
        pose_leg_ratio = max(
            (float(f.get("photo_pose_leg_ratio", 0.0) or 0.0) for f in real_faces), default=0.0
        )
        pose_body_coverage = max(
            (float(f.get("photo_pose_body_coverage", 0.0) or 0.0) for f in real_faces), default=0.0
        )
    else:
        pose_shoulder_width = 0.0
        pose_hip_width = 0.0
        pose_shoulder_hip_ratio = 0.5
        pose_torso_visibility = 0.0
        pose_torso_height = 0.0
        pose_upper_body_ratio = 0.5
        pose_leg_ratio = 0.0
        pose_body_coverage = 0.0

    body_skin_ratio = max(
        (float(f.get("photo_body_skin_ratio", 0.0) or 0.0) for f in real_faces), default=0.0
    )

    smile_score = max(float(f.get("photo_face_smile_score", 0.5) or 0.5) for f in real_faces)
    image_brightness = _avg_result_feature(results, "photo_image_brightness", 0.5)
    image_contrast = _avg_result_feature(results, "photo_image_contrast", 0.5)
    image_sharpness = _avg_result_feature(results, "photo_image_sharpness", 0.5)
    image_colorfulness = _avg_result_feature(results, "photo_image_colorfulness", 0.5)

    import numpy as np
    raw_embeddings = [f["_embedding"] for f in real_faces if f["_embedding"] is not None]
    mean_emb = None
    if raw_embeddings:
        mean_emb = np.mean(raw_embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

    aggregated = {
        "photo_has_face": 1,
        "photo_woman_confidence": round(woman_conf, 4),
        "photo_skin_lightness": round(lightness, 4),
        "photo_face_similarity": round(similarity, 4),
        "photo_faces_ratio": round(len(real_faces) / max(len(to_analyze), 1), 4),
        "photo_failure_ratio": round(failed_count / max(len(to_analyze), 1), 4),
        "photo_gender_certainty": round(abs(woman_conf - 0.5) * 2, 4),
        **{k: round(v, 4) for k, v in race_avgs.items()},
        "photo_body_visible": round(body_visible, 4),
        "photo_body_full_length": round(body_full, 4),
        "photo_body_upper_length": round(body_upper, 4),
        "photo_body_closeup": round(body_closeup, 4),
        "photo_body_width_ratio": round(body_width_ratio, 4),
        "photo_body_signal_quality": round(body_signal_quality, 4),
        **body_width_buckets,
        "photo_body_skin_ratio": round(_clamp01(body_skin_ratio), 4),
        "photo_pose_shoulder_width": round(_clamp01(pose_shoulder_width), 4),
        "photo_pose_hip_width": round(_clamp01(pose_hip_width), 4),
        "photo_pose_shoulder_hip_ratio": round(pose_shoulder_hip_ratio, 4),
        "photo_pose_torso_visibility": round(_clamp01(pose_torso_visibility), 4),
        "photo_pose_torso_height": round(_clamp01(pose_torso_height), 4),
        "photo_pose_upper_body_ratio": round(pose_upper_body_ratio, 4),
        "photo_pose_leg_ratio": round(_clamp01(pose_leg_ratio), 4),
        "photo_pose_body_coverage": round(_clamp01(pose_body_coverage), 4),
        "photo_face_smile_score": round(_clamp01(smile_score), 4),
        "photo_image_brightness": round(_clamp01(image_brightness), 4),
        "photo_image_contrast": round(_clamp01(image_contrast), 4),
        "photo_image_sharpness": round(_clamp01(image_sharpness), 4),
        "photo_image_colorfulness": round(_clamp01(image_colorfulness), 4),
        **semantic_carousel_features,
        "_dominant_race": dominant_race,
        "_embedding": mean_emb,
        "_semantic_embedding": semantic_embedding,
        "_faces_found": len(real_faces),
        "_photos_analyzed": len(to_analyze),
        "_photos_failed": failed_count,
        "_photos_timed_out": timed_out_count,
        "_photo_analysis_incomplete": failed_count > 0,
        "_best_face_photo_url": face_url,
        "_best_body_photo_url": body_url,
        "_review_photo_url": review_url,
        "_best_face_photo_score": round(face_score, 4),
        "_best_body_photo_score": round(body_score, 4),
        "_best_content_photo_url": content_url,
        "_best_content_photo_score": round(content_score, 4),
    }
    logger.info(
        "Analise de conjunto concluida: faces=%s/%s failed=%s timed_out=%s woman=%.3f race=%s face_save=%s body_save=%s elapsed=%.2fs",
        len(real_faces),
        len(to_analyze),
        failed_count,
        timed_out_count,
        float(aggregated["photo_woman_confidence"]),
        dominant_race,
        bool(face_url),
        bool(body_url),
        time.perf_counter() - started_at,
    )
    return aggregated
