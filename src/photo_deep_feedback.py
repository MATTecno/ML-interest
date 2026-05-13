"""
Rotulos opcionais e ricos sobre fotos salvas (treino extra / auditoria).

Os registros vao para data/photo_deep_feedback.jsonl — separado de profiles.csv.
O treino principal (model_training) nao depende deste arquivo; falhas aqui
nunca devem quebrar o retreino.
"""

from __future__ import annotations

import json
import hashlib
import pickle
import threading
import csv
from datetime import datetime, timezone
from pathlib import Path

from body_photo_rules import BODY_MEASUREMENT_MIN_STRENGTH, body_measurement_strength
from config import ROOT_DIR
from features import PHOTO_FEATURE_NAMES
from logging_config import get_logger

logger = get_logger(__name__)

PHOTO_DEEP_JSONL = ROOT_DIR / "data" / "photo_deep_feedback.jsonl"
PHOTO_DEEP_MODEL_PATH = ROOT_DIR / "models" / "photo_deep_classifier.pkl"
PHOTO_DEEP_SCHEMA_VERSION = 1

# Grupos: (titulo, [(tag_id, rotulo_a_favor, rotulo_contra), ...])
# tag_id unico; mesmo id nos dois lados com exclusao mutua na UI.
PHOTO_DEEP_TAG_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Rosto, expressão e cabelo (sinais visuais)",
        [
            ("d_face_harmony", "harmonia geral do rosto — a favor", "harmonia geral do rosto — contra"),
            ("d_face_eyes", "olhos / olhar — puxa a favor", "olhos / olhar — puxa contra"),
            ("d_face_brows", "sobrancelhas / moldura do olhar — ok", "sobrancelhas / moldura — incomoda"),
            ("d_face_skin", "pele / textura na foto — agrada", "pele / textura na foto — incomoda"),
            ("d_face_smile", "sorriso / boca expressiva — positivo", "sorriso / boca — negativo ou tenso"),
            ("d_face_jaw_profile", "perfil / mandíbula — positivo", "perfil / mandíbula — negativo"),
            ("d_expression_warm", "expressão acolhedora / viva", "expressão fria / fechada"),
            ("d_makeup", "maquiagem bem integrada na foto", "maquiagem pesada ou destoante"),
            ("d_hair", "cabelo arrumado / combina com o look", "cabelo bagunçado ou distraindo na foto"),
        ],
    ),
    (
        "Corpo, pose e enquadramento (composição visual, não medida real)",
        [
            ("d_body_silhouette", "silhueta / proporções na foto — a favor", "silhueta / proporções — não combina comigo"),
            ("d_body_tone", "tonalidade / definição visível — positivo", "tonalidade / definição — neutro ou negativo"),
            ("d_body_posture", "postura confiante na imagem", "postura aparentemente desconfortável"),
            ("d_body_framing", "enquadramento do corpo na foto — equilibrado", "enquadramento do corpo — estranho ou cortado demais"),
            ("d_body_visibility", "quantidade de corpo mostrada — ok pra mim", "quantidade de corpo mostrada — incomoda"),
            ("d_pose_natural", "pose natural / espontânea", "pose rígida ou forçada"),
        ],
    ),
    (
        "Ambiente, contexto e estilo de vida sugerido",
        [
            ("d_ctx_clean", "fundo / ambiente limpo e simples", "fundo bagunçado ou poluído"),
            ("d_ctx_outdoor", "ambiente externo / natureza — positivo", "ambiente externo — negativo ou irrelevante"),
            ("d_ctx_urban", "cidade / urbano — combina", "cidade / urbano — não combina"),
            ("d_ctx_home", "casa / intimidade — vibe boa", "casa / intimidade — vibe fraca"),
            ("d_ctx_gym", "academia / fitness no contexto — ok", "academia / espelho — incomoda"),
            ("d_ctx_bathroom", "banheiro / espelho — aceitável", "banheiro / espelho — destoa"),
            ("d_ctx_car", "carro / selfie no carro — ok", "carro — destoa ou incomoda"),
            ("d_ctx_nightlife", "bar / festa / nightlife — positivo", "bar / festa — negativo"),
            ("d_ctx_travel", "viagem / destino — positivo", "viagem / destino — neutro ou negativo"),
            ("d_ctx_work", "ambiente profissional / estudo — positivo", "ambiente profissional — neutro ou negativo"),
        ],
    ),
    (
        "Roupa, cor e estética pessoal",
        [
            ("d_outfit_fit", "roupa cai bem / valoriza a foto", "roupa não valoriza ou destoa"),
            ("d_outfit_style", "estilo do look combina com o que curto", "estilo do look não combina"),
            ("d_outfit_casual", "casual bem resolvido", "casual desleixado na imagem"),
            ("d_outfit_formal", "elegância / formal — positivo", "formal demais ou destoante"),
            ("d_colors", "harmonia de cores na foto", "cores destoando ou gritando demais"),
            ("d_accessories", "acessórios discretos e legais", "acessórios excessivos ou destoando"),
        ],
    ),
    (
        "Luz, foco e qualidade de imagem",
        [
            ("d_light_soft", "luz suave / flattering", "luz dura / sombras ruins"),
            ("d_sharp", "foto nítida onde importa", "desfoque indesejado ou borrão"),
            ("d_noise", "ruído / granulação controlados", "muito ruído ou baixa qualidade"),
            ("d_angle", "ângulo da câmera favorece", "ângulo pouco favorece"),
            ("d_crop_face", "corte / close no rosto — bom", "corte no rosto — apertado ou estranho"),
            ("d_dof_bg", "desfoque de fundo agradável", "fundo competindo com o sujeito"),
            ("d_filters", "filtros leves / naturais", "filtros pesados ou pele artificial"),
        ],
    ),
    (
        "Composição geral e leitura da foto",
        [
            ("d_comp_balance", "composição equilibrada (peso visual)", "composição desbalanceada"),
            ("d_comp_story", "foto “conta algo” / tem personalidade", "foto genérica ou sem alma"),
            ("d_multi_people", "outras pessoas não atrapalham", "outras pessoas confundem o foco"),
            ("d_energy_frame", "energia da pose / momento — positiva", "energia da pose — fraca ou negativa"),
        ],
    ),
    (
        "Impressão subjetiva rápida (opcional)",
        [
            ("d_vibe_confident", "passa confiança", "passa insegurança na imagem"),
            ("d_vibe_fun", "passa diversão / leveza", "passa seriedade excessiva (se incomoda)"),
            ("d_vibe_authentic", "parece autêntica / natural", "parece muito “montada” ou performática"),
        ],
    ),
]

_ALL_DEEP_TAG_IDS: frozenset[str] = frozenset(
    tid for _, items in PHOTO_DEEP_TAG_GROUPS for tid, _, _ in items
)

_write_lock = threading.Lock()


def _safe_photo_path(path: str) -> Path | None:
    if not path or ".." in path:
        return None
    p = (ROOT_DIR / path).resolve()
    try:
        p.relative_to((ROOT_DIR / "data" / "photos").resolve())
    except ValueError:
        return None
    if not p.is_file():
        return None
    return p


def is_allowed_photo_rel(path: str) -> bool:
    """True se path relativo aponta para arquivo sob data/photos/(liked|disliked)."""
    p = _safe_photo_path(path)
    if p is None:
        return False
    parts = p.parts
    if len(parts) < 2:
        return False
    folder = parts[-2].lower() if len(parts) >= 2 else ""
    return folder in ("liked", "disliked") and p.suffix.lower() in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    )


def _normalize_photo_name(text: str) -> str:
    import re
    import unicodedata

    base = unicodedata.normalize("NFD", (text or "").strip().lower())
    base = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", base.replace("_", " ")).strip()


def _measured_body_profile_keys() -> set[tuple[str, str]]:
    from dataset import PROFILES_PATH

    keys: set[tuple[str, str]] = set()
    if not PROFILES_PATH.exists():
        return keys
    try:
        with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("source", "") or "").strip().lower() != "real":
                    continue
                name = _normalize_photo_name(row.get("name", ""))
                age = str(row.get("age", "") or "").strip()
                if not name or not age:
                    continue
                if body_measurement_strength(row) >= BODY_MEASUREMENT_MIN_STRENGTH:
                    keys.add((name, age))
    except OSError:
        logger.debug("Falha ao montar índice de perfis corporais medidos", exc_info=True)
    return keys


def _photo_profile_key_from_rel(rel: str) -> tuple[str, str] | None:
    import re

    name = Path(str(rel or "")).name
    match = re.match(r"^[^_]+_(.+)_(\d+)(?:_(?:face|body))?\.[^.]+$", name, re.IGNORECASE)
    if not match:
        return None
    return (_normalize_photo_name(match.group(1)), match.group(2))


def _is_body_review_photo(rel: str, measured_keys: set[tuple[str, str]]) -> bool:
    path_name = Path(rel).name.lower()
    if "_body." not in path_name:
        return False
    key = _photo_profile_key_from_rel(rel)
    return bool(key and (not measured_keys or key in measured_keys))


def _photo_sha256_for_rel(rel: str) -> str:
    p = _safe_photo_path(rel)
    if p is None:
        return ""
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def list_saved_photo_paths(limit: int | None = 96, body_only: bool = False) -> list[str]:
    """Mais recentes primeiro (por mtime). Use limit=None para listar todas."""
    root = ROOT_DIR / "data" / "photos"
    items: list[tuple[float, str]] = []
    measured_keys = _measured_body_profile_keys() if body_only else set()
    for bucket in ("liked", "disliked"):
        d = root / bucket
        if not d.is_dir():
            continue
        try:
            for p in d.iterdir():
                if not p.is_file():
                    continue
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                try:
                    rel = str(p.relative_to(ROOT_DIR))
                except ValueError:
                    continue
                if body_only and not _is_body_review_photo(rel, measured_keys):
                    continue
                items.append((p.stat().st_mtime, rel))
        except OSError:
            logger.debug("list_saved_photo_paths: skip dir %s", d)
    items.sort(key=lambda x: -x[0])

    if body_only:
        deduped: list[tuple[float, str]] = []
        seen_profile_keys: set[tuple[str, str]] = set()
        seen_hashes: set[str] = set()
        for item in items:
            rel = item[1]
            profile_key = _photo_profile_key_from_rel(rel)
            if profile_key and profile_key in seen_profile_keys:
                continue
            sha = _photo_sha256_for_rel(rel)
            if sha and sha in seen_hashes:
                continue
            if profile_key:
                seen_profile_keys.add(profile_key)
            if sha:
                seen_hashes.add(sha)
            deduped.append(item)
        if len(deduped) != len(items):
            logger.info("Fotos corporais duplicadas ocultadas do review: %s", len(items) - len(deduped))
        items = deduped

    if limit is None:
        return [rel for _, rel in items]
    return [rel for _, rel in items[: max(0, int(limit))]]


def _photo_file_metadata(photo_path: str) -> dict:
    p = _safe_photo_path(photo_path)
    if p is None:
        return {}
    try:
        stat = p.stat()
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return {
            "photo_sha256": h.hexdigest(),
            "photo_size_bytes": stat.st_size,
            "photo_mtime": stat.st_mtime,
        }
    except OSError:
        logger.debug("Falha ao gerar metadata da foto: %s", photo_path, exc_info=True)
        return {}


def load_deep_records() -> list[dict]:
    if not PHOTO_DEEP_JSONL.exists():
        return []
    records = []
    try:
        with open(PHOTO_DEEP_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
    except OSError:
        logger.debug("Falha ao carregar feedback visual profundo", exc_info=True)
    return records


def photo_review_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in load_deep_records():
        path = str(rec.get("photo_path") or "")
        if not path:
            continue
        counts[path] = counts.get(path, 0) + 1
    return counts


def next_unreviewed_photo(current_path: str, paths: list[str]) -> str:
    """Próxima foto ainda sem registro; se todas estiverem revisadas, próxima da lista."""
    if not paths:
        return ""
    counts = photo_review_counts()
    try:
        start = paths.index(current_path) + 1
    except ValueError:
        start = 0

    ordered = paths[start:] + paths[:start]
    for path in ordered:
        if counts.get(path, 0) == 0:
            return path
    return ordered[0] if ordered else paths[0]


def count_deep_records() -> int:
    if not PHOTO_DEEP_JSONL.exists():
        return 0
    try:
        n = 0
        with open(PHOTO_DEEP_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    except OSError:
        return 0


def append_deep_record(
    photo_path: str,
    impression: str,
    alignment: str,
    positive_tags: list[str],
    negative_tags: list[str],
    note: str,
) -> None:
    """Anexa uma linha JSON ao jsonl (thread-safe)."""
    pos_set = {t for t in positive_tags if t in _ALL_DEEP_TAG_IDS}
    neg = [t for t in negative_tags if t in _ALL_DEEP_TAG_IDS and t not in pos_set]
    pos = sorted(pos_set)
    file_meta = _photo_file_metadata(photo_path)
    rec = {
        "schema_version": PHOTO_DEEP_SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "photo_path": photo_path,
        **file_meta,
        "impression": impression or "",
        "alignment": alignment or "",
        "positive_tags": pos,
        "negative_tags": neg,
        "note": (note or "")[:4000],
        "source": "review_ui_photo_deep",
    }
    PHOTO_DEEP_JSONL.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _write_lock:
        with open(PHOTO_DEEP_JSONL, "a", encoding="utf-8") as f:
            f.write(line)
    logger.info(
        "photo_deep_feedback: salvo photo_path=%r tags+=%s tags-=%s",
        photo_path,
        len(pos),
        len(neg),
    )


def log_dataset_stats_for_training() -> None:
    """Chamado após treino bem-sucedido — só log, nunca levanta."""
    try:
        n = count_deep_records()
        if n:
            logger.info(
                "Treino extra de fotos (opcional): %s registros em %s — pipeline principal inalterado.",
                n,
                PHOTO_DEEP_JSONL.name,
            )
    except Exception:
        logger.debug("log_dataset_stats_for_training: ignorado", exc_info=True)


def _label_from_deep_record(rec: dict) -> int | None:
    impression = str(rec.get("impression") or "").strip().lower()
    if impression == "like":
        return 1
    if impression == "dislike":
        return 0
    return None


def _latest_labeled_records() -> list[dict]:
    latest: dict[str, dict] = {}
    for rec in load_deep_records():
        if _label_from_deep_record(rec) is None:
            continue
        path = str(rec.get("photo_path") or "")
        if not is_allowed_photo_rel(path):
            continue
        latest[path] = rec
    return list(latest.values())


def train_photo_deep_model() -> dict:
    """
    Treina um modelo visual opcional separado usando apenas os registros da aba
    de rotulagem visual. Não altera o ensemble principal.
    """
    records = _latest_labeled_records()
    if len(records) < 8:
        return {"ok": False, "reason": "mínimo de 8 fotos com impressão geral like/dislike", "n_samples": len(records)}

    labels = [_label_from_deep_record(rec) for rec in records]
    if sorted(set(labels)) != [0, 1]:
        return {"ok": False, "reason": "precisa ter pelo menos uma foto positiva e uma negativa", "n_samples": len(records)}

    from photo_features import analyze_local_photo
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = []
    y = []
    used_paths = []
    for rec, label in zip(records, labels):
        if label is None:
            continue
        path_rel = str(rec.get("photo_path") or "")
        path = _safe_photo_path(path_rel)
        if path is None:
            continue
        try:
            feats = analyze_local_photo(path)
            X.append([float(feats.get(name, 0.0) or 0.0) for name in PHOTO_FEATURE_NAMES])
            y.append(label)
            used_paths.append(path_rel)
        except Exception:
            logger.exception("Falha ao extrair features para treino visual extra: %s", path_rel)

    if len(y) < 8 or sorted(set(y)) != [0, 1]:
        return {"ok": False, "reason": "dados válidos insuficientes após analisar fotos", "n_samples": len(y)}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X, y)

    PHOTO_DEEP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PHOTO_DEEP_MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "schema_version": PHOTO_DEEP_SCHEMA_VERSION,
                "model_type": "PhotoDeep LogisticRegression",
                "pipeline": model,
                "feature_names": PHOTO_FEATURE_NAMES,
                "n_samples": len(y),
                "positive_samples": int(sum(y)),
                "negative_samples": int(len(y) - sum(y)),
                "photo_paths": used_paths,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
        )

    logger.info("Modelo visual extra salvo: path=%s samples=%s", PHOTO_DEEP_MODEL_PATH, len(y))
    return {
        "ok": True,
        "path": str(PHOTO_DEEP_MODEL_PATH.relative_to(ROOT_DIR)),
        "n_samples": len(y),
        "positive_samples": int(sum(y)),
        "negative_samples": int(len(y) - sum(y)),
    }
