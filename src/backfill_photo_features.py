"""
Preenche/recalcula features de foto no data/profiles.csv usando as fotos já salvas em data/photos.

Uso:
  python3 src/backfill_photo_features.py                    # preenche todos os faltantes
  python3 src/backfill_photo_features.py --limit 10
  python3 src/backfill_photo_features.py --no-retrain
  python3 src/backfill_photo_features.py --mode pose        # backfill rápido de pose MediaPipe
  python3 src/backfill_photo_features.py --mode semantic    # backfill CLIP/PCA visual
  python3 src/backfill_photo_features.py --mode body-alias  # cria *_body.jpg faltantes
  python3 src/backfill_photo_features.py --mode diagnose    # relatório de cobertura de features
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import model as mdl
from body_photo_rules import body_measurement_strength
from config import get_photos_config
from features import PHOTO_FEATURE_NAMES, SEMANTIC_EMBEDDING_FEATURE_NAMES
from logging_config import get_logger, setup_logging
from photo_features import analyze_local_photo, analyze_pose_only


ROOT_DIR = Path(__file__).parent.parent
PHOTOS_DIR = ROOT_DIR / "data" / "photos"
PROFILES_PATH = ROOT_DIR / "data" / "profiles.csv"
REVIEW_PATH = ROOT_DIR / "data" / "review_queue.csv"
logger = get_logger(__name__)

_POSE_COLS = [
    "photo_pose_shoulder_width",
    "photo_pose_hip_width",
    "photo_pose_shoulder_hip_ratio",
    "photo_pose_torso_visibility",
    "photo_pose_torso_height",
    "photo_pose_upper_body_ratio",
    "photo_pose_leg_ratio",
    "photo_pose_body_coverage",
]

_COVERAGE_GROUPS = {
    "Rosto / raça (DeepFace)":    ["photo_has_face", "photo_woman_confidence", "photo_face_similarity"],
    "Qualidade de imagem":         ["photo_image_brightness", "photo_image_contrast", "photo_image_sharpness", "photo_image_colorfulness"],
    "Sorriso / emoção":            ["photo_face_smile_score"],
    "Corpo visível (Canny)":       ["photo_body_visible", "photo_body_width_ratio", "photo_body_signal_quality"],
    "Largura — buckets (Canny)":   ["photo_body_width_bucket_narrow", "photo_body_width_bucket_medium", "photo_body_width_bucket_wide"],
    "Pose (MediaPipe)":            _POSE_COLS,
    "Skin ratio":                  ["photo_body_skin_ratio"],
    "Embeddings PCA":              ["photo_emb_pc_01", "photo_emb_pc_02", "photo_emb_pc_03"],
    "CLIP visual semantico":        ["photo_semantic_embedding_saved", "photo_clip_pc_01", "photo_carousel_visual_diversity"],
}


def _normalize_name(text: str) -> str:
    base = unicodedata.normalize("NFD", (text or "").strip().lower())
    base = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
    base = base.replace("_", " ")
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _parse_photo_filename(path: Path) -> tuple[str, int] | None:
    match = re.match(r"^[^_]+_(.+)_(\d+)(?:_(?:face|body))?\.jpg$", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    raw_name = match.group(1)
    age = int(match.group(2))
    return _normalize_name(raw_name), age


def _is_body_photo(path: Path) -> bool:
    return "_body." in path.name.lower()


def _row_has_any_photo_features(row: dict) -> bool:
    saved = str(row.get("photo_features_saved", "")).strip().lower()
    if saved in {"1", "1.0", "true"}:
        return True
    return any(str(row.get(col, "")).strip() for col in PHOTO_FEATURE_NAMES)


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _body_measurement_strength(row: dict) -> float:
    """Mesma regra conservadora da aba Treino corporal."""
    return body_measurement_strength(row)


def _has_neutral_similarity(row: dict) -> bool:
    """Similaridade 0.5 com rosto salvo costuma ser fallback antigo."""
    similarity = _as_float(row.get("photo_face_similarity", ""), -1.0)
    has_face = _as_float(row.get("photo_has_face", ""), 0.0)
    return similarity == 0.5 and has_face > 0


def _row_needs_photo_features(row: dict) -> bool:
    if not _row_has_any_photo_features(row):
        return True
    if any(not str(row.get(col, "")).strip() for col in PHOTO_FEATURE_NAMES):
        return True
    return _has_neutral_similarity(row)


def _row_needs_pose_features(row: dict) -> bool:
    """True se o perfil tem corpo visível e a coluna de pose nunca foi preenchida."""
    body_visible = _as_float(row.get("photo_body_visible", ""), 0.0)
    # Se a coluna já tem qualquer valor (mesmo 0.0), já foi tentado — não retentar.
    # Assim evitamos loop infinito em fotos onde MediaPipe nunca detecta landmarks.
    pose_col_present = str(row.get("photo_pose_torso_visibility", "")).strip() != ""
    return body_visible > 0 and not pose_col_present


def _row_needs_semantic_features(row: dict) -> bool:
    saved = str(row.get("photo_features_saved", "")).strip().lower()
    if saved not in {"1", "1.0", "true"}:
        return False
    if str(row.get("photo_semantic_embedding_saved", "")).strip() not in {"1", "1.0", "true"}:
        return True
    return any(str(row.get(col, "")).strip() == "" for col in SEMANTIC_EMBEDDING_FEATURE_NAMES)


def _load_rows() -> list[dict]:
    mdl._ensure_profiles_schema()
    if not PROFILES_PATH.exists():
        return []
    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(rows: list[dict]) -> None:
    with open(PROFILES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mdl.CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_map(rows: list[dict]) -> dict[tuple[str, int, int], list[int]]:
    mapping: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if str(row.get("source", "real")).strip().lower() != "real":
            continue
        try:
            age = int(float(row.get("age", 0) or 0))
            label = int(row.get("label", 0))
        except Exception:
            continue
        key = (_normalize_name(row.get("name", "")), age, label)
        mapping[key].append(idx)

    for key, indices in mapping.items():
        indices.sort(key=lambda i: (0 if _row_needs_photo_features(rows[i]) else 1, i))
    return mapping


def _photo_files() -> list[tuple[Path, int]]:
    files = []
    for subdir, label in (("liked", 1), ("disliked", 0)):
        for path in sorted((PHOTOS_DIR / subdir).glob("*.jpg")):
            files.append((path, label))
    return files


def _build_photo_map() -> dict[tuple[str, int], list[Path]]:
    """
    Mapeia (nome_normalizado, idade) → lista de fotos (sem label).
    Fotos corporais (*_body.jpg) ficam primeiro — melhores para pose.
    Label excluído do key porque fotos permanecem no diretório original
    mesmo após re-avaliação do perfil.
    """
    mapping: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for photo_path, _label in _photo_files():
        parsed = _parse_photo_filename(photo_path)
        if not parsed:
            continue
        norm_name, age = parsed
        key = (norm_name, age)
        if _is_body_photo(photo_path):
            mapping[key].insert(0, photo_path)
        else:
            mapping[key].append(photo_path)
    return mapping


# ---------------------------------------------------------------------------
# Modo 1: backfill completo (comportamento original)
# ---------------------------------------------------------------------------

def backfill(limit: int | None = None, retrain: bool = True) -> dict:
    logger.info("Backfill iniciado: limit=%s retrain=%s", limit, retrain)
    rows = _load_rows()
    if not rows:
        logger.warning("Backfill sem linhas em profiles.csv")
        return {"updated": 0, "skipped": 0, "photos_seen": 0}

    candidates = _candidate_map(rows)
    updated = 0
    skipped = 0
    photos_seen = 0

    for photo_path, label in _photo_files():
        if limit is not None and photos_seen >= limit:
            break

        parsed = _parse_photo_filename(photo_path)
        if not parsed:
            skipped += 1
            continue

        norm_name, age = parsed
        key = (norm_name, age, label)
        indices = candidates.get(key, [])
        target_idx = next((i for i in indices if _row_needs_photo_features(rows[i])), None)
        if target_idx is None:
            skipped += 1
            continue

        photos_seen += 1
        logger.info("Backfill analisando foto: %s label=%s target_idx=%s", photo_path, label, target_idx)
        print(f"  [backfill] Analisando {photo_path.name} ...")
        feat = analyze_local_photo(photo_path, age)

        row = rows[target_idx]
        for col in PHOTO_FEATURE_NAMES:
            value = feat.get(col, "")
            row[col] = "" if value in ("", None) else round(float(value), 6)
        row["photo_features_saved"] = "1"
        updated += 1

    if updated:
        _write_rows(rows)
        logger.info("Backfill escreveu rows atualizadas: updated=%s", updated)

    if retrain and updated:
        logger.info("Backfill retreinando modelo")
        print("\n  [backfill] Retreinando modelo com as novas features de foto...")
        mdl.train_model()

    logger.info("Backfill concluido: updated=%s skipped=%s photos_seen=%s", updated, skipped, photos_seen)
    return {
        "updated": updated,
        "skipped": skipped,
        "photos_seen": photos_seen,
    }


# ---------------------------------------------------------------------------
# Modo 2: backfill rápido só de pose (MediaPipe, sem DeepFace)
# ---------------------------------------------------------------------------

def backfill_pose(limit: int | None = None, retrain: bool = True) -> dict:
    """
    Preenche photo_pose_* para perfis com corpo visível mas sem pose detectada.
    Muito mais rápido que o backfill completo — não roda DeepFace.
    Prioriza fotos corporais (*_body.jpg) sobre fotos de rosto.
    """
    logger.info("Backfill de pose iniciado: limit=%s", limit)
    rows = _load_rows()
    if not rows:
        return {"updated": 0, "skipped": 0}

    photo_map = _build_photo_map()

    # Índice reverso: (nome, idade) → índices de linha no CSV (sem label)
    row_map: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if str(row.get("source", "real")).strip().lower() != "real":
            continue
        try:
            age = int(float(row.get("age", 0) or 0))
        except Exception:
            continue
        key = (_normalize_name(row.get("name", "")), age)
        row_map[key].append(idx)

    updated = 0
    skipped = 0

    for key, indices in row_map.items():
        if limit is not None and updated >= limit:
            break

        photos = photo_map.get(key, [])
        if not photos:
            skipped += 1
            continue

        # Encontra linha que precisa de pose
        target_idx = next((i for i in indices if _row_needs_pose_features(rows[i])), None)
        if target_idx is None:
            skipped += 1
            continue

        # Tenta cada foto em ordem (body photos primeiro), para na primeira boa
        best_pose: dict | None = None
        best_vis = -1.0
        for photo_path in photos:
            pose = analyze_pose_only(photo_path)
            vis = pose.get("photo_pose_torso_visibility", 0.0)
            if vis > best_vis:
                best_vis = vis
                best_pose = pose
            if vis >= 0.35:
                break  # bom o suficiente, não precisa continuar

        if best_pose is None:
            skipped += 1
            continue

        row = rows[target_idx]
        for col in _POSE_COLS:
            val = best_pose.get(col, 0.0)
            # Salva 0.0 explicitamente mesmo quando detecção falhou, para que
            # _row_needs_pose_features reconheça que já foi tentado e não reinsira
            # na fila (evita loop infinito em fotos sem corpo detectável).
            row[col] = round(float(val), 6)

        log_name = row.get("name", "?")
        logger.info("Pose backfill: %s torso_vis=%.3f", log_name, best_vis)
        print(f"  [pose] {log_name} ({row.get('age','?')}) — torso_vis={best_vis:.3f}")
        updated += 1

    if updated:
        _write_rows(rows)
        logger.info("Pose backfill escreveu %s linhas", updated)

    if retrain and updated:
        print("\n  [pose] Retreinando modelo com as novas features de pose...")
        mdl.train_model()

    logger.info("Pose backfill concluido: updated=%s skipped=%s", updated, skipped)
    return {"updated": updated, "skipped": skipped}


# ---------------------------------------------------------------------------
# Modo 3: backfill de embeddings visuais CLIP/PCA
# ---------------------------------------------------------------------------

def backfill_semantic(limit: int | None = None, retrain: bool = True, dry_run: bool = False) -> dict:
    """
    Preenche photo_clip_pc_* e métricas de carrossel usando fotos salvas.

    O embedding bruto fica em data/photo_semantic_cache.sqlite; o CSV recebe só
    as features compactadas por PCA e métricas agregadas por perfil.
    """
    logger.info("Backfill semantico CLIP iniciado: limit=%s retrain=%s dry_run=%s", limit, retrain, dry_run)
    rows = _load_rows()
    if not rows:
        return {"updated": 0, "skipped": 0, "candidates": 0, "no_embedding": 0}

    try:
        from photo_semantic_embeddings import (
            aggregate_embeddings,
            apply_to_embedding,
            embeddings_for_paths,
            fit_and_save_pca as fit_semantic_pca,
            load_embeddings_from_cache as load_semantic_embeddings,
            load_pca as load_semantic_pca,
        )
    except Exception:
        logger.exception("Dependencias de embedding semantico indisponiveis")
        return {"updated": 0, "skipped": 0, "candidates": 0, "no_embedding": 0, "error": "semantic_dependencies"}

    photo_map = _build_photo_map()
    cfg = (get_photos_config().get("semantic_embedding", {}) or {})
    try:
        max_photos = max(1, int(cfg.get("max_photos_per_profile", 4) or 4))
    except Exception:
        max_photos = 4

    targets = []
    for idx, row in enumerate(rows):
        if str(row.get("source", "real")).strip().lower() != "real":
            continue
        if not _row_needs_semantic_features(row):
            continue
        try:
            age = int(float(row.get("age", 0) or 0))
        except Exception:
            continue
        key = (_normalize_name(row.get("name", "")), age)
        photos = photo_map.get(key, [])[:max_photos]
        if not photos:
            continue
        targets.append((idx, photos))
        if limit is not None and len(targets) >= limit:
            break

    if dry_run:
        print(f"  [semantic] Candidatos: {len(targets)}")
        return {"updated": 0, "skipped": 0, "candidates": len(targets), "no_embedding": 0}

    prepared: list[tuple[int, object, dict]] = []
    no_embedding = 0
    skipped = 0
    unique_photo_paths: list[Path] = []
    seen_photo_paths: set[str] = set()
    for _, photos in targets:
        for photo_path in photos:
            label = str(Path(photo_path))
            if label in seen_photo_paths:
                continue
            seen_photo_paths.add(label)
            unique_photo_paths.append(Path(photo_path))

    embedding_by_path = embeddings_for_paths(unique_photo_paths)
    logger.info(
        "Backfill semantico preparou embeddings em lote: targets=%s unique_photos=%s",
        len(targets),
        len(unique_photo_paths),
    )

    for done, (idx, photos) in enumerate(targets, start=1):
        row = rows[idx]
        embeddings = []
        for photo_path in photos:
            emb = embedding_by_path.get(str(Path(photo_path)))
            if emb is not None:
                embeddings.append(emb)
        mean_emb, aggregate = aggregate_embeddings(embeddings)
        if mean_emb is None:
            no_embedding += 1
            continue
        prepared.append((idx, mean_emb, aggregate))
        print(
            f"  [semantic] {done}/{len(targets)} {row.get('name','?')} ({row.get('age','?')}) "
            f"fotos={len(embeddings)}"
        )

    semantic_pca = fit_semantic_pca(load_semantic_embeddings()) or load_semantic_pca()
    updated = 0
    for idx, mean_emb, aggregate in prepared:
        row = rows[idx]
        row.update({key: round(float(value), 6) for key, value in aggregate.items()})
        pc_features = apply_to_embedding(mean_emb, semantic_pca)
        for col in SEMANTIC_EMBEDDING_FEATURE_NAMES:
            val = pc_features.get(col)
            try:
                val_f = float(val)
                row[col] = "" if val_f != val_f else round(val_f, 6)
            except Exception:
                row[col] = ""
        updated += 1

    skipped = len(targets) - updated - no_embedding
    if updated:
        _write_rows(rows)
        logger.info("Backfill semantico escreveu rows atualizadas: updated=%s", updated)

    if retrain and updated:
        print("\n  [semantic] Retreinando modelo com embeddings visuais CLIP...")
        mdl.train_model()

    logger.info(
        "Backfill semantico concluido: updated=%s skipped=%s candidates=%s no_embedding=%s",
        updated,
        skipped,
        len(targets),
        no_embedding,
    )
    return {
        "updated": updated,
        "skipped": skipped,
        "candidates": len(targets),
        "no_embedding": no_embedding,
    }


# ---------------------------------------------------------------------------
# Modo 4: backfill de aliases *_body.jpg já medidos no review_queue
# ---------------------------------------------------------------------------

def _body_alias_for_photo_path(photo_path: str) -> tuple[Path, Path] | None:
    rel = str(photo_path or "").strip()
    if not rel:
        return None
    src = (ROOT_DIR / rel).resolve()
    try:
        src.relative_to(ROOT_DIR.resolve())
    except ValueError:
        return None
    if src.suffix.lower() != ".jpg":
        return None
    name_lower = src.name.lower()
    if "_body." in name_lower or "_face." in name_lower:
        return None
    dest = src.with_name(f"{src.stem}_body{src.suffix}")
    return src, dest


def _write_body_alias_reason(src: Path, dest: Path, dry_run: bool) -> bool:
    dest_txt = dest.with_suffix(".txt")
    if dest_txt.exists():
        return False

    source_txt = src.with_suffix(".txt")
    text = ""
    if source_txt.exists():
        try:
            text = source_txt.read_text(encoding="utf-8").rstrip()
        except OSError:
            text = ""
    marker = "Foto salva para: analise de corpo (backfill)"
    text = f"{text}\n\n{marker}\n" if text else f"{marker}\n"
    if not dry_run:
        dest_txt.write_text(text, encoding="utf-8")
    return True


def backfill_body_aliases(limit: int | None = None, dry_run: bool = False) -> dict:
    """
    Cria *_body.jpg para fotos principais que ja tinham medicao corporal boa.

    Usa review_queue.csv porque nele temos o caminho exato da foto salva. Isso
    evita copiar por nome/idade quando existem perfis diferentes com mesmo nome.
    """
    logger.info("Backfill body aliases iniciado: limit=%s dry_run=%s", limit, dry_run)
    if not REVIEW_PATH.exists():
        logger.warning("review_queue.csv nao encontrado para body-alias")
        return {"created": 0, "candidates": 0, "existing": 0, "missing_source": 0, "weak": 0, "bad_path": 0}

    with open(REVIEW_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    created = 0
    candidates = 0
    existing = 0
    missing_source = 0
    weak = 0
    bad_path = 0
    txt_created = 0
    seen_dest: set[Path] = set()

    for row in rows:
        strength = _body_measurement_strength(row)
        if strength < 0.45:
            weak += 1
            continue

        pair = _body_alias_for_photo_path(row.get("photo_path", ""))
        if pair is None:
            bad_path += 1
            continue
        src, dest = pair
        candidates += 1

        if dest in seen_dest or dest.exists():
            existing += 1
            continue
        if not src.exists():
            missing_source += 1
            continue

        if limit is not None and created >= limit:
            break

        print(f"  [body-alias] {src.relative_to(ROOT_DIR)} -> {dest.name}")
        if not dry_run:
            shutil.copy2(src, dest)
        if _write_body_alias_reason(src, dest, dry_run):
            txt_created += 1
        seen_dest.add(dest)
        created += 1

    logger.info(
        "Backfill body aliases concluido: created=%s txt_created=%s candidates=%s existing=%s missing_source=%s weak=%s bad_path=%s dry_run=%s",
        created,
        txt_created,
        candidates,
        existing,
        missing_source,
        weak,
        bad_path,
        dry_run,
    )
    return {
        "created": created,
        "txt_created": txt_created,
        "candidates": candidates,
        "existing": existing,
        "missing_source": missing_source,
        "weak": weak,
        "bad_path": bad_path,
    }


# ---------------------------------------------------------------------------
# Modo 5: relatório de cobertura
# ---------------------------------------------------------------------------

def coverage_report() -> None:
    """Imprime cobertura (% preenchido) de cada grupo de features no CSV."""
    rows = _load_rows()
    if not rows:
        print("  profiles.csv vazio ou inexistente.")
        return

    real_rows = [r for r in rows if str(r.get("source", "real")).strip().lower() == "real"]
    n = len(real_rows)
    if n == 0:
        print("  Nenhuma linha real encontrada.")
        return

    print(f"\n  Cobertura de features ({n} perfis reais)\n")
    print(f"  {'Grupo':<38} {'Cobertura':>10}  {'c/ dado':>8}  {'sem dado':>8}")
    print("  " + "-" * 68)

    for group_name, cols in _COVERAGE_GROUPS.items():
        # Considera "preenchido" se a coluna principal (primeira) tem valor != "" e != 0
        col = cols[0]
        filled = sum(
            1 for r in real_rows
            if str(r.get(col, "")).strip() not in ("", "0", "0.0")
        )
        pct = filled / n * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {group_name:<38} {pct:>9.1f}%  {filled:>8}  {n - filled:>8}")

    print()

    # Perfis que podem se beneficiar de pose backfill
    needs_pose = sum(1 for r in real_rows if _row_needs_pose_features(r))
    has_body = sum(1 for r in real_rows if _as_float(r.get("photo_body_visible", ""), 0.0) > 0)
    has_full = sum(1 for r in real_rows if _as_float(r.get("photo_body_full_length", ""), 0.0) > 0)

    print(f"  Fotos com corpo visível              : {has_body:>5} ({has_body/n*100:.0f}%)")
    print(f"  Fotos corpo inteiro (full-length)    : {has_full:>5} ({has_full/n*100:.0f}%)")
    print(f"  Candidatos ao pose backfill          : {needs_pose:>5}")
    print()

    import subprocess
    r = subprocess.run(
        ["find", str(PHOTOS_DIR), "-name", "*.jpg"],
        capture_output=True, text=True
    )
    all_photos = [l for l in r.stdout.strip().split("\n") if l]
    body_photos = [p for p in all_photos if "_body." in p.lower()]
    print(f"  Fotos salvas no disco                : {len(all_photos):>5}")
    print(f"  Fotos corporais (*_body.jpg)         : {len(body_photos):>5}")
    print()
    if needs_pose > 0:
        print(f"  → Rode --mode pose para preencher até {needs_pose} perfis sem pose.")
    else:
        print("  → Pose já está preenchida em todos os perfis com corpo visível.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["all", "pose", "semantic", "body-alias", "diagnose"],
        default="all",
        help="all=backfill completo, pose=só MediaPipe rápido, semantic=CLIP/PCA, body-alias=cria *_body.jpg faltantes, diagnose=relatório de cobertura",
    )
    args = parser.parse_args()

    print("\n  Tinder-IA — Backfill de features de foto")
    print("  " + "-" * 42)

    if args.mode == "diagnose":
        coverage_report()
    elif args.mode == "body-alias":
        print("  Modo: aliases *_body.jpg para fotos ja medidas\n")
        result = backfill_body_aliases(limit=args.limit, dry_run=args.dry_run)
        print()
        print(f"  Criados      : {result['created']}")
        print(f"  TXT criados  : {result['txt_created']}")
        print(f"  Candidatos   : {result['candidates']}")
        print(f"  Ja existiam  : {result['existing']}")
        print(f"  Sem origem   : {result['missing_source']}")
        print(f"  Fracos       : {result['weak']}")
        print(f"  Caminho ruim : {result['bad_path']}")
        print()
    elif args.mode == "pose":
        print("  Modo: pose rápido (MediaPipe, sem DeepFace)\n")
        result = backfill_pose(limit=args.limit, retrain=not args.no_retrain)
        print()
        print(f"  Atualizados : {result['updated']}")
        print(f"  Ignorados   : {result['skipped']}")
        print()
    elif args.mode == "semantic":
        print("  Modo: embeddings visuais CLIP/PCA\n")
        result = backfill_semantic(limit=args.limit, retrain=not args.no_retrain, dry_run=args.dry_run)
        print()
        print(f"  Atualizados : {result['updated']}")
        print(f"  Candidatos  : {result['candidates']}")
        print(f"  Sem embed   : {result['no_embedding']}")
        print(f"  Ignorados   : {result['skipped']}")
        print()
    else:
        result = backfill(limit=args.limit, retrain=not args.no_retrain)
        print()
        print(f"  Atualizados : {result['updated']}")
        print(f"  Ignorados   : {result['skipped']}")
        print(f"  Processados : {result['photos_seen']}")
        print()

    # MediaPipe 0.10.x tem um bug no destrutor C++ (free(): invalid pointer) ao
    # encerrar o processo Python. Dados já salvos — saída antecipada evita o crash.
    import os as _os
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)


if __name__ == "__main__":
    main()
