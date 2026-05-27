"""UI local para revisao pos-sessao das decisoes automaticas."""

from __future__ import annotations

import html
import hashlib
import json
import math
import mimetypes
import re
import threading
import unicodedata
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time

import yaml

from body_photo_rules import BODY_MEASUREMENT_MIN_STRENGTH, body_measurement_strength
from config import ROOT_DIR, CONFIG_PATH
from logging_config import get_logger, setup_logging
from model_training import load_model, train_model
from model_evaluation import LATEST_JSON
from text_preferences import (
    append_signal_feedback,
    append_signal_feedback_batch,
    score_profile_text,
    signal_feedback_counts,
    signal_training_candidates,
)
from photo_deep_feedback import (
    PHOTO_DEEP_TAG_GROUPS,
    append_deep_record,
    count_deep_records,
    is_allowed_photo_rel,
    list_saved_photo_paths,
    next_unreviewed_photo,
    photo_review_counts,
    train_photo_deep_model,
)
from review_queue import (
    REVIEW_PATH,
    _review_dedupe_key_from_row,
    apply_review,
    enqueue_history_review_candidates,
    history_review_candidate_stats,
    load_reviews,
    skip_absolute_filter_reviews,
    skip_duplicate_history_reviews,
    skip_duplicate_pending_reviews,
    skip_all_pending,
    skip_review,
    undo_review,
)


PORT = 5055
REVIEW_PAGE_LIMIT = 5
logger = get_logger(__name__)
_FILTER_CLEANUP_DONE = False
_LAST_CLEANUP_AT = 0.0
CLEANUP_THROTTLE_SECONDS = 30
_RETRAIN_LOCK = threading.Lock()
_RETRAIN_STATUS_LOCK = threading.Lock()
_RETRAIN_STATUS: dict[str, str | bool] = {
    "running": False,
    "type": "idle",
    "message": "",
    "started_at": "",
    "finished_at": "",
    "error": "",
}


def _mark_retrain_started() -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with _RETRAIN_STATUS_LOCK:
        _RETRAIN_STATUS.update({
            "running": True,
            "type": "running",
            "message": "Retreinando modelo em background. Pode continuar revisando.",
            "started_at": now,
            "finished_at": "",
            "error": "",
        })


def _mark_retrain_finished(ok: bool, error: str = "") -> None:
    now = datetime.now().isoformat(timespec="seconds")
    message = (
        "Retreino concluído. As próximas decisões já podem usar o modelo atualizado."
        if ok
        else "Retreino falhou. Veja o terminal/logs para detalhes."
    )
    with _RETRAIN_STATUS_LOCK:
        _RETRAIN_STATUS.update({
            "running": False,
            "type": "ok" if ok else "err",
            "message": message,
            "finished_at": now,
            "error": error[:500],
        })


def _get_retrain_status() -> dict[str, str | bool]:
    with _RETRAIN_STATUS_LOCK:
        status = dict(_RETRAIN_STATUS)
    if _RETRAIN_LOCK.locked():
        status["running"] = True
        status["type"] = "running"
        if not status.get("message"):
            status["message"] = "Retreinando modelo em background. Pode continuar revisando."
    return status


def _domains_from_feedback_details(details: dict) -> list[str]:
    domains: list[str] = []

    def add(domain: str) -> None:
        if domain and domain not in domains:
            domains.append(domain)

    if (
        details.get("photo_score_adjustment")
        or details.get("photo_reason")
        or details.get("photo_positive_details")
        or details.get("photo_negative_details")
        or details.get("body_frame_correction")
        or details.get("body_build_correction")
        or details.get("visual_face_label")
        or details.get("visual_body_label")
        or details.get("visual_style_label")
        or details.get("visual_overall_label")
    ):
        add("photo")
    if details.get("descriptor_detail") or details.get("descriptor_positive_details") or details.get("descriptor_negative_details"):
        add("descriptors")
    if details.get("descriptor_not_positive") or details.get("descriptor_not_negative"):
        add("descriptors")
    if details.get("selected_interests") or details.get("interest_not_positive") or details.get("interest_not_negative"):
        add("interests")
    if details.get("bio_detail") or details.get("bio_not_positive") or details.get("bio_not_negative"):
        add("bio")
    return domains


def _clean_visual_label(value: str) -> str:
    clean = str(value or "").strip().lower()
    return clean if clean in {"positive", "neutral", "negative"} else ""


def _visual_label_select() -> str:
    options = [
        ("", "sem marcar"),
        ("positive", "gostei"),
        ("neutral", "neutro"),
        ("negative", "não gostei"),
    ]
    return "\n".join(f'<option value="{value}">{_esc(label)}</option>' for value, label in options)


def _primary_domain_from_details(details: dict, fallback: str = "other") -> str:
    domains = _domains_from_feedback_details(details)
    return domains[0] if domains else fallback


def _maybe_start_review_auto_retrain(reason: str = "review_saved") -> bool:
    cfg = _load_raw_config()
    retrain_every = int((cfg.get("model", {}) or {}).get("retrain_every", 10) or 10)
    if retrain_every <= 0:
        return False
    try:
        from model import count_trainable_real_profiles

        current_count = count_trainable_real_profiles()
        model_data = load_model()
        trained_count = int((model_data or {}).get("n_samples", 0) or 0)
    except Exception:
        logger.debug("Auto-retreino pos-review indisponivel", exc_info=True)
        return False

    pending = current_count - trained_count
    if pending < retrain_every:
        logger.info(
            "Auto-retreino pos-review aguardando mais dados: pending=%s retrain_every=%s reason=%s",
            pending,
            retrain_every,
            reason,
        )
        return False
    if not _RETRAIN_LOCK.acquire(blocking=False):
        logger.info("Auto-retreino pos-review ignorado: retreino ja em andamento")
        return False

    _mark_retrain_started()

    def _run() -> None:
        try:
            logger.info(
                "Auto-retreino pos-review iniciado: pending=%s current=%s trained=%s reason=%s",
                pending,
                current_count,
                trained_count,
                reason,
            )
            train_model()
            logger.info("Auto-retreino pos-review concluido")
            _mark_retrain_finished(True)
        except Exception as exc:
            logger.exception("Erro no auto-retreino pos-review")
            _mark_retrain_finished(False, str(exc))
        finally:
            _RETRAIN_LOCK.release()

    threading.Thread(target=_run, daemon=True, name="review-auto-retrain").start()
    return True


def _render_retrain_status_notice() -> str:
    status = _get_retrain_status()
    if not status.get("running"):
        return '<div id="retrain-status" class="notice retrain" hidden aria-live="polite"></div>'
    message = _esc(str(status.get("message") or "Retreinando modelo em background."))
    return (
        '<div id="retrain-status" class="notice retrain running" aria-live="polite">'
        '<span class="retrain-dot" aria-hidden="true"></span>'
        f"<span>{message}</span>"
        "</div>"
    )


def _render_evaluation_summary() -> str:
    if not LATEST_JSON.exists():
        return ""
    try:
        report = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Falha ao ler relatório de avaliação", exc_info=True)
        return ""
    if report.get("status") != "ok":
        return ""

    dataset = report.get("dataset", {}) or {}
    metrics = report.get("metrics", {}) or {}
    thresholds = report.get("thresholds", {}) or {}
    threshold = thresholds.get("like_threshold", 0.48)
    rows = dataset.get("evaluated_rows", 0)
    coverage = float(dataset.get("probability_coverage", 0.0) or 0.0) * 100
    precision = metrics.get("precision", "")
    recall = metrics.get("recall", "")
    false_neg = metrics.get("fn", "")
    brier = report.get("brier_score", "")
    generated = report.get("generated_at", "")

    return f"""
    <section class="eval-summary" aria-label="Avaliação do modelo">
      <div>
        <strong>Avaliação offline</strong>
        <span>{_esc(generated)} · {rows} perfis no holdout ({coverage:.0f}% com probabilidade)</span>
      </div>
      <div class="eval-metrics">
        <span>threshold <b>{_esc(threshold)}</b></span>
        <span>precision <b>{_esc(precision)}</b></span>
        <span>recall boas <b>{_esc(recall)}</b></span>
        <span>FN <b>{_esc(false_neg)}</b></span>
        <span>brier <b>{_esc(brier)}</b></span>
      </div>
    </section>"""


def _cleanup_review_queue_once() -> tuple[int, int]:
    """Limpezas caras de pendências antigas; não devem rodar em todo clique."""
    global _FILTER_CLEANUP_DONE
    if _FILTER_CLEANUP_DONE:
        return 0, 0
    _FILTER_CLEANUP_DONE = True
    hidden_filters = skip_absolute_filter_reviews()
    hidden_duplicates = skip_duplicate_pending_reviews()
    hidden_duplicates += skip_duplicate_history_reviews()
    return hidden_filters, hidden_duplicates


def _cleanup_review_queue() -> tuple[int, int]:
    """Limpa pendências duplicadas ou filtros absolutos sempre que a UI reenfileira a lista.

    Throttled to reduce frequent I/O: calls to this helper may occur often
    during UI rendering, so only run the expensive dedupe work at most once
    every `CLEANUP_THROTTLE_SECONDS`. When the cleanup actually runs we log
    the result to help debugging.
    """
    global _LAST_CLEANUP_AT
    now = time.time()
    if now - _LAST_CLEANUP_AT < CLEANUP_THROTTLE_SECONDS:
        return 0, 0
    _LAST_CLEANUP_AT = now
    hidden_filters = skip_absolute_filter_reviews()
    hidden_duplicates = skip_duplicate_pending_reviews()
    hidden_duplicates += skip_duplicate_history_reviews()
    if hidden_filters or hidden_duplicates:
        logger.info(
            "Review cleanup executed: hidden_filters=%d hidden_duplicates=%d",
            hidden_filters,
            hidden_duplicates,
        )
    return hidden_filters, hidden_duplicates


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de config
# ──────────────────────────────────────────────────────────────────────────────

def _load_raw_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_raw_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _photo_deep_enabled() -> bool:
    cfg = _load_raw_config()
    return bool((cfg.get("review_ui") or {}).get("photo_deep_enabled", False))


def _load_prefs() -> dict:
    cfg = _load_raw_config()
    prefs = cfg.get("preferences", {}) or {}
    hf = cfg.get("hard_filters", {}) or {}
    age_range = prefs.get("age_range") or [18, 30]

    def _norm_list(lst) -> list[str]:
        seen: set[str] = set()
        result = []
        for s in (lst or []):
            if s:
                n = _normalize(s)
                if n not in seen:
                    seen.add(n)
                    result.append(n)
        return result

    return {
        "age_min": int(age_range[0]) if len(age_range) > 0 else 18,
        "age_max": int(age_range[1]) if len(age_range) > 1 else 30,
        "preferred_interests": _norm_list(prefs.get("preferred_interests")),
        "positive_bio_keywords": _norm_list(prefs.get("positive_bio_keywords")),
        "negative_bio_keywords": _norm_list(prefs.get("negative_bio_keywords")),
        "disliked_names": _norm_list(prefs.get("disliked_names")),
        "extra_male_names": _norm_list(hf.get("extra_male_names")),
        "desc_neg_suppressed": set(_norm_list(prefs.get("desc_neg_suppressed"))),
        # raw lists for display
        "_raw_preferred_interests": list(prefs.get("preferred_interests") or []),
        "_raw_positive_bio_keywords": list(prefs.get("positive_bio_keywords") or []),
        "_raw_negative_bio_keywords": list(prefs.get("negative_bio_keywords") or []),
        "_raw_disliked_names": list(prefs.get("disliked_names") or []),
        "_raw_extra_male_names": list(hf.get("extra_male_names") or []),
        "_raw_age_range": list(age_range),
    }


def _config_list_update(section: str, field: str, action: str, value: str) -> bool:
    """Adiciona ou remove um item de uma lista no config.yaml."""
    cfg = _load_raw_config()
    try:
        obj = cfg
        for key in section.split("."):
            if key not in obj:
                obj[key] = {}
            obj = obj[key]
        lst = obj.get(field) or []
        value = value.strip()
        if not value:
            return False
        if action == "add" and value not in lst:
            lst.append(value)
            obj[field] = lst
        elif action == "remove" and value in lst:
            lst.remove(value)
            obj[field] = lst
        else:
            return False
        _save_raw_config(cfg)
        return True
    except Exception:
        logger.exception("Falha ao atualizar config: section=%r field=%r", section, field)
        return False


def _config_age_update(min_age: int, max_age: int) -> bool:
    cfg = _load_raw_config()
    try:
        prefs = cfg.setdefault("preferences", {})
        prefs["age_range"] = [min_age, max_age]
        _save_raw_config(cfg)
        return True
    except Exception:
        logger.exception("Falha ao atualizar faixa de idade")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de normalização e cor
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    base = unicodedata.normalize("NFD", (text or "").strip().lower())
    return "".join(ch for ch in base if unicodedata.category(ch) != "Mn")


def _esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _interest_class(interest: str, prefs: dict) -> str:
    norm = _normalize(interest)
    for p in prefs["preferred_interests"]:
        if norm == p or norm in p or p in norm:
            return "tag tag-match"
    for n in prefs["negative_bio_keywords"]:
        if norm == n or norm in n or n in norm:
            return "tag tag-neg"
    return "tag"


def _bio_matches(bio: str, prefs: dict) -> tuple[list[str], list[str]]:
    """Retorna (palavras_positivas_encontradas, palavras_negativas_encontradas)."""
    norm_bio = _normalize(bio)
    pos_found, neg_found = [], []
    for kw in prefs["positive_bio_keywords"]:
        pattern = r"\b" + re.escape(kw) + r"\b" if " " not in kw else re.escape(kw)
        if re.search(pattern, norm_bio):
            pos_found.append(kw)
    for kw in prefs["negative_bio_keywords"]:
        pattern = r"\b" + re.escape(kw) + r"\b" if " " not in kw else re.escape(kw)
        if re.search(pattern, norm_bio):
            neg_found.append(kw)
    return pos_found, neg_found


def _matching_interests(interests: list[str], prefs: dict) -> list[str]:
    matched = []
    for interest in interests:
        norm = _normalize(interest)
        for p in prefs["preferred_interests"]:
            if norm == p or norm in p or p in norm:
                matched.append(interest)
                break
    return matched


def _highlight_bio_html(bio: str, prefs: dict) -> str:
    """Retorna HTML do texto da bio com palavras-chave destacadas."""
    if not bio:
        return '<em class="muted">sem bio</em>'

    norm_bio = _normalize(bio)
    marks: list[tuple[int, int, str]] = []

    for kw in prefs["negative_bio_keywords"]:
        pattern = r"\b" + re.escape(kw) + r"\b" if " " not in kw else re.escape(kw)
        for m in re.finditer(pattern, norm_bio):
            marks.append((m.start(), m.end(), "bio-neg"))

    for kw in prefs["positive_bio_keywords"]:
        pattern = r"\b" + re.escape(kw) + r"\b" if " " not in kw else re.escape(kw)
        for m in re.finditer(pattern, norm_bio):
            marks.append((m.start(), m.end(), "bio-pos"))

    if not marks:
        return html.escape(bio)

    marks.sort(key=lambda x: x[0])
    # remove overlaps
    clean: list[tuple[int, int, str]] = []
    for start, end, cls in marks:
        if clean and start < clean[-1][1]:
            continue
        clean.append((start, end, cls))

    result = []
    prev = 0
    for start, end, cls in clean:
        result.append(html.escape(bio[prev:start]))
        result.append(f'<mark class="{cls}">{html.escape(bio[start:end])}</mark>')
        prev = end
    result.append(html.escape(bio[prev:]))
    return "".join(result)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de foto / URL
# ──────────────────────────────────────────────────────────────────────────────

def _photo_exists(row: dict) -> bool:
    rel = row.get("photo_path") or ""
    if not rel:
        return False
    path = (ROOT_DIR / rel).resolve()
    try:
        path.relative_to(ROOT_DIR)
    except ValueError:
        return False
    return path.exists()


def _photo_abs_path(rel: str) -> Path | None:
    if not rel:
        return None
    path = (ROOT_DIR / rel).resolve()
    try:
        path.relative_to(ROOT_DIR)
    except ValueError:
        return None
    return path if path.exists() else None


def _file_sha1(path: Path) -> str:
    try:
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 128), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _same_local_photo(rel_a: str, rel_b: str) -> bool:
    path_a = _photo_abs_path(rel_a)
    path_b = _photo_abs_path(rel_b)
    if not path_a or not path_b:
        return False
    try:
        if path_a.samefile(path_b):
            return True
    except Exception:
        pass
    try:
        if path_a.stat().st_size != path_b.stat().st_size:
            return False
    except Exception:
        return False
    hash_a = _file_sha1(path_a)
    return bool(hash_a and hash_a == _file_sha1(path_b))


def _remote_photo_url(row: dict) -> str:
    url = (row.get("photo_url") or "").strip()
    if not url.startswith(("https://", "http://")):
        return ""
    return url


def _photo_url(row: dict) -> str:
    rel = row.get("photo_path") or ""
    if rel and _photo_exists(row):
        return "/photo?path=" + urllib.parse.quote(rel)
    return _remote_photo_url(row)


def _photo_available(row: dict) -> bool:
    return _photo_exists(row) or bool(_remote_photo_url(row))


def _body_photo_path(row: dict) -> str:
    rel = row.get("photo_path") or ""
    if not rel.endswith(".jpg"):
        return ""
    if body_measurement_strength(row) < BODY_MEASUREMENT_MIN_STRENGTH:
        return ""
    body_rel = rel[:-4] + "_body.jpg"
    path = (ROOT_DIR / body_rel).resolve()
    try:
        path.relative_to(ROOT_DIR)
    except ValueError:
        return ""
    if not path.exists():
        return ""
    if _same_local_photo(rel, body_rel):
        logger.debug("Foto corporal omitida da review porque duplica a principal: %s", body_rel)
        return ""
    return body_rel


def _body_photo_url(row: dict) -> str:
    body_rel = _body_photo_path(row)
    return "/photo?path=" + urllib.parse.quote(body_rel) if body_rel else ""


def _body_correction_html(row: dict) -> str:
    # A review principal mostra a inferência corporal em _body_inference_html.
    # Correções de enquadramento/silhueta ficam só no fluxo de treino corporal.
    return ""


def _body_inference_html(row: dict) -> str:
    body_vis = str(row.get("photo_body_visible", "")).strip()
    if body_vis in ("", "0", "0.0"):
        return ""

    chips = []

    # enquadramento
    if str(row.get("photo_body_full_length", "")).strip() not in ("", "0", "0.0"):
        chips.append('<span class="bchip frame">corpo inteiro</span>')
    elif str(row.get("photo_body_upper_length", "")).strip() not in ("", "0", "0.0"):
        chips.append('<span class="bchip frame">meio corpo</span>')
    elif str(row.get("photo_body_closeup", "")).strip() not in ("", "0", "0.0"):
        chips.append('<span class="bchip frame">close-up</span>')

    # silhueta
    narrow = str(row.get("photo_body_width_bucket_narrow", "")).strip() not in ("", "0", "0.0")
    medium = str(row.get("photo_body_width_bucket_medium", "")).strip() not in ("", "0", "0.0")
    wide = str(row.get("photo_body_width_bucket_wide", "")).strip() not in ("", "0", "0.0")

    if narrow and medium and not wide:
        chips.append('<span class="bchip build">silhueta média-estreita</span>')
    elif medium and wide and not narrow:
        chips.append('<span class="bchip build">silhueta média-ampla</span>')
    elif narrow:
        chips.append('<span class="bchip build">silhueta estreita</span>')
    elif medium:
        chips.append('<span class="bchip build">silhueta média</span>')
    elif wide:
        chips.append('<span class="bchip build">silhueta ampla</span>')

    # proporção de largura
    try:
        wr = float(row.get("photo_body_width_ratio") or 0)
        if wr > 0:
            chips.append(f'<span class="bchip ratio">largura {wr * 100:.0f}%</span>')
    except Exception:
        pass

    # qualidade do sinal
    try:
        sq = float(row.get("photo_body_signal_quality") or 0)
        if sq > 0:
            sq_cls = "sq-ok" if sq >= 0.6 else "sq-mid" if sq >= 0.3 else "sq-low"
            chips.append(f'<span class="bchip quality {sq_cls}">sinal {sq * 100:.0f}%</span>')
    except Exception:
        pass

    if not chips:
        return ""

    return '<div class="body-inference"><span class="body-label">corpo</span>' + "".join(chips) + "</div>"


def _pct(row: dict, key: str, default: str = "--") -> str:
    try:
        raw = row.get(key)
        if raw in ("", None):
            return default
        v = float(raw)
        return f"{v * 100:.0f}%"
    except Exception:
        return default


def _distance_text(row: dict) -> str:
    try:
        raw = row.get("distance_km")
        if raw in ("", None):
            return ""
        km = float(raw)
        if km != km:
            return ""
        return f"{km:.0f} km"
    except Exception:
        return ""


def _parse_interests(raw: str) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _parse_descriptors(raw: str) -> dict:
    try:
        d = json.loads(raw or "{}")
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _decision_label(label: str) -> str:
    raw = str(label).strip().upper()
    if raw in {"SUPER_LIKE", "SUPER LIKE"}:
        return "SUPER LIKE"
    if raw in {"TALVEZ", "MAYBE"}:
        return "TALVEZ"
    return "CURTIR" if raw in {"1", "1.0", "CURTIR"} else "NÃO CURTIR"


def _descriptor_polarity(key: str, value: str, suppressed: set | None = None) -> tuple[str, str]:
    key_n = _normalize(key)
    val_n = _normalize(value)
    detail_n = _normalize(f"{key}: {value}")
    if suppressed and (key_n in suppressed or detail_n in suppressed):
        return "", ""

    if key_n in {"familia", "família"}:
        does_not_want = "nao quero" in val_n
        unsure = "ainda nao sei" in val_n
        wants = not does_not_want and "quero filhos" in val_n
        has_children = (
            "tenho filho" in val_n
            or "tenho filhos" in val_n
            or ("ja tenho" in val_n and "filh" in val_n)
            or "meus filhos" in val_n
            or re.search(r"\b(mae|pai)\s+(solo|de|do|da)\b", val_n) is not None
        )
        if wants:
            return "like", "IA: pró-curtir"
        if does_not_want or unsure or has_children:
            return "pass", "IA: pró-passar"
    if key_n in {"voce fuma?", "você fuma?", "fumo"}:
        if val_n and "nao fumo" not in val_n and "não fumo" not in val_n:
            return "pass", "IA: pró-passar"
    if key_n == "atividade fisica" and any(x in val_n for x in ["frequentemente", "todo dia"]):
        return "like", "IA: pró-curtir"
    if key_n == "pets" and any(x in val_n for x in ["cachorro", "gato", "gosto", "amo"]):
        return "like", "IA: pró-curtir"
    if key_n in {"formacao", "formação"} and any(x in val_n for x in ["superior", "faculdade", "pos", "pós"]):
        return "like", "IA: pró-curtir"
    if key_n in {"religiao", "religião"} and any(x in val_n for x in ["crista", "cristã", "catolic", "evangelic"]):
        return "like", "IA: pró-curtir"
    if key_n in {"tipo de relacionamento", "relacionamento"} and ("nao-monog" in val_n or "não-monog" in val_n):
        return "pass", "IA: pró-passar"
    if key_n in {"estilo de comunicacao", "estilo de comunicação"} and any(x in val_n for x in ["odeio", "demoro"]):
        return "pass", "IA: pró-passar"
    if key_n == "bebida" and val_n and "nao curto" not in val_n and "não curto" not in val_n and "parei" not in val_n:
        return "neutral", "IA: neutro"
    if key_n in {"habitos de sono", "hábitos de sono"} and "noturna" in val_n:
        return "neutral", "IA: neutro"
    return "", ""


def _descriptor_tag_html(key: str, value: str, suppressed: set | None = None) -> str:
    polarity, _ = _descriptor_polarity(key, value, suppressed)
    cls = "tag tag-soft"
    if polarity == "like":
        cls = "tag tag-match"
    elif polarity == "pass":
        cls = "tag tag-neg"
    return f'<span class="{cls}">{_esc(key)}: <b>{_esc(value)}</b></span>'


def _descriptor_correction_rows(descriptors: dict) -> str:
    if not descriptors:
        return '<span class="muted compact-note">sem descritores neste perfil</span>'

    rows = []
    for key, value in list(descriptors.items())[:20]:
        if not key and not value:
            continue
        detail = f"{key}: {value}"
        polarity, hint = _descriptor_polarity(str(key), str(value))
        cls = f"descriptor-choice inferred-{polarity or 'none'}"  # no suppressed here — correction rows always show all
        hint_html = f"<small>{_esc(hint)}</small>" if hint else ""
        rows.append(
            f'<div class="{cls}">'
            f'<span class="descriptor-label">{_esc(key)}: <b>{_esc(value)}</b>{hint_html}</span>'
            f'<label class="chip-check positive descriptor-signal-opt">'
            f'<input type="checkbox" class="descriptor-signal-cb" name="descriptor_positive_detail" value="{_esc(detail)}" data-aspect="{_esc(detail)}">'
            f"<span>gosto disso</span></label>"
            f'<label class="chip-check negative descriptor-signal-opt">'
            f'<input type="checkbox" class="descriptor-signal-cb" name="descriptor_negative_detail" value="{_esc(detail)}" data-aspect="{_esc(detail)}">'
            f"<span>não gosto</span></label>"
            f"</div>"
        )
    return "\n".join(rows) or '<span class="muted compact-note">sem descritores neste perfil</span>'


# Mesmo valor interno nos dois lados; rótulos separados para não soar absurdo
# em “não gostei” (ex.: nunca “rosto bonito” como algo que se “não gosta”).
# Quarto campo = valor do <select name="photo_detail"> em que o chip aparece;
# "photo_general" mostra todos os chips na UI.
PHOTO_ASPECT_OPTIONS: list[tuple[str, str, str, str]] = [
    ("face_beauty", "rosto / traços (combinou)", "rosto / traços (não combinou)", "photo_face"),
    ("face_expression", "sorriso / expressão (a favor)", "sorriso / expressão (contra)", "photo_face"),
    (
        "gender_presentation",
        "gênero visual/feminilidade (ok)",
        "aparência masculina ou ambígua",
        "photo_gender",
    ),
    ("body_shape", "corpo / silhueta (a favor)", "corpo / silhueta (contra)", "photo_body"),
    ("body_fitness", "forma física / fitness (a favor)", "forma física / fitness (contra)", "photo_body"),
    ("body_visibility", "quanto de corpo na foto (ok)", "quanto de corpo na foto (incomoda)", "photo_body"),
    ("context_lifestyle", "lugar / vibe / estilo de vida (positivo)", "lugar / vibe / estilo de vida (negativo)", "photo_context"),
    ("style_outfit", "roupa / estética (positivo)", "roupa / estética (negativo)", "photo_style"),
    ("black_photo", "foto legível / com conteúdo", "foto preta ou sem conteúdo", "photo_style"),
    ("photo_quality", "pose / enquadramento / qualidade (positivo)", "pose / enquadramento / qualidade (negativo)", "photo_style"),
]


def _photo_aspect_keys_for_detail(photo_detail: str) -> set[str] | None:
    """None = qualquer aspecto permitido (detalhe geral ou vazio)."""
    pd = (photo_detail or "").strip()
    if not pd or pd == "photo_general":
        return None
    return {key for key, _, _, det in PHOTO_ASPECT_OPTIONS if det == pd}


def _photo_fine_choices(field_name: str, chip_class: str, positive: bool) -> str:
    rows = (
        (value, pos_lbl if positive else neg_lbl, detail_cat)
        for value, pos_lbl, neg_lbl, detail_cat in PHOTO_ASPECT_OPTIONS
    )
    return "\n".join(
        f'<label class="chip-check {chip_class} photo-aspect-opt" data-for-details="{_esc(detail_cat)}">'
        f'<input type="checkbox" class="photo-aspect-cb" name="{field_name}" value="{_esc(value)}" data-aspect="{_esc(value)}">'
        f"<span>{_esc(label)}</span></label>"
        for value, label, detail_cat in rows
    )


# ──────────────────────────────────────────────────────────────────────────────
# Seção de raciocínio da IA
# ──────────────────────────────────────────────────────────────────────────────

def _render_ai_reasoning(row: dict, prefs: dict) -> str:
    interests = _parse_interests(row.get("interests", ""))
    bio = row.get("bio", "") or ""
    distance_label = _distance_text(row)
    age_str = str(row.get("age", "")).strip()

    lines = []

    # Idade
    try:
        age = int(float(age_str))
        in_range = prefs["age_min"] <= age <= prefs["age_max"]
        age_cls = "signal-ok" if in_range else "signal-no"
        age_text = f"dentro de [{prefs['age_min']}, {prefs['age_max']}]" if in_range else f"fora de [{prefs['age_min']}, {prefs['age_max']}]"
        lines.append(f'<span class="{age_cls}">Idade {age} — {age_text}</span>')
    except Exception:
        pass

    # Interesses combinando
    matched = _matching_interests(interests, prefs)
    if matched:
        tags = " ".join(f'<span class="mini-tag match">{_esc(i)}</span>' for i in matched)
        lines.append(f'<span class="signal-ok">Interesses em comum ({len(matched)}/{len(interests)}):</span> {tags}')
    elif interests:
        lines.append(f'<span class="signal-no">Nenhum interesse em comum de {len(interests)} listados</span>')

    # Palavras na bio
    pos_found, neg_found = _bio_matches(bio, prefs)
    if pos_found:
        tags = " ".join(f'<span class="mini-tag match">{_esc(k)}</span>' for k in pos_found[:6])
        lines.append(f'<span class="signal-ok">Bio — palavras positivas:</span> {tags}')
    if neg_found:
        tags = " ".join(f'<span class="mini-tag neg">{_esc(k)}</span>' for k in neg_found[:6])
        lines.append(f'<span class="signal-no">Bio — palavras negativas:</span> {tags}')

    # Foto
    w_conf = row.get("photo_woman_confidence", "")
    sim = row.get("photo_face_similarity", "")
    has_face = str(row.get("photo_has_face", "")).strip()
    body_vis = str(row.get("photo_body_visible", "")).strip()
    smile = row.get("photo_face_smile_score", "")
    brightness = row.get("photo_image_brightness", "")
    sharpness = row.get("photo_image_sharpness", "")

    try:
        wc = float(w_conf) * 100
        wc_cls = "signal-ok" if wc >= 70 else "signal-mid" if wc >= 50 else "signal-no"
        lines.append(f'<span class="{wc_cls}">Foto — confiança mulher: {wc:.0f}%</span>')
    except Exception:
        pass

    try:
        sv = float(sim) * 100
        sv_cls = "signal-ok" if sv >= 60 else "signal-mid" if sv >= 40 else "signal-no"
        lines.append(f'<span class="{sv_cls}">Foto — similaridade ao seu gosto: {sv:.0f}%</span>')
    except Exception:
        pass

    try:
        if float(has_face) > 0:
            lines.append('<span class="signal-ok">Foto — rosto detectado</span>')
        else:
            lines.append('<span class="signal-no">Foto — nenhum rosto detectado</span>')
    except Exception:
        pass

    try:
        if float(body_vis) > 0:
            lines.append('<span class="signal-ok">Foto — corpo visível</span>')
    except Exception:
        pass

    try:
        sm = float(smile)
        if sm >= 0.62:
            lines.append(f'<span class="signal-ok">Foto — sorriso/expressão presente: {sm * 100:.0f}%</span>')
        elif sm <= 0.25:
            lines.append(f'<span class="signal-mid">Foto — expressão mais neutra/séria: {sm * 100:.0f}%</span>')
    except Exception:
        pass

    try:
        br = float(brightness)
        if br <= 0.28:
            lines.append(f'<span class="signal-mid">Foto — iluminação escura: {br * 100:.0f}%</span>')
        elif br >= 0.78:
            lines.append(f'<span class="signal-mid">Foto — muito clara: {br * 100:.0f}%</span>')
    except Exception:
        pass

    try:
        sh = float(sharpness)
        if sh >= 0.62:
            lines.append(f'<span class="signal-ok">Foto — nítida: {sh * 100:.0f}%</span>')
        elif sh <= 0.28:
            lines.append(f'<span class="signal-mid">Foto — pouca nitidez: {sh * 100:.0f}%</span>')
    except Exception:
        pass

    if not lines:
        return ""

    inner = "".join(f"<div>{l}</div>" for l in lines)
    return f'<section class="reasoning"><h3>Por que a IA decidiu assim</h3><div class="reason-lines">{inner}</div></section>'


def _signal_list(items: list[str], empty: str, limit: int | None = None) -> str:
    if limit is not None:
        items = items[:limit]
    if not items:
        return f'<li class="muted">{_esc(empty)}</li>'
    return "".join(f"<li>{_esc(item)}</li>" for item in items)


def _as_float_display(value, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _fmt_pct_value(value, default: str = "--") -> str:
    try:
        if value in ("", None):
            return default
        return f"{float(value) * 100:.0f}%"
    except Exception:
        return default


def _score_chip(label: str, value, cls: str = "", title: str = "") -> str:
    pct = _fmt_pct_value(value)
    if pct == "--":
        return ""
    title_attr = f' title="{_esc(title)}"' if title else ""
    cls_attr = f" score-chip {cls}".strip()
    return f'<span class="{_esc(cls_attr)}"{title_attr}>{_esc(label)} <b>{pct}</b></span>'


def _model_chip(label: str, value, detail: str = "") -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    detail_html = f"<small>{_esc(detail)}</small>" if detail else ""
    return f'<span class="model-chip"><b>{_esc(label)}</b>{_esc(value)}{detail_html}</span>'


def _row_visual_subscores(row: dict) -> list[dict]:
    try:
        from explainer import build_visual_subscores
        from features import PHOTO_FEATURE_NAMES

        features = {
            key: row.get(key)
            for key in PHOTO_FEATURE_NAMES
            if row.get(key) not in ("", None)
        }
        return build_visual_subscores({"features": features})
    except Exception:
        return []


def _render_visual_breakdown(snapshot: dict, row: dict) -> str:
    items = snapshot.get("visual_subscores")
    if not isinstance(items, list) or not items:
        items = _row_visual_subscores(row)
    if not items:
        return ""

    parts: list[str] = []
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        score = _as_float_display(item.get("score"), -1.0)
        if score < 0:
            continue
        key = _normalize(str(item.get("key") or item.get("label") or "visual"))
        key = re.sub(r"[^a-z0-9_-]+", "-", key).strip("-") or "visual"
        label = str(item.get("label") or key).strip()
        note = str(item.get("note") or "").strip()
        pct = max(0, min(100, int(round(score * 100))))
        tone = "high" if score >= 0.62 else "mid" if score >= 0.43 else "low"
        title = f"{label}: {note}" if note else label
        parts.append(
            f'<span class="visual-score {tone} {key}" title="{_esc(title)}">'
            f'<strong>{_esc(label)}</strong>'
            f'<i style="--v:{pct}%"></i>'
            f'<b>{pct}%</b>'
            f'</span>'
        )

    if not parts:
        return ""
    return '<div class="visual-breakdown">{} </div>'.format("".join(parts))


def _signal_box(title: str, items: list[str], cls: str, empty: str, limit: int = 5) -> str:
    return f"""
      <div class="signal-box {cls}">
        <h3>{_esc(title)}</h3>
        <ul>{_signal_list(items, empty, limit)}</ul>
      </div>
    """


def _load_ai_snapshot(row: dict) -> dict:
    try:
        data = json.loads(row.get("ai_snapshot") or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _snapshot_from_current_row(row: dict, prefs: dict) -> dict:
    """Monta uma leitura leve para registros antigos sem ai_snapshot."""
    interests = _parse_interests(row.get("interests", ""))
    bio = row.get("bio", "") or ""
    descriptors = _parse_descriptors(row.get("descriptors", ""))
    pro_like: list[str] = []
    pro_pass: list[str] = []
    photo_lines: list[str] = []
    body_lines: list[str] = []
    text_lines: list[str] = []
    top: list[str] = []
    uncertainty: list[str] = ["registro antigo sem snapshot completo; leitura reconstruída com os dados salvos"]

    def add(target: list[str], text: str) -> None:
        if text and text not in target:
            target.append(text)

    try:
        age = int(float(row.get("age") or 0))
        if prefs["age_min"] <= age <= prefs["age_max"]:
            add(pro_like, f"idade {age} dentro da faixa configurada")
            add(text_lines, f"idade dentro da faixa preferida ({prefs['age_min']}-{prefs['age_max']})")
        else:
            add(pro_pass, f"idade {age} fora da faixa configurada")
            add(text_lines, f"idade fora da faixa preferida ({prefs['age_min']}-{prefs['age_max']})")
    except Exception:
        pass

    dist_label = _distance_text(row)
    if dist_label:
        add(text_lines, f"distância informada: {dist_label}")
    else:
        add(uncertainty, "distância não veio no perfil salvo")

    matched = _matching_interests(interests, prefs)
    if matched:
        add(pro_like, "interesses preferidos: " + ", ".join(matched[:4]))
        add(text_lines, f"{len(matched)} interesse(s) batem com preferências explícitas")
    elif interests:
        add(uncertainty, "sem interesse preferido explícito")
        add(text_lines, f"{len(interests)} interesse(s), mas sem match explícito")
    else:
        add(text_lines, "perfil sem interesses listados")

    pos_bio, neg_bio = _bio_matches(bio, prefs)
    if pos_bio:
        add(pro_like, "bio com palavra positiva: " + ", ".join(pos_bio[:4]))
        add(text_lines, "bio ativou palavra(s) positiva(s): " + ", ".join(pos_bio[:4]))
    if neg_bio:
        add(pro_pass, "bio com alerta configurado: " + ", ".join(neg_bio[:4]))
        add(text_lines, "bio ativou alerta(s): " + ", ".join(neg_bio[:4]))
    if not bio.strip():
        add(uncertainty, "perfil sem bio")
        add(text_lines, "perfil sem bio")
    else:
        add(text_lines, f"bio com {len(bio)} caracteres")

    suppressed = prefs.get("desc_neg_suppressed", set())
    for key, raw_value in list(descriptors.items())[:12]:
        detail = f"{key}: {raw_value}"
        polarity, _ = _descriptor_polarity(str(key), str(raw_value), suppressed)
        if polarity == "like":
            add(pro_like, "descritor favorável: " + detail)
            add(text_lines, "descritor favorável: " + detail)
        elif polarity == "pass":
            add(pro_pass, "descritor de alerta: " + detail)
            add(text_lines, "descritor de alerta: " + detail)

    try:
        face_similarity = float(row.get("photo_face_similarity") or 0.5)
        add(photo_lines, f"similaridade visual salva: {face_similarity * 100:.0f}%")
        if face_similarity >= 0.62:
            add(pro_like, f"rosto parecido com curtidas anteriores ({face_similarity * 100:.0f}%)")
        elif face_similarity <= 0.42:
            add(pro_pass, f"similaridade visual baixa ({face_similarity * 100:.0f}%)")
    except Exception:
        pass

    try:
        woman_conf = float(row.get("photo_woman_confidence") or 0.5)
        add(photo_lines, f"confiança de perfil feminino: {woman_conf * 100:.0f}%")
        if woman_conf >= 0.70:
            add(pro_like, f"foto com alta confiança de perfil feminino ({woman_conf * 100:.0f}%)")
        elif woman_conf <= 0.55:
            add(pro_pass, f"confiança feminina baixa/ambígua ({woman_conf * 100:.0f}%)")
    except Exception:
        pass

    try:
        has_face = float(row.get("photo_has_face") or 0.0)
        add(photo_lines, "rosto detectado nas fotos" if has_face > 0 else "nenhum rosto confiável detectado")
        if has_face <= 0:
            add(pro_pass, "nenhum rosto confiável foi detectado nas fotos analisadas")
    except Exception:
        pass

    try:
        faces_ratio = float(row.get("photo_faces_ratio") or 0.0)
        if faces_ratio > 0:
            add(photo_lines, f"rostos aproveitáveis em {faces_ratio * 100:.0f}% das fotos")
    except Exception:
        pass

    body_visible = str(row.get("photo_body_visible", "")).strip()
    if body_visible not in ("", "0", "0.0"):
        add(body_lines, f"corpo visível na foto salva ({_pct(row, 'photo_body_visible')})")
        if str(row.get("photo_body_width_bucket_narrow", "")).strip() not in ("", "0", "0.0"):
            add(body_lines, "silhueta visual estreita")
        elif str(row.get("photo_body_width_bucket_medium", "")).strip() not in ("", "0", "0.0"):
            add(body_lines, "silhueta visual média")
        elif str(row.get("photo_body_width_bucket_wide", "")).strip() not in ("", "0", "0.0"):
            add(body_lines, "silhueta visual ampla")
        try:
            body_quality = float(row.get("photo_body_signal_quality") or 0.0)
            if body_quality > 0:
                add(body_lines, f"qualidade do sinal corporal: {body_quality * 100:.0f}%")
        except Exception:
            pass
    else:
        add(body_lines, "pouco corpo visível; leitura visual focou mais em rosto/texto")

    try:
        smile_score = float(row.get("photo_face_smile_score") or 0.5)
        if smile_score >= 0.62:
            add(photo_lines, f"sorriso/expressão presente ({smile_score * 100:.0f}%)")
        elif smile_score <= 0.25:
            add(photo_lines, f"expressão mais neutra/séria ({smile_score * 100:.0f}%)")
    except Exception:
        pass

    try:
        sharpness = float(row.get("photo_image_sharpness") or 0.5)
        brightness = float(row.get("photo_image_brightness") or 0.5)
        if sharpness >= 0.62:
            add(photo_lines, f"foto nítida ({sharpness * 100:.0f}%)")
        elif sharpness <= 0.28:
            add(uncertainty, f"foto pouco nítida ({sharpness * 100:.0f}%)")
        if brightness <= 0.28:
            add(uncertainty, f"foto escura ({brightness * 100:.0f}%)")
        elif 0.40 <= brightness <= 0.68:
            add(photo_lines, f"iluminação equilibrada ({brightness * 100:.0f}%)")
    except Exception:
        pass

    try:
        pose_vis = float(row.get("photo_pose_torso_visibility") or 0.0)
        if pose_vis >= 0.15:
            add(body_lines, f"pose corporal detectou tronco visível ({pose_vis * 100:.0f}%)")
            sw = float(row.get("photo_pose_shoulder_width") or 0.0)
            hw = float(row.get("photo_pose_hip_width") or 0.0)
            shr = float(row.get("photo_pose_shoulder_hip_ratio") or 0.5)
            if sw > 0 or hw > 0:
                add(body_lines, f"pose: ombro {sw * 100:.0f}% / quadril {hw * 100:.0f}% / proporção {shr * 100:.0f}%")
    except Exception:
        pass

    try:
        seg_cov = float(row.get("photo_seg_body_coverage") or 0.0)
        if seg_cov >= 0.08:
            ratio = float(row.get("photo_seg_shoulder_waist_ratio") or 0.5)
            add(body_lines, f"segmentação corporal: cobertura {seg_cov * 100:.0f}% / ombro-cintura {ratio * 100:.0f}%")
    except Exception:
        pass

    if photo_lines:
        top.append(photo_lines[0])
    if body_lines:
        top.append(body_lines[0])
    if text_lines:
        top.append(text_lines[0])

    return {
        "decision": _decision_label(row.get("original_label", row.get("label", ""))),
        "confidence": 0.0,
        "photo_score": 0.5,
        "text_score": 0.5,
        "text_preference_score": 0.5,
        "photo_score_mode": "salvo",
        "weights": {},
        "groups": {
            "pro_like": pro_like[:7],
            "pro_pass": pro_pass[:7],
            "photo": photo_lines[:8],
            "body": body_lines[:8],
            "text": text_lines[:8],
            "top": top[:6],
            "uncertainty": uncertainty[:5],
        },
        "_current_fallback": True,
    }


def _render_signal_panel(row: dict, prefs: dict) -> str:
    snapshot = _load_ai_snapshot(row)
    if not snapshot:
        snapshot = _snapshot_from_current_row(row, prefs)

    decision = snapshot.get("decision", "?")
    is_like = decision in {"CURTIR", "SUPER LIKE", "SUPER_LIKE"}
    confidence = _as_float_display(snapshot.get("confidence"), 0.0)
    decision_cls = "like" if is_like else "pass"
    fallback = bool(snapshot.get("_current_fallback"))
    safety = snapshot.get("probability_safety", {}) or {}
    safety_note = " calibrado" if safety.get("applied") else ""
    groups = snapshot.get("groups", {}) or {}
    weights = snapshot.get("weights", {}) or {}
    photo_score = _as_float_display(snapshot.get("photo_score"), 0.5)
    text_score = _as_float_display(snapshot.get("text_score"), 0.5)
    photo_weight = _as_float_display(weights.get("photo_weight"), 0.5)
    text_weight = _as_float_display(weights.get("text_weight"), 0.5)

    pro_like = list(groups.get("pro_like", []) or [])
    pro_pass = list(groups.get("pro_pass", []) or [])
    primary = pro_like if is_like else pro_pass
    counter = pro_pass if is_like else pro_like
    primary_title = "Por que curtiu" if is_like else "Por que passou"
    counter_title = "Contrapontos contra a decisão" if is_like else "Contrapontos a favor"

    score_chips: list[str] = []
    if fallback:
        score_chips.extend([
            _score_chip("similaridade", row.get("photo_face_similarity"), "photo"),
            _score_chip("mulher", row.get("photo_woman_confidence"), "photo"),
            _score_chip("sorriso", row.get("photo_face_smile_score"), "photo"),
            _score_chip("corpo", row.get("photo_body_visible"), "body"),
        ])
        badge = _esc(decision)
        source_label = "Leitura reconstruída"
        source_detail = "dados salvos + regras atuais"
    else:
        score_chips.extend([
            _score_chip(
                "certeza da decisão",
                confidence,
                "confidence",
                "Confiança calibrada na decisão mostrada no badge: CURTIR ou PASSAR.",
            ),
            _score_chip(
                "score visual",
                photo_score,
                "photo",
                "Pontuação final da parte visual, misturando modelo de foto e heurísticas de rosto/corpo/qualidade.",
            ),
            _score_chip(
                "score do perfil",
                text_score,
                "text",
                "Pontuação final da parte textual do perfil: bio, interesses, descritores e idade.",
            ),
            _score_chip(
                "histórico textual",
                snapshot.get("text_preference_score"),
                "text",
                "Score por sinais textuais que você já curtiu ou passou antes, como interesses, termos da bio e descritores.",
            ),
            _score_chip(
                "ML perfil",
                snapshot.get("text_model_probability"),
                "text",
                "Probabilidade bruta do modelo treinado só com features de perfil/texto.",
            ),
            _score_chip(
                "ML foto",
                snapshot.get("photo_model_probability"),
                "photo",
                "Probabilidade bruta do modelo treinado só com features visuais.",
            ),
        ])
        if is_like:
            score_chips.append(_score_chip(
                "chance super like",
                snapshot.get("superlike_probability"),
                "super",
                "Probabilidade do classificador separado de curtir forte, treinado só em perfis que você curtiu.",
            ))
        badge = f"{_esc(decision)} {confidence * 100:.0f}%{_esc(safety_note)}"
        source_label = "Snapshot do swipe"
        source_detail = "explicação congelada na decisão"
    score_chips = [chip for chip in score_chips if chip]

    if fallback:
        model_chips = []
        model_items = ["sem snapshot original de modelo; reconstruído a partir das features salvas"]
    else:
        model_type = str(snapshot.get("model_type", "") or "").strip()
        if model_type.startswith("Ensemble (") and model_type.endswith(")"):
            model_type = model_type[len("Ensemble ("):-1]
        model_chips = [
            _model_chip("Ensemble", model_type or "indisponível", f"{snapshot.get('n_samples') or '?'} treino"),
            _model_chip("Texto", snapshot.get("text_model_type"), f"{snapshot.get('text_n_samples') or '?'} amostras"),
            _model_chip("Foto", snapshot.get("photo_model_type") or snapshot.get("photo_score_mode"), f"{snapshot.get('photo_n_samples') or '?'} com foto"),
            _model_chip(
                "Super like",
                snapshot.get("superlike_model_type"),
                f"{snapshot.get('superlike_positive_samples') or 0}/{snapshot.get('superlike_n_samples') or 0} fortes",
            ),
        ]
        model_chips = [chip for chip in model_chips if chip]

        model_items = [
            f"pesos usados na época: foto {photo_weight * 100:.0f}% / texto {text_weight * 100:.0f}%",
        ]
        distance_info = snapshot.get("distance", {}) or {}
        if not distance_info.get("missing") and distance_info.get("km") not in ("", None):
            model_items.append(
                f"distância: {float(distance_info.get('km')):.0f} km "
                f"(score {_fmt_pct_value(distance_info.get('score'))})"
            )
        distance_adjustment = snapshot.get("distance_adjustment", {}) or {}
        if distance_adjustment.get("delta") not in ("", None):
            delta = float(distance_adjustment.get("delta") or 0.0)
            direction = "ajudou" if delta > 0 else "pesou contra"
            model_items.append(f"ajuste de distância {direction}: {delta:+.3f} na probabilidade")
        if snapshot.get("photo_score_mode"):
            model_items.append(f"score de foto: {snapshot.get('photo_score_mode')}")
        if safety.get("applied"):
            raw = _fmt_pct_value(safety.get("raw_probability") or snapshot.get("raw_probability"))
            safe = _fmt_pct_value(safety.get("safe_probability") or snapshot.get("probability"))
            reason = safety.get("reason") or "calibração de segurança"
            model_items.append(f"calibração ajustou prob. CURTIR de {raw} para {safe}: {reason}")
    model_items.extend(list(groups.get("top", []) or [])[:4])
    model_items.extend(list(groups.get("uncertainty", []) or [])[:3])

    photo_items = list(groups.get("photo", []) or [])
    body_items = list(groups.get("body", []) or [])
    text_items = list(groups.get("text", []) or [])

    scores_html = f'<div class="score-strip">{"".join(score_chips)}</div>' if score_chips else ""
    visual_html = _render_visual_breakdown(snapshot, row)
    models_html = f'<div class="model-strip">{"".join(model_chips)}</div>' if model_chips else ""

    details_title = (
        f"ver detalhes técnicos ({len(photo_items)} foto, {len(body_items)} corpo, {len(text_items)} texto)"
        if (photo_items or body_items or text_items)
        else "ver detalhes técnicos"
    )

    return f"""
      <section class="signal-panel decision-story">
        <div class="signal-head">
          <div>
            <span>{source_label}</span>
            <small>{source_detail}</small>
          </div>
          <b class="{decision_cls}">{badge}</b>
        </div>
        {scores_html}
        {visual_html}
        <div class="decision-summary">
          {_signal_box(primary_title, primary, f"primary {decision_cls}", "sem motivo dominante salvo", 3)}
          {_signal_box(counter_title, counter, "counter", "sem contraponto forte", 2)}
        </div>
        <details class="ai-details-drawer">
          <summary>{_esc(details_title)}</summary>
          {models_html}
          <div class="signal-grid detailed">
            {_signal_box("Foto / rosto", photo_items, "photo", "sem leitura visual detalhada", 4)}
            {_signal_box("Corpo / pose", body_items, "body", "sem sinal corporal relevante", 4)}
            {_signal_box("Texto / perfil", text_items, "text", "sem detalhe textual forte", 4)}
            {_signal_box("Modelos e pesos", model_items, "model", "modelo sem peso claro neste perfil", 5)}
          </div>
        </details>
      </section>
    """


def _signal_tail_values(line: str) -> list[str]:
    """Extrai valores depois de ':' em frases salvas no snapshot da IA."""
    if ":" not in line:
        return []
    tail = line.split(":", 1)[1]
    tail = re.sub(r"\([^)]*\)", "", tail)
    return [x.strip(" .") for x in tail.split(",") if x.strip(" .")]


def _classify_text_signal(value: str, interests: list[str], descriptors: dict) -> tuple[str, str]:
    """Mapeia um sinal textual para bio/interesse/descritor usando o perfil atual."""
    norm = _normalize(value)
    if not norm:
        return "bio", value

    interest_map = {_normalize(i): i for i in interests}
    if norm in interest_map:
        return "interest", interest_map[norm]

    descriptor_map = {}
    for key, raw_value in descriptors.items():
        detail = f"{key}: {raw_value}"
        descriptor_map[_normalize(detail)] = detail

    if norm in descriptor_map:
        return "descriptor", descriptor_map[norm]
    for desc_norm, detail in descriptor_map.items():
        if norm and (norm in desc_norm or desc_norm in norm):
            return "descriptor", detail
    if ":" in value:
        return "descriptor", value
    return "bio", value


def _render_signal_corrections(bio: str, interests: list[str], descriptors: dict, prefs: dict, row: dict) -> str:
    """Renderiza chips para neutralizar sinais textuais que a IA leu forte demais."""
    candidates: dict[str, dict[tuple[str, str], tuple[str, str, str]]] = {
        "positive": {},
        "negative": {},
    }

    def add(sentiment: str, field: str, value: str, label: str | None = None) -> None:
        clean = str(value or "").strip()
        if not clean:
            return
        key = (field, _normalize(clean))
        candidates[sentiment][key] = (field, clean, label or clean)

    # Sinais explícitos do config que aparecem no card.
    for interest in _matching_interests(interests, prefs):
        add("positive", "interest_not_positive", interest)
    for interest in interests:
        if "tag-neg" in _interest_class(interest, prefs):
            add("negative", "interest_not_negative", interest)

    pos_bio, neg_bio = _bio_matches(bio, prefs)
    for token in pos_bio:
        add("positive", "bio_not_positive", token)
    for token in neg_bio:
        add("negative", "bio_not_negative", token)

    suppressed = prefs.get("desc_neg_suppressed", set())
    for key, raw_value in descriptors.items():
        detail = f"{key}: {raw_value}"
        polarity, _ = _descriptor_polarity(str(key), str(raw_value), suppressed)
        if polarity == "like":
            add("positive", "descriptor_not_positive", detail)
        elif polarity == "pass":
            add("negative", "descriptor_not_negative", detail)

    # Sinais textuais aprendidos que estavam no snapshot salvo no momento do swipe.
    snapshot = _load_ai_snapshot(row)
    groups = snapshot.get("groups", {}) if snapshot else {}
    for line in groups.get("pro_like", []) or []:
        if "sinais textuais" not in _normalize(line):
            continue
        for value in _signal_tail_values(line):
            domain, detail = _classify_text_signal(value, interests, descriptors)
            field = {
                "interest": "interest_not_positive",
                "descriptor": "descriptor_not_positive",
            }.get(domain, "bio_not_positive")
            add("positive", field, detail)
    for line in groups.get("pro_pass", []) or []:
        if "sinais textuais" not in _normalize(line):
            continue
        for value in _signal_tail_values(line):
            domain, detail = _classify_text_signal(value, interests, descriptors)
            field = {
                "interest": "interest_not_negative",
                "descriptor": "descriptor_not_negative",
            }.get(domain, "bio_not_negative")
            add("negative", field, detail)

    # Fallback útil para registros antigos sem snapshot: mostra sinais textuais
    # que o aprendizado atual reconhece naquele mesmo perfil.
    if not snapshot:
        try:
            learned = score_profile_text(bio, interests, descriptors)
        except Exception:
            learned = {}
        for value in learned.get("interest_pref_positive_signals", []) or []:
            add("positive", "interest_not_positive", value)
        for value in learned.get("interest_pref_negative_signals", []) or []:
            add("negative", "interest_not_negative", value)
        for value in learned.get("bio_pref_positive_signals", []) or []:
            add("positive", "bio_not_positive", value)
        for value in learned.get("bio_pref_negative_signals", []) or []:
            add("negative", "bio_not_negative", value)
        for value in learned.get("descriptor_pref_positive_signals", []) or []:
            _, detail = _classify_text_signal(value, interests, descriptors)
            add("positive", "descriptor_not_positive", detail)
        for value in learned.get("descriptor_pref_negative_signals", []) or []:
            _, detail = _classify_text_signal(value, interests, descriptors)
            add("negative", "descriptor_not_negative", detail)

    def chips(sentiment: str, empty: str) -> str:
        items = list(candidates[sentiment].values())
        if not items:
            return f'<span class="muted compact-note">{_esc(empty)}</span>'
        cls = "positive" if sentiment == "positive" else "negative"
        title = "Clique para salvar que não deveria contar como positivo" if sentiment == "positive" else "Clique para salvar que não deveria contar como negativo"
        parts = []
        for field, value, label in items:
            ban_btn = ""
            if sentiment == "negative" and field == "descriptor_not_negative":
                ban_key = _normalize(value.split(":")[0]) if ":" in value else _normalize(value)
                ban_btn = (
                    f'<button type="button" class="ban-desc-btn" '
                    f'onclick="suppressDescriptor(\'{_esc(ban_key)}\')" '
                    f'title="Nunca mais mostrar este descritor como negativo">🚫 banir</button>'
                )
            parts.append(
                f'<span class="veto-chip-wrap">'
                f'<label class="veto-chip {cls}" title="{_esc(title)}">'
                f'<input type="checkbox" name="{_esc(field)}" value="{_esc(value)}">'
                f'<span class="veto-x">×</span><span>{_esc(label)}</span>'
                f'<span class="veto-state" aria-hidden="true"></span>'
                f"</label>{ban_btn}</span>"
            )
        return "".join(parts)

    if not candidates["positive"] and not candidates["negative"]:
        return ""

    return f"""
      <details class="signal-correction-panel" open>
        <summary>Corrigir sinais de texto que a IA interpretou errado <small>marcado = será salvo</small></summary>
        <div class="signal-correction-grid">
          <div class="signal-correction-box positive">
            <h3>Não era positivo</h3>
            <div class="veto-grid">{chips("positive", "sem positivo textual destacado")}</div>
          </div>
          <div class="signal-correction-box negative">
            <h3>Não era negativo</h3>
            <div class="veto-grid">{chips("negative", "sem negativo textual destacado")}</div>
          </div>
        </div>
      </details>
    """


_SIMILAR_CORE_FEATURES: list[tuple[str, float]] = [
    ("photo_face_similarity", 1.00),
    ("photo_faces_ratio", 0.35),
    ("photo_face_smile_score", 0.30),
    ("photo_woman_confidence", 0.65),
    ("photo_gender_certainty", 0.25),
    ("photo_body_visible", 0.80),
    ("photo_body_full_length", 0.45),
    ("photo_body_upper_length", 0.40),
    ("photo_body_closeup", 0.35),
    ("photo_body_width_ratio", 0.70),
    ("photo_body_signal_quality", 0.65),
    ("photo_pose_torso_visibility", 0.70),
    ("photo_pose_torso_height", 0.35),
    ("photo_pose_body_coverage", 0.50),
    ("photo_pose_upper_body_ratio", 0.25),
    ("photo_pose_leg_ratio", 0.20),
    ("photo_seg_body_coverage", 0.45),
    ("photo_image_brightness", 0.35),
    ("photo_image_contrast", 0.25),
    ("photo_image_sharpness", 0.30),
    ("photo_image_colorfulness", 0.20),
]

_SIMILAR_BODY_KEYS = ("photo_body_visible", "photo_body_width_ratio", "photo_body_signal_quality")
_SIMILAR_POSE_KEYS = ("photo_pose_torso_visibility", "photo_pose_body_coverage", "photo_pose_torso_height")
_SIMILAR_QUALITY_KEYS = ("photo_image_brightness", "photo_image_contrast", "photo_image_sharpness")

_PHOTO_ASPECT_TO_DETAIL = {key: detail for key, _, _, detail in PHOTO_ASPECT_OPTIONS}
_PHOTO_DETAIL_LABELS = {
    "photo_face": "rosto",
    "photo_gender": "gênero visual",
    "photo_body": "corpo",
    "photo_context": "contexto",
    "photo_style": "estilo/pose/qualidade",
}
_PHOTO_DETAIL_KEYS = {
    "photo_face": ("photo_has_face", "photo_face_similarity", "photo_faces_ratio", "photo_face_smile_score"),
    "photo_gender": ("photo_woman_confidence", "photo_gender_certainty"),
    "photo_body": _SIMILAR_BODY_KEYS + ("photo_body_full_length", "photo_body_upper_length", "photo_body_closeup"),
    "photo_context": ("photo_image_brightness", "photo_image_contrast", "photo_image_colorfulness"),
    "photo_style": _SIMILAR_QUALITY_KEYS + ("photo_image_colorfulness",),
}


def _row_float(row: dict, key: str, default: float | None = None) -> float | None:
    try:
        raw = row.get(key)
        if raw in ("", None):
            return default
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _avg_row_values(row: dict, keys: tuple[str, ...]) -> float | None:
    values = [_row_float(row, key) for key in keys]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _embedding_vector(row: dict) -> list[float]:
    pairs: list[tuple[str, float]] = []
    for key, raw in row.items():
        if not str(key).startswith("photo_emb_pc_"):
            continue
        value = _row_float(row, key)
        if value is None:
            continue
        pairs.append((str(key), value))
    pairs.sort(key=lambda item: item[0])
    return [value for _, value in pairs]


def _cosine_similarity01(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 3:
        return None
    a = a[:n]
    b = b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return None
    return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))


def _weighted_feature_similarity(a: dict, b: dict) -> float | None:
    total = 0.0
    score = 0.0
    for key, weight in _SIMILAR_CORE_FEATURES:
        av = _row_float(a, key)
        bv = _row_float(b, key)
        if av is None or bv is None:
            continue
        score += max(0.0, 1.0 - min(1.0, abs(av - bv))) * weight
        total += weight
    if total <= 0:
        return None
    return score / total


def _visual_similarity_score(a: dict, b: dict) -> tuple[float, bool]:
    core = _weighted_feature_similarity(a, b)
    emb = _cosine_similarity01(_embedding_vector(a), _embedding_vector(b))
    if emb is not None and core is not None:
        return emb * 0.58 + core * 0.42, True
    if emb is not None:
        return emb, True
    if core is not None:
        return core, False
    return 0.0, False


def _close_avg(a: dict, b: dict, keys: tuple[str, ...], tolerance: float) -> bool:
    av = _avg_row_values(a, keys)
    bv = _avg_row_values(b, keys)
    return av is not None and bv is not None and abs(av - bv) <= tolerance


def _visual_similarity_notes(source: dict, candidate: dict, used_embedding: bool) -> list[str]:
    notes: list[str] = []
    if used_embedding:
        notes.append("embedding visual próximo")

    face_a = _row_float(source, "photo_face_similarity")
    face_b = _row_float(candidate, "photo_face_similarity")
    if face_a is not None and face_b is not None and abs(face_a - face_b) <= 0.10:
        notes.append("rosto/gosto parecido")

    woman_a = _row_float(source, "photo_woman_confidence")
    woman_b = _row_float(candidate, "photo_woman_confidence")
    if woman_a is not None and woman_b is not None and abs(woman_a - woman_b) <= 0.10:
        notes.append("apresentação visual próxima")

    if _close_avg(source, candidate, _SIMILAR_BODY_KEYS, 0.13):
        notes.append("corpo parecido")
    if _close_avg(source, candidate, _SIMILAR_POSE_KEYS, 0.13):
        notes.append("pose parecida")
    if _close_avg(source, candidate, _SIMILAR_QUALITY_KEYS, 0.12):
        notes.append("qualidade parecida")

    return notes[:4] or ["features visuais próximas"]


def _detail_values(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def _row_descriptor_norms(row: dict) -> set[str]:
    descriptors = _parse_descriptors(row.get("descriptors", ""))
    norms: set[str] = set()
    for key, value in descriptors.items():
        key_n = _normalize(str(key))
        value_n = _normalize(str(value))
        detail_n = _normalize(f"{key}: {value}")
        if key_n:
            norms.add(key_n)
        if value_n:
            norms.add(value_n)
        if detail_n:
            norms.add(detail_n)
    return norms


def _bio_target_matches(row: dict, target: str) -> bool:
    bio = _normalize(str(row.get("bio", "") or ""))
    target_n = _normalize(target)
    if not target_n:
        return False
    if target_n in {"bio vazia", "vazia", "sem bio", "bio em branco"}:
        return len(bio) < 8
    if target_n in bio:
        return True
    tokens = [tok for tok in re.findall(r"[a-z0-9]{3,}", target_n) if tok]
    return bool(tokens) and all(tok in bio for tok in tokens[:4])


def _photo_detail_score(source: dict, candidate: dict, detail: str, aspects: set[str] | None = None) -> float | None:
    detail = detail if detail in _PHOTO_DETAIL_KEYS else ""
    if not detail:
        return None

    if aspects and "black_photo" in aspects:
        cand_brightness = _row_float(candidate, "photo_image_brightness")
        source_brightness = _row_float(source, "photo_image_brightness")
        if cand_brightness is None:
            return None
        if cand_brightness <= 0.14:
            return 1.0
        if source_brightness is not None and source_brightness <= 0.18 and abs(cand_brightness - source_brightness) <= 0.08:
            return 0.80
        return None

    keys = _PHOTO_DETAIL_KEYS.get(detail, ())
    total = 0.0
    score = 0.0
    for key in keys:
        av = _row_float(source, key)
        bv = _row_float(candidate, key)
        if av is None or bv is None:
            continue
        score += max(0.0, 1.0 - min(1.0, abs(av - bv)))
        total += 1.0

    if total <= 0:
        return None

    focused = score / total
    if detail == "photo_face":
        if (_row_float(candidate, "photo_has_face", 0.0) or 0.0) <= 0:
            return None
        emb = _cosine_similarity01(_embedding_vector(source), _embedding_vector(candidate))
        if emb is not None:
            focused = max(focused, emb)
    if detail == "photo_gender":
        woman_a = _row_float(source, "photo_woman_confidence")
        woman_b = _row_float(candidate, "photo_woman_confidence")
        if aspects and "gender_presentation" in aspects and woman_a is not None and woman_a <= 0.65 and woman_b is not None and woman_b > 0.72:
            return None
    if detail == "photo_body":
        body_a = _row_float(source, "photo_body_visible", 0.0) or 0.0
        body_b = _row_float(candidate, "photo_body_visible", 0.0) or 0.0
        if aspects and "body_visibility" not in aspects and max(body_a, body_b) < 0.15:
            return None
        if max(body_a, body_b) >= 0.15 and abs(body_a - body_b) > 0.35:
            focused *= 0.75
    return focused


def _calibration_targets(details: dict | None, final_decision: str) -> list[dict]:
    details = details or {}
    final_like = _decision_label(final_decision) in {"CURTIR", "SUPER LIKE"}
    domains = set(_detail_values(details.get("selected_domains")))
    targets: list[dict] = []

    photo_aspects = _detail_values(
        details.get("photo_positive_details") if final_like else details.get("photo_negative_details")
    )
    if not photo_aspects:
        photo_aspects = _detail_values(details.get("photo_positive_details")) + _detail_values(details.get("photo_negative_details"))
    photo_details = {
        _PHOTO_ASPECT_TO_DETAIL.get(aspect, "")
        for aspect in photo_aspects
        if _PHOTO_ASPECT_TO_DETAIL.get(aspect, "")
    }
    photo_reason = str(details.get("photo_reason") or "").strip()
    if "photo" in domains and photo_reason and photo_reason != "photo_general":
        photo_details.add(photo_reason)
    for detail in sorted(d for d in photo_details if d in _PHOTO_DETAIL_LABELS):
        aspects_for_detail = {a for a in photo_aspects if _PHOTO_ASPECT_TO_DETAIL.get(a) == detail}
        targets.append({
            "type": "photo",
            "detail": detail,
            "aspects": aspects_for_detail,
            "label": _PHOTO_DETAIL_LABELS.get(detail, detail),
        })

    if "interests" in domains or details.get("selected_interests"):
        for interest in _detail_values(details.get("selected_interests")):
            targets.append({"type": "interest", "value": interest, "label": interest})

    if "bio" in domains and str(details.get("bio_detail") or "").strip():
        value = str(details.get("bio_detail") or "").strip()
        targets.append({"type": "bio", "value": value, "label": value[:40]})

    descriptor_values = _detail_values(
        details.get("descriptor_positive_details") if final_like else details.get("descriptor_negative_details")
    )
    if not descriptor_values:
        descriptor_values = _detail_values(details.get("descriptor_positive_details")) + _detail_values(details.get("descriptor_negative_details"))
    if not descriptor_values and ("descriptors" in domains or details.get("descriptor_detail")):
        descriptor_values = _detail_values(details.get("descriptor_detail"))
    for descriptor in descriptor_values:
        targets.append({"type": "descriptor", "value": descriptor, "label": descriptor[:48]})

    # Deduplica mantendo ordem.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for target in targets:
        key = (str(target.get("type")), _normalize(str(target.get("detail") or target.get("value") or target.get("label") or "")))
        if key[1] and key not in seen:
            seen.add(key)
            unique.append(target)
    return unique


def _candidate_target_matches(source: dict, candidate: dict, targets: list[dict]) -> tuple[float, list[str]]:
    if not targets:
        return 0.0, []

    candidate_interests = {_normalize(i) for i in _parse_interests(candidate.get("interests", ""))}
    candidate_descriptors = _row_descriptor_norms(candidate)
    matched_scores: list[float] = []
    notes: list[str] = []

    for target in targets:
        kind = target.get("type")
        label = str(target.get("label") or target.get("value") or target.get("detail") or "").strip()
        if kind == "photo":
            score = _photo_detail_score(source, candidate, str(target.get("detail") or ""), set(target.get("aspects") or []))
            if score is not None and score >= 0.58:
                matched_scores.append(score)
                notes.append(f"mesmo ponto na foto: {label}")
        elif kind == "interest":
            norm = _normalize(str(target.get("value") or ""))
            if norm and norm in candidate_interests:
                matched_scores.append(0.86)
                notes.append(f"mesmo interesse: {label}")
        elif kind == "bio":
            if _bio_target_matches(candidate, str(target.get("value") or "")):
                matched_scores.append(0.80)
                notes.append(f"bio parecida: {label}")
        elif kind == "descriptor":
            norm = _normalize(str(target.get("value") or ""))
            if norm and norm in candidate_descriptors:
                matched_scores.append(0.88)
                notes.append(f"mesmo descritor: {label}")

    if not matched_scores:
        return 0.0, []
    score = sum(matched_scores) / len(matched_scores)
    score = min(1.0, score + max(0, len(matched_scores) - 1) * 0.035)
    return score, notes[:4]


def _blocked_prior_review_keys(rows: list[dict]) -> set[tuple]:
    blocked: set[tuple] = set()
    for row in rows:
        if row.get("review_status", "pending") == "pending":
            continue
        key = _review_dedupe_key_from_row(row)
        if key:
            blocked.add(key)
    return blocked


def _review_row_by_id(review_id: str, rows: list[dict] | None = None) -> dict | None:
    review_id = str(review_id or "").strip()
    if not review_id:
        return None
    rows = rows if rows is not None else load_reviews(None)
    for row in rows:
        if row.get("review_id") == review_id:
            return row
    return None


def _review_confidence(row: dict) -> float:
    try:
        snap = json.loads(row.get("ai_snapshot") or "{}")
        if snap.get("confidence") not in ("", None):
            return max(0.0, min(1.0, float(snap.get("confidence"))))
        prob = float(snap.get("probability", 0.5) or 0.5)
        decision = snap.get("decision") or _decision_label(row.get("original_label", row.get("label", "")))
        return prob if decision in {"CURTIR", "SUPER LIKE", "SUPER_LIKE"} else 1.0 - prob
    except Exception:
        return 0.0


def _decision_disagrees_with_ai(row: dict, final_decision: str) -> bool:
    original = _decision_label(row.get("original_label", row.get("label", "")))
    final = _decision_label(final_decision)
    return original != final


def _should_offer_visual_calibration(
    row: dict | None,
    final_decision: str,
    feedback_details: dict | None = None,
    threshold: float = 0.80,
) -> bool:
    if not row or not _photo_available(row):
        return False
    if _review_confidence(row) < threshold or not _decision_disagrees_with_ai(row, final_decision):
        return False
    return bool(_calibration_targets(feedback_details, final_decision)) if feedback_details is not None else True


def _similar_review_candidates(
    source_row: dict,
    limit: int = 6,
    feedback_details: dict | None = None,
    final_decision: str = "",
) -> list[tuple[float, dict, list[str]]]:
    source_id = source_row.get("review_id", "")
    targets = _calibration_targets(feedback_details, final_decision) if feedback_details is not None else []
    all_rows = load_reviews(None)
    blocked_keys = _blocked_prior_review_keys(all_rows)
    candidates: list[tuple[float, dict, list[str]]] = []
    for row in all_rows:
        if row.get("review_status", "pending") != "pending":
            continue
        if row.get("review_id") == source_id or not _photo_available(row):
            continue
        key = _review_dedupe_key_from_row(row)
        if key and key in blocked_keys:
            continue
        score, used_embedding = _visual_similarity_score(source_row, row)
        notes = _visual_similarity_notes(source_row, row, used_embedding)
        if targets:
            target_score, target_notes = _candidate_target_matches(source_row, row, targets)
            if not target_notes:
                continue
            score = min(1.0, target_score * 0.70 + score * 0.30)
            notes = target_notes + [n for n in notes if n not in target_notes]
        elif score < 0.46:
            continue
        candidates.append((score, row, notes[:4]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:limit]


def _render_similar_panel(
    source_row: dict,
    mode: str = "manual",
    limit: int = 6,
    feedback_details: dict | None = None,
    final_decision: str = "",
) -> str:
    targets = _calibration_targets(feedback_details, final_decision) if feedback_details is not None else []
    candidates = _similar_review_candidates(
        source_row,
        limit=limit,
        feedback_details=feedback_details,
        final_decision=final_decision,
    )
    source_name = str(source_row.get("name") or "este perfil").strip() or "este perfil"
    confidence = _review_confidence(source_row)
    hot = mode == "calibration"

    if candidates:
        items = []
        for score, row, notes in candidates:
            rid = row.get("review_id", "")
            photo = _photo_url(row)
            decision = _decision_label(row.get("original_label", row.get("label", "")))
            dcls = "like" if decision in {"CURTIR", "SUPER LIKE"} else "pass"
            img_html = (
                f'<img src="{_esc(photo)}" alt="" loading="lazy" decoding="async">'
                if photo else
                '<span class="similar-no-photo"></span>'
            )
            items.append(
                f'<button type="button" class="similar-card" data-review-id="{_esc(rid)}">'
                f'{img_html}'
                f'<span class="similar-info">'
                f'<b>{_esc(row.get("name"))} <small>{_esc(row.get("age"))}</small></b>'
                f'<em>{score * 100:.0f}% parecido</em>'
                f'<small>{_esc(" · ".join(notes))}</small>'
                f'</span>'
                f'<span class="similar-decision {dcls}">{_esc(decision)}</span>'
                f'</button>'
            )
        body = f'<div class="similar-list">{"".join(items)}</div>'
    else:
        if targets:
            body = '<p class="similar-empty">Não achei pendentes com esse mesmo ponto marcado agora.</p>'
        else:
            body = '<p class="similar-empty">Não achei pendentes visualmente próximos o suficiente agora.</p>'

    title = "Calibrar erro confiante" if hot else "Perfis visualmente parecidos"
    detail = (
        f"A IA tinha {confidence * 100:.0f}% de certeza em {source_name}. "
        "Filtrei pelos pontos que você marcou para confirmar esse sinal específico."
        if hot else
        f"Pendentes mais próximos de {source_name} por embedding/features visuais."
    )
    hot_cls = " hot" if hot else ""
    return f"""
      <section class="similar-panel{hot_cls}" data-source-review="{_esc(source_row.get("review_id", ""))}">
        <div class="similar-head">
          <div>
            <b>{_esc(title)}</b>
            <span>{_esc(detail)}</span>
          </div>
          <button type="button" class="similar-close" title="fechar">×</button>
        </div>
        {body}
      </section>
    """


def _render_similar_trigger(row: dict) -> str:
    if not _photo_available(row):
        return ""
    confidence = _review_confidence(row)
    strong = confidence >= 0.80
    cls = " strong" if strong else ""
    hint = (
        "bom para auditar erro confiante"
        if strong else
        "compara rosto/corpo/pose/qualidade"
    )
    return f"""
      <div class="similar-trigger{cls}">
        <button type="button" class="btn ghost btn-sm similar-btn" data-review-id="{_esc(row.get("review_id", ""))}">
          ver parecidos
        </button>
        <span>{_esc(hint)}</span>
      </div>
      <div class="similar-slot" id="similar-slot-{_esc(row.get("review_id", ""))}"></div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Renderização dos cards
# ──────────────────────────────────────────────────────────────────────────────

def _render_card(row: dict, prefs: dict) -> str:
    review_id = row.get("review_id", "")
    interests = _parse_interests(row.get("interests", ""))
    descriptors = _parse_descriptors(row.get("descriptors", ""))
    original = _decision_label(row.get("original_label", row.get("label", "")))
    bio = row.get("bio", "") or ""
    distance_label = _distance_text(row)

    _face_url = _photo_url(row)
    _body_url = _body_photo_url(row)
    _name_esc = _esc(row.get("name"))
    _rid = _esc(review_id)
    if not _photo_available(row):
        photo_html = '<div class="photo-missing">foto indisponível<br><small>não foi salva neste perfil</small></div>'
    elif _body_url:
        photo_html = (
            f'<div class="carousel" id="carousel-{_rid}">'
            f'<div class="carousel-slides">'
            f'<img class="cs active" src="{_esc(_face_url)}" alt="Rosto - {_name_esc}" loading="lazy" decoding="async"'
            f' onclick="btLightbox(this.src)"'
            f' onerror="this.replaceWith(Object.assign(document.createElement(\'div\'),{{className:\'photo-missing\',innerHTML:\'foto rosto indisponível\'}}))">'
            f'<img class="cs" src="{_esc(_body_url)}" alt="Corpo - {_name_esc}" loading="lazy" decoding="async"'
            f' onclick="btLightbox(this.src)"'
            f' onerror="this.replaceWith(Object.assign(document.createElement(\'div\'),{{className:\'photo-missing\',innerHTML:\'foto corpo indisponível\'}}))">'
            f'</div>'
            f'<div class="carousel-dots">'
            f'<span class="cdot active" onclick="carouselGo(\'{_rid}\',0)" title="Rosto"></span>'
            f'<span class="cdot" onclick="carouselGo(\'{_rid}\',1)" title="Corpo"></span>'
            f'</div>'
            f'<button class="carousel-arrow left" onclick="carouselStep(\'{_rid}\',-1)">&#8249;</button>'
            f'<button class="carousel-arrow right" onclick="carouselStep(\'{_rid}\',1)">&#8250;</button>'
            f'</div>'
        )
    else:
        photo_html = (
            f'<img src="{_esc(_face_url)}" alt="Foto de {_name_esc}" loading="lazy" decoding="async" onclick="btLightbox(this.src)" '
            f'onerror="this.replaceWith(Object.assign(document.createElement(\'div\'),{{className:\'photo-missing\',innerHTML:\'foto indisponível<br><small>URL expirou ou arquivo foi removido</small>\'}}))">'
        )

    interest_tags = "".join(
        f'<span class="{_interest_class(i, prefs)}">{_esc(i)}</span>'
        for i in interests[:20]
    ) or '<span class="muted">sem interesses</span>'

    _suppressed = prefs.get("desc_neg_suppressed", set())
    descriptor_tags = "".join(
        _descriptor_tag_html(str(k), str(v), _suppressed)
        for k, v in list(descriptors.items())[:14] if k or v
    ) or '<span class="muted">sem descritores</span>'
    descriptor_corrections = _descriptor_correction_rows(descriptors)

    bio_html = _highlight_bio_html(bio, prefs)
    reasoning_html = _render_ai_reasoning(row, prefs)
    signal_panel_html = _render_signal_panel(row, prefs)
    similar_trigger_html = _render_similar_trigger(row)
    signal_corrections_html = _render_signal_corrections(bio, interests, descriptors, prefs, row)
    body_correction_html = _body_correction_html(row)

    decision_cls = "like" if original in {"CURTIR", "SUPER LIKE"} else "pass"
    ai_label = "curtiu" if original in {"CURTIR", "SUPER LIKE"} else "passou"
    review_mode = str(row.get("review_mode") or "").strip().lower()
    origin_text = (
        f"concordância antiga para reavaliar em {_esc(row.get('created_at',''))}"
        if review_mode == "quick_agree_recheck"
        else (
            f"histórico salvo em {_esc(row.get('created_at',''))}"
            if review_mode == "history"
            else f"IA {ai_label} automaticamente em {_esc(row.get('created_at',''))}"
        )
    )

    if descriptors:
        descriptor_options = '<option value="" selected>escolha o descritor</option>\n' + "\n".join(
            f'<option value="{_esc(k)}: {_esc(v)}">{_esc(k)}: {_esc(v)}</option>'
            for k, v in list(descriptors.items())[:20] if k
        )
    else:
        descriptor_options = '<option value="">sem descritores neste perfil</option>'

    if interests:
        interest_choices = "\n".join(
            f'<label class="chip-check"><input type="checkbox" name="interest_detail" value="{_esc(i)}"><span>{_esc(i)}</span></label>'
            for i in interests[:20]
        )
    else:
        interest_choices = '<span class="muted compact-note">sem interesses neste perfil</span>'

    photo_positive_choices = _photo_fine_choices("photo_positive_detail", "positive", True)
    photo_negative_choices = _photo_fine_choices("photo_negative_detail", "negative", False)
    visual_label_options = _visual_label_select()

    metrics = (
        f'<span>mulher <b>{_pct(row, "photo_woman_confidence")}</b></span>'
        f'<span>similar <b>{_pct(row, "photo_face_similarity")}</b></span>'
        f'<span>sorriso <b>{_pct(row, "photo_face_smile_score")}</b></span>'
        f'<span>faces <b>{_pct(row, "photo_faces_ratio")}</b></span>'
    )
    body_inference_html = _body_inference_html(row)
    compact_profile = (
        f'<span>bio <b>{len(bio)}</b> chars</span>'
        f'<span>interesses <b>{len(interests)}</b></span>'
        f'<span>descritores <b>{len(descriptors)}</b></span>'
        + (f'<span>distância <b>{_esc(distance_label)}</b></span>' if distance_label else "")
    )

    _data_decision = "like" if original in {"CURTIR", "SUPER LIKE"} else "pass"
    _data_has_photo = "1" if _photo_available(row) else "0"
    _data_has_body = "1" if _body_photo_path(row) or str(row.get("photo_body_visible","")) not in ("","0","0.0") else "0"
    _data_descriptors = " ".join(_normalize(f"{k} {v}") for k, v in descriptors.items())[:400]
    _data_interests = " ".join(_normalize(i) for i in interests)[:400]

    return f"""
      <article class="card" id="card-{_esc(review_id)}"
        data-decision="{_data_decision}"
        data-has-photo="{_data_has_photo}"
        data-has-body="{_data_has_body}"
        data-descriptors="{_esc(_data_descriptors)}"
        data-interests="{_esc(_data_interests)}">
        <div class="photo">{photo_html}</div>
        <div class="content">
          <div class="topline">
            <div>
              <h2>{_esc(row.get("name"))} <span>{_esc(row.get("age"))}</span><form method="post" action="/add-male-name" style="display:inline;margin-left:8px"><input type="hidden" name="name" value="{_esc(row.get('name',''))}"><button type="submit" class="male-flag-btn" title="Adicionar '{_esc(row.get('name',''))}' à lista de nomes masculinos para ignorar">♂</button></form></h2>
              <p class="muted">{origin_text}{(" · " + _esc(distance_label)) if distance_label else ""}</p>
            </div>
            <div class="decision {decision_cls}">{original}</div>
          </div>

          <div class="metrics">{metrics}</div>
          {body_inference_html}
          <div class="compact-profile">{compact_profile}</div>
          {signal_panel_html}
          {similar_trigger_html}

          <form class="review-form" method="post" action="/apply">
            <input type="hidden" name="review_id" value="{_esc(review_id)}">
            <div class="form-title">Ensinar o que pesou</div>
            {signal_corrections_html}

            <div class="fields">
              <div class="detail-slot visual-label-wrap">
                <span class="field-title">Avaliação visual separada</span>
                <div class="visual-label-grid">
                  <label>Rosto
                    <select name="visual_face_label">{visual_label_options}</select>
                  </label>
                  <label>Corpo
                    <select name="visual_body_label">{visual_label_options}</select>
                  </label>
                  <label>Estilo / qualidade
                    <select name="visual_style_label">{visual_label_options}</select>
                  </label>
                  <label>Foto geral
                    <select name="visual_overall_label">{visual_label_options}</select>
                  </label>
                </div>
              </div>
              <div class="reason-picker">
                <input type="hidden" name="feedback_domain" class="primary-domain-input" value="">
                <span class="field-title">O que pesou <small>marque um ou mais</small></span>
                <div class="reason-chip-grid">
                  <label class="chip-check reason-chip"><input type="checkbox" name="also" value="photo"><span>foto</span></label>
                  <label class="chip-check reason-chip"><input type="checkbox" name="also" value="interests"><span>interesses</span></label>
                  <label class="chip-check reason-chip"><input type="checkbox" name="also" value="bio"><span>bio / descrição</span></label>
                  <label class="chip-check reason-chip"><input type="checkbox" name="also" value="descriptors"><span>descritores</span></label>
                  <label class="chip-check reason-chip"><input type="checkbox" name="also" value="other"><span>outro</span></label>
                </div>
              </div>
              <div class="detail-slot photo-detail-wrap" style="display:none">
                <label>Detalhe da foto
                  <select name="photo_detail" class="photo-detail-select">
                    <option value="photo_general">foto geral / sem certeza</option>
                    <option value="photo_face">rosto / traços faciais</option>
                    <option value="photo_gender">gênero visual / feminilidade</option>
                    <option value="photo_body">corpo / forma física</option>
                    <option value="photo_context">contexto / cenário</option>
                    <option value="photo_style">estilo / pose / qualidade</option>
                  </select>
                </label>
                <div class="fine-group positive">
                  <span class="field-title">O que puxou a favor (foto)</span>
                  <div class="choice-grid photo-aspects-pos">{photo_positive_choices}</div>
                </div>
                <div class="fine-group negative">
                  <span class="field-title">O que puxou contra (foto)</span>
                  <div class="choice-grid photo-aspects-neg">{photo_negative_choices}</div>
                </div>
              </div>
              <div class="detail-slot interests-detail-wrap" style="display:none">
                <span class="field-title">Interesses que pesaram</span>
                <div class="choice-grid">{interest_choices}</div>
              </div>
              <div class="detail-slot bio-detail-wrap" style="display:none">
                <label>Trecho ou palavra da bio
                  <textarea name="bio_detail" rows="2" placeholder="ex: quer filhos, chama para bar, bio vazia"></textarea>
                </label>
              </div>
              <div class="detail-slot desc-detail-wrap" style="display:none">
                <label>Qual descritor pesou
                  <select name="descriptor_detail">
                    {descriptor_options}
                  </select>
                </label>
                <div class="descriptor-corrections">
                  <span class="field-title">Corrigir sinal do descritor</span>
                  {descriptor_corrections}
                </div>
              </div>
              <div class="detail-slot other-detail-wrap" style="display:none">
                <label>Observação rápida
                  <textarea name="feedback_note" rows="2" placeholder="ex: combo geral, algo subjetivo, dúvida"></textarea>
                </label>
              </div>
              <label>Intensidade
                <select name="feedback_intensity" required>
                  <option value="1">1 - pouco</option>
                  <option value="2" selected>2 - médio</option>
                  <option value="3">3 - muito</option>
                </select>
              </label>
            </div>

            {body_correction_html}

            <div class="actions">
              <button class="btn like" name="final_decision" value="CURTIR">✓ Curtir</button>
              <button class="btn super" name="final_decision" value="SUPER_LIKE" title="Salva como curtir forte para treinar futuro super like">★ Super like</button>
              <button class="btn ghost" name="final_decision" value="TALVEZ" title="Salva como talvez: entra nos relatórios e na prioridade, mas não treina o binário">Talvez</button>
              <button class="btn pass" name="final_decision" value="NÃO CURTIR">✗ Passar</button>
              <button class="btn agree" type="submit" formaction="/agree" formnovalidate name="final_decision" value="">Concordar com IA</button>
              <button class="btn ghost" type="submit" formaction="/skip" formnovalidate>Pular</button>
            </div>
          </form>

          <details class="profile-details">
            <summary>Dados do perfil e explicação da IA</summary>
            <div class="profile-details-body">
              {reasoning_html}

              <section>
                <h3>Bio</h3>
                <p class="bio">{bio_html}</p>
              </section>

              <section>
                <h3>Interesses <small class="legend"><span class="mini-tag match">✓ combina</span> <span class="mini-tag neg">✗ negativo</span></small></h3>
                <div class="tags">{interest_tags}</div>
              </section>

              <section>
                <h3>Descritores</h3>
                <div class="tags">{descriptor_tags}</div>
              </section>
            </div>
          </details>
        </div>
      </article>
    """


def _render_reviewed_row(row: dict) -> str:
    original = _decision_label(row.get("original_label", row.get("label", "")))
    final = _decision_label(row.get("final_decision", row.get("label", "")))
    corrected = str(row.get("manual_corrected", "0")).strip() == "1"
    corrected_tag = ' <span class="corrected-badge">corrigido</span>' if corrected else ""
    review_id = row.get("review_id", "")
    return f"""
      <div class="reviewed-row">
        <span class="rname">{_esc(row.get("name"))} <span class="muted">({_esc(row.get("age"))})</span></span>
        <span class="rdecision {('like' if final in {'CURTIR', 'SUPER LIKE'} else 'pass')}">{final}</span>
        {corrected_tag}
        <span class="rdate muted">{_esc(row.get("reviewed_at","")[:16])}</span>
        <form method="post" action="/undo" style="display:inline">
          <input type="hidden" name="review_id" value="{_esc(review_id)}">
          <button class="btn ghost btn-sm">↩ Desfazer</button>
        </form>
      </div>
    """


def _render_photo_deep_groups_html() -> str:
    parts: list[str] = []
    for idx, (title, items) in enumerate(PHOTO_DEEP_TAG_GROUPS):
        rows = []
        for tid, lp, ln in items:
            rows.append(
                f'<div class="deep-row">'
                f'<label class="chip-check positive deep-chip">'
                f'<input type="checkbox" name="deep_pos" value="{_esc(tid)}" class="deep-pos-cb">'
                f"<span>{_esc(lp)}</span></label>"
                f'<label class="chip-check negative deep-chip">'
                f'<input type="checkbox" name="deep_neg" value="{_esc(tid)}" class="deep-neg-cb">'
                f"<span>{_esc(ln)}</span></label>"
                f"</div>"
            )
        open_attr = " open" if idx < 2 else ""
        parts.append(
            f'<details class="deep-group"{open_attr}><summary>{_esc(title)}</summary><div class="deep-rows">{"".join(rows)}</div></details>'
        )
    return "\n".join(parts)


def _render_photo_deep_panel(saved_paths: list[str], selected_path: str = "") -> str:
    groups_html = _render_photo_deep_groups_html()
    n_paths = len(saved_paths)
    if n_paths == 0:
        return f"""
    <div class="deep-intro notice" style="margin-bottom:16px">
      <p><strong>Rotulagem visual — fotos corporais medidas</strong> grava em <code>data/photo_deep_feedback.jsonl</code>.
      O retreino principal <strong>não depende</strong> deste arquivo até você treinar/integrar um modelo visual extra.</p>
      <p class="muted" style="margin:0">Ainda não há imagens corporais <code>_body</code> com medição útil. Rode o swiper para salvar fotos.</p>
    </div>
    <div class="empty"><h2>Nenhuma foto local</h2><p>Quando existirem arquivos JPG/PNG/WebP nas pastas de fotos, elas aparecerão aqui.</p></div>
    """

    counts = photo_review_counts()
    reviewed_paths = {p for p in saved_paths if counts.get(p, 0) > 0}
    reviewed_total = len(reviewed_paths)
    selected_path = selected_path if selected_path in saved_paths else saved_paths[0]

    opts = []
    for i, p in enumerate(saved_paths):
        sel = " selected" if p == selected_path else ""
        label = Path(p).name
        folder = Path(p).parent.name
        opts.append(f'<option value="{_esc(p)}"{sel}>{_esc(folder)} — {_esc(label)}</option>')
    options_html = "\n            ".join(opts)

    thumbs = []
    for p in saved_paths:
        name = Path(p).name
        folder = Path(p).parent.name
        reviewed = counts.get(p, 0)
        cls = " reviewed" if reviewed else " unreviewed"
        cls += " selected" if p == selected_path else ""
        folder_label = "curtida" if folder == "liked" else "passada"
        badge = f'<b>{reviewed}</b>' if reviewed else ""
        thumbs.append(
            f'<button type="button" class="deep-thumb{cls}" data-path="{_esc(p)}" data-folder="{_esc(folder)}" data-reviewed="{1 if reviewed else 0}">'
            f'<img loading="lazy" src="/photo?path={urllib.parse.quote(p)}" alt="">'
            f'<span>{_esc(folder_label)} {badge}</span>'
            f'</button>'
        )
    thumbs_html = "\n".join(thumbs)
    preview_src = f"/photo?path={urllib.parse.quote(selected_path)}" if selected_path else ""
    return f"""
    <div class="deep-intro notice" style="margin-bottom:16px">
      <p><strong>Rotulagem visual — fotos corporais medidas</strong>. Cada envio acrescenta uma linha em
      <code>data/photo_deep_feedback.jsonl</code>. Isso cria um dataset visual separado; o modelo principal só muda quando você decidir integrar esse sinal.</p>
      <p class="muted" style="margin:0">A lista prioriza arquivos <code>_body</code> com sinal de ombro/quadril, segmentação ou largura corporal útil. Use rótulos como “sinal na foto”, não como medida objetiva da pessoa.
      Marque o que fizer sentido; quanto mais consistente, mais útil fica o treino visual.</p>
    </div>
    <div class="deep-toolbar">
      <div class="deep-filterbar" aria-label="Filtros de fotos">
        <button type="button" class="deep-filter active" data-filter="all">todas <b>{n_paths}</b></button>
        <button type="button" class="deep-filter" data-filter="unreviewed">não rotuladas <b>{n_paths - reviewed_total}</b></button>
        <button type="button" class="deep-filter" data-filter="reviewed">rotuladas <b>{reviewed_total}</b></button>
        <button type="button" class="deep-filter" data-filter="liked">curtidas</button>
        <button type="button" class="deep-filter" data-filter="disliked">passadas</button>
      </div>
      <form method="post" action="/photo-deep-train">
        <button class="btn ghost btn-sm">Treinar modelo visual extra</button>
      </form>
    </div>
    <div id="deep-gallery" class="deep-gallery">
      {thumbs_html}
    </div>
    <form id="photo-deep-form" class="deep-form" method="post" action="/photo-deep-save">
      <div class="deep-top">
        <div class="deep-preview-col">
          <label>Foto salva
            <select name="deep_photo_path" id="deep-photo-path" class="deep-photo-select" required>
            {options_html}
            </select>
          </label>
          <div class="deep-preview-wrap">
            <img id="deep-photo-preview" class="deep-preview-img" src="{_esc(preview_src)}" alt="Pré-visualização">
          </div>
        </div>
        <div class="deep-meta-col">
          <fieldset class="deep-fieldset">
            <legend>Impressão geral nesta foto</legend>
            <label class="radio-line"><input type="radio" name="deep_impression" value="like"> Atraente / quero registrar como positiva</label>
            <label class="radio-line"><input type="radio" name="deep_impression" value="neutral"> Neutra / mista</label>
            <label class="radio-line"><input type="radio" name="deep_impression" value="dislike"> Não atraente / negativa pra mim</label>
            <label class="radio-line"><input type="radio" name="deep_impression" value="" checked> Só detalhes (sem rótulo geral)</label>
          </fieldset>
          <label>(Opcional) Alinhamento com o que costumo curtir
            <select name="deep_alignment">
              <option value="">não informar</option>
              <option value="0">0 — pouco alinhada</option>
              <option value="1">1</option>
              <option value="2">2 — média</option>
              <option value="3">3</option>
              <option value="4">4 — muito alinhada</option>
            </select>
          </label>
        </div>
      </div>
      <h3 class="deep-tags-title">Detalhes por eixo (a favor × contra — mesmo tema não nos dois)</h3>
      <div class="deep-groups-wrap">
        {groups_html}
      </div>
      <label class="note-field">Observação livre (opcional)
        <textarea name="deep_note" rows="3" placeholder="Ex.: gostei da luz natural, mas o ângulo não favorece o rosto."></textarea>
      </label>
      <div class="actions">
        <button type="submit" class="btn agree" name="deep_after" value="next">Salvar e próxima não rotulada</button>
        <button type="submit" class="btn ghost" name="deep_after" value="stay">Salvar nesta foto</button>
      </div>
    </form>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Estilos globais da UI
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
    :root {
      --bg: #f6f7f4;
      --ink: #1d1b16;
      --muted: #746f66;
      --card: #ffffff;
      --line: #d8ded3;
      --like: #138a52;
      --pass: #c23b34;
      --agree: #5b7fb5;
      --super: #8b5cf6;
      --accent: #c47c2c;
      --match: #166534;
      --match-bg: #dcfce7;
      --neg: #991b1b;
      --neg-bg: #fee2e2;
      --mid: #92400e;
      --mid-bg: #fef3c7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky; top: 0; z-index: 2;
      backdrop-filter: blur(12px);
      background: rgba(246,247,244,.90);
      border-bottom: 1px solid var(--line);
      padding: 14px clamp(16px, 4vw, 48px);
      display: flex; gap: 16px; align-items: center; justify-content: space-between; flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: clamp(22px, 3.5vw, 34px); letter-spacing: 0; }
    .subtitle { color: var(--muted); margin: 2px 0 0; font-size: 14px; }
    .header-right { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .stats { display: flex; gap: 8px; flex-wrap: wrap; }
    .pill {
      border: 1px solid var(--line); background: rgba(255,255,255,.76);
      border-radius: 999px; padding: 6px 12px; font-size: 13px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    main { width: min(1200px, calc(100% - 28px)); margin: 24px auto 80px; }
    .notice {
      background: #1d1b16; color: #fffaf0;
      border-radius: 8px; padding: 12px 16px; margin-bottom: 18px;
    }
    .notice.ok { background: #14532d; }
    .notice.err { background: #7f1d1d; }
    .notice.retrain {
      display: flex; align-items: center; gap: 9px;
      padding: 10px 12px; font-size: 13px; line-height: 1.35;
      background: #1e3a5f;
    }
    .notice.retrain[hidden] { display: none; }
    .notice.retrain.running { background: #1d4ed8; }
    .notice.retrain.ok { background: #14532d; }
    .notice.retrain.err { background: #7f1d1d; }
    .eval-summary {
      display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
      border: 1px solid var(--line); background: rgba(255,255,255,.78);
      border-radius: 8px; padding: 12px 14px; margin-bottom: 16px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .eval-summary strong { display: block; font-size: 14px; }
    .eval-summary span { color: var(--muted); font-size: 12px; }
    .eval-metrics { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .eval-metrics span {
      border: 1px solid var(--line); border-radius: 999px;
      padding: 5px 9px; background: rgba(246,247,244,.9);
    }
    .eval-metrics b { color: var(--ink); }
    .retrain-dot {
      width: 8px; height: 8px; flex: 0 0 auto;
      border-radius: 999px; background: #bfdbfe;
      animation: retrainPulse 1.05s ease-in-out infinite;
    }
    @keyframes retrainPulse {
      0%, 100% { opacity: .38; transform: scale(.82); }
      50% { opacity: 1; transform: scale(1.18); }
    }
    .toolbar { display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
    .more-pending {
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.7);
      color: var(--muted);
      padding: 12px 14px;
      margin: 14px 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 13px;
      text-align: center;
    }
    .more-pending a { color: var(--agree); font-weight: 800; }
    /* Card */
    .card {
      display: grid; grid-template-columns: minmax(200px, 300px) 1fr;
      gap: 16px; padding: 14px; margin: 16px 0;
      background: var(--card); border: 1px solid var(--line);
      border-radius: 8px; box-shadow: 0 8px 24px rgba(38,48,32,.08);
    }
    .card.focus-flash {
      outline: 3px solid #93c5fd;
      outline-offset: 2px;
    }
    .photo {
      height: clamp(360px, 64vh, 620px);
      max-height: 620px;
      align-self: start;
      background: #eef1ec;
      border-radius: 8px;
      overflow: hidden; display: grid; place-items: center;
    }
    .photo > img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      cursor: zoom-in;
    }
    .photo-missing { color: var(--muted); text-align: center; line-height: 1.5; padding: 20px; font-size: 13px; }
    .carousel { position: relative; width: 100%; height: 100%; }
    .carousel-slides { width: 100%; height: 100%; }
    .cs {
      display: none;
      width: 100%;
      height: 100%;
      object-fit: contain;
      cursor: zoom-in;
    }
    .cs.active { display: block; }
    .carousel-dots { position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); display: flex; gap: 6px; z-index: 2; }
    .cdot { width: 9px; height: 9px; border-radius: 50%; background: rgba(255,255,255,.45); cursor: pointer; border: 1px solid rgba(0,0,0,.2); transition: background .15s; }
    .cdot.active { background: #fff; }
    .carousel-arrow { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,.35); color: #fff; border: none; border-radius: 50%; width: 32px; height: 32px; font-size: 22px; line-height: 1; cursor: pointer; z-index: 2; display: flex; align-items: center; justify-content: center; padding: 0; }
    .carousel-arrow.left { left: 8px; }
    .carousel-arrow.right { right: 8px; }
    .carousel-arrow:hover { background: rgba(0,0,0,.6); }
    .topline { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }
    h2 { margin: 0; font-size: 24px; letter-spacing: 0; }
    h2 span { color: var(--muted); font-weight: 400; }
    h3 { margin: 14px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted);
         display: flex; align-items: center; gap: 10px; }
    .muted { color: var(--muted); }
    .decision {
      font-family: ui-sans-serif, system-ui, sans-serif; font-weight: 800;
      border-radius: 999px; padding: 8px 14px; color: white; white-space: nowrap; font-size: 14px;
    }
    .decision.like, .btn.like { background: var(--like); }
    .decision.pass, .btn.pass { background: var(--pass); }
    .btn.agree { background: var(--agree); }
    .btn.super { background: var(--super); }
    .metrics { display: flex; gap: 6px; flex-wrap: wrap; margin: 12px 0; }
    .metrics span {
      display: inline-block; border: 1px solid var(--line);
      background: rgba(255,255,255,.56); border-radius: 999px;
      padding: 5px 10px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px;
    }
    .filter-bar { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 14px; padding: 12px 14px; background: rgba(255,255,255,.7); border: 1px solid var(--line); border-radius: 10px; align-items: flex-start; }
    .filter-group { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; }
    .filter-group + .filter-group { padding-left: 12px; border-left: 1px solid var(--line); }
    .filter-label { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin-right: 2px; white-space: nowrap; }
    .fchip { font-size: 12px; padding: 4px 10px; border-radius: 99px; border: 1px solid var(--line); background: rgba(255,255,255,.5); cursor: pointer; transition: all .12s; white-space: nowrap; }
    .fchip:hover { border-color: #93c5fd; background: #eff6ff; }
    .fchip.active { background: #1e3a5f; color: #fff; border-color: #1e3a5f; }
    .fchip.desc-chip.active { background: #7c3aed; border-color: #7c3aed; color: #fff; }
    .filter-search { font-size: 12px; padding: 4px 10px; border-radius: 99px; border: 1px solid var(--line); background: rgba(255,255,255,.7); outline: none; width: 160px; }
    .filter-search:focus { border-color: #93c5fd; }
    .filter-count { font-size: 12px; color: var(--muted); margin: -8px 0 10px; padding: 0 2px; }
    .card.filter-hidden { display: none; }
    .skip-all-btn { color: #b45309; border-color: #fcd34d; }
    .skip-all-btn:hover { background: #fef3c7; }
    .veto-chip-wrap { display: inline-flex; align-items: center; gap: 3px; }
    .ban-desc-btn { background: none; border: none; font-size: 11px; color: var(--muted); cursor: pointer; padding: 2px 4px; border-radius: 4px; white-space: nowrap; }
    .ban-desc-btn:hover { background: #fee2e2; color: #991b1b; }
    .male-flag-btn { background: none; border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px; font-size: 11px; color: var(--muted); cursor: pointer; vertical-align: middle; line-height: 1.4; }
    .male-flag-btn:hover { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }
    .body-corr-panel { margin: 4px 0 8px; }
    .body-corr-panel > summary { cursor: pointer; font-size: 12px; color: var(--muted); user-select: none; list-style: none; display: flex; align-items: center; gap: 6px; }
    .body-corr-panel > summary::before { content: "▸"; font-size: 10px; }
    .body-corr-panel[open] > summary::before { content: "▾"; }
    .body-corr-panel > summary:hover { color: var(--fg); }
    .body-corr-panel > summary small { font-size: 11px; opacity: .7; }
    .body-corr-inner { display: flex; flex-direction: column; gap: 8px; padding: 10px 0 4px; }
    .corr-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .corr-label { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); min-width: 130px; }
    .corr-chips { display: flex; gap: 5px; flex-wrap: wrap; }
    .corr-chip { cursor: pointer; }
    .corr-chip input { position: absolute; opacity: 0; width: 0; height: 0; }
    .corr-chip span { display: inline-block; font-size: 12px; padding: 3px 9px; border-radius: 99px; border: 1px solid var(--line); background: rgba(255,255,255,.5); cursor: pointer; transition: all .12s; }
    .corr-chip input:checked + span { background: #1e3a5f; color: #fff; border-color: #1e3a5f; }
    .corr-chip span:hover { border-color: #93c5fd; background: #eff6ff; }
    .body-inference { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin: 2px 0 8px; }
    .body-label { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin-right: 2px; }
    .bchip { font-size: 12px; padding: 3px 9px; border-radius: 99px; }
    .bchip.frame { background: #dbeafe; color: #1e40af; }
    .bchip.build { background: #fef3c7; color: #92400e; }
    .bchip.ratio { background: #f3f4f6; color: #374151; }
    .bchip.quality.sq-ok { background: #dcfce7; color: #166534; }
    .bchip.quality.sq-mid { background: #fef9c3; color: #854d0e; }
    .bchip.quality.sq-low { background: #fee2e2; color: #991b1b; }
    .compact-profile {
      display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 10px;
      color: var(--muted); font-size: 12px;
    }
    .compact-profile span {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8faf7;
      padding: 5px 8px;
    }
    .signal-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8faf7;
      padding: 8px;
      margin: 8px 0 10px;
    }
    .signal-panel.decision-story {
      background: linear-gradient(180deg, #ffffff 0%, #f8faf7 100%);
      border-color: #cbd5c4;
    }
    .signal-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .signal-head > div {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }
    .signal-head span {
      color: var(--ink);
      font-size: 13px;
      font-weight: 900;
    }
    .signal-head small {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
    }
    .signal-head b {
      border-radius: 999px;
      color: white;
      padding: 4px 9px;
      white-space: nowrap;
    }
    .signal-head b.like { background: var(--like); }
    .signal-head b.pass { background: var(--pass); }
    .score-strip,
    .model-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      margin: 5px 0;
    }
    .score-chip,
    .model-chip {
      display: inline-flex;
      align-items: center;
      align-self: flex-start;
      gap: 5px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      width: auto;
      min-height: 24px;
      height: auto;
      padding: 4px 8px;
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 12px;
      line-height: 1.1;
      white-space: nowrap;
    }
    .score-chip b,
    .model-chip b {
      color: var(--ink);
      font-weight: 900;
    }
    .score-chip.confidence { border-color: #bfdbfe; background: #eff6ff; }
    .score-chip.photo { border-color: #bbf7d0; background: #f0fdf4; }
    .score-chip.text { border-color: #fed7aa; background: #fff7ed; }
    .score-chip.body { border-color: #fde68a; background: #fefce8; }
    .score-chip.super { border-color: #ddd6fe; background: #f5f3ff; }
    .visual-breakdown {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      align-items: center;
      margin: 3px 0 6px;
    }
    .visual-score {
      display: inline-grid;
      grid-template-columns: auto 42px 31px;
      align-items: center;
      gap: 5px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      min-height: 22px;
      padding: 3px 6px;
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 11px;
      line-height: 1;
      white-space: nowrap;
    }
    .visual-score strong {
      color: var(--ink);
      font-size: 11px;
      font-weight: 850;
    }
    .visual-score i {
      position: relative;
      display: block;
      width: 42px;
      height: 5px;
      overflow: hidden;
      border-radius: 999px;
      background: #e5e7eb;
    }
    .visual-score i::before {
      content: "";
      display: block;
      width: var(--v);
      height: 100%;
      border-radius: inherit;
      background: #94a3b8;
    }
    .visual-score b {
      color: var(--ink);
      font-size: 11px;
      font-weight: 900;
      text-align: right;
    }
    .visual-score.high { border-color: #bbf7d0; background: #f0fdf4; }
    .visual-score.high i::before { background: #16a34a; }
    .visual-score.mid { border-color: #fde68a; background: #fffdf0; }
    .visual-score.mid i::before { background: #ca8a04; }
    .visual-score.low { border-color: #fecaca; background: #fff1f2; }
    .visual-score.low i::before { background: #dc2626; }
    .similar-trigger {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 4px 0 8px;
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 12px;
    }
    .similar-trigger.strong span {
      color: #854d0e;
      font-weight: 750;
    }
    .similar-btn {
      color: var(--agree) !important;
      background: #eff6ff !important;
      border-color: #bfdbfe !important;
    }
    .similar-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      padding: 9px;
      margin: 8px 0 10px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .similar-panel.hot {
      border-color: #f59e0b;
      background: #fffdf0;
      box-shadow: inset 3px 0 0 #f59e0b;
    }
    .similar-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 8px;
    }
    .similar-head div {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }
    .similar-head b {
      color: var(--ink);
      font-size: 13px;
      font-weight: 900;
    }
    .similar-head span {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
    }
    .similar-close {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      border-radius: 999px;
      width: 22px;
      height: 22px;
      cursor: pointer;
      line-height: 1;
      font-weight: 900;
    }
    .similar-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 7px;
    }
    .similar-card {
      display: grid;
      grid-template-columns: 44px minmax(0, 1fr) auto;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 6px;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
      min-width: 0;
    }
    .similar-card:hover {
      border-color: #93c5fd;
      background: #eff6ff;
    }
    .similar-card img,
    .similar-no-photo {
      width: 44px;
      height: 52px;
      border-radius: 6px;
      object-fit: cover;
      background: #e5e7eb;
      display: block;
    }
    .similar-info {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }
    .similar-info b,
    .similar-info small,
    .similar-info em {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .similar-info b {
      font-size: 12px;
      font-weight: 900;
    }
    .similar-info b small {
      color: var(--muted);
      font-weight: 650;
    }
    .similar-info em {
      color: var(--agree);
      font-size: 11px;
      font-style: normal;
      font-weight: 850;
    }
    .similar-info small {
      color: var(--muted);
      font-size: 11px;
    }
    .similar-decision {
      border-radius: 999px;
      color: #fff;
      font-size: 10px;
      font-weight: 900;
      padding: 3px 6px;
      white-space: nowrap;
    }
    .similar-decision.like { background: var(--like); }
    .similar-decision.pass { background: var(--pass); }
    .similar-empty {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
    }
    .model-strip { margin-top: 4px; }
    .model-chip {
      border-radius: 8px;
      align-items: baseline;
      background: #f8fafc;
      color: var(--ink);
    }
    .model-chip b {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: 10px;
    }
    .model-chip small {
      color: var(--muted);
      border-left: 1px solid var(--line);
      padding-left: 5px;
      font-size: 11px;
    }
    .decision-summary {
      display: grid;
      grid-template-columns: 1.25fr .95fr;
      gap: 8px;
      margin-top: 6px;
    }
    .signal-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .signal-grid.detailed {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 8px;
    }
    .signal-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 7px 8px;
      min-width: 0;
    }
    .signal-box.like { border-color: #86efac; background: #f0fdf4; }
    .signal-box.pass { border-color: #fca5a5; background: #fff1f2; }
    .signal-box.primary.like,
    .signal-box.primary.pass {
      box-shadow: inset 3px 0 0 currentColor;
    }
    .signal-box.primary.like { color: var(--match); }
    .signal-box.primary.pass { color: var(--neg); }
    .signal-box.counter { background: #fffdf8; }
    .signal-box.photo { border-color: #bbf7d0; background: #f8fff9; }
    .signal-box.body { border-color: #fde68a; background: #fffdf0; }
    .signal-box.text { border-color: #fed7aa; background: #fff9f3; }
    .signal-box.model { background: #f8fafc; }
    .signal-box h3 { margin-top: 0; }
    .signal-box ul {
      margin: 0;
      padding-left: 16px;
      font-size: 12px;
      line-height: 1.28;
    }
    .signal-box li + li { margin-top: 3px; }
    .ai-details-drawer {
      margin-top: 6px;
      border-top: 1px solid var(--line);
      padding-top: 5px;
    }
    .ai-details-drawer > summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      user-select: none;
      list-style: none;
    }
    .ai-details-drawer > summary::before {
      content: "▸";
      display: inline-block;
      margin-right: 5px;
      font-size: 10px;
    }
    .ai-details-drawer[open] > summary::before { content: "▾"; }
    .ai-details-drawer[open] > summary { margin-bottom: 6px; }
    .snapshot-missing {
      margin: 0;
      font-size: 12px;
      line-height: 1.35;
    }
    .signal-correction-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffdf8;
      padding: 9px 10px;
      margin: 0 0 10px;
    }
    .signal-correction-panel summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      user-select: none;
    }
    .signal-correction-panel summary small {
      margin-left: 6px;
      color: var(--muted);
      font-weight: 600;
    }
    .signal-correction-panel[open] summary { margin-bottom: 8px; }
    .signal-correction-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .signal-correction-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
      min-width: 0;
    }
    .signal-correction-box.positive { border-color: #86efac; background: #f0fdf4; }
    .signal-correction-box.negative { border-color: #fca5a5; background: #fff1f2; }
    .signal-correction-box h3 { margin-top: 0; }
    .veto-grid { display: flex; flex-wrap: wrap; gap: 6px; }
    .veto-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 8px;
      background: #fffdf8;
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 12px;
      line-height: 1.2;
      cursor: pointer;
    }
    .veto-chip input {
      position: absolute;
      opacity: 0;
      width: 1px;
      height: 1px;
      margin: 0;
      pointer-events: none;
    }
    .veto-chip.positive { border-color: #86efac; background: #f0fdf4; color: var(--match); }
    .veto-chip.negative { border-color: #fca5a5; background: #fff1f2; color: var(--neg); }
    .veto-state {
      margin-left: auto;
      border: 1px solid currentColor;
      border-radius: 999px;
      padding: 1px 6px;
      font-size: 10px;
      font-weight: 800;
      opacity: .72;
      white-space: nowrap;
    }
    .veto-state::before { content: "marcar"; }
    .veto-chip:has(input:checked) {
      outline: 2px solid currentColor;
      outline-offset: 1px;
      font-weight: 800;
    }
    .veto-chip:has(input:checked) .veto-state {
      background: currentColor;
      color: #fff;
      opacity: 1;
    }
    .veto-chip:has(input:checked) .veto-state::before { content: "salvar"; }
    .veto-chip:has(input:checked) .veto-x {
      background: currentColor;
      border-color: currentColor;
      color: #fff;
    }
    .veto-x {
      display: inline-grid;
      place-items: center;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 1px solid currentColor;
      font-weight: 900;
      line-height: 1;
    }
    /* Tags */
    .tag {
      display: inline-block; border: 1px solid var(--line);
      background: rgba(255,255,255,.56); border-radius: 999px;
      padding: 5px 10px; margin: 3px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px;
    }
    .tag-match { background: var(--match-bg); border-color: #86efac; color: var(--match); font-weight: 600; }
    .tag-neg { background: var(--neg-bg); border-color: #fca5a5; color: var(--neg); font-weight: 600; }
    .tag-soft { background: rgba(196,124,44,.10); }
    /* Bio highlights */
    mark.bio-pos { background: var(--match-bg); color: var(--match); border-radius: 4px; padding: 0 3px; }
    mark.bio-neg { background: var(--neg-bg); color: var(--neg); border-radius: 4px; padding: 0 3px; font-weight: 600; }
    .bio { white-space: pre-wrap; line-height: 1.5; }
    /* Reasoning */
    .reasoning { margin: 4px 0 14px; }
    .reason-lines { display: flex; flex-direction: column; gap: 6px; font-size: 13px; font-family: ui-sans-serif, system-ui, sans-serif; }
    .reason-lines > div { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
    .signal-ok { color: var(--match); font-weight: 600; }
    .signal-no { color: var(--neg); font-weight: 600; }
    .signal-mid { color: var(--mid); font-weight: 600; }
    /* Mini tags in reasoning */
    .mini-tag {
      display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 11px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      border: 1px solid currentColor;
    }
    .mini-tag.match { background: var(--match-bg); color: var(--match); }
    .mini-tag.neg { background: var(--neg-bg); color: var(--neg); }
    .legend { font-size: 11px; text-transform: none; letter-spacing: 0; }
    /* Form */
    .review-form {
      margin-top: 10px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }
    .form-title {
      margin: 0 0 10px;
      color: var(--ink);
      font-weight: 800;
      font-size: 14px;
    }
    .fields { display: grid; grid-template-columns: minmax(180px, .9fr) minmax(130px, .45fr); gap: 10px; margin-bottom: 10px; align-items: start; }
    .reason-picker {
      grid-column: 1 / -1;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.62);
      padding: 8px;
    }
    .reason-picker .field-title {
      margin-bottom: 6px;
      font-weight: 700;
      color: var(--ink);
    }
    .reason-picker .field-title small {
      color: var(--muted);
      font-weight: 500;
      margin-left: 6px;
    }
    .reason-chip-grid { display: flex; flex-wrap: wrap; gap: 6px; }
    .reason-chip {
      padding: 5px 8px;
      border-radius: 999px;
      font-size: 12px;
    }
    .reason-chip.is-checked {
      border-color: #86efac;
      background: var(--match-bg);
      color: var(--match);
      font-weight: 700;
    }
    .reason-picker.needs-choice {
      border-color: #fca5a5;
      box-shadow: 0 0 0 2px rgba(248,113,113,.14);
    }
    .detail-slot {
      grid-column: 1 / -1;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #f8faf7;
    }
    .visual-label-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 8px;
    }
    .visual-label-grid label { min-width: 0; }
    .photo-detail-wrap { grid-column: 1 / -1; display: grid; grid-template-columns: minmax(180px, 240px) repeat(2, minmax(180px, 1fr)); gap: 10px; }
    .photo-detail-wrap > label { grid-column: auto; }
    .interests-detail-wrap, .bio-detail-wrap, .desc-detail-wrap, .other-detail-wrap { grid-column: 1 / -1; }
    .field-title {
      display: block; margin-bottom: 6px;
      font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; color: var(--muted);
    }
    .choice-grid { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip-check {
      display: inline-flex; align-items: center; gap: 5px;
      border: 1px solid var(--line); background: #fffdf8;
      border-radius: 8px; padding: 6px 9px; color: var(--ink);
    }
    .chip-check.positive { border-color: #86efac; background: var(--match-bg); color: var(--match); }
    .chip-check.negative { border-color: #fca5a5; background: var(--neg-bg); color: var(--neg); }
    .chip-check input { width: auto; display: inline; margin: 0; }
    .fine-group { min-width: 0; }
    .descriptor-corrections {
      margin-top: 10px;
    }
    .descriptor-choice {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      margin-top: 6px;
    }
    .descriptor-choice.inferred-like { border-color: #86efac; background: #f0fdf4; }
    .descriptor-choice.inferred-pass { border-color: #fca5a5; background: #fff1f2; }
    .descriptor-label {
      color: var(--ink);
      font-size: 12px;
      line-height: 1.3;
    }
    .descriptor-label small {
      display: block;
      color: var(--muted);
      margin-top: 2px;
    }
    .compact-note { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; }
    label { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; color: var(--muted); }
    label > * { display: block; margin-top: 4px; }
    select, input[type="number"], input[type="text"], textarea {
      width: 100%; border: 1px solid var(--line); border-radius: 10px;
      padding: 8px 10px; background: #fffdf8; color: var(--ink); font-size: 13px;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    textarea {
      resize: vertical;
      min-height: 38px;
      line-height: 1.35;
    }
    .optional-feedback,
    .profile-details {
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 8px;
    }
    .optional-feedback summary,
    .profile-details summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      user-select: none;
    }
    .optional-feedback[open] summary,
    .profile-details[open] summary { margin-bottom: 10px; }
    .profile-details-body section { margin-top: 10px; }
    .secondary-reasons { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
    .secondary-label { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; color: var(--muted); margin: 0; }
    .check-label {
      font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; color: var(--ink);
      display: inline-flex; align-items: center; gap: 5px; cursor: pointer;
    }
    .check-label input { width: auto; display: inline; margin: 0; padding: 0; }
    .photo-score-adjust {
      border: 1px solid var(--line);
      background: rgba(255,255,255,.52);
      border-radius: 8px;
      padding: 10px;
      margin: 8px 0 10px;
      max-width: 760px;
    }
    .photo-score-options,
    .photo-score-fields {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: flex-end;
      margin-top: 8px;
    }
    .photo-score-fields label { min-width: 190px; max-width: 240px; }
    .note-field { display: block; margin: 6px 0 10px; max-width: 680px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .btn {
      border: 0; border-radius: 999px; padding: 10px 16px; color: white;
      font-weight: 800; cursor: pointer; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
    }
    .btn:disabled { opacity: .58; cursor: wait; }
    .btn.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
    .btn.primary { background: var(--agree); color: #fff; }
    .btn-sm { padding: 5px 10px; font-size: 12px; font-weight: 600; }
    /* Reviewed section */
    .reviewed-section { margin-top: 40px; }
    .reviewed-section summary {
      cursor: pointer; font-size: 14px; font-family: ui-sans-serif, system-ui, sans-serif;
      color: var(--muted); padding: 10px 0; user-select: none;
    }
    .reviewed-row {
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      padding: 8px 14px; border: 1px solid var(--line); border-radius: 12px;
      background: rgba(255,250,240,.55); margin: 6px 0;
      font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
    }
    .rname { font-weight: 600; }
    .rdecision { font-weight: 800; padding: 2px 10px; border-radius: 999px; color: white; font-size: 12px; }
    .rdecision.like { background: var(--like); }
    .rdecision.pass { background: var(--pass); }
    .corrected-badge {
      background: #fef3c7; color: var(--mid); border: 1px solid #fde68a;
      border-radius: 999px; padding: 2px 8px; font-size: 11px;
    }
    .rdate { font-size: 11px; flex: 1; }
    /* Empty */
    .empty {
      text-align: center; padding: 60px 20px;
      border: 1px dashed var(--line); border-radius: 8px;
      background: rgba(255,255,255,.72);
    }
    /* Settings page */
    .settings-section { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 20px; margin: 18px 0; }
    .settings-section h2 { margin: 0 0 16px; font-size: 20px; }
    .list-editor { display: flex; flex-direction: column; gap: 6px; }
    .list-item {
      display: flex; align-items: center; gap: 8px; padding: 6px 10px;
      background: rgba(255,255,255,.55); border: 1px solid var(--line);
      border-radius: 8px; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
    }
    .list-item span { flex: 1; }
    .list-add { display: flex; gap: 8px; margin-top: 8px; }
    .list-add input { flex: 1; }
    .review-tabs {
      display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px;
      border-bottom: 1px solid var(--line); padding-bottom: 12px;
    }
    .tab-btn {
      border: 1px solid var(--line);
      background: rgba(255,250,240,.72);
      border-radius: 999px;
      padding: 8px 16px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 13px;
      cursor: pointer;
      color: var(--ink);
    }
    .tab-btn.active {
      background: var(--ink);
      color: #fffaf0;
      border-color: var(--ink);
    }
    .tab-panel[hidden] { display: none !important; }
    .signal-train-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 12px;
    }
    .signal-train-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }
    .signal-train-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
    }
    .signal-train-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--card);
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-width: 0;
    }
    .signal-train-main h2 {
      margin: 6px 0 8px;
      font-size: 18px;
      line-height: 1.18;
      overflow-wrap: anywhere;
    }
    .signal-kind {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 800;
      color: var(--muted);
      background: #fffdf8;
    }
    .signal-train-meta,
    .signal-examples {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    .signal-train-meta span,
    .signal-example {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      background: rgba(255,255,255,.72);
    }
    .signal-train-meta .like { color: var(--match); border-color: #86efac; background: #f0fdf4; }
    .signal-train-meta .pass { color: var(--neg); border-color: #fca5a5; background: #fff1f2; }
    .signal-train-meta .mid { color: var(--mid); border-color: #fde68a; background: #fffdf0; }
    .signal-train-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      align-items: center;
      margin-top: auto;
    }
	    .signal-batch-actions {
	      display: flex;
	      flex-wrap: wrap;
	      gap: 8px;
	      align-items: center;
	    }
    .signal-explicit {
      flex: 1 1 100%;
      color: var(--muted);
      font-size: 12px;
    }
	    .signal-train-card.queued {
	      border-color: var(--agree);
	      box-shadow: 0 0 0 2px rgba(91,127,181,.16);
	    }
	    .signal-train-card.saving {
	      border-color: var(--agree);
	      opacity: .68;
	      pointer-events: none;
	      box-shadow: 0 0 0 2px rgba(91,127,181,.14);
	    }
	    .signal-train-card.save-error {
	      border-color: var(--neg);
	      box-shadow: 0 0 0 2px rgba(185,74,72,.14);
	    }
	    .signal-choice:disabled {
	      cursor: wait;
	      opacity: .72;
	    }
	    .signal-choice.selected {
	      outline: 3px solid rgba(29,27,22,.18);
	      outline-offset: 2px;
	    }
    .deep-intro code { font-size: 12px; font-family: ui-monospace, monospace; }
    .deep-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .deep-filterbar {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .deep-filter {
      border: 1px solid var(--line);
      background: rgba(255,250,240,.72);
      border-radius: 999px;
      padding: 7px 11px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 12px;
      color: var(--ink);
      cursor: pointer;
    }
    .deep-filter.active {
      background: var(--ink);
      color: #fffaf0;
      border-color: var(--ink);
    }
    .deep-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(92px, 1fr));
      gap: 8px;
      max-height: 360px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px;
      margin-bottom: 16px;
      background: rgba(255,255,255,.35);
    }
    .deep-thumb {
      border: 2px solid transparent;
      border-radius: 12px;
      padding: 0;
      overflow: hidden;
      background: #fffdf8;
      cursor: pointer;
      min-height: 118px;
      display: flex;
      flex-direction: column;
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 11px;
    }
    .deep-thumb img {
      width: 100%;
      aspect-ratio: 3 / 4;
      object-fit: cover;
      background: #ded3bf;
      display: block;
    }
    .deep-thumb span {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 4px;
      padding: 5px 6px;
    }
    .deep-thumb b {
      display: inline-grid;
      place-items: center;
      min-width: 18px;
      height: 18px;
      border-radius: 999px;
      background: var(--agree);
      color: white;
      font-size: 10px;
    }
    .deep-thumb.reviewed { opacity: .72; }
    .deep-thumb.selected {
      border-color: var(--agree);
      box-shadow: 0 0 0 2px rgba(91,127,181,.18);
      opacity: 1;
    }
    .deep-thumb.hidden-filter { display: none; }
    .deep-top {
      display: grid;
      grid-template-columns: minmax(200px, 320px) 1fr;
      gap: 20px;
      align-items: start;
      margin-bottom: 20px;
    }
    .deep-preview-wrap {
      margin-top: 10px;
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #ded3bf;
      min-height: 260px;
      display: grid;
      place-items: center;
    }
    .deep-preview-img {
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      display: block;
    }
    .deep-fieldset {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      margin: 0 0 12px;
      background: rgba(255,255,255,.38);
    }
    .deep-fieldset legend {
      padding: 0 6px;
      font-size: 12px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      color: var(--muted);
    }
    .radio-line {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 13px;
      color: var(--ink);
    }
    .radio-line input { width: auto; margin: 0; }
    .deep-tags-title {
      margin: 18px 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .1em;
      color: var(--muted);
    }
    .deep-groups-wrap {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .deep-group {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255,250,240,.65);
    }
    .deep-group summary {
      margin: 0 0 10px;
      font-size: 15px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
      user-select: none;
    }
    .deep-rows { display: flex; flex-direction: column; gap: 8px; }
    .deep-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      align-items: stretch;
    }
    .deep-chip span { font-size: 11px; line-height: 1.25; }
      @media (max-width: 820px) {
      header { align-items: flex-start; flex-direction: column; }
      .card { grid-template-columns: 1fr; }
      .photo { height: clamp(320px, 70vh, 620px); }
      .fields { grid-template-columns: 1fr; }
      .visual-label-grid { grid-template-columns: 1fr 1fr; }
      .photo-detail-wrap { grid-column: 1 / -1; grid-template-columns: 1fr; }
      .decision-summary { grid-template-columns: 1fr; }
      .signal-grid { grid-template-columns: 1fr; }
      .signal-grid.detailed { grid-template-columns: 1fr; }
      .signal-correction-grid { grid-template-columns: 1fr; }
      .descriptor-choice { grid-template-columns: 1fr; }
      .deep-top { grid-template-columns: 1fr; }
      .deep-row { grid-template-columns: 1fr; }
      .bt-grid { grid-template-columns: 1fr; }
    }
    /* ── Body Training Tab ── */
    .bt-header { display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:8px; margin-bottom:16px; font-family:ui-sans-serif,system-ui,sans-serif; font-size:13px; color:var(--muted); }
    .bt-empty { text-align:center; padding:48px 24px; color:var(--muted); font-family:ui-sans-serif,system-ui,sans-serif; }
    .bt-empty p { margin:0 0 8px; }
    .bt-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:12px; margin-bottom:24px; }
    .bt-card { background:rgba(255,250,240,.85); border:1px solid var(--line); border-radius:12px; overflow:hidden; display:flex; flex-direction:column; }
    .bt-photo-wrap { aspect-ratio:3/5; overflow:hidden; background:#e8e0d0; flex-shrink:0; }
    .bt-photo-wrap img { width:100%; height:100%; object-fit:cover; display:block; }
    .bt-no-photo { width:100%; height:100%; min-height:80px; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:12px; font-family:ui-sans-serif,system-ui,sans-serif; }
    .bt-meta { padding:8px 10px 3px; display:flex; justify-content:space-between; align-items:center; }
    .bt-name { font-size:13px; font-weight:600; color:var(--ink); font-family:ui-sans-serif,system-ui,sans-serif; }
    .bt-chip-c { font-size:10px; padding:2px 6px; border-radius:999px; font-weight:600; font-family:ui-sans-serif,system-ui,sans-serif; background:#d4edda; color:#155724; }
    .bt-chip-p { font-size:10px; padding:2px 6px; border-radius:999px; font-weight:600; font-family:ui-sans-serif,system-ui,sans-serif; background:#f8d7da; color:#721c24; }
    .bt-ai { padding:2px 10px 8px; font-size:11px; color:var(--muted); font-family:ui-sans-serif,system-ui,sans-serif; line-height:1.4; min-height:24px; }
    .bt-form { padding:0 8px 10px; margin-top:auto; }
    .bt-btns { display:grid; grid-template-columns:repeat(5,1fr); gap:4px; }
    .bt-btns button { font-size:10px; padding:5px 3px; border-radius:6px; cursor:pointer; border:1px solid var(--line); background:rgba(255,250,240,.9); color:var(--ink); font-family:ui-sans-serif,system-ui,sans-serif; white-space:nowrap; }
    .bt-e:hover,.bt-e:focus   { background:#cfe2ff; border-color:#9ec5fe; }
    .bt-me:hover,.bt-me:focus { background:#dbeafe; border-color:#93c5fd; }
    .bt-m:hover,.bt-m:focus   { background:#d1e7dd; border-color:#a3cfbb; }
    .bt-ma:hover,.bt-ma:focus { background:#fef9c3; border-color:#fde047; }
    .bt-a:hover,.bt-a:focus   { background:#fff3cd; border-color:#ffc107; }
    .bt-nobody { grid-column:span 5; font-size:10px !important; color:#856404 !important; padding:4px !important; background:#fff3cd !important; border-color:#ffc107 !important; border-radius:5px !important; }
    .bt-nobody:hover { background:#ffe69c !important; border-color:#fd7e14 !important; }
    .bt-skip { grid-column:span 5; font-size:10px !important; color:var(--muted) !important; padding:3px !important; background:transparent !important; border-color:transparent !important; }
    .bt-skip:hover { color:var(--ink) !important; background:rgba(0,0,0,.04) !important; border-color:var(--line) !important; }
    .bt-pagination { display:flex; justify-content:center; gap:16px; padding:8px 0 24px; font-family:ui-sans-serif,system-ui,sans-serif; }
    .bt-photo-wrap img { cursor:zoom-in; }
    #bt-lightbox { display:none; position:fixed; inset:0; background:rgba(0,0,0,.82); z-index:9999; align-items:center; justify-content:center; cursor:zoom-out; }
    #bt-lightbox.open { display:flex; }
    #bt-lightbox img { width:min(90vw,700px); height:min(85vh,700px); object-fit:contain; border-radius:10px; box-shadow:0 8px 40px rgba(0,0,0,.6); cursor:default; }
    #bt-lightbox-close { position:absolute; top:16px; right:20px; font-size:28px; color:#fff; cursor:pointer; line-height:1; background:none; border:none; opacity:.8; }
    #bt-lightbox-close:hover { opacity:1; }
"""


# ──────────────────────────────────────────────────────────────────────────────
# Body Training Tab
# ──────────────────────────────────────────────────────────────────────────────

import csv as _csv_module

_BT_PHOTO_INDEX: dict | None = None


def _bt_normalize_name(text: str) -> str:
    import unicodedata as _ud
    base = _ud.normalize("NFD", (text or "").strip().lower())
    base = "".join(ch for ch in base if _ud.category(ch) != "Mn")
    base = re.sub(r"\s+", " ", base.replace("_", " ")).strip()
    return base


def _bt_build_photo_index() -> dict:
    # Chave: (nome_normalizado, idade) sem label — a foto fica em liked/ ou
    # disliked/ de acordo com a decisão ORIGINAL da IA, que pode divergir do
    # label corrigido pelo usuário no profiles.csv.
    index: dict = {}
    _pat = re.compile(r"^[^_]+_(.+)_(\d+)(?:_(face|body))?\.jpg$", re.IGNORECASE)
    for label_dir in ("liked", "disliked"):
        d = ROOT_DIR / "data" / "photos" / label_dir
        if not d.exists():
            continue
        for p in d.glob("*.jpg"):
            m = _pat.match(p.name)
            if not m:
                continue
            key = (_bt_normalize_name(m.group(1)), int(m.group(2)))
            entry = index.setdefault(key, {"face": "", "body": ""})
            rel = f"data/photos/{label_dir}/{p.name}"
            suffix = (m.group(3) or "").lower()
            if suffix == "body":
                entry["body"] = rel
            elif suffix != "face" and not entry["face"]:
                entry["face"] = rel
    return index


def _bt_photo_index() -> dict:
    global _BT_PHOTO_INDEX
    if _BT_PHOTO_INDEX is None:
        _BT_PHOTO_INDEX = _bt_build_photo_index()
    return _BT_PHOTO_INDEX


def _bt_clear_photo_index() -> None:
    global _BT_PHOTO_INDEX
    _BT_PHOTO_INDEX = None


def _as_float_safe(v, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "", "nan") else default
    except Exception:
        return default


def _bt_photo_url(row: dict) -> str:
    rel = _bt_photo_rel(row)
    return ("/photo?path=" + urllib.parse.quote(rel)) if rel else ""


def _bt_photo_rel(row: dict, allow_main_fallback: bool = True) -> str:
    try:
        name = row.get("name") or ""
        age = int(float(row.get("age", 0) or 0))
    except Exception:
        return ""
    entry = _bt_photo_index().get((_bt_normalize_name(name), age), {})
    rel = entry.get("body") or ""
    if not rel and allow_main_fallback:
        rel = entry.get("face") or ""
    return rel


def _bt_profile_key(row: dict) -> tuple:
    name = _bt_normalize_name(row.get("name") or "")
    age = str(row.get("age") or "").strip()
    return (name, age)


def _body_measurement_strength(row: dict) -> float:
    return body_measurement_strength(row)


def _has_body_measurement(row: dict) -> bool:
    return _body_measurement_strength(row) >= BODY_MEASUREMENT_MIN_STRENGTH


def _body_train_profiles(page: int = 0, per_page: int = 24) -> tuple[list[dict], int]:
    """Perfis com medição corporal útil e sem correção de silhueta salva."""
    from model import PROFILES_PATH
    if not PROFILES_PATH.exists():
        return [], 0
    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        rows = list(_csv_module.DictReader(f))
    by_profile: dict[tuple, dict] = {}
    for row in rows:
        if str(row.get("source", "real")).strip().lower() != "real":
            continue
        strength = _body_measurement_strength(row)
        if strength < 0.45:
            continue
        try:
            details = json.loads(row.get("feedback_details") or "{}")
        except Exception:
            details = {}
        if details.get("body_build_correction"):
            continue
        rel = _bt_photo_rel(row, allow_main_fallback=False)
        if not rel:
            continue
        row["_bt_body_strength"] = strength
        row["_bt_photo_rel"] = rel
        key = _bt_profile_key(row)
        current = by_profile.get(key)
        if current is None:
            by_profile[key] = row
            continue
        current_has_body = 1 if str(current.get("_bt_photo_rel", "")).lower().endswith("_body.jpg") else 0
        row_has_body = 1 if str(rel).lower().endswith("_body.jpg") else 0
        current_score = (current_has_body, _as_float_safe(current.get("_bt_body_strength")), current.get("photo_features_saved", ""))
        row_score = (row_has_body, strength, row.get("photo_features_saved", ""))
        if row_score > current_score:
            by_profile[key] = row

    result = list(by_profile.values())
    # Pose/segmentação primeiro (sinal mais preciso), depois likes antes de passes.
    result.sort(key=lambda r: (
        -_as_float_safe(r.get("_bt_body_strength")),
        0 if _as_float_safe(r.get("photo_pose_torso_visibility")) >= 0.2 else 1,
        0 if int(r.get("label", 0) or 0) else 1,
    ))
    total = len(result)
    return result[page * per_page: (page + 1) * per_page], total


def _render_bt_card(row: dict, page: int = 0) -> str:
    name = _esc(row.get("name") or "?")
    age = row.get("age", "?")
    label = int(row.get("label", 0) or 0)
    chip = ('<span class="bt-chip-c">curtiu</span>' if label
            else '<span class="bt-chip-p">passou</span>')
    photo_rel = row.get("_bt_photo_rel") or _bt_photo_rel(row, allow_main_fallback=False)
    photo_url = ("/photo?path=" + urllib.parse.quote(photo_rel)) if photo_rel else ""
    img_html = (
        f'<img src="{photo_url}" loading="lazy" alt="" onclick="btLightbox(this.src)">'
        if photo_url else
        '<div class="bt-no-photo">sem foto</div>'
    )
    ai_parts = []
    if _as_float_safe(row.get("photo_body_width_bucket_narrow")) > 0:
        ai_parts.append("Canny: estreita")
    elif _as_float_safe(row.get("photo_body_width_bucket_medium")) > 0:
        ai_parts.append("Canny: média")
    elif _as_float_safe(row.get("photo_body_width_bucket_wide")) > 0:
        ai_parts.append("Canny: ampla")
    pose_vis = _as_float_safe(row.get("photo_pose_torso_visibility"))
    if pose_vis > 0.2:
        sw = _as_float_safe(row.get("photo_pose_shoulder_width"))
        hw = _as_float_safe(row.get("photo_pose_hip_width"))
        ai_parts.append(f"pose: ombro {sw:.2f} · quadril {hw:.2f}")
    seg_cov = _as_float_safe(row.get("photo_seg_body_coverage"))
    if seg_cov >= 0.08:
        ss = _as_float_safe(row.get("photo_seg_shoulder_width"))
        sh = _as_float_safe(row.get("photo_seg_hip_width"))
        ai_parts.append(f"segmentação: ombro {ss:.2f} · quadril {sh:.2f}")
    strength = _as_float_safe(row.get("_bt_body_strength"), _body_measurement_strength(row))
    if strength:
        ai_parts.append(f"sinal {strength * 100:.0f}%")
    ai_info = " · ".join(ai_parts) if ai_parts else "sinal corporal visível"
    h_name = _esc(row.get("name") or "")
    h_age = _esc(str(row.get("age") or ""))
    bt_id = _esc(f"{row.get('name','').strip()}|{row.get('age','').strip()}")
    return (
        f'<div class="bt-card" data-bt-id="{bt_id}">'
        f'<div class="bt-photo-wrap">{img_html}</div>'
        f'<div class="bt-meta"><span class="bt-name">{name}, {age}</span>{chip}</div>'
        f'<div class="bt-ai">{ai_info}</div>'
        f'<form method="post" action="/body-train-save" class="bt-form">'
        f'<input type="hidden" name="bt_name" value="{h_name}">'
        f'<input type="hidden" name="bt_age" value="{h_age}">'
        f'<input type="hidden" name="bt_page" value="{page}">'
        f'<input type="hidden" name="ajax" value="1">'
        f'<div class="bt-btns">'
        f'<button type="button" class="bt-e"  onclick="btCardClick(this,\'estreita\')">Estreita</button>'
        f'<button type="button" class="bt-me" onclick="btCardClick(this,\'media_estreita\')">Média-Estreita</button>'
        f'<button type="button" class="bt-m"  onclick="btCardClick(this,\'media\')">Média</button>'
        f'<button type="button" class="bt-ma" onclick="btCardClick(this,\'media_ampla\')">Média-Ampla</button>'
        f'<button type="button" class="bt-a"  onclick="btCardClick(this,\'ampla\')">Ampla</button>'
        f'<button type="button" class="bt-nobody" onclick="btCardClick(this,\'no_body\')" title="A ML detectou sinal corporal, mas não há corpo visível nesta foto — corrige os dados de treino">⚠ Sem corpo visível</button>'
        f'<button type="button" class="bt-skip" onclick="btCardClick(this,\'skip\')">pular (não sei)</button>'
        f'</div></form></div>'
    )


def _render_body_train_panel(page: int = 0) -> str:
    per_page = 24
    profiles, total = _body_train_profiles(page=page, per_page=per_page)
    if total == 0:
        return (
            '<div class="bt-empty">'
            '<p>Nenhum perfil com sinal corporal pendente de correção.</p>'
            '<p style="font-size:13px">Continue usando o swiper — novos perfis com fotos corporais aparecerão aqui.</p>'
            '</div>'
        )
    cards = "".join(_render_bt_card(r, page) for r in profiles)
    start = page * per_page + 1
    end = min(start + len(profiles) - 1, total)
    prev_btn = (
        f'<a href="/?tab=body-train&bt_page={page - 1}" class="btn ghost">← Anterior</a>'
        if page > 0 else ""
    )
    next_btn = (
        f'<a href="/?tab=body-train&bt_page={page + 1}" class="btn ghost">Próximos →</a>'
        if end < total else ""
    )
    retrain_btn = (
        '<form method="post" action="/retrain" style="display:inline">'
        '<input type="hidden" name="redirect_tab" value="body-train">'
        '<button class="btn primary" style="font-size:12px;padding:4px 12px" '
        'title="Aplica todas as correções corporais salvas e retreina o modelo">↻ Retreinar agora</button>'
        '</form>'
    )
    return (
        f'<div class="bt-header">'
        f'<span>{total} perfis pendentes · mostrando {start}–{end}</span>'
        f'<span style="display:flex;align-items:center;gap:12px">'
        f'<span>Clique na silhueta que descreve o corpo visível na foto</span>'
        f'{retrain_btn}'
        f'</span>'
        f'</div>'
        f'<div class="bt-grid">{cards}</div>'
        f'<div class="bt-pagination">{prev_btn}{next_btn}</div>'
    )


_SIGNAL_TYPE_LABELS = {
    "all": "todos",
    "interest": "interesses",
    "bio": "bio",
    "descriptor": "descritores",
}

_SIGNAL_POLARITY_LABELS = {
    "positive": "positivo",
    "neutral": "neutro",
    "negative": "negativo",
}


def _signal_type_label(signal_type: str) -> str:
    return _SIGNAL_TYPE_LABELS.get(str(signal_type or "").strip().lower(), signal_type or "")


def _signal_train_stats() -> dict:
    counts = signal_feedback_counts()
    by_type = {"interest": 0, "bio": 0, "descriptor": 0}
    total = 0
    for (kind, _norm), item in counts.items():
        value = int(item.get("total", 0) or 0)
        total += value
        if kind in by_type:
            by_type[kind] += value
    return {"total": total, **by_type}


def _render_signal_examples(examples: list[dict]) -> str:
    if not examples:
        return '<span class="muted">sem exemplos</span>'
    bits = []
    for ex in examples[:3]:
        name = str(ex.get("name") or "").strip() or "perfil"
        age = str(ex.get("age") or "").strip()
        label = "curtiu" if str(ex.get("label")) == "1" else "passou"
        shown = f"{name}, {age}" if age else name
        bits.append(f'<span class="signal-example">{_esc(shown)} · {_esc(label)}</span>')
    return "".join(bits)


def _signal_train_key(item: dict) -> str:
    kind = str(item.get("signal_type") or "").strip().lower()
    value = str(item.get("signal_value") or "").strip()
    signal_norm = str(item.get("signal_norm") or value)
    return f"{kind}:{signal_norm}"


def _parse_signal_shown_keys(values: list[str]) -> set[str]:
    shown: set[str] = set()
    for raw in values:
        raw_value = str(raw or "")
        if not raw_value:
            continue
        # Compatibilidade com a versao anterior do JS, que juntava ids com
        # virgula. Alguns ids tambem tem virgula, como "Ingles, Portugues".
        parts = re.split(r",(?=(?:interest|bio|descriptor):)", raw_value)
        shown.update(part.strip() for part in parts if part.strip())
    return shown


def _render_signal_train_card(item: dict, signal_filter: str, page: int) -> str:
    kind = str(item.get("signal_type") or "").strip().lower()
    value = str(item.get("signal_value") or "").strip()
    signal_key = _signal_train_key(item)
    score = float(item.get("learned_score", 0.5) or 0.5)
    score_pct = int(round(score * 100))
    explicit = item.get("explicit_feedback") or {}
    explicit_total = int(explicit.get("total", 0) or 0)
    pos = int(explicit.get("positive", 0) or 0)
    neu = int(explicit.get("neutral", 0) or 0)
    neg = int(explicit.get("negative", 0) or 0)
    occurrences = int(item.get("occurrences", 0) or 0)
    likes = int(item.get("likes", 0) or 0)
    dislikes = int(item.get("dislikes", 0) or 0)
    examples = _render_signal_examples(item.get("examples") or [])
    score_cls = "like" if score >= 0.62 else "pass" if score <= 0.38 else "mid"
    return f"""
    <article class="signal-train-card" data-signal-key="{_esc(signal_key)}">
      <div class="signal-train-main">
        <span class="signal-kind">{_esc(_signal_type_label(kind))}</span>
        <h2>{_esc(value)}</h2>
        <div class="signal-train-meta">
          <span>{occurrences} ocorrência(s)</span>
          <span>{likes} curtida(s)</span>
          <span>{dislikes} passada(s)</span>
          <span class="{score_cls}">score {score_pct}%</span>
        </div>
        <div class="signal-examples">{examples}</div>
      </div>
      <div class="signal-train-actions"
           data-signal-type="{_esc(kind)}"
           data-signal-value="{_esc(value)}"
           data-occurrences="{occurrences}"
           data-likes="{likes}"
           data-dislikes="{dislikes}">
        <span class="signal-explicit">marcado: {explicit_total} · +{pos} · ={neu} · -{neg}</span>
        <button type="button" class="btn agree signal-choice" data-polarity="positive">Gosto</button>
        <button type="button" class="btn ghost signal-choice" data-polarity="neutral">Neutro</button>
        <button type="button" class="btn pass signal-choice" data-polarity="negative">Não gosto</button>
      </div>
    </article>
    """


def _render_signal_train_panel(signal_filter: str = "all", page: int = 0) -> str:
    signal_filter = str(signal_filter or "all").strip().lower()
    if signal_filter not in {"all", "interest", "bio", "descriptor"}:
        signal_filter = "all"
    page = max(0, int(page or 0))
    per_page = 24
    candidates = signal_training_candidates(signal_filter, limit=600)
    total = len(candidates)
    start_idx = page * per_page
    visible = candidates[start_idx:start_idx + per_page]
    end = start_idx + len(visible)
    stats = _signal_train_stats()

    filter_links = []
    for key in ("all", "interest", "bio", "descriptor"):
        active = " active" if key == signal_filter else ""
        href = "/?tab=signal-train" if key == "all" else f"/?tab=signal-train&signal_type={key}"
        count = stats.get(key, stats.get("total", 0)) if key != "all" else stats.get("total", 0)
        filter_links.append(
            f'<a class="fchip{active}" href="{href}">{_esc(_signal_type_label(key))} <b>{count}</b></a>'
        )

    if not visible:
        cards = """
        <div class="empty">
          <h2>Nenhum sinal pendente</h2>
          <p>Os sinais deste tipo já têm feedback suficiente ou ainda aparecem pouco no histórico.</p>
        </div>"""
    else:
        cards = "".join(_render_signal_train_card(item, signal_filter, page) for item in visible)

    prev_btn = (
        f'<a href="/?tab=signal-train&signal_type={signal_filter}&signal_page={page - 1}" class="btn ghost">← Anterior</a>'
        if page > 0 else ""
    )
    next_btn = (
        f'<a href="/?tab=signal-train&signal_type={signal_filter}&signal_page={page + 1}" class="btn ghost">Próximos →</a>'
        if end < total else ""
    )
    shown = f"{start_idx + 1}-{end}" if visible else "0"
    return f"""
    <div class="signal-train-head">
      <div class="filter-group">
        <span class="filter-label">Sinais</span>
        {"".join(filter_links)}
      </div>
      <div class="signal-batch-actions">
        <form method="post" action="/retrain" style="display:inline">
          <input type="hidden" name="redirect_tab" value="signal-train">
          <button class="btn ghost">Retreinar modelo agora</button>
        </form>
      </div>
    </div>
    <div class="signal-train-summary">
      <span class="pill">candidatos: <b id="signal-candidate-total">{total}</b></span>
      <span class="pill">mostrando: <b>{shown}</b></span>
      <span class="pill">feedbacks atômicos: <b id="signal-feedback-total">{stats.get("total", 0)}</b></span>
    </div>
    <div class="signal-train-grid">{cards}</div>
    <div class="bt-pagination">{prev_btn}{next_btn}</div>
    """


def _render_filter_bar(pending: list[dict], top_descriptors: list[str], sort_mode: str = "priority") -> str:
    n_like = sum(1 for r in pending if _decision_label(r.get("original_label", r.get("label", ""))) in {"CURTIR", "SUPER LIKE"})
    n_pass = len(pending) - n_like
    n_photo = sum(1 for r in pending if _photo_available(r))
    n_body = sum(1 for r in pending if _body_photo_path(r) or str(r.get("photo_body_visible","")) not in ("","0","0.0"))

    desc_chips = "".join(
        f'<button type="button" class="fchip desc-chip" data-desc="{_esc(_normalize(d))}" title="{_esc(d)}">{_esc(d)}</button>'
        for d in top_descriptors
    )

    sort_priority_active = " active" if sort_mode == "priority" else ""
    sort_uncertain_active = " active" if sort_mode == "uncertain" else ""
    sort_confident_active = " active" if sort_mode == "confident" else ""
    sort_recent_active = " active" if sort_mode == "recent" else ""

    return f"""
    <div class="filter-bar" id="filter-bar">
      <div class="filter-group">
        <span class="filter-label">Decisão</span>
        <button type="button" class="fchip active" data-filter="decision" data-value="all">todos <b>{len(pending)}</b></button>
        <button type="button" class="fchip" data-filter="decision" data-value="like">IA curtiu <b>{n_like}</b></button>
        <button type="button" class="fchip" data-filter="decision" data-value="pass">IA passou <b>{n_pass}</b></button>
      </div>
      <div class="filter-group">
        <span class="filter-label">Foto</span>
        <button type="button" class="fchip" data-filter="photo" data-value="has-photo">com foto <b>{n_photo}</b></button>
        <button type="button" class="fchip" data-filter="photo" data-value="has-body">com corpo <b>{n_body}</b></button>
      </div>
      <div class="filter-group">
        <span class="filter-label">Ordenar</span>
        <a href="?sort=priority" class="fchip{sort_priority_active}" title="Maior prioridade de revisão calculada pelo modelo">prioridade</a>
        <a href="?sort=uncertain" class="fchip{sort_uncertain_active}" title="Perfis em que a IA teve mais dúvida aparecem primeiro — aproveita melhor cada revisão">incertos primeiro</a>
        <a href="?sort=confident" class="fchip{sort_confident_active}" title="Perfis em que a IA teve mais certeza aparecem primeiro — bom para achar erros confiantes">mais certeza primeiro</a>
        <a href="?sort=recent" class="fchip{sort_recent_active}" title="Mais recentes primeiro">mais recentes</a>
      </div>
      <div class="filter-group" id="desc-filter-group">
        <span class="filter-label">Descritor</span>
        <input type="search" id="desc-search" class="filter-search" placeholder="buscar descritor…" autocomplete="off">
        {desc_chips}
      </div>
    </div>"""


def _sort_pending_reviews(pending: list[dict], sort_mode: str = "priority") -> list[dict]:
    sort_mode = sort_mode if sort_mode in ("priority", "uncertain", "confident", "recent") else "priority"

    def _uncertainty(row: dict) -> float:
        try:
            snap = json.loads(row.get("ai_snapshot") or "{}")
            prob = float(snap.get("probability", 0.5))
            return abs(prob - 0.5)
        except Exception:
            return 0.5

    rows = list(pending)
    if sort_mode == "priority":
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        rows.sort(key=lambda row: _as_float_safe(row.get("review_priority", ""), 0.0), reverse=True)
        rows.sort(key=lambda row: 0 if _photo_available(row) else 1)
    elif sort_mode == "uncertain":
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        rows.sort(key=_uncertainty)
        rows.sort(key=lambda row: 0 if _photo_available(row) else 1)
    elif sort_mode == "confident":
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        rows.sort(key=_uncertainty, reverse=True)
        rows.sort(key=lambda row: 0 if _photo_available(row) else 1)
    else:
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        rows.sort(key=lambda row: 0 if _photo_available(row) else 1)
    return rows


def _render_next_review_card(shown: set[str], sort_mode: str = "priority") -> bytes:
  prefs = _load_prefs()
  _cleanup_review_queue_once()
  pending = [
    row for row in load_reviews("pending")
    if row.get("review_id", "") not in shown
  ]
  pending = _sort_pending_reviews(pending, sort_mode)
  html = _render_card(pending[0], prefs) if pending else ""
  return html.encode("utf-8")


def _render_page(
    message: str = "",
    msg_type: str = "",
    initial_tab: str = "",
    selected_photo: str = "",
    show_all: bool = False,
    sort_mode: str = "priority",
    bt_page: int = 0,
    signal_type: str = "all",
    signal_page: int = 0,
) -> bytes:
    prefs = _load_prefs()
    photo_deep_enabled = _photo_deep_enabled()
    if not photo_deep_enabled and initial_tab == "photo-deep":
        initial_tab = ""
    if initial_tab not in ("photo-deep", "body-train", "signal-train"):
        initial_tab = initial_tab if initial_tab == "photo-deep" else ""
    if sort_mode not in ("priority", "uncertain", "confident", "recent"):
        sort_mode = "priority"
    hidden_filters, hidden_duplicates = _cleanup_review_queue_once()
    all_reviews = load_reviews(None)
    pending = [row for row in all_reviews if row.get("review_status", "pending") == "pending"]

    pending = _sort_pending_reviews(pending, sort_mode)
    reviewed = [row for row in all_reviews if row.get("review_status", "pending") == "reviewed"]
    skipped = [row for row in all_reviews if row.get("review_status", "pending") == "skipped"]
    reviewed_sorted = sorted(reviewed + skipped, key=lambda r: r.get("reviewed_at", ""), reverse=True)
    deep_n = count_deep_records() if photo_deep_enabled else 0
    _, bt_total = _body_train_profiles(page=0, per_page=1)
    initial_tab_json = json.dumps(initial_tab or "")
    main_hidden = initial_tab in ("photo-deep", "body-train", "signal-train")
    deep_hidden = initial_tab != "photo-deep"
    bt_hidden = initial_tab != "body-train"
    signal_hidden = initial_tab != "signal-train"
    main_attr = " hidden" if main_hidden else ""
    deep_attr = " hidden" if deep_hidden else ""
    bt_attr = " hidden" if bt_hidden else ""
    signal_attr = " hidden" if signal_hidden else ""
    tab_main_active = "" if main_hidden else " active"
    tab_deep_active = " active" if not deep_hidden else ""
    tab_bt_active = " active" if not bt_hidden else ""
    tab_signal_active = " active" if not signal_hidden else ""
    if photo_deep_enabled and initial_tab == "photo-deep":
        deep_paths = list_saved_photo_paths(None, body_only=True)
        deep_panel = _render_photo_deep_panel(deep_paths, selected_photo)
        deep_lazy_attr = ""
    elif photo_deep_enabled:
        deep_panel = """
        <div class="empty">
          <h2>Rotulagem visual</h2>
          <p>Abra esta aba para carregar a galeria de fotos.</p>
        </div>"""
        deep_lazy_attr = ' data-lazy="1"'
    else:
        deep_panel = ""
        deep_lazy_attr = ""

    # Coleta descritores mais frequentes entre todos os pendentes para chips de filtro
    _desc_counter: dict[str, int] = {}
    for _row in pending:
        for _k, _v in _parse_descriptors(_row.get("descriptors", "")).items():
            _key = f"{_k}: {_v}"
            _desc_counter[_key] = _desc_counter.get(_key, 0) + 1
    top_descriptors = sorted(_desc_counter, key=lambda k: _desc_counter[k], reverse=True)[:12]

    visible_pending = pending if show_all else pending[:REVIEW_PAGE_LIMIT]
    hidden_pending = max(0, len(pending) - len(visible_pending))
    cards = "\n".join(_render_card(row, prefs) for row in visible_pending)
    if not cards:
        cards = """
        <div class="empty">
          <h2>Nada pendente</h2>
          <p>Rode o swiper em modo automático para preencher a fila de revisão.</p>
        </div>"""
    elif hidden_pending:
        cards += f"""
        <div class="more-pending">
          Mostrando {len(visible_pending)} de {len(pending)} pendentes para manter a tela rápida.
          <a href="/?all=1">mostrar todos</a>
        </div>"""

    reviewed_rows = "".join(_render_reviewed_row(r) for r in reviewed_sorted[:20])
    if not reviewed_rows:
        reviewed_rows = '<p class="muted" style="font-family:ui-sans-serif;font-size:13px;padding:8px">Nenhuma revisão feita ainda.</p>'

    notice_html = ""
    if message:
        cls = {"ok": "ok", "err": "err"}.get(msg_type, "")
        notice_html = f'<div id="ajax-notice" class="notice {cls}">{_esc(message)}</div>'
    elif hidden_filters:
        notice_html = f'<div id="ajax-notice" class="notice ok">{hidden_filters} filtro(s) absoluto(s) ocultado(s) da revisão.</div>'

    retrain_notice_html = _render_retrain_status_notice()
    eval_summary_html = _render_evaluation_summary()
    visual_stats_html = f'<span class="pill">rótulos visuais: <b>{deep_n}</b></span>' if photo_deep_enabled else ""
    tab_deep_button = (
        f'<button type="button" class="tab-btn{tab_deep_active}" data-tab="photo-deep" '
        f'role="tab" aria-controls="tab-photo-deep" aria-selected="{str(not deep_hidden).lower()}">'
        "Rotulagem visual — fotos</button>"
    ) if photo_deep_enabled else ""
    deep_section_html = (
        f'<div id="tab-photo-deep" class="tab-panel"{deep_attr}{deep_lazy_attr} role="tabpanel">\n'
        f'{deep_panel}\n'
        "</div>"
    ) if photo_deep_enabled else ""
    bt_lazy_attr = ' data-lazy="1"' if bt_hidden else ""
    bt_panel_html = _render_body_train_panel(page=bt_page) if not bt_hidden else (
        '<div class="bt-empty"><p>Abra esta aba para carregar os perfis.</p></div>'
    )
    signal_lazy_attr = ' data-lazy="1"' if signal_hidden else ""
    signal_panel_html = _render_signal_train_panel(signal_type, signal_page) if not signal_hidden else (
        '<div class="bt-empty"><p>Abra esta aba para carregar os sinais.</p></div>'
    )
    bt_count_pill = f' <span style="font-size:10px;opacity:.7">({bt_total})</span>' if bt_total else ""
    subtitle = (
        "Correção pós-swipe para o treino principal."
        if not photo_deep_enabled
        else "Correção pós-swipe (treino principal) e avaliação extra só de fotos (opcional)."
    )
    history_stats = history_review_candidate_stats()
    history_actionable = int(history_stats.get("actionable", 0) or 0)
    history_pending = int(history_stats.get("pending_history", 0) or 0)
    history_title = (
        f"{history_actionable} perfil(is) antigos ainda fora da fila; "
        f"{history_pending} já estão pendentes na fila."
    )
    history_label = f"Revisar histórico ({history_actionable})"

    page = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tinder-IA Revisão</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <div>
      <h1>Tinder-IA — revisão</h1>
      <p class="subtitle">{subtitle}</p>
    </div>
    <div class="header-right">
      <div class="stats">
        <span class="pill">pendentes: <b>{len(pending)}</b></span>
        <span class="pill">revisados: <b>{len(reviewed)}</b></span>
        {visual_stats_html}
      </div>
      <a href="/settings" class="btn ghost btn-sm" style="text-decoration:none">⚙ Configurações</a>
    </div>
  </header>
  <main>
    {notice_html}
    {retrain_notice_html}
    {eval_summary_html}
    <nav class="review-tabs" role="tablist" aria-label="Modos de revisão">
      <button type="button" class="tab-btn{tab_main_active}" data-tab="main" role="tab" aria-controls="tab-main" aria-selected="{str(not main_hidden).lower()}">Revisão pós-swipe</button>
      {tab_deep_button}
      <button type="button" class="tab-btn{tab_bt_active}" data-tab="body-train" role="tab" aria-controls="tab-body-train" aria-selected="{str(not bt_hidden).lower()}">Treino corporal<span id="bt-count-pill">{bt_total if bt_total else ""}</span></button>
      <button type="button" class="tab-btn{tab_signal_active}" data-tab="signal-train" role="tab" aria-controls="tab-signal-train" aria-selected="{str(not signal_hidden).lower()}">Treino de sinais</button>
    </nav>
    <div id="tab-main" class="tab-panel"{main_attr} role="tabpanel">
    <div class="toolbar">
      <span class="pill">mostrando <b>{len(visible_pending)}</b> de <b>{len(pending)}</b></span>
      <form method="post" action="/skip-all" onsubmit="return confirm('Descartar todos os {len(pending)} perfis pendentes?\\nEles não entrarão no treino — você poderá rever perfis futuros com o modelo atualizado.')">
        <button class="btn ghost skip-all-btn">Descartar todos os pendentes ({len(pending)})</button>
      </form>
      <form method="post" action="/retrain">
        <button class="btn ghost" id="retrain-button">Retreinar modelo agora</button>
      </form>
      <form method="post" action="/enqueue-history">
        <input type="hidden" name="limit" value="40">
        <button class="btn ghost" title="{_esc(history_title)}">{_esc(history_label)}</button>
      </form>
    </div>
    {_render_filter_bar(pending, top_descriptors, sort_mode)}
    <div id="filter-count" class="filter-count" style="display:none"></div>
    {cards}
    <details class="reviewed-section">
      <summary>↩ Revisados recentemente — clique para desfazer</summary>
      {reviewed_rows}
    </details>
    </div>
    {deep_section_html}
    <div id="tab-body-train" class="tab-panel"{bt_attr}{bt_lazy_attr} role="tabpanel">
      {bt_panel_html}
    </div>
    <div id="tab-signal-train" class="tab-panel"{signal_attr}{signal_lazy_attr} role="tabpanel">
      {signal_panel_html}
    </div>
  </main>
  <script>
    function setReviewTab(which) {{
      ['tab-main','tab-photo-deep','tab-body-train','tab-signal-train'].forEach(function(id) {{
        var el = document.getElementById(id);
        if (el) el.hidden = (id !== 'tab-' + which);
      }});
      document.querySelectorAll('.review-tabs .tab-btn').forEach(function(btn) {{
        var on = btn.getAttribute('data-tab') === which;
        btn.classList.toggle('active', on);
        btn.setAttribute('aria-selected', on ? 'true' : 'false');
      }});
    }}
    document.querySelectorAll('.review-tabs .tab-btn').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        var target = btn.getAttribute('data-tab') || 'main';
        if (target === 'photo-deep') {{
          var panel = document.getElementById('tab-photo-deep');
          if (panel && panel.getAttribute('data-lazy') === '1') {{
            var lazyUrl = new URL(location.href);
            lazyUrl.searchParams.set('tab', 'photo-deep');
            location.href = lazyUrl.pathname + lazyUrl.search + lazyUrl.hash;
            return;
          }}
        }}
        if (target === 'body-train') {{
          var panel = document.getElementById('tab-body-train');
          if (panel && panel.getAttribute('data-lazy') === '1') {{
            var lazyUrl = new URL(location.href);
            lazyUrl.searchParams.set('tab', 'body-train');
            location.href = lazyUrl.pathname + lazyUrl.search + lazyUrl.hash;
            return;
          }}
        }}
        if (target === 'signal-train') {{
          var panel = document.getElementById('tab-signal-train');
          if (panel && panel.getAttribute('data-lazy') === '1') {{
            var lazyUrl = new URL(location.href);
            lazyUrl.searchParams.set('tab', 'signal-train');
            location.href = lazyUrl.pathname + lazyUrl.search + lazyUrl.hash;
            return;
          }}
        }}
        setReviewTab(target);
        try {{
          var u = new URL(location.href);
          if (target === 'photo-deep' || target === 'body-train' || target === 'signal-train') u.searchParams.set('tab', target);
          else u.searchParams.delete('tab');
          history.replaceState(null, '', u.pathname + u.search + u.hash);
        }} catch (e) {{}}
      }});
    }});
    var tabInit = {initial_tab_json};
    if (tabInit === 'photo-deep') setReviewTab('photo-deep');
    if (tabInit === 'body-train') setReviewTab('body-train');
    if (tabInit === 'signal-train') setReviewTab('signal-train');

    var dsel = document.getElementById('deep-photo-path');
    var dimg = document.getElementById('deep-photo-preview');
    var deepGallery = document.getElementById('deep-gallery');
    var activeDeepFilter = 'all';
    function syncDeepSelectedThumb() {{
      if (!deepGallery || !dsel) return;
      deepGallery.querySelectorAll('.deep-thumb').forEach(function(btn) {{
        btn.classList.toggle('selected', btn.getAttribute('data-path') === dsel.value);
      }});
    }}
    function updDeepImg() {{
      if (!dsel || !dimg) return;
      var p = dsel.value;
      if (!p) {{ dimg.removeAttribute('src'); return; }}
      dimg.src = '/photo?path=' + encodeURIComponent(p);
      syncDeepSelectedThumb();
    }}
    if (dsel) {{ dsel.addEventListener('change', updDeepImg); updDeepImg(); }}
    if (deepGallery && dsel) {{
      deepGallery.querySelectorAll('.deep-thumb').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          dsel.value = btn.getAttribute('data-path') || '';
          updDeepImg();
          document.getElementById('photo-deep-form')?.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }});
      }});
    }}
    function applyDeepFilter(filter) {{
      activeDeepFilter = filter || 'all';
      document.querySelectorAll('.deep-filter').forEach(function(btn) {{
        btn.classList.toggle('active', btn.getAttribute('data-filter') === activeDeepFilter);
      }});
      if (!deepGallery) return;
      deepGallery.querySelectorAll('.deep-thumb').forEach(function(btn) {{
        var folder = btn.getAttribute('data-folder');
        var reviewed = btn.getAttribute('data-reviewed') === '1';
        var show = activeDeepFilter === 'all'
          || (activeDeepFilter === 'reviewed' && reviewed)
          || (activeDeepFilter === 'unreviewed' && !reviewed)
          || (activeDeepFilter === folder);
        btn.classList.toggle('hidden-filter', !show);
      }});
    }}
    document.querySelectorAll('.deep-filter').forEach(function(btn) {{
      btn.addEventListener('click', function() {{ applyDeepFilter(btn.getAttribute('data-filter') || 'all'); }});
    }});
    applyDeepFilter('all');

    var dform = document.getElementById('photo-deep-form');
    if (dform) {{
      dform.querySelectorAll('.deep-pos-cb').forEach(function(inp) {{
        inp.addEventListener('change', function() {{
          if (!inp.checked) return;
          dform.querySelectorAll('.deep-neg-cb').forEach(function(o) {{
            if (o.value === inp.value) o.checked = false;
          }});
        }});
      }});
      dform.querySelectorAll('.deep-neg-cb').forEach(function(inp) {{
        inp.addEventListener('change', function() {{
          if (!inp.checked) return;
          dform.querySelectorAll('.deep-pos-cb').forEach(function(o) {{
            if (o.value === inp.value) o.checked = false;
          }});
        }});
      }});
    }}

    function clearFeedbackControls(root) {{
      if (!root) return;
      root.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach(function(inp) {{
        inp.checked = false;
      }});
      root.querySelectorAll('textarea').forEach(function(txt) {{
        txt.value = '';
      }});
      root.querySelectorAll('select').forEach(function(sel) {{
        sel.selectedIndex = 0;
      }});
    }}
    function syncPhotoAspectChips(photoWrap) {{
      if (!photoWrap) return;
      var detailSel = photoWrap.querySelector('select.photo-detail-select');
      var selected = detailSel ? (detailSel.value || 'photo_general') : 'photo_general';
      photoWrap.querySelectorAll('label.photo-aspect-opt').forEach(function(lab) {{
        var detail = lab.getAttribute('data-for-details') || '';
        var show = selected === 'photo_general' || detail === selected;
        lab.style.display = show ? '' : 'none';
        if (!show) {{
          var input = lab.querySelector('input.photo-aspect-cb');
          if (input) input.checked = false;
        }}
      }});
    }}
    function selectedReasonDomains(form) {{
      if (!form) return [];
      return Array.from(form.querySelectorAll('input[name="also"]:checked'))
        .map(function(inp) {{ return inp.value || ''; }})
        .filter(Boolean);
    }}
    function syncReasonDomains(form) {{
      if (!form) return [];
      var selected = selectedReasonDomains(form);
      var selectedSet = new Set(selected);
      var primary = form.querySelector('.primary-domain-input');
      if (primary) primary.value = selected[0] || '';
      form.querySelectorAll('.reason-chip').forEach(function(label) {{
        var input = label.querySelector('input[name="also"]');
        label.classList.toggle('is-checked', !!input && input.checked);
      }});
      [
        ['photo', '.photo-detail-wrap'],
        ['interests', '.interests-detail-wrap'],
        ['bio', '.bio-detail-wrap'],
        ['descriptors', '.desc-detail-wrap'],
        ['other', '.other-detail-wrap']
      ].forEach(function(pair) {{
        var wrap = form.querySelector(pair[1]);
        var show = selectedSet.has(pair[0]);
        if (!wrap) return;
        wrap.style.display = show ? '' : 'none';
        if (!show) clearFeedbackControls(wrap);
      }});
      var photoWrap = form.querySelector('.photo-detail-wrap');
      if (photoWrap && photoWrap.style.display !== 'none') syncPhotoAspectChips(photoWrap);
      return selected;
    }}
    function requireReasonDomains(form) {{
      var selected = syncReasonDomains(form);
      if (selected.length) return true;
      var picker = form ? form.querySelector('.reason-picker') : null;
      if (picker) {{
        picker.classList.add('needs-choice');
        setTimeout(function() {{ picker.classList.remove('needs-choice'); }}, 1200);
      }}
      _showAjaxNotice('Marque ao menos um sinal que pesou nessa decisão.', 'err');
      return false;
    }}
    function bindReasonDomains(root) {{
      (root || document).querySelectorAll('.review-form').forEach(function(form) {{
        if (form.dataset.reasonBound === '1') {{
          syncReasonDomains(form);
          return;
        }}
        form.dataset.reasonBound = '1';
        form.querySelectorAll('input[name="also"]').forEach(function(inp) {{
          inp.addEventListener('change', function() {{ syncReasonDomains(form); }});
        }});
        syncReasonDomains(form);
      }});
    }}
    bindReasonDomains(document);
    document.querySelectorAll('.photo-detail-wrap').forEach(function(wrap) {{
      var ds = wrap.querySelector('select.photo-detail-select');
      if (ds) {{
        ds.addEventListener('change', function() {{ syncPhotoAspectChips(wrap); }});
        syncPhotoAspectChips(wrap);
      }}
    }});
    document.querySelectorAll('.review-form').forEach(function(form) {{
      var pos = form.querySelector('.photo-aspects-pos');
      var neg = form.querySelector('.photo-aspects-neg');
      function bind(grid, other) {{
        if (!grid || !other) return;
        grid.querySelectorAll('input.photo-aspect-cb').forEach(function(inp) {{
          inp.addEventListener('change', function() {{
            if (!inp.checked) return;
            var v = inp.value;
            other.querySelectorAll('input.photo-aspect-cb').forEach(function(o) {{
              if (o.value === v) o.checked = false;
            }});
          }});
        }});
      }}
      bind(pos, neg);
      bind(neg, pos);
      form.addEventListener('submit', function(ev) {{
        var submitter = ev.submitter;
        if (!submitter || submitter.getAttribute('formaction') === '/agree' || submitter.getAttribute('formaction') === '/skip') return;
        if (!requireReasonDomains(form)) {{
          ev.preventDefault();
          return;
        }}
        var decision = submitter.value || '';
        if (!decision) return;
        var posMarked = form.querySelectorAll('input[name="photo_positive_detail"]:checked').length;
        var negMarked = form.querySelectorAll('input[name="photo_negative_detail"]:checked').length;
        var asksPassWithPositive = decision === 'NÃO CURTIR' && posMarked > 0;
        var asksLikeWithNegative = (decision === 'CURTIR' || decision === 'SUPER_LIKE') && negMarked > 0;
        if (asksPassWithPositive || asksLikeWithNegative) {{
          var msg = asksPassWithPositive
            ? 'Você marcou pontos positivos na foto, mas escolheu passar. Salvar mesmo assim?'
            : 'Você marcou pontos negativos na foto, mas escolheu curtir/super like. Salvar mesmo assim?';
          if (!window.confirm(msg)) ev.preventDefault();
        }}
      }});
      form.querySelectorAll('input[name="descriptor_positive_detail"]').forEach(function(inp) {{
        inp.addEventListener('change', function() {{
          if (!inp.checked) return;
          form.querySelectorAll('input[name="descriptor_negative_detail"]').forEach(function(o) {{
            if (o.value === inp.value) o.checked = false;
          }});
        }});
      }});
      form.querySelectorAll('input[name="descriptor_negative_detail"]').forEach(function(inp) {{
        inp.addEventListener('change', function() {{
          if (!inp.checked) return;
          form.querySelectorAll('input[name="descriptor_positive_detail"]').forEach(function(o) {{
            if (o.value === inp.value) o.checked = false;
          }});
        }});
      }});
    }});

    var _reviewSeen = new Set(
      Array.from(document.querySelectorAll('.card[id^="card-"]'))
        .map(function(card) {{ return card.id.replace(/^card-/, ''); }})
    );
    var _reviewPageLimit = {REVIEW_PAGE_LIMIT};
    var _nextReviewInFlight = false;
    var _reviewNextExhausted = false;

    function _visibleReviewCardCount() {{
      return document.querySelectorAll('#tab-main > .card[id^="card-"]').length;
    }}

    function _bindReviewAjax(form) {{
      if (!form || form.dataset.ajaxBound === '1') return;
      form.dataset.ajaxBound = '1';
      form.addEventListener('submit', function(ev) {{
        if (ev.defaultPrevented) return;
        var submitter = ev.submitter;
        if (!submitter) return;
        ev.preventDefault();
        if (form.dataset.submitting === '1') return;
        form.dataset.submitting = '1';
        var card = form.closest('.card');
        var action = submitter.getAttribute('formaction') || form.getAttribute('action') || '/apply';
        if (action.indexOf('/apply') >= 0 && !requireReasonDomains(form)) {{
          form.dataset.submitting = '';
          return;
        }}
        var params = new URLSearchParams(new FormData(form));
        if (submitter.name) params.set(submitter.name, submitter.value || '');
        params.set('ajax', '1');
        form.querySelectorAll('button').forEach(function(btn) {{ btn.disabled = true; }});
        fetch(action, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
          body: params.toString()
        }})
          .then(function(r) {{ return r.json(); }})
          .then(function(j) {{
            if (!j.ok) throw new Error(j.message || 'Falha ao salvar revisão');
            if (card) {{
              var rid = card.id.replace(/^card-/, '');
              _reviewSeen.add(rid);
              if (j.similar_html) _insertSimilarPanel(j.similar_html, card);
              card.style.transition = 'opacity .18s, transform .18s';
              card.style.opacity = '0';
              card.style.transform = 'translateY(-6px)';
              setTimeout(function() {{
                card.remove();
                _fillReviewSlots();
              }}, 190);
            }}
            _showAjaxNotice(j.message || 'Revisão salva.', j.type || 'ok');
          }})
          .catch(function(err) {{
            form.dataset.submitting = '';
            form.querySelectorAll('button').forEach(function(btn) {{ btn.disabled = false; }});
            _showAjaxNotice(err.message || 'Erro ao salvar revisão.', 'err');
          }});
      }});
    }}

    function _showAjaxNotice(message, type) {{
      var notice = document.getElementById('ajax-notice');
      if (!notice) {{
        notice = document.createElement('div');
        notice.id = 'ajax-notice';
        notice.className = 'notice';
        var main = document.querySelector('main');
        if (main) main.insertBefore(notice, main.firstChild);
      }}
      notice.className = 'notice ' + (type || 'ok');
      notice.textContent = message || '';
    }}

	    var _signalSeen = new Set(
	      Array.from(document.querySelectorAll('.signal-train-card[data-signal-key]'))
	        .map(function(card) {{ return card.getAttribute('data-signal-key') || ''; }})
	        .filter(Boolean)
	    );
	    var _signalInFlight = new Set();
	    var _signalNextInFlight = false;
	    var _signalNextQueued = false;

	    function _currentSignalFilter() {{
	      try {{
	        return new URL(location.href).searchParams.get('signal_type') || 'all';
	      }} catch (e) {{
	        return 'all';
	      }}
	    }}

	    function _bumpSignalCount(id, delta) {{
	      var el = document.getElementById(id);
	      if (!el) return;
	      var n = parseInt(el.textContent || '0', 10);
	      if (Number.isNaN(n)) return;
	      el.textContent = String(Math.max(0, n + delta));
	    }}

	    function _signalCardsForKey(key, fallbackCard) {{
	      var cards = Array.from(document.querySelectorAll('.signal-train-card[data-signal-key]'))
	        .filter(function(card) {{ return card.getAttribute('data-signal-key') === key; }});
	      if (!cards.length && fallbackCard) cards = [fallbackCard];
	      return cards;
	    }}

	    function _setSignalCardsSaving(key, selectedButton, fallbackCard) {{
	      _signalCardsForKey(key, fallbackCard).forEach(function(card) {{
	        card.classList.remove('save-error');
	        card.classList.add('saving');
	        card.querySelectorAll('.signal-choice').forEach(function(other) {{
	          other.disabled = true;
	          other.classList.toggle('selected', other === selectedButton);
	        }});
	      }});
	    }}

	    function _setSignalCardsError(key, fallbackCard) {{
	      _signalCardsForKey(key, fallbackCard).forEach(function(card) {{
	        card.classList.remove('saving');
	        card.classList.add('save-error');
	        card.querySelectorAll('.signal-choice').forEach(function(other) {{
	          other.disabled = false;
	        }});
	      }});
	    }}

	    function _signalAppendNext() {{
	      var grid = document.querySelector('.signal-train-grid');
	      if (!grid) return;
	      if (_signalNextInFlight) {{
	        _signalNextQueued = true;
	        return;
	      }}
	      _signalNextInFlight = true;
	      var params = new URLSearchParams();
	      params.set('signal_type', _currentSignalFilter());
	      Array.from(_signalSeen).forEach(function(key) {{
	        params.append('shown', key);
	      }});
	      fetch('/signal-train-next?' + params.toString())
	        .then(function(r) {{ return r.text(); }})
	        .then(function(html) {{
	          var trimmed = html.trim();
	          if (!trimmed) return;
	          var div = document.createElement('div');
	          div.innerHTML = trimmed;
	          var card = div.firstElementChild;
	          if (!card) return;
	          var key = card.getAttribute('data-signal-key') || '';
	          if (key) _signalSeen.add(key);
	          card.style.opacity = '0';
	          card.style.transform = 'translateY(6px)';
	          grid.appendChild(card);
	          _bindSignalTrain(card);
	          requestAnimationFrame(function() {{
	            card.style.transition = 'opacity .2s, transform .2s';
	            card.style.opacity = '1';
	            card.style.transform = 'translateY(0)';
	          }});
	        }})
	        .catch(function() {{}})
	        .finally(function() {{
	          _signalNextInFlight = false;
	          if (_signalNextQueued) {{
	            _signalNextQueued = false;
	            _signalAppendNext();
	          }}
	        }});
	    }}

	    function _removeSignalCardsByKey(key, fallbackCard) {{
	      var cards = _signalCardsForKey(key, fallbackCard);
	      cards = cards.filter(function(card) {{
	        if (!card || card.dataset.signalRemoving === '1') return false;
	        card.dataset.signalRemoving = '1';
	        return true;
	      }});
	      if (!cards.length) {{
	        _signalAppendNext();
	        return;
	      }}
	      var pending = cards.length;
	      function done() {{
	        pending -= 1;
	        if (pending <= 0) {{
	          _bumpSignalCount('signal-candidate-total', -1);
	          _signalAppendNext();
	        }}
	      }}
	      cards.forEach(function(card) {{
	        card.style.transition = 'opacity .18s, transform .18s';
	        card.style.opacity = '0';
	        card.style.transform = 'translateY(-6px)';
	        setTimeout(function() {{
	          card.remove();
	          done();
	        }}, 190);
	      }});
	    }}

	    function _bindSignalTrain(root) {{
	      (root || document).querySelectorAll('.signal-choice').forEach(function(btn) {{
	        if (btn.dataset.signalBound === '1') return;
	        btn.dataset.signalBound = '1';
	        btn.addEventListener('click', function() {{
	          var actions = btn.closest('.signal-train-actions');
	          var card = btn.closest('.signal-train-card');
	          if (!actions || !card) return;
	          var key = card.getAttribute('data-signal-key') || '';
	          if (!key || _signalInFlight.has(key)) return;
	          _signalInFlight.add(key);
	          _setSignalCardsSaving(key, btn, card);

	          var params = new URLSearchParams();
	          params.set('ajax', '1');
	          params.set('signal_type', actions.dataset.signalType || '');
	          params.set('signal_value', actions.dataset.signalValue || '');
	          params.set('polarity', btn.dataset.polarity || '');
	          params.set('signal_filter', _currentSignalFilter());
	          params.set('occurrences', actions.dataset.occurrences || '');
	          params.set('likes', actions.dataset.likes || '');
	          params.set('dislikes', actions.dataset.dislikes || '');

	          fetch('/signal-train-save', {{
	            method: 'POST',
	            headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
	            body: params.toString()
	          }})
	          .then(function(r) {{
	            return r.json().then(function(j) {{
	              if (!r.ok || !j.ok) throw new Error(j.message || 'Falha ao salvar sinal.');
	              return j;
	            }});
	          }})
	          .then(function(j) {{
	            _signalInFlight.delete(key);
	            _bumpSignalCount('signal-feedback-total', 1);
	            _showAjaxNotice(j.message || 'Sinal salvo.', j.type || 'ok');
	            _removeSignalCardsByKey(key, card);
	          }})
	          .catch(function(err) {{
	            _signalInFlight.delete(key);
	            _setSignalCardsError(key, card);
	            _showAjaxNotice(err.message || 'Erro ao salvar sinal.', 'err');
	          }});
	        }});
	      }});
	    }}
	    _bindSignalTrain(document);

    var _lastRetrainRunning = (function() {{
      var el = document.getElementById('retrain-status');
      return !!(el && el.className.indexOf('running') !== -1);
    }})();

    function _ensureRetrainNotice() {{
      var el = document.getElementById('retrain-status');
      if (el) return el;
      el = document.createElement('div');
      el.id = 'retrain-status';
      el.className = 'notice retrain';
      el.hidden = true;
      el.setAttribute('aria-live', 'polite');
      var main = document.querySelector('main');
      if (main) {{
        var ajaxNotice = document.getElementById('ajax-notice');
        if (ajaxNotice && ajaxNotice.nextSibling) {{
          main.insertBefore(el, ajaxNotice.nextSibling);
        }} else {{
          main.insertBefore(el, main.firstChild);
        }}
      }}
      return el;
    }}

    function _updateRetrainNotice(status) {{
      status = status || {{}};
      var running = !!status.running;
      var shouldShow = running || _lastRetrainRunning;
      var el = _ensureRetrainNotice();
      var button = document.getElementById('retrain-button');
      if (button) {{
        button.disabled = running;
        button.textContent = running ? 'Retreinando...' : 'Retreinar modelo agora';
      }}
      _lastRetrainRunning = running;
      if (!shouldShow) {{
        el.hidden = true;
        return running;
      }}

      var type = running ? 'running' : (status.type === 'err' ? 'err' : 'ok');
      el.hidden = false;
      el.className = 'notice retrain ' + type;
      el.textContent = '';
      if (running) {{
        var dot = document.createElement('span');
        dot.className = 'retrain-dot';
        dot.setAttribute('aria-hidden', 'true');
        el.appendChild(dot);
      }}
      var text = document.createElement('span');
      text.textContent = status.message || (running ? 'Retreinando modelo em background.' : 'Retreino concluído.');
      el.appendChild(text);
      if (!running) {{
        window.setTimeout(function() {{
          if (!_lastRetrainRunning) el.hidden = true;
        }}, 9000);
      }}
      return running;
    }}

    function _pollRetrainStatus() {{
      fetch('/retrain-status', {{headers: {{'Accept': 'application/json'}}}})
        .then(function(r) {{
          if (!r.ok) throw new Error('Falha ao consultar retreino');
          return r.json();
        }})
        .then(function(status) {{
          var running = _updateRetrainNotice(status);
          window.setTimeout(_pollRetrainStatus, running ? 2500 : 10000);
        }})
        .catch(function() {{
          window.setTimeout(_pollRetrainStatus, 5000);
        }});
    }}
    _pollRetrainStatus();

    function _activateSimilarPanel(panel) {{
      if (!panel || panel.dataset.bound === '1') return;
      panel.dataset.bound = '1';
      var close = panel.querySelector('.similar-close');
      if (close) close.addEventListener('click', function() {{ panel.remove(); }});
      panel.querySelectorAll('.similar-card').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          var rid = btn.getAttribute('data-review-id') || '';
          if (rid) _loadReviewCardById(rid, btn);
        }});
      }});
    }}

    function _insertSimilarPanel(html, anchor) {{
      var trimmed = (html || '').trim();
      if (!trimmed) return null;
      var tmp = document.createElement('div');
      tmp.innerHTML = trimmed;
      var panel = tmp.firstElementChild;
      if (!panel) return null;
      if (anchor && anchor.parentNode) {{
        anchor.parentNode.insertBefore(panel, anchor);
      }} else {{
        var main = document.querySelector('main');
        if (main) main.insertBefore(panel, main.firstChild);
      }}
      _activateSimilarPanel(panel);
      return panel;
    }}

    function _loadSimilar(reviewId, slot, mode) {{
      if (!reviewId) return;
      if (slot) {{
        slot.innerHTML = '<div class="similar-panel"><p class="similar-empty">buscando parecidos...</p></div>';
      }}
      fetch('/review-similar?review_id=' + encodeURIComponent(reviewId) + '&mode=' + encodeURIComponent(mode || 'manual'))
        .then(function(r) {{ return r.text(); }})
        .then(function(html) {{
          if (slot) {{
            slot.innerHTML = html;
            _activateSimilarPanel(slot.querySelector('.similar-panel'));
          }} else {{
            _insertSimilarPanel(html, null);
          }}
        }})
        .catch(function() {{
          if (slot) slot.innerHTML = '<div class="similar-panel"><p class="similar-empty">não consegui buscar parecidos agora.</p></div>';
        }});
    }}

    function _loadReviewCardById(reviewId, sourceEl) {{
      if (!reviewId) return;
      var existing = document.getElementById('card-' + reviewId);
      if (existing) {{
        existing.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        existing.classList.add('focus-flash');
        setTimeout(function() {{ existing.classList.remove('focus-flash'); }}, 900);
        return;
      }}
      fetch('/review-card?review_id=' + encodeURIComponent(reviewId))
        .then(function(r) {{ return r.text(); }})
        .then(function(html) {{
          var trimmed = html.trim();
          if (!trimmed) {{
            _showAjaxNotice('Esse parecido não está mais pendente.', 'err');
            return;
          }}
          var tmp = document.createElement('div');
          tmp.innerHTML = trimmed;
          var card = tmp.firstElementChild;
          if (!card) return;
          var morePending = document.querySelector('.more-pending');
          var details = document.querySelector('.reviewed-section');
          var anchor = morePending || details;
          if (sourceEl) {{
            var panel = sourceEl.closest('.similar-panel');
            if (panel && panel.parentNode) anchor = panel.nextSibling || anchor;
          }}
          if (anchor && anchor.parentNode) {{
            anchor.parentNode.insertBefore(card, anchor);
          }} else {{
            document.getElementById('tab-main')?.appendChild(card);
          }}
          _reviewSeen.add(reviewId);
          _initReviewCard(card);
          card.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
          card.classList.add('focus-flash');
          setTimeout(function() {{ card.classList.remove('focus-flash'); }}, 900);
        }});
    }}

    function _bindSimilarButtons(root) {{
      (root || document).querySelectorAll('.similar-btn').forEach(function(btn) {{
        if (btn.dataset.bound === '1') return;
        btn.dataset.bound = '1';
        btn.addEventListener('click', function() {{
          var rid = btn.getAttribute('data-review-id') || '';
          var scope = btn.closest('.card') || document;
          var slot = scope.querySelector('#similar-slot-' + rid);
          _loadSimilar(rid, slot, 'manual');
        }});
      }});
    }}

    function _appendNextReviewCard() {{
      if (_nextReviewInFlight || _reviewNextExhausted) return;
      _nextReviewInFlight = true;
      var u = new URL(location.href);
      var sort = u.searchParams.get('sort') || 'priority';
      fetch('/review-next?sort=' + encodeURIComponent(sort) + '&shown=' + encodeURIComponent(Array.from(_reviewSeen).join(',')))
        .then(function(r) {{ return r.text(); }})
        .then(function(html) {{
          var trimmed = html.trim();
          if (!trimmed) {{
            _reviewNextExhausted = true;
            return;
          }}
          var tmp = document.createElement('div');
          tmp.innerHTML = trimmed;
          var card = tmp.firstElementChild;
          if (!card) return;
          var rid = card.id.replace(/^card-/, '');
          if (rid) _reviewSeen.add(rid);
          var morePending = document.querySelector('.more-pending');
          var details = document.querySelector('.reviewed-section');
          var anchor = morePending || details;
          if (anchor && anchor.parentNode) {{
            anchor.parentNode.insertBefore(card, anchor);
          }} else {{
            document.getElementById('tab-main')?.appendChild(card);
          }}
          _initReviewCard(card);
        }})
        .catch(function() {{
          _reviewNextExhausted = true;
          _showAjaxNotice('Não consegui carregar o próximo perfil agora.', 'err');
        }})
        .finally(function() {{
          _nextReviewInFlight = false;
          if (!_reviewNextExhausted && _visibleReviewCardCount() < _reviewPageLimit) {{
            _appendNextReviewCard();
          }}
        }});
    }}

    function _fillReviewSlots() {{
      if (_reviewNextExhausted || _visibleReviewCardCount() >= _reviewPageLimit) return;
      _appendNextReviewCard();
    }}

    function _initReviewCard(root) {{
      _bindSimilarButtons(root);
      bindReasonDomains(root);
      root.querySelectorAll('.photo-detail-wrap').forEach(function(wrap) {{
        var ds = wrap.querySelector('select.photo-detail-select');
        if (ds) {{
          ds.addEventListener('change', function() {{ syncPhotoAspectChips(wrap); }});
          syncPhotoAspectChips(wrap);
        }}
      }});
      root.querySelectorAll('.review-form').forEach(function(form) {{
        var pos = form.querySelector('.photo-aspects-pos');
        var neg = form.querySelector('.photo-aspects-neg');
        function bind(grid, other) {{
          if (!grid || !other) return;
          grid.querySelectorAll('input.photo-aspect-cb').forEach(function(inp) {{
            inp.addEventListener('change', function() {{
              if (!inp.checked) return;
              other.querySelectorAll('input.photo-aspect-cb').forEach(function(o) {{
                if (o.value === inp.value) o.checked = false;
              }});
            }});
          }});
        }}
        bind(pos, neg);
        bind(neg, pos);
        form.addEventListener('submit', function(ev) {{
          var submitter = ev.submitter;
          if (!submitter || submitter.getAttribute('formaction') === '/agree' || submitter.getAttribute('formaction') === '/skip') return;
          if (!requireReasonDomains(form)) {{
            ev.preventDefault();
            return;
          }}
          var decision = submitter.value || '';
          if (!decision) return;
          var posMarked = form.querySelectorAll('input[name="photo_positive_detail"]:checked').length;
          var negMarked = form.querySelectorAll('input[name="photo_negative_detail"]:checked').length;
          var asksPassWithPositive = decision === 'NÃO CURTIR' && posMarked > 0;
          var asksLikeWithNegative = (decision === 'CURTIR' || decision === 'SUPER_LIKE') && negMarked > 0;
          if (asksPassWithPositive || asksLikeWithNegative) {{
            var msg = asksPassWithPositive
              ? 'Você marcou pontos positivos na foto, mas escolheu passar. Salvar mesmo assim?'
              : 'Você marcou pontos negativos na foto, mas escolheu curtir/super like. Salvar mesmo assim?';
            if (!window.confirm(msg)) ev.preventDefault();
          }}
        }});
        form.querySelectorAll('input[name="descriptor_positive_detail"]').forEach(function(inp) {{
          inp.addEventListener('change', function() {{
            if (!inp.checked) return;
            form.querySelectorAll('input[name="descriptor_negative_detail"]').forEach(function(o) {{
              if (o.value === inp.value) o.checked = false;
            }});
          }});
        }});
        form.querySelectorAll('input[name="descriptor_negative_detail"]').forEach(function(inp) {{
          inp.addEventListener('change', function() {{
            if (!inp.checked) return;
            form.querySelectorAll('input[name="descriptor_positive_detail"]').forEach(function(o) {{
              if (o.value === inp.value) o.checked = false;
            }});
          }});
        }});
        _bindReviewAjax(form);
      }});
    }}

    _bindSimilarButtons(document);
    document.querySelectorAll('.review-form').forEach(_bindReviewAjax);
  </script>
  <script>
    (function() {{
      var activeDecision = 'all';
      var activePhoto = new Set();
      var activeDescs = new Set();
      var descSearch = '';

      function applyFilters() {{
        var cards = document.querySelectorAll('.card');
        var visible = 0;
        cards.forEach(function(card) {{
          var dec = card.dataset.decision;
          var hasPhoto = card.dataset.hasPhoto === '1';
          var hasBody = card.dataset.hasBody === '1';
          var descData = card.dataset.descriptors || '';

          var ok = true;
          if (activeDecision !== 'all' && dec !== activeDecision) ok = false;
          if (activePhoto.has('has-photo') && !hasPhoto) ok = false;
          if (activePhoto.has('has-body') && !hasBody) ok = false;
          activeDescs.forEach(function(d) {{ if (descData.indexOf(d) === -1) ok = false; }});
          if (descSearch && descData.indexOf(descSearch) === -1) ok = false;
          card.classList.toggle('filter-hidden', !ok);
          if (ok) visible++;
        }});
        var total = cards.length;
        var countEl = document.getElementById('filter-count');
        if (countEl) {{
          var isFiltered = activeDecision !== 'all' || activePhoto.size > 0 || activeDescs.size > 0 || descSearch;
          countEl.style.display = isFiltered ? '' : 'none';
          countEl.textContent = 'Mostrando ' + visible + ' de ' + total + ' perfis';
        }}
      }}

      document.querySelectorAll('.fchip[data-filter="decision"]').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          activeDecision = this.dataset.value;
          document.querySelectorAll('.fchip[data-filter="decision"]').forEach(function(b) {{
            b.classList.toggle('active', b.dataset.value === activeDecision);
          }});
          applyFilters();
        }});
      }});

      document.querySelectorAll('.fchip[data-filter="photo"]').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          var val = this.dataset.value;
          if (activePhoto.has(val)) {{ activePhoto.delete(val); this.classList.remove('active'); }}
          else {{ activePhoto.add(val); this.classList.add('active'); }}
          applyFilters();
        }});
      }});

      document.querySelectorAll('.desc-chip').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          var val = this.dataset.desc;
          if (activeDescs.has(val)) {{ activeDescs.delete(val); this.classList.remove('active'); }}
          else {{ activeDescs.add(val); this.classList.add('active'); }}
          applyFilters();
        }});
      }});

      var searchEl = document.getElementById('desc-search');
      if (searchEl) {{
        searchEl.addEventListener('input', function() {{
          descSearch = this.value.trim().toLowerCase();
          applyFilters();
        }});
      }}
    }})();
  </script>
  <script>
    function suppressDescriptor(key) {{
      fetch('/suppress-descriptor', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
        body: 'key=' + encodeURIComponent(key)
      }}).then(function() {{ location.reload(); }});
    }}
  </script>
  <script>
    function carouselGo(id, idx) {{
      var c = document.getElementById('carousel-' + id);
      if (!c) return;
      c.querySelectorAll('.cs').forEach(function(s, i) {{ s.classList.toggle('active', i === idx); }});
      c.querySelectorAll('.cdot').forEach(function(d, i) {{ d.classList.toggle('active', i === idx); }});
    }}
    function carouselStep(id, dir) {{
      var c = document.getElementById('carousel-' + id);
      if (!c) return;
      var slides = Array.from(c.querySelectorAll('.cs'));
      var cur = slides.findIndex(function(s) {{ return s.classList.contains('active'); }});
      carouselGo(id, (cur + dir + slides.length) % slides.length);
    }}
  </script>
  <div id="bt-lightbox" onclick="this.classList.remove('open')">
    <button id="bt-lightbox-close" onclick="document.getElementById('bt-lightbox').classList.remove('open')">✕</button>
    <img id="bt-lightbox-img" src="" alt="">
  </div>
  <script>
    function btLightbox(src) {{
      var lb = document.getElementById('bt-lightbox');
      document.getElementById('bt-lightbox-img').src = src;
      lb.classList.add('open');
    }}
    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') document.getElementById('bt-lightbox').classList.remove('open');
    }});

    // Set de IDs já exibidos ou em trânsito — evita duplicatas por race condition
    var _btSeen = new Set(
      Array.from(document.querySelectorAll('.bt-card[data-bt-id]'))
        .map(function(c) {{ return c.getAttribute('data-bt-id'); }})
    );

    function btAppendNext() {{
      fetch('/body-train-next?shown=' + encodeURIComponent(Array.from(_btSeen).join(',')))
        .then(function(r) {{ return r.text(); }})
        .then(function(html) {{
          var trimmed = html.trim();
          if (!trimmed) return;
          var div = document.createElement('div');
          div.innerHTML = trimmed;
          var newCard = div.firstElementChild;
          if (!newCard) return;
          var newId = newCard.getAttribute('data-bt-id');
          if (newId) _btSeen.add(newId); // registra imediatamente, antes de inserir no DOM
          var grid = document.querySelector('.bt-grid');
          if (!grid) return;
          newCard.style.opacity = '0';
          newCard.style.transform = 'scale(.97)';
          grid.appendChild(newCard);
          requestAnimationFrame(function() {{
            newCard.style.transition = 'opacity .25s, transform .25s';
            newCard.style.opacity = '1';
            newCard.style.transform = 'scale(1)';
          }});
        }});
    }}

    function btCardClick(btn, correction) {{
      var form = btn.closest('form');
      var card = form.closest('.bt-card');
      var params = new URLSearchParams(new FormData(form));
      params.set('correction', correction);
      btn.disabled = true;
      fetch('/body-train-save', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
        body: params.toString()
      }})
        .then(function(r) {{ return r.json(); }})
        .then(function(j) {{
          if (j.ok) {{
            card.style.transition = 'opacity .22s, transform .22s';
            card.style.opacity = '0';
            card.style.transform = 'scale(.96)';
            setTimeout(function() {{
              var removedId = card.getAttribute('data-bt-id');
              if (removedId) _btSeen.delete(removedId);
              card.remove();
              var pill = document.getElementById('bt-count-pill');
              if (pill) {{
                var n = parseInt(pill.textContent || '0') - 1;
                pill.textContent = n > 0 ? ' (' + n + ')' : '';
              }}
              btAppendNext();
            }}, 230);
          }} else {{
            btn.disabled = false;
          }}
        }})
        .catch(function() {{ btn.disabled = false; }});
    }}
  </script>
</body>
</html>"""
    return page.encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Página de configurações
# ──────────────────────────────────────────────────────────────────────────────

def _list_editor_html(title: str, items: list[str], field_section: str, field_name: str) -> str:
    rows = "".join(
        f"""<div class="list-item">
          <span>{_esc(item)}</span>
          <form method="post" action="/settings/update" style="display:inline">
            <input type="hidden" name="section" value="{_esc(field_section)}">
            <input type="hidden" name="field" value="{_esc(field_name)}">
            <input type="hidden" name="action" value="remove">
            <input type="hidden" name="value" value="{_esc(item)}">
            <button class="btn ghost btn-sm">✕</button>
          </form>
        </div>"""
        for item in items
    ) or '<p class="muted" style="font-family:ui-sans-serif;font-size:13px">Nenhum item.</p>'
    return f"""
    <div class="settings-section">
      <h2>{_esc(title)}</h2>
      <div class="list-editor">{rows}</div>
      <form method="post" action="/settings/update" class="list-add">
        <input type="hidden" name="section" value="{_esc(field_section)}">
        <input type="hidden" name="field" value="{_esc(field_name)}">
        <input type="hidden" name="action" value="add">
        <input type="text" name="value" placeholder="Adicionar novo item..." required>
        <button class="btn agree" style="white-space:nowrap">+ Adicionar</button>
      </form>
    </div>"""


def _render_settings_page(message: str = "", msg_type: str = "") -> bytes:
    prefs = _load_prefs()
    notice_html = ""
    if message:
        cls = {"ok": "ok", "err": "err"}.get(msg_type, "")
        notice_html = f'<div class="notice {cls}">{_esc(message)}</div>'

    age_min = prefs["age_min"]
    age_max = prefs["age_max"]

    page = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Configurações — Tinder-IA</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <div>
      <h1>⚙ Configurações</h1>
      <p class="subtitle">Edita o config.yaml em tempo real.</p>
    </div>
    <a href="/" class="btn ghost btn-sm" style="text-decoration:none">← Voltar às revisões</a>
  </header>
  <main>
    {notice_html}

    <div class="settings-section">
      <h2>Faixa de idade</h2>
      <form method="post" action="/settings/age" style="display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap">
        <label>Mínima <input type="number" name="age_min" value="{age_min}" min="18" max="99" style="width:100px"></label>
        <label>Máxima <input type="number" name="age_max" value="{age_max}" min="18" max="99" style="width:100px"></label>
        <button class="btn agree">Salvar</button>
      </form>
    </div>

    {_list_editor_html("Interesses preferidos", prefs["_raw_preferred_interests"], "preferences", "preferred_interests")}
    {_list_editor_html("Palavras positivas na bio", prefs["_raw_positive_bio_keywords"], "preferences", "positive_bio_keywords")}
    {_list_editor_html("Palavras negativas na bio", prefs["_raw_negative_bio_keywords"], "preferences", "negative_bio_keywords")}
    {_list_editor_html("Nomes a ignorar (disliked_names)", prefs["_raw_disliked_names"], "preferences", "disliked_names")}
    {_list_editor_html("Nomes masculinos extras (hard_filters)", prefs["_raw_extra_male_names"], "hard_filters", "extra_male_names")}
  </main>
</body>
</html>"""
    return page.encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# HTTP Handler
# ──────────────────────────────────────────────────────────────────────────────

class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("Review UI: " + fmt, *args)

    def _send_html(self, body: bytes, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Cliente fechou a conexão antes do envio completo da Review UI")

    def _send_json(self, body: dict, status: int = 200) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Cliente fechou a conexão antes do envio JSON da Review UI")

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Cliente fechou a conexão antes do envio de bytes da Review UI")

    def _redirect(
        self,
        message: str = "",
        msg_type: str = "",
        to: str = "/",
        extra_params: dict[str, str] | None = None,
    ) -> None:
        location = to
        params: dict[str, str] = {}
        if extra_params:
            params.update(extra_params)
        if message:
            params["msg"] = message
        if msg_type:
            params["mt"] = msg_type
        if params:
            sep = "&" if "?" in location else "?"
            location += sep + urllib.parse.urlencode(params)
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _read_form(self) -> dict[str, list[str]]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        parsed = urllib.parse.parse_qs(raw)
        return parsed

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    def _form_one(self, form: dict, key: str) -> str:
        vals = form.get(key, [])
        return vals[-1].strip() if vals else ""

    def _form_all(self, form: dict, key: str) -> list[str]:
        return form.get(key, [])

    def _signal_veto_details(self, form: dict) -> dict:
        fields = (
            "interest_not_positive",
            "interest_not_negative",
            "bio_not_positive",
            "bio_not_negative",
            "descriptor_not_positive",
            "descriptor_not_negative",
        )
        details = {}
        for key in fields:
            values = []
            seen = set()
            for raw in self._form_all(form, key):
                value = str(raw or "").strip()
                norm = _normalize(value)
                if not value or norm in seen:
                    continue
                seen.add(norm)
                values.append(value)
            if values:
                details[key] = values
        return details

    def _signal_veto_summary(self, details: dict) -> str:
        not_positive = []
        not_negative = []
        for key in ("interest_not_positive", "bio_not_positive", "descriptor_not_positive"):
            not_positive.extend(details.get(key, []))
        for key in ("interest_not_negative", "bio_not_negative", "descriptor_not_negative"):
            not_negative.extend(details.get(key, []))

        parts = []
        if not_positive:
            parts.append("não era positivo: " + ", ".join(not_positive[:8]))
        if not_negative:
            parts.append("não era negativo: " + ", ".join(not_negative[:8]))
        return "; ".join(parts)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        msg = qs.get("msg", [""])[0]
        mt = qs.get("mt", [""])[0]

        if parsed.path == "/":
            tab = (qs.get("tab", [""])[0] or "").strip().lower()
            if tab not in ("photo-deep", "body-train", "signal-train"):
                tab = ""
            initial_tab = tab
            selected = qs.get("selected", [""])[0]
            show_all = (qs.get("all", [""])[0] or "").strip() == "1"
            sort_mode = (qs.get("sort", ["priority"])[0] or "priority").strip()
            try:
                bt_page = max(0, int(qs.get("bt_page", ["0"])[0]))
            except Exception:
                bt_page = 0
            signal_type = (qs.get("signal_type", ["all"])[0] or "all").strip().lower()
            if signal_type not in ("all", "interest", "bio", "descriptor"):
                signal_type = "all"
            try:
                signal_page = max(0, int(qs.get("signal_page", ["0"])[0]))
            except Exception:
                signal_page = 0
            if selected and not is_allowed_photo_rel(selected):
                selected = ""
            self._send_html(_render_page(
                msg,
                mt,
                initial_tab=initial_tab,
                selected_photo=selected,
                show_all=show_all,
                sort_mode=sort_mode,
                bt_page=bt_page,
                signal_type=signal_type,
                signal_page=signal_page,
            ))
            return

        if parsed.path == "/settings":
            self._send_html(_render_settings_page(msg, mt))
            return

        if parsed.path == "/retrain-status":
            self._send_json(_get_retrain_status())
            return

        if parsed.path == "/review-next":
            shown_raw = qs.get("shown", [""])[0]
            shown = {x for x in shown_raw.split(",") if x}
            sort_mode = (qs.get("sort", ["priority"])[0] or "priority").strip()
            self._send_html(_render_next_review_card(shown, sort_mode=sort_mode))
            return

        if parsed.path == "/review-card":
            review_id = qs.get("review_id", [""])[0]
            row = _review_row_by_id(review_id, load_reviews("pending"))
            html = _render_card(row, _load_prefs()) if row else ""
            self._send_html(html.encode("utf-8"))
            return

        if parsed.path == "/review-similar":
            review_id = qs.get("review_id", [""])[0]
            mode = (qs.get("mode", ["manual"])[0] or "manual").strip()
            row = _review_row_by_id(review_id, load_reviews(None))
            html = _render_similar_panel(row, mode=mode) if row else ""
            self._send_html(html.encode("utf-8"))
            return

        if parsed.path == "/photo":
            rel = qs.get("path", [""])[0]
            path = (ROOT_DIR / rel).resolve()
            try:
                path.relative_to(ROOT_DIR / "data" / "photos")
            except ValueError:
                self.send_error(403)
                return
            if not path.exists():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self._send_bytes(data, content_type)
            return

        if parsed.path == "/body-train-next":
            shown_raw = qs.get("shown", [""])[0]
            shown = set(shown_raw.split(",")) if shown_raw else set()
            profiles, _ = _body_train_profiles(page=0, per_page=500)
            card_html = ""
            for row in profiles:
                bt_id = f"{(row.get('name') or '').strip()}|{(row.get('age') or '').strip()}"
                if bt_id not in shown:
                    card_html = _render_bt_card(row, page=0)
                    break
            data = card_html.encode("utf-8")
            self._send_bytes(data, "text/html; charset=utf-8")
            return

        if parsed.path == "/signal-train-next":
            signal_type = (qs.get("signal_type", ["all"])[0] or "all").strip().lower()
            if signal_type not in ("all", "interest", "bio", "descriptor"):
                signal_type = "all"
            shown = _parse_signal_shown_keys(qs.get("shown", []))
            card_html = ""
            for item in signal_training_candidates(signal_type, limit=600):
                key = _signal_train_key(item)
                if key not in shown:
                    card_html = _render_signal_train_card(item, signal_type, page=0)
                    break
            self._send_html(card_html.encode("utf-8"))
            return

        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/signal-train-batch":
            try:
                payload = self._read_json()
                items = payload.get("items") if isinstance(payload, dict) else []
                if not isinstance(items, list):
                    items = []
                result = append_signal_feedback_batch(items)
                saved = int(result.get("saved", 0) or 0)
                errors = result.get("errors") or []
                msg = f"{saved} sinal(is) salvo(s) no SQLite em um lote."
                if errors:
                    msg += f" Ignorados: {len(errors)}."
                self._send_json({"ok": saved > 0, "message": msg, **result}, status=200 if saved > 0 else 400)
            except Exception as exc:
                logger.exception("Falha ao salvar lote de treino de sinais")
                self._send_json({"ok": False, "message": f"Falha ao salvar lote: {exc}"}, status=400)
            return

        form = self._read_form()

        if self.path == "/signal-train-save":
            ajax = self._form_one(form, "ajax") == "1"
            signal_type = self._form_one(form, "signal_type").strip().lower()
            signal_value = self._form_one(form, "signal_value").strip()
            polarity = self._form_one(form, "polarity").strip().lower()
            signal_filter = self._form_one(form, "signal_filter").strip().lower() or "all"
            signal_page = self._form_one(form, "signal_page").strip() or "0"
            if signal_filter not in ("all", "interest", "bio", "descriptor"):
                signal_filter = "all"
            extra = {"tab": "signal-train", "signal_type": signal_filter, "signal_page": signal_page}
            try:
                context = {
                    "occurrences": self._form_one(form, "occurrences"),
                    "likes": self._form_one(form, "likes"),
                    "dislikes": self._form_one(form, "dislikes"),
                }
                record = append_signal_feedback(signal_type, signal_value, polarity, context=context)
                msg = (
                    f"Sinal salvo: {_signal_type_label(signal_type)} · "
                    f"{signal_value[:80]} · {_SIGNAL_POLARITY_LABELS.get(polarity, polarity)}."
                )
                if ajax:
                    self._send_json({"ok": True, "message": msg, "type": "ok", "record": record})
                    return
                self._redirect(msg, "ok", extra_params=extra)
            except Exception as exc:
                logger.exception("Falha ao salvar treino de sinal")
                if ajax:
                    self._send_json({"ok": False, "message": f"Falha ao salvar sinal: {exc}", "type": "err"}, status=400)
                    return
                self._redirect(f"Falha ao salvar sinal: {exc}", "err", extra_params=extra)
            return

        if self.path == "/apply":
            ajax = self._form_one(form, "ajax") == "1"
            review_id = self._form_one(form, "review_id")
            raw_final_decision = self._form_one(form, "final_decision")
            target_action = "super_like" if raw_final_decision == "SUPER_LIKE" else ""
            final_decision = "CURTIR" if target_action else raw_final_decision
            primary_domain = self._form_one(form, "feedback_domain")
            allowed_domains = {"photo", "interests", "bio", "descriptors", "other"}
            selected_domains: list[str] = []
            for raw_domain in self._form_all(form, "also"):
                domain_value = raw_domain.strip().lower()
                if domain_value in allowed_domains and domain_value not in selected_domains:
                    selected_domains.append(domain_value)
            primary_domain = primary_domain.strip().lower()
            if primary_domain not in allowed_domains:
                primary_domain = selected_domains[0] if selected_domains else ""
            elif primary_domain not in selected_domains:
                selected_domains.insert(0, primary_domain)
            else:
                selected_domains = [primary_domain] + [d for d in selected_domains if d != primary_domain]
            if not primary_domain:
                msg = "Marque ao menos um sinal que pesou nessa decisão."
                if ajax:
                    self._send_json({"ok": False, "message": msg, "type": "err"}, status=400)
                else:
                    self._redirect(msg, "err")
                return
            photo_detail = self._form_one(form, "photo_detail")
            photo_positive_details = [x.strip() for x in self._form_all(form, "photo_positive_detail") if x.strip()]
            photo_negative_details = [x.strip() for x in self._form_all(form, "photo_negative_detail") if x.strip()]
            # Mesmo aspecto não pode ser a favor e contra ao mesmo tempo
            pos_set = set(photo_positive_details)
            photo_negative_details = [x for x in photo_negative_details if x not in pos_set]
            selected_interests = [x.strip() for x in self._form_all(form, "interest_detail") if x.strip()]
            bio_detail = self._form_one(form, "bio_detail")
            intensity = self._form_one(form, "feedback_intensity")
            also = selected_domains
            descriptor_detail = self._form_one(form, "descriptor_detail")
            descriptor_positive_details = [x.strip() for x in self._form_all(form, "descriptor_positive_detail") if x.strip()]
            descriptor_negative_details = [x.strip() for x in self._form_all(form, "descriptor_negative_detail") if x.strip()]
            desc_pos_set = set(descriptor_positive_details)
            descriptor_negative_details = [x for x in descriptor_negative_details if x not in desc_pos_set]
            feedback_note = self._form_one(form, "feedback_note")
            photo_score_adjustment = self._form_one(form, "photo_score_adjustment")
            if photo_score_adjustment not in {"higher", "lower"}:
                photo_score_adjustment = ""
            photo_score_reason = self._form_one(form, "photo_score_reason")
            if photo_score_reason not in {"photo_general", "photo_face", "photo_gender", "photo_body", "photo_context", "photo_style"}:
                photo_score_reason = photo_detail or "photo_general"
            photo_score_intensity = self._form_one(form, "photo_score_intensity")
            if photo_score_intensity not in {"1", "2", "3"}:
                photo_score_intensity = intensity if intensity in {"1", "2", "3"} else "2"
            visual_face_label = _clean_visual_label(self._form_one(form, "visual_face_label"))
            visual_body_label = _clean_visual_label(self._form_one(form, "visual_body_label"))
            visual_style_label = _clean_visual_label(self._form_one(form, "visual_style_label"))
            visual_overall_label = _clean_visual_label(self._form_one(form, "visual_overall_label"))
            visual_label_summary = [
                f"{label}={value}"
                for label, value in (
                    ("rosto", visual_face_label),
                    ("corpo", visual_body_label),
                    ("estilo", visual_style_label),
                    ("geral", visual_overall_label),
                )
                if value
            ]
            signal_veto_details = self._signal_veto_details(form)
            body_frame_correction = ""
            body_build_correction = ""

            # Primary reason keeps the old canonical fields useful; detailed
            # signals go to feedback_details for newer learning/audit.
            if primary_domain == "photo" and photo_detail:
                feedback_reason = photo_detail
            elif primary_domain == "interests" and selected_interests:
                feedback_reason = ", ".join(selected_interests)
            elif primary_domain == "bio" and bio_detail:
                feedback_reason = bio_detail
            elif primary_domain == "descriptors":
                descriptor_parts = []
                if descriptor_detail:
                    descriptor_parts.append(descriptor_detail)
                if descriptor_positive_details:
                    descriptor_parts.append("gosto: " + ", ".join(descriptor_positive_details))
                if descriptor_negative_details:
                    descriptor_parts.append("não gosto: " + ", ".join(descriptor_negative_details))
                feedback_reason = "; ".join(descriptor_parts)
            elif primary_domain == "other" and feedback_note:
                feedback_reason = feedback_note
            else:
                feedback_reason = ""

            # Append secondary reasons
            secondary = [d for d in also if d != primary_domain]
            if secondary:
                feedback_reason = (feedback_reason or primary_domain) + "; também: " + ", ".join(secondary)
            if photo_score_adjustment:
                direction = "subir" if photo_score_adjustment == "higher" else "baixar"
                feedback_reason = (
                    feedback_reason or primary_domain or "ajuste de foto"
                ) + f"; score_foto_{direction}: {photo_score_reason}"
            if visual_label_summary:
                feedback_reason = (
                    feedback_reason or primary_domain or "avaliação visual"
                ) + "; visual: " + ", ".join(visual_label_summary)
            veto_summary = self._signal_veto_summary(signal_veto_details)
            if veto_summary:
                feedback_reason = (feedback_reason or primary_domain or "correção de sinais") + "; " + veto_summary

            details = {
                "target_action": target_action,
                "primary_domain": primary_domain,
                "selected_domains": selected_domains,
                "secondary_domains": secondary,
                "photo_reason": photo_detail,
                "photo_score_adjustment": photo_score_adjustment,
                "photo_score_reason": photo_score_reason if photo_score_adjustment else "",
                "photo_score_intensity": photo_score_intensity if photo_score_adjustment else "",
                "photo_positive_details": photo_positive_details,
                "photo_negative_details": photo_negative_details,
                "visual_face_label": visual_face_label,
                "visual_body_label": visual_body_label,
                "visual_style_label": visual_style_label,
                "visual_overall_label": visual_overall_label,
                "selected_interests": selected_interests,
                "bio_detail": bio_detail,
                "descriptor_detail": descriptor_detail,
                "descriptor_positive_details": descriptor_positive_details,
                "descriptor_negative_details": descriptor_negative_details,
                "body_frame_correction": body_frame_correction,
                "body_build_correction": body_build_correction,
                "note": feedback_note,
            }
            details.update(signal_veto_details)
            details = {k: v for k, v in details.items() if v not in ("", [], {})}
            if target_action:
                feedback_reason = ("super_like: " + (feedback_reason or primary_domain or "sinal forte")).strip()
            feedback_details = json.dumps(details, ensure_ascii=False, sort_keys=True)

            source_row_for_similar = _review_row_by_id(review_id, load_reviews(None))
            offer_similar = _should_offer_visual_calibration(source_row_for_similar, final_decision, details)
            ok = apply_review(
                review_id=review_id,
                final_decision=final_decision,
                feedback_domain=primary_domain,
                feedback_reason=feedback_reason,
                feedback_intensity=intensity,
                feedback_secondary=",".join(secondary),
                feedback_details=feedback_details,
            )
            if ok:
                _maybe_start_review_auto_retrain("apply_review")
            msg = "Revisão salva no treino." if ok else "Revisão não encontrada."
            if ok and veto_summary:
                msg += " Correções salvas: " + veto_summary[:180]
            mt = "ok" if ok else "err"
            if ajax:
                similar_html = (
                    _render_similar_panel(
                        source_row_for_similar,
                        mode="calibration",
                        feedback_details=details,
                        final_decision=final_decision,
                    )
                    if ok and offer_similar and source_row_for_similar else
                    ""
                )
                if similar_html:
                    msg += " Abri parecidos para calibrar esse erro confiante."
                self._send_json({
                    "ok": ok,
                    "message": msg,
                    "review_id": review_id,
                    "type": mt,
                    "similar_html": similar_html,
                })
                return
            self._redirect(msg, mt)
            return

        if self.path == "/photo-deep-save":
            if not _photo_deep_enabled():
                self._redirect("Rotulagem visual extra está desativada no config.yaml.", "err")
                return
            photo_path = self._form_one(form, "deep_photo_path")
            if not is_allowed_photo_rel(photo_path):
                self._redirect("Caminho de foto inválido.", "err", extra_params={"tab": "photo-deep"})
                return
            impression = self._form_one(form, "deep_impression")
            if impression not in ("like", "neutral", "dislike", ""):
                impression = ""
            alignment = self._form_one(form, "deep_alignment")
            if alignment not in ("", "0", "1", "2", "3", "4"):
                alignment = ""
            pos = [x.strip() for x in self._form_all(form, "deep_pos") if x.strip()]
            neg = [x.strip() for x in self._form_all(form, "deep_neg") if x.strip()]
            note = self._form_one(form, "deep_note")
            after = self._form_one(form, "deep_after")
            has_signal = bool(impression) or bool(pos) or bool(neg) or len(note.strip()) >= 4
            if not has_signal:
                self._redirect(
                    "Escolha impressão geral, marque detalhes ou escreva uma nota (mín. 4 caracteres).",
                    "err",
                    extra_params={"tab": "photo-deep"},
                )
                return
            try:
                append_deep_record(photo_path, impression, alignment, pos, neg, note)
                next_path = photo_path
                if after == "next":
                    next_path = next_unreviewed_photo(photo_path, list_saved_photo_paths(None, body_only=True))
                self._redirect(
                    "Registro extra salvo em data/photo_deep_feedback.jsonl. Treino principal inalterado.",
                    "ok",
                    extra_params={"tab": "photo-deep", "selected": next_path},
                )
            except Exception:
                logger.exception("photo-deep-save falhou")
                self._redirect("Erro ao salvar. Tente de novo.", "err", extra_params={"tab": "photo-deep"})
            return

        if self.path == "/photo-deep-train":
            if not _photo_deep_enabled():
                self._redirect("Treino visual extra está desativado no config.yaml.", "err")
                return

            def _run_photo_deep_train():
                try:
                    result = train_photo_deep_model()
                    if result.get("ok"):
                        logger.info("Treino visual extra concluído: %s", result)
                    else:
                        logger.warning("Treino visual extra não executado: %s", result)
                except Exception:
                    logger.exception("Erro no treino visual extra")

            threading.Thread(target=_run_photo_deep_train, daemon=True).start()
            self._redirect(
                "Treino visual extra iniciado em background. Veja o log para o resultado.",
                "ok",
                extra_params={"tab": "photo-deep"},
            )
            return

        if self.path == "/agree":
            ajax = self._form_one(form, "ajax") == "1"
            # Concordar com IA: salva a decisão original sem correção.
            # Se houver ajuste visual opcional, ele treina só o submodelo de foto.
            review_id = self._form_one(form, "review_id")
            signal_veto_details = self._signal_veto_details(form)
            photo_score_adjustment = self._form_one(form, "photo_score_adjustment")
            if photo_score_adjustment not in {"higher", "lower"}:
                photo_score_adjustment = ""
            photo_score_reason = self._form_one(form, "photo_score_reason")
            if photo_score_reason not in {"photo_general", "photo_face", "photo_gender", "photo_body", "photo_context", "photo_style"}:
                photo_score_reason = "photo_general"
            photo_score_intensity = self._form_one(form, "photo_score_intensity")
            if photo_score_intensity not in {"1", "2", "3"}:
                photo_score_intensity = "2"
            visual_face_label = _clean_visual_label(self._form_one(form, "visual_face_label"))
            visual_body_label = _clean_visual_label(self._form_one(form, "visual_body_label"))
            visual_style_label = _clean_visual_label(self._form_one(form, "visual_style_label"))
            visual_overall_label = _clean_visual_label(self._form_one(form, "visual_overall_label"))
            reviews = load_reviews(None)
            original_decision = "NÃO CURTIR"
            for r in reviews:
                if r.get("review_id") == review_id:
                    original_decision = r.get("final_decision") or r.get("ai_decision", "NÃO CURTIR")
                    break
            body_frame_correction = ""
            body_build_correction = ""
            details = {
                "quick_agree": True,
                "photo_score_adjustment": photo_score_adjustment,
                "photo_score_reason": photo_score_reason if photo_score_adjustment else "",
                "photo_score_intensity": photo_score_intensity if photo_score_adjustment else "",
                "visual_face_label": visual_face_label,
                "visual_body_label": visual_body_label,
                "visual_style_label": visual_style_label,
                "visual_overall_label": visual_overall_label,
                "body_frame_correction": body_frame_correction,
                "body_build_correction": body_build_correction,
            }
            if str(original_decision).strip().upper() in {"SUPER_LIKE", "SUPER LIKE"}:
                details["target_action"] = "super_like"
            details.update(signal_veto_details)
            inferred_domains = _domains_from_feedback_details(details)
            primary_domain = inferred_domains[0] if inferred_domains else "other"
            secondary_domains = inferred_domains[1:]
            if inferred_domains:
                details["primary_domain"] = primary_domain
                details["selected_domains"] = inferred_domains
                if secondary_domains:
                    details["secondary_domains"] = secondary_domains
            feedback_reason = (
                "concordou_com_super_like_ia"
                if details.get("target_action") == "super_like"
                else "concordou_com_ia"
            )
            if photo_score_adjustment:
                direction = "subir" if photo_score_adjustment == "higher" else "baixar"
                feedback_reason += f"; score_foto_{direction}: {photo_score_reason}"
            visual_label_summary = [
                f"{label}={value}"
                for label, value in (
                    ("rosto", visual_face_label),
                    ("corpo", visual_body_label),
                    ("estilo", visual_style_label),
                    ("geral", visual_overall_label),
                )
                if value
            ]
            if visual_label_summary:
                feedback_reason += "; visual: " + ", ".join(visual_label_summary)
            veto_summary = self._signal_veto_summary(signal_veto_details)
            if veto_summary:
                feedback_reason += "; " + veto_summary
            details = {k: v for k, v in details.items() if v not in ("", [], {})}
            ok = apply_review(
                review_id=review_id,
                final_decision=original_decision,
                feedback_domain=primary_domain,
                feedback_reason=feedback_reason,
                feedback_intensity="1",
                feedback_secondary=",".join(secondary_domains),
                feedback_details=json.dumps(details, ensure_ascii=False, sort_keys=True),
            )
            if ok:
                _maybe_start_review_auto_retrain("agree_review")
            msg = "Concordância com IA salva." if ok else "Revisão não encontrada."
            if ok and veto_summary:
                msg += " Correções salvas: " + veto_summary[:180]
            mt = "ok" if ok else "err"
            if ajax:
                self._send_json({"ok": ok, "message": msg, "review_id": review_id, "type": mt})
                return
            self._redirect(msg, mt)
            return

        if self.path == "/skip":
            # skip sem salvar (realmente ignorar)
            ajax = self._form_one(form, "ajax") == "1"
            review_id = self._form_one(form, "review_id")
            ok = skip_review(review_id)
            msg = "Revisão pulada (não salva no treino)." if ok else "Revisão não encontrada."
            if ajax:
                self._send_json({"ok": ok, "message": msg, "review_id": review_id, "type": "ok" if ok else "err"})
                return
            self._redirect(msg, "ok" if ok else "err")
            return

        if self.path == "/undo":
            review_id = self._form_one(form, "review_id")
            ok = undo_review(review_id)
            msg = "Revisão desfeita — voltou para pendentes." if ok else "Não foi possível desfazer."
            mt = "ok" if ok else "err"
            self._redirect(msg, mt)
            return

        if self.path == "/retrain":
            redirect_tab = self._form_one(form, "redirect_tab").strip()
            extra: dict[str, str] = {"tab": redirect_tab} if redirect_tab else {}
            if not _RETRAIN_LOCK.acquire(blocking=False):
                self._redirect(
                    "Retreino já está em andamento. Aguarde ele terminar antes de iniciar outro.",
                    "err",
                    extra_params=extra,
                )
                return

            _mark_retrain_started()

            def _run():
                try:
                    logger.info("Retreino iniciado pela UI de revisao")
                    train_model()
                    logger.info("Retreino pela UI concluido")
                    _mark_retrain_finished(True)
                except Exception as exc:
                    logger.exception("Erro no retreino pela UI")
                    _mark_retrain_finished(False, str(exc))
                finally:
                    _RETRAIN_LOCK.release()

            threading.Thread(target=_run, daemon=True).start()
            self._redirect("Retreino iniciado em background.", "ok", extra_params=extra)
            return

        if self.path == "/settings/update":
            section = self._form_one(form, "section")
            field = self._form_one(form, "field")
            action = self._form_one(form, "action")
            value = self._form_one(form, "value")
            ok = _config_list_update(section, field, action, value)
            msg = f"'{value}' {'adicionado' if action == 'add' else 'removido'} com sucesso." if ok else "Operação falhou."
            mt = "ok" if ok else "err"
            self._redirect(msg, mt, to="/settings")
            return

        if self.path == "/settings/age":
            try:
                age_min = int(self._form_one(form, "age_min"))
                age_max = int(self._form_one(form, "age_max"))
                ok = _config_age_update(age_min, age_max)
                msg = f"Faixa de idade atualizada: [{age_min}, {age_max}]." if ok else "Falha ao salvar."
                mt = "ok" if ok else "err"
            except ValueError:
                msg, mt = "Valores de idade inválidos.", "err"
            self._redirect(msg, mt, to="/settings")
            return

        if self.path == "/skip-all":
            count = skip_all_pending()
            self._redirect(f"{count} perfis pendentes descartados. Retreine o modelo e os novos perfis serão avaliados com o modelo atualizado.", "ok")
            return

        if self.path == "/enqueue-history":
            try:
                limit = max(1, min(200, int(self._form_one(form, "limit") or "40")))
            except Exception:
                limit = 40
            result = enqueue_history_review_candidates(limit=limit)
            msg = (
                f"{result.get('enqueued', 0)} perfil(is) históricos adicionados à revisão "
                f"de {result.get('scanned', 0)} analisados."
            )
            if result.get("skipped_no_photo"):
                msg += f" Sem foto local: {result.get('skipped_no_photo')}."
            self._redirect(msg, "ok" if result.get("enqueued", 0) else "err")
            return

        if self.path == "/suppress-descriptor":
            key = self._form_one(form, "key").strip()
            ok = _config_list_update("preferences", "desc_neg_suppressed", "add", key) if key else False
            msg = f"Descritor '{key}' banido dos negativos permanentemente." if ok else ("Já estava banido." if key else "Chave inválida.")
            self._redirect(msg, "ok" if ok else "err")
            return

        if self.path == "/add-male-name":
            name = self._form_one(form, "name").strip()
            ok = _config_list_update("hard_filters", "extra_male_names", "add", name) if name else False
            msg = f"'{name}' adicionado aos nomes masculinos." if ok else ("Nome já está na lista." if name else "Nome inválido.")
            self._redirect(msg, "ok" if ok else "err")
            return

        if self.path == "/body-train-save":
            bt_name = self._form_one(form, "bt_name").strip()
            bt_age_raw = self._form_one(form, "bt_age").strip()
            correction = self._form_one(form, "correction").strip()
            bt_page_raw = self._form_one(form, "bt_page").strip() or "0"
            is_ajax = self._form_one(form, "ajax") == "1"
            redirect_params: dict[str, str] = {"tab": "body-train", "bt_page": bt_page_raw}

            def _ajax_err(msg: str):
                self._send_json({"ok": False, "msg": msg}, status=400)

            if correction not in ("estreita", "media_estreita", "media", "media_ampla", "ampla", "skip", "no_body"):
                if is_ajax: _ajax_err("Correção inválida."); return
                self._redirect("Correção inválida.", "err", extra_params=redirect_params)
                return

            try:
                bt_age = int(float(bt_age_raw))
            except Exception:
                if is_ajax: _ajax_err("Idade inválida."); return
                self._redirect("Idade inválida.", "err", extra_params=redirect_params)
                return

            from model import PROFILES_PATH, CSV_FIELDNAMES as _CSV_FN
            if not PROFILES_PATH.exists():
                if is_ajax: _ajax_err("profiles.csv não encontrado."); return
                self._redirect("profiles.csv não encontrado.", "err", extra_params=redirect_params)
                return

            with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
                rows = list(_csv_module.DictReader(f))

            updated = 0
            for row in rows:
                if row.get("name", "").strip() != bt_name:
                    continue
                try:
                    if int(float(row.get("age", 0) or 0)) != bt_age:
                        continue
                except Exception:
                    continue
                try:
                    details = json.loads(row.get("feedback_details") or "{}")
                except Exception:
                    details = {}
                if details.get("body_build_correction"):
                    continue
                details["body_build_correction"] = correction
                inferred_domains = _domains_from_feedback_details(details)
                if inferred_domains and row.get("feedback_domain", "").strip().lower() in {"", "other"}:
                    row["feedback_domain"] = inferred_domains[0]
                    row["feedback_secondary"] = ",".join(inferred_domains[1:])
                    reason = row.get("feedback_reason", "").strip()
                    if not reason or reason in {"concordou_com_ia", "concordou_com_super_like_ia", "other"}:
                        row["feedback_reason"] = f"{reason or 'concordou_com_ia'}; correcao_corpo: {correction}"
                    details["primary_domain"] = inferred_domains[0]
                    details["selected_domains"] = inferred_domains
                    if inferred_domains[1:]:
                        details["secondary_domains"] = inferred_domains[1:]
                row["feedback_details"] = json.dumps(details, ensure_ascii=False)
                updated += 1

            if updated:
                with open(PROFILES_PATH, "w", encoding="utf-8", newline="") as f:
                    writer = _csv_module.DictWriter(f, fieldnames=_CSV_FN)
                    writer.writeheader()
                    writer.writerows(rows)
                _bt_clear_photo_index()
                logger.info("body-train-save: %s → %s (%d linhas)", bt_name, correction, updated)

            if is_ajax:
                self._send_json({"ok": True})
                return

            if correction == "skip":
                msg = ""
            elif correction == "no_body":
                msg = f"Marcado 'sem corpo' para {bt_name} — sinais corporais zerados no treino." if updated else f"{bt_name} já tinha correção ou não foi encontrado."
            else:
                msg = f"Silhueta '{correction}' salva para {bt_name}." if updated else f"{bt_name} já tinha correção ou não foi encontrado."
            self._redirect(msg, "ok" if (updated or correction == "skip") else "err", extra_params=redirect_params)
            return

        self.send_error(404)


# ──────────────────────────────────────────────────────────────────────────────
# Entrada principal
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    hidden_filters, hidden_duplicates = _cleanup_review_queue_once()
    server = ReviewHTTPServer(("localhost", PORT), ReviewHandler)
    print("\n  Tinder-IA — Revisão pós-swipe")
    print("  --------------------------------------------")
    if hidden_filters:
        print(f"  Filtros absolutos ocultados: {hidden_filters}")
    if hidden_duplicates:
        print(f"  Duplicados ocultados: {hidden_duplicates}")
    print(f"  Abra: http://localhost:{PORT}")
    print(f"  Configurações: http://localhost:{PORT}/settings")
    print("  Ctrl+C para encerrar\n")
    logger.info("Review UI ouvindo em localhost:%s", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Revisão encerrada.\n")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
