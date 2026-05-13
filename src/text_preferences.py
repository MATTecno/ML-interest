"""
Aprendizado simples de preferências de texto a partir dos perfis reais rotulados.

Objetivo:
  - aprender quais interesses, descritores e palavras de bio aparecem mais nos perfis curtidos
  - gerar um score 0..1 para um novo perfil sem depender de listas fixas
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from logging_config import get_logger

PROFILES_PATH = Path(__file__).parent.parent / "data" / "profiles.csv"
logger = get_logger(__name__)

_CACHE = {
    "mtime_ns": None,
    "size": None,
    "tables": None,
}

VETO_TABLE_NAMES = (
    "interest_not_positive",
    "interest_not_negative",
    "bio_not_positive",
    "bio_not_negative",
    "descriptor_not_positive",
    "descriptor_not_negative",
)

_STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na",
    "nos", "nas", "um", "uma", "uns", "umas", "com", "sem", "pra", "para", "por",
    "que", "me", "te", "se", "eu", "vc", "voce", "você", "ele", "ela", "eles",
    "elas", "sou", "ser", "estar", "to", "tô", "bem", "mais", "menos", "muito",
    "muita", "muitas", "muitos", "não", "nao", "sim", "mas", "ou", "ao", "aos",
    "minha", "meu", "meus", "minhas", "sua", "seu", "seus", "suas", "gosto",
    "curto", "amo", "vida", "aqui", "ali", "isso", "isto", "tipo", "tem",
    # "tenho" removido para permitir bigrams como "tenho filhos", "tenho filho"
}


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", (text or "").strip().lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", t).strip()


def _tokenize_bio(text: str) -> list[str]:
    normalized = _normalize(text)
    raw_tokens = re.findall(r"[a-z]{3,}", normalized)
    unigrams = [t for t in raw_tokens if t not in _STOPWORDS]
    bigrams = [
        f"{raw_tokens[i]} {raw_tokens[i + 1]}"
        for i in range(len(raw_tokens) - 1)
    ]
    return unigrams + bigrams


def _token_score(likes: int, dislikes: int) -> float:
    return (likes + 1) / (likes + dislikes + 2)


def _parse_feedback_details(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _detail_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def _feedback_domains(row: dict, details: dict) -> set[str]:
    allowed = {"photo", "interests", "bio", "descriptors", "other"}
    domains: set[str] = set()

    def add(value) -> None:
        domain = str(value or "").strip().lower()
        if domain in allowed:
            domains.add(domain)

    add(row.get("feedback_domain", ""))
    for item in _detail_list(details.get("selected_domains")):
        add(item)
    for item in _detail_list(details.get("secondary_domains")):
        add(item)
    for item in _detail_list(row.get("feedback_secondary", "")):
        add(item)
    return domains


def _parse_descriptors(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw in ("", None):
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _descriptor_items(raw) -> list[str]:
    descriptors = _parse_descriptors(raw)
    result = []
    for key, value in descriptors.items():
        label = f"{key}: {value}".strip()
        norm = _normalize(label)
        if norm:
            result.append(norm)
    return result


def _load_real_rows() -> list[dict]:
    if not PROFILES_PATH.exists():
        return []

    with open(PROFILES_PATH, encoding="utf-8", newline="") as f:
        try:
            rows = list(csv.DictReader(f))
        except Exception:
            logger.exception("Falha ao ler preferencias textuais: %s", PROFILES_PATH)
            raise

    real_rows = []
    for row in rows:
        if str(row.get("source", "real")).strip().lower() != "real":
            continue
        try:
            label = int(row.get("label", ""))
        except Exception:
            continue
        real_rows.append({
            "bio": row.get("bio", "") or "",
            "interests": row.get("interests", "") or "",
            "descriptors": row.get("descriptors", "") or "",
            "label": label,
            "feedback_domain": (row.get("feedback_domain", "") or "").strip().lower(),
            "feedback_details": row.get("feedback_details", "") or "",
        })
    return real_rows


def _build_tables() -> dict:
    interest_counts: dict[str, dict[str, int]] = {}
    bio_counts: dict[str, dict[str, int]] = {}
    descriptor_counts: dict[str, dict[str, int]] = {}
    veto_tables = {name: {} for name in VETO_TABLE_NAMES}

    def _add_veto(table_name: str, values) -> None:
        table = veto_tables[table_name]
        for item in _detail_list(values):
            norm = _normalize(item)
            if norm:
                table[norm] = table.get(norm, 0) + 1

    for row in _load_real_rows():
        label = row["label"]
        bucket = "likes" if label == 1 else "dislikes"
        details = _parse_feedback_details(row.get("feedback_details", ""))
        feedback_domains = _feedback_domains(row, details)

        _add_veto("interest_not_positive", details.get("interest_not_positive"))
        _add_veto("interest_not_negative", details.get("interest_not_negative"))
        _add_veto("bio_not_positive", details.get("bio_not_positive"))
        _add_veto("bio_not_negative", details.get("bio_not_negative"))
        _add_veto("descriptor_not_positive", details.get("descriptor_not_positive"))
        _add_veto("descriptor_not_negative", details.get("descriptor_not_negative"))

        # So aprende interesse quando o usuario disse explicitamente que
        # interesse pesou. Motivo vazio/sem certeza não deve contaminar texto.
        if "interests" in feedback_domains:
            selected = _detail_list(details.get("selected_interests"))
            source_interests = selected or str(row["interests"]).split(",")
            interests = [_normalize(x) for x in source_interests if _normalize(x)]
            for interest in set(interests):
                stats = interest_counts.setdefault(interest, {"likes": 0, "dislikes": 0})
                stats[bucket] += 1

        # Mesmo raciocinio: palavras da bio so entram quando bio foi o motivo.
        if "bio" in feedback_domains:
            bio_signal = str(details.get("bio_detail") or "").strip()
            tokens = set(_tokenize_bio(bio_signal or row["bio"]))
            for token in tokens:
                stats = bio_counts.setdefault(token, {"likes": 0, "dislikes": 0})
                stats[bucket] += 1

        descriptor_pos = set(_normalize(x) for x in _detail_list(details.get("descriptor_positive_details")))
        descriptor_neg = set(_normalize(x) for x in _detail_list(details.get("descriptor_negative_details")))
        for item in descriptor_pos:
            stats = descriptor_counts.setdefault(item, {"likes": 0, "dislikes": 0})
            stats["likes"] += 1
        for item in descriptor_neg - descriptor_pos:
            stats = descriptor_counts.setdefault(item, {"likes": 0, "dislikes": 0})
            stats["dislikes"] += 1

        if "descriptors" in feedback_domains and not descriptor_pos and not descriptor_neg:
            selected = _detail_list(details.get("descriptor_detail"))
            source_descriptors = [_normalize(x) for x in selected if _normalize(x)]
            if not source_descriptors:
                source_descriptors = _descriptor_items(row.get("descriptors", ""))
            for item in set(source_descriptors):
                stats = descriptor_counts.setdefault(item, {"likes": 0, "dislikes": 0})
                stats[bucket] += 1

    return {
        "interest_counts": interest_counts,
        "bio_counts": bio_counts,
        "descriptor_counts": descriptor_counts,
        **veto_tables,
    }


def _get_tables() -> dict:
    if not PROFILES_PATH.exists():
        return {
            "interest_counts": {},
            "bio_counts": {},
            "descriptor_counts": {},
            **{name: {} for name in VETO_TABLE_NAMES},
        }

    stat = PROFILES_PATH.stat()
    if (
        _CACHE["tables"] is not None
        and _CACHE["mtime_ns"] == stat.st_mtime_ns
        and _CACHE["size"] == stat.st_size
    ):
        return _CACHE["tables"]

    tables = _build_tables()
    logger.info(
        "Preferencias textuais atualizadas: interests=%s bio_tokens=%s descriptors=%s",
        len(tables["interest_counts"]),
        len(tables["bio_counts"]),
        len(tables["descriptor_counts"]),
    )
    _CACHE["mtime_ns"] = stat.st_mtime_ns
    _CACHE["size"] = stat.st_size
    _CACHE["tables"] = tables
    return tables


def signal_veto_tables(min_count: int = 1) -> dict[str, set[str]]:
    """Retorna os termos que o usuario pediu para neutralizar."""
    tables = _get_tables()
    return {
        name: {
            term
            for term, count in tables.get(name, {}).items()
            if int(count or 0) >= min_count
        }
        for name in VETO_TABLE_NAMES
    }


def _score_items(
    items: list[str],
    table: dict[str, dict[str, int]],
    prefix: str,
    positive_veto: dict[str, int] | None = None,
    negative_veto: dict[str, int] | None = None,
) -> dict:
    if not items:
        return {
            f"{prefix}_score": 0.5,
            f"{prefix}_positive_hits": 0,
            f"{prefix}_negative_hits": 0,
            f"{prefix}_positive_signals": [],
            f"{prefix}_negative_signals": [],
        }

    scores = []
    positive_hits = []
    negative_hits = []

    positive_veto = positive_veto or {}
    negative_veto = negative_veto or {}

    for item in items:
        stats = table.get(item)
        if not stats:
            continue

        likes = stats.get("likes", 0)
        dislikes = stats.get("dislikes", 0)
        total = likes + dislikes
        score = _token_score(likes, dislikes)
        veto_pos = positive_veto.get(item, 0)
        veto_neg = negative_veto.get(item, 0)
        if veto_pos:
            score = min(score, 0.55)
        if veto_neg:
            score = max(score, 0.45)
        if veto_pos and veto_neg:
            score = 0.5
        scores.append(score)

        if total >= 2 and score >= 0.70 and not veto_pos:
            positive_hits.append(item)
        elif total >= 2 and score <= 0.30 and not veto_neg:
            negative_hits.append(item)

    return {
        f"{prefix}_score": round(sum(scores) / len(scores), 4) if scores else 0.5,
        f"{prefix}_positive_hits": len(positive_hits),
        f"{prefix}_negative_hits": len(negative_hits),
        f"{prefix}_positive_signals": positive_hits[:5],
        f"{prefix}_negative_signals": negative_hits[:5],
    }


def score_profile_text(bio: str, interests: list[str] | None, descriptors=None) -> dict:
    tables = _get_tables()
    interest_items = [_normalize(x) for x in (interests or []) if _normalize(x)]
    bio_tokens = _tokenize_bio(bio)
    descriptor_items = _descriptor_items(descriptors)

    interest_data = _score_items(
        interest_items,
        tables["interest_counts"],
        "interest_pref",
        tables.get("interest_not_positive", {}),
        tables.get("interest_not_negative", {}),
    )
    bio_data = _score_items(
        bio_tokens,
        tables["bio_counts"],
        "bio_pref",
        tables.get("bio_not_positive", {}),
        tables.get("bio_not_negative", {}),
    )
    descriptor_data = _score_items(
        descriptor_items,
        tables["descriptor_counts"],
        "descriptor_pref",
        tables.get("descriptor_not_positive", {}),
        tables.get("descriptor_not_negative", {}),
    )

    if descriptor_data["descriptor_pref_score"] != 0.5:
        combined = round(
            interest_data["interest_pref_score"] * 0.45
            + bio_data["bio_pref_score"] * 0.35
            + descriptor_data["descriptor_pref_score"] * 0.20,
            4,
        )
    else:
        combined = round(
            interest_data["interest_pref_score"] * 0.55
            + bio_data["bio_pref_score"] * 0.45,
            4,
        )

    return {
        **interest_data,
        **bio_data,
        **descriptor_data,
        "text_preference_score": combined,
    }


def summary_snapshot() -> dict:
    tables = _get_tables()

    def top_items(table: dict[str, dict[str, int]], threshold: float, reverse: bool) -> list[str]:
        scored = []
        for item, stats in table.items():
            likes = stats.get("likes", 0)
            dislikes = stats.get("dislikes", 0)
            total = likes + dislikes
            if total < 2:
                continue
            score = _token_score(likes, dislikes)
            if reverse and score <= threshold:
                scored.append((score, total, item))
            elif not reverse and score >= threshold:
                scored.append((score, total, item))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=not reverse)
        return [item for _, _, item in scored[:8]]

    return {
        "liked_interests": top_items(tables["interest_counts"], 0.70, False),
        "disliked_interests": top_items(tables["interest_counts"], 0.30, True),
        "liked_bio_tokens": top_items(tables["bio_counts"], 0.70, False),
        "disliked_bio_tokens": top_items(tables["bio_counts"], 0.30, True),
        "liked_descriptors": top_items(tables["descriptor_counts"], 0.70, False),
        "disliked_descriptors": top_items(tables["descriptor_counts"], 0.30, True),
    }
