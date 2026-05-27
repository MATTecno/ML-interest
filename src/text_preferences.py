"""
Aprendizado simples de preferências de texto a partir dos perfis reais rotulados.

Objetivo:
  - aprender quais interesses, descritores e palavras de bio aparecem mais nos perfis curtidos
  - gerar um score 0..1 para um novo perfil sem depender de listas fixas
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from logging_config import get_logger

PROFILES_PATH = Path(__file__).parent.parent / "data" / "profiles.csv"
TEXT_SIGNAL_FEEDBACK_PATH = Path(__file__).parent.parent / "data" / "text_signal_feedback.jsonl"
TEXT_SIGNAL_FEEDBACK_DB_PATH = Path(__file__).parent.parent / "data" / "text_signal_feedback.sqlite"
logger = get_logger(__name__)

_CACHE = {
    "profiles_mtime_ns": None,
    "profiles_size": None,
    "signals_mtime_ns": None,
    "signals_size": None,
    "legacy_mtime_ns": None,
    "legacy_size": None,
    "tables": None,
}
_SQLITE_LOCK = threading.RLock()

SIGNAL_TYPES = ("interest", "bio", "descriptor")
SIGNAL_POLARITIES = ("positive", "neutral", "negative")
_SIGNAL_WEIGHT = 3

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

_BIO_SIGNAL_CANDIDATE_BLOCKLIST = {
    "tenho anos",
    "sou uma",
    "sou um",
    "estou aqui",
    "por aqui",
    "nao sei",
    "não sei",
    "tudo bem",
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


def _is_bio_signal_candidate(token: str) -> bool:
    normalized = _normalize(token)
    if " " not in normalized:
        return False
    parts = normalized.split()
    if len(parts) == 2 and (parts[0] in _STOPWORDS or parts[1] in _STOPWORDS):
        return False
    if normalized in _BIO_SIGNAL_CANDIDATE_BLOCKLIST:
        return False
    return len(normalized) >= 7


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


def _invalidate_cache() -> None:
    _CACHE["profiles_mtime_ns"] = None
    _CACHE["profiles_size"] = None
    _CACHE["signals_mtime_ns"] = None
    _CACHE["signals_size"] = None
    _CACHE["legacy_mtime_ns"] = None
    _CACHE["legacy_size"] = None
    _CACHE["tables"] = None


def _json_default(value):
    try:
        return str(value)
    except Exception:
        return ""


def _context_json(context: dict | None) -> str:
    return json.dumps(context or {}, ensure_ascii=False, sort_keys=True, default=_json_default)


def _legacy_record_key(record: dict) -> str:
    payload = {
        "created_at": record.get("created_at", ""),
        "signal_type": record.get("signal_type", ""),
        "signal_value": record.get("signal_value", record.get("value", "")),
        "polarity": record.get("polarity", ""),
        "context": record.get("context", {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sqlite_connect() -> sqlite3.Connection:
    TEXT_SIGNAL_FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TEXT_SIGNAL_FEEDBACK_DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            signal_value TEXT NOT NULL,
            signal_norm TEXT NOT NULL,
            polarity TEXT NOT NULL,
            weight INTEGER NOT NULL,
            context_json TEXT NOT NULL DEFAULT '{}',
            legacy_key TEXT UNIQUE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_signal_feedback_type_norm ON signal_feedback(signal_type, signal_norm)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_signal_feedback_created ON signal_feedback(created_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_feedback_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    return conn


def _meta_get(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT value FROM signal_feedback_meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else ""


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO signal_feedback_meta(key, value) VALUES(?, ?)",
        (key, value),
    )


def _clean_signal_record(record: dict) -> dict | None:
    signal_type = str(record.get("signal_type") or "").strip().lower()
    polarity = str(record.get("polarity") or "").strip().lower()
    value = str(record.get("signal_value") or record.get("value") or "").strip()
    if signal_type not in SIGNAL_TYPES or polarity not in SIGNAL_POLARITIES or not value:
        return None
    try:
        weight = int(record.get("weight") or _SIGNAL_WEIGHT)
    except Exception:
        weight = _SIGNAL_WEIGHT
    weight = max(1, min(20, weight))
    return {
        "created_at": str(record.get("created_at") or datetime.now(timezone.utc).isoformat()),
        "source": str(record.get("source") or "review_ui_signal_train"),
        "signal_type": signal_type,
        "signal_value": value,
        "signal_norm": _normalize(value),
        "polarity": polarity,
        "weight": weight,
        "context": record.get("context") if isinstance(record.get("context"), dict) else {},
    }


def _insert_signal_record(conn: sqlite3.Connection, record: dict, legacy_key: str | None = None) -> bool:
    clean = _clean_signal_record(record)
    if not clean or not clean["signal_norm"]:
        return False
    before = conn.total_changes
    conn.execute(
        """
        INSERT OR IGNORE INTO signal_feedback(
            created_at, source, signal_type, signal_value, signal_norm,
            polarity, weight, context_json, legacy_key
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clean["created_at"],
            clean["source"],
            clean["signal_type"],
            clean["signal_value"],
            clean["signal_norm"],
            clean["polarity"],
            clean["weight"],
            _context_json(clean["context"]),
            legacy_key,
        ),
    )
    return conn.total_changes > before


def _legacy_jsonl_signature() -> str:
    if not TEXT_SIGNAL_FEEDBACK_PATH.exists():
        return "missing"
    stat = TEXT_SIGNAL_FEEDBACK_PATH.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _import_legacy_jsonl(conn: sqlite3.Connection) -> int:
    if not TEXT_SIGNAL_FEEDBACK_PATH.exists():
        _meta_set(conn, "legacy_jsonl_signature", "missing")
        return 0
    signature = _legacy_jsonl_signature()
    if _meta_get(conn, "legacy_jsonl_signature") == signature:
        return 0

    imported = 0
    before = conn.total_changes
    with TEXT_SIGNAL_FEEDBACK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                logger.debug("Linha invalida em text_signal_feedback.jsonl ignorada", exc_info=True)
                continue
            if not isinstance(raw, dict):
                continue
            _insert_signal_record(conn, raw, legacy_key=_legacy_record_key(raw))
    imported = max(0, conn.total_changes - before)
    _meta_set(conn, "legacy_jsonl_signature", signature)
    if imported:
        logger.info("Feedback atomico legado importado para SQLite: %s registro(s)", imported)
    return imported


def _ensure_signal_db_imported() -> None:
    with _SQLITE_LOCK:
        with _sqlite_connect() as conn:
            _import_legacy_jsonl(conn)


def _signal_storage_signature() -> tuple[int, int, int, int]:
    _ensure_signal_db_imported()
    db_mtime = 0
    db_size = 0
    for path in (
        TEXT_SIGNAL_FEEDBACK_DB_PATH,
        TEXT_SIGNAL_FEEDBACK_DB_PATH.with_name(TEXT_SIGNAL_FEEDBACK_DB_PATH.name + "-wal"),
        TEXT_SIGNAL_FEEDBACK_DB_PATH.with_name(TEXT_SIGNAL_FEEDBACK_DB_PATH.name + "-shm"),
    ):
        if path.exists():
            stat = path.stat()
            db_mtime = max(db_mtime, stat.st_mtime_ns)
            db_size += stat.st_size
    if TEXT_SIGNAL_FEEDBACK_PATH.exists():
        legacy_stat = TEXT_SIGNAL_FEEDBACK_PATH.stat()
        legacy_mtime = legacy_stat.st_mtime_ns
        legacy_size = legacy_stat.st_size
    else:
        legacy_mtime = 0
        legacy_size = 0
    return db_mtime, db_size, legacy_mtime, legacy_size


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


def _descriptor_display_items(raw) -> list[tuple[str, str]]:
    descriptors = _parse_descriptors(raw)
    result = []
    for key, value in descriptors.items():
        label = f"{key}: {value}".strip()
        norm = _normalize(label)
        if norm:
            result.append((label, norm))
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
            "name": row.get("name", "") or "",
            "age": row.get("age", "") or "",
            "bio": row.get("bio", "") or "",
            "interests": row.get("interests", "") or "",
            "descriptors": row.get("descriptors", "") or "",
            "label": label,
            "feedback_domain": (row.get("feedback_domain", "") or "").strip().lower(),
            "feedback_details": row.get("feedback_details", "") or "",
        })
    return real_rows


def _load_signal_feedback_records() -> list[dict]:
    records: list[dict] = []
    with _SQLITE_LOCK:
        with _sqlite_connect() as conn:
            _import_legacy_jsonl(conn)
            rows = conn.execute(
                """
                SELECT created_at, source, signal_type, signal_value, signal_norm,
                       polarity, weight, context_json
                FROM signal_feedback
                ORDER BY id ASC
                """
            ).fetchall()
    for row in rows:
        try:
            context = json.loads(row["context_json"] or "{}")
            if not isinstance(context, dict):
                context = {}
        except Exception:
            context = {}
        records.append({
            "created_at": row["created_at"],
            "source": row["source"],
            "signal_type": row["signal_type"],
            "signal_value": row["signal_value"],
            "signal_norm": row["signal_norm"],
            "polarity": row["polarity"],
            "weight": int(row["weight"] or _SIGNAL_WEIGHT),
            "context": context,
        })
    return records


def append_signal_feedback(
    signal_type: str,
    signal_value: str,
    polarity: str,
    context: dict | None = None,
) -> dict:
    """Salva uma avaliacao atomica de interesse/bio/descritor sem alterar label de perfil."""
    clean_type = str(signal_type or "").strip().lower()
    clean_polarity = str(polarity or "").strip().lower()
    clean_value = str(signal_value or "").strip()
    if clean_type not in SIGNAL_TYPES:
        raise ValueError(f"signal_type invalido: {signal_type!r}")
    if clean_polarity not in SIGNAL_POLARITIES:
        raise ValueError(f"polarity invalido: {polarity!r}")
    if not clean_value:
        raise ValueError("signal_value vazio")

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "review_ui_signal_train",
        "signal_type": clean_type,
        "signal_value": clean_value,
        "signal_norm": _normalize(clean_value),
        "polarity": clean_polarity,
        "weight": _SIGNAL_WEIGHT,
        "context": context or {},
    }
    with _SQLITE_LOCK:
        with _sqlite_connect() as conn:
            _import_legacy_jsonl(conn)
            _insert_signal_record(conn, record)
    _invalidate_cache()
    logger.info(
        "Feedback atomico de sinal salvo: type=%s value=%r polarity=%s",
        clean_type,
        clean_value,
        clean_polarity,
    )
    return record


def append_signal_feedback_batch(items: list[dict]) -> dict:
    """Salva varios feedbacks atomicos em uma unica transacao SQLite."""
    saved = 0
    errors: list[str] = []
    records: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    with _SQLITE_LOCK:
        with _sqlite_connect() as conn:
            _import_legacy_jsonl(conn)
            for idx, item in enumerate(items or []):
                signal_type = str(item.get("signal_type") or "").strip().lower()
                polarity = str(item.get("polarity") or "").strip().lower()
                value = str(item.get("signal_value") or item.get("value") or "").strip()
                if signal_type not in SIGNAL_TYPES:
                    errors.append(f"{idx}: tipo invalido")
                    continue
                if polarity not in SIGNAL_POLARITIES:
                    errors.append(f"{idx}: polaridade invalida")
                    continue
                if not value:
                    errors.append(f"{idx}: valor vazio")
                    continue
                record = {
                    "created_at": now,
                    "source": "review_ui_signal_train_batch",
                    "signal_type": signal_type,
                    "signal_value": value,
                    "signal_norm": _normalize(value),
                    "polarity": polarity,
                    "weight": _SIGNAL_WEIGHT,
                    "context": item.get("context") if isinstance(item.get("context"), dict) else {},
                }
                if _insert_signal_record(conn, record):
                    saved += 1
                    records.append(record)
    if saved:
        _invalidate_cache()
        logger.info(
            "Feedback atomico em lote salvo: items=%s saved=%s errors=%s",
            len(items or []),
            saved,
            len(errors),
        )
    return {"saved": saved, "errors": errors, "records": records}


def signal_feedback_counts() -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for record in _load_signal_feedback_records():
        key = (record["signal_type"], record["signal_norm"])
        bucket = counts.setdefault(key, {"positive": 0, "neutral": 0, "negative": 0, "total": 0})
        bucket[record["polarity"]] += 1
        bucket["total"] += 1
    return counts


def _apply_signal_feedback(tables: dict) -> None:
    count_tables = {
        "interest": tables["interest_counts"],
        "bio": tables["bio_counts"],
        "descriptor": tables["descriptor_counts"],
    }
    not_positive_tables = {
        "interest": tables["interest_not_positive"],
        "bio": tables["bio_not_positive"],
        "descriptor": tables["descriptor_not_positive"],
    }
    not_negative_tables = {
        "interest": tables["interest_not_negative"],
        "bio": tables["bio_not_negative"],
        "descriptor": tables["descriptor_not_negative"],
    }

    for record in _load_signal_feedback_records():
        signal_type = record["signal_type"]
        norm = record["signal_norm"]
        polarity = record["polarity"]
        weight = int(record.get("weight") or _SIGNAL_WEIGHT)
        stats = count_tables[signal_type].setdefault(norm, {"likes": 0, "dislikes": 0})
        if polarity == "positive":
            stats["likes"] += weight
            not_negative_tables[signal_type][norm] = not_negative_tables[signal_type].get(norm, 0) + 1
        elif polarity == "negative":
            stats["dislikes"] += weight
            not_positive_tables[signal_type][norm] = not_positive_tables[signal_type].get(norm, 0) + 1
        else:
            not_positive_tables[signal_type][norm] = not_positive_tables[signal_type].get(norm, 0) + 1
            not_negative_tables[signal_type][norm] = not_negative_tables[signal_type].get(norm, 0) + 1


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

    tables = {
        "interest_counts": interest_counts,
        "bio_counts": bio_counts,
        "descriptor_counts": descriptor_counts,
        **veto_tables,
    }
    _apply_signal_feedback(tables)
    return tables


def _get_tables() -> dict:
    if not PROFILES_PATH.exists():
        return {
            "interest_counts": {},
            "bio_counts": {},
            "descriptor_counts": {},
            **{name: {} for name in VETO_TABLE_NAMES},
        }

    stat = PROFILES_PATH.stat()
    signal_mtime, signal_size, legacy_mtime, legacy_size = _signal_storage_signature()
    if (
        _CACHE["tables"] is not None
        and _CACHE["profiles_mtime_ns"] == stat.st_mtime_ns
        and _CACHE["profiles_size"] == stat.st_size
        and _CACHE["signals_mtime_ns"] == signal_mtime
        and _CACHE["signals_size"] == signal_size
        and _CACHE.get("legacy_mtime_ns") == legacy_mtime
        and _CACHE.get("legacy_size") == legacy_size
    ):
        return _CACHE["tables"]

    tables = _build_tables()
    logger.info(
        "Preferencias textuais atualizadas: interests=%s bio_tokens=%s descriptors=%s",
        len(tables["interest_counts"]),
        len(tables["bio_counts"]),
        len(tables["descriptor_counts"]),
    )
    _CACHE["profiles_mtime_ns"] = stat.st_mtime_ns
    _CACHE["profiles_size"] = stat.st_size
    _CACHE["signals_mtime_ns"] = signal_mtime
    _CACHE["signals_size"] = signal_size
    _CACHE["legacy_mtime_ns"] = legacy_mtime
    _CACHE["legacy_size"] = legacy_size
    _CACHE["tables"] = tables
    return tables


def signal_training_candidates(
    signal_type: str = "all",
    limit: int = 60,
    min_occurrences: int = 3,
    max_explicit_feedback: int = 2,
) -> list[dict]:
    requested = str(signal_type or "all").strip().lower()
    allowed = set(SIGNAL_TYPES)
    if requested not in allowed:
        requested = "all"

    rows = _load_real_rows()
    explicit = signal_feedback_counts()
    candidates: dict[tuple[str, str], dict] = {}

    def add(kind: str, display: str, label: int, row: dict) -> None:
        if requested != "all" and kind != requested:
            return
        norm = _normalize(display)
        if not norm:
            return
        key = (kind, norm)
        item = candidates.setdefault(key, {
            "signal_type": kind,
            "signal_value": display.strip(),
            "signal_norm": norm,
            "occurrences": 0,
            "likes": 0,
            "dislikes": 0,
            "examples": [],
        })
        item["occurrences"] += 1
        if label == 1:
            item["likes"] += 1
        else:
            item["dislikes"] += 1
        if len(item["examples"]) < 3:
            item["examples"].append({
                "name": row.get("name", ""),
                "age": row.get("age", ""),
                "label": label,
            })

    for row in rows:
        label = int(row["label"])
        for interest in str(row.get("interests", "") or "").split(","):
            interest = interest.strip()
            if interest:
                add("interest", interest, label, row)
        for token in set(_tokenize_bio(row.get("bio", ""))):
            if _is_bio_signal_candidate(token):
                add("bio", token, label, row)
        for display, _norm in _descriptor_display_items(row.get("descriptors", "")):
            add("descriptor", display, label, row)

    result = []
    for key, item in candidates.items():
        kind, norm = key
        explicit_counts = explicit.get(key, {"positive": 0, "neutral": 0, "negative": 0, "total": 0})
        if item["occurrences"] < min_occurrences:
            continue
        if explicit_counts.get("total", 0) > max_explicit_feedback:
            continue
        total = item["likes"] + item["dislikes"]
        score = _token_score(item["likes"], item["dislikes"]) if total else 0.5
        item = dict(item)
        item["explicit_feedback"] = dict(explicit_counts)
        item["learned_score"] = round(score, 4)
        item["priority"] = round(
            (item["occurrences"] / (1 + explicit_counts.get("total", 0)))
            * (1.0 + abs(score - 0.5)),
            4,
        )
        result.append(item)

    result.sort(
        key=lambda item: (
            item["explicit_feedback"].get("total", 0),
            -item["priority"],
            -item["occurrences"],
            item["signal_type"],
            item["signal_norm"],
        )
    )
    return result[: max(1, int(limit or 60))]


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
