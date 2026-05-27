"""Avaliação offline, calibração operacional e relatórios do modelo."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import ROOT_DIR
from dataset import PROFILES_PATH, _ensure_profiles_schema, _load_csv
from decision_policy import policy_config
from preference_labels import clamp01


REPORT_DIR = ROOT_DIR / "data" / "reports"
LATEST_JSON = REPORT_DIR / "model_eval_latest.json"
LATEST_MD = REPORT_DIR / "model_eval_latest.md"


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        f = float(value)
        if not np.isfinite(f):
            return default
        return f
    except Exception:
        return default


def _details(raw: object) -> dict:
    try:
        value = json.loads(str(raw or "{}"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _truthy_token(value) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return False
    try:
        number = float(text)
        if number.is_integer():
            text = str(int(number))
    except Exception:
        pass
    return text.lower() in {"1", "true", "yes", "sim"}


def _confusion(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "like_recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def optimize_like_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    precision_floor: float = 0.55,
    target_like_recall: float = 0.88,
) -> dict:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) < 30 or len(set(y.tolist())) < 2:
        return {
            "like_threshold": 0.48,
            "source": "fallback_insufficient_data",
            "candidates": 0,
        }

    candidates = np.round(np.arange(0.45, 0.5501, 0.005), 4)
    scored = []
    for threshold in candidates:
        metrics = _confusion(y, p, float(threshold))
        scored.append(metrics)

    viable = [m for m in scored if m["precision"] >= precision_floor]
    if viable:
        viable.sort(
            key=lambda m: (
                m["recall"] >= target_like_recall,
                m["recall"],
                m["precision"],
                -abs(m["threshold"] - 0.48),
            ),
            reverse=True,
        )
        chosen = viable[0]
        source = "optimized_precision_floor"
    else:
        scored.sort(key=lambda m: (m["recall"], m["f1"], -abs(m["threshold"] - 0.48)), reverse=True)
        chosen = scored[0]
        source = "optimized_recall_no_precision_floor"

    return {
        "like_threshold": chosen["threshold"],
        "source": source,
        "precision_floor": round(float(precision_floor), 4),
        "target_like_recall": round(float(target_like_recall), 4),
        "metrics_at_threshold": chosen,
        "candidates": len(scored),
    }


def _calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        lo, hi = float(edges[idx]), float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count <= 0:
            bins.append({"range": [round(lo, 2), round(hi, 2)], "count": 0})
            continue
        bins.append({
            "range": [round(lo, 2), round(hi, 2)],
            "count": count,
            "avg_probability": round(float(p[mask].mean()), 4),
            "observed_like_rate": round(float(y[mask].mean()), 4),
        })
    return bins


def _range_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> list[dict]:
    ranges = [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0)]
    result = []
    for idx, (lo, hi) in enumerate(ranges):
        mask = (p >= lo) & (p <= hi if idx == len(ranges) - 1 else p < hi)
        if not mask.any():
            result.append({"range": [lo, hi], "count": 0})
            continue
        item = _confusion(y[mask], p[mask], threshold)
        item["range"] = [lo, hi]
        item["count"] = int(mask.sum())
        result.append(item)
    return result


def _counts(series: pd.Series) -> dict:
    return {
        str(key): int(value)
        for key, value in series.fillna("").astype(str).value_counts().items()
        if str(key)
    }


def evaluate_profiles(config: dict | None = None, write_reports: bool = True) -> dict:
    _ensure_profiles_schema()
    df = _load_csv(PROFILES_PATH)
    if len(df) == 0:
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "status": "empty_dataset",
        }
        if write_reports:
            write_report_files(report)
        return report

    df = df[df["source"].astype(str).str.lower().eq("real")].copy()
    label_num = pd.to_numeric(df["label"], errors="coerce")
    trainable = df[label_num.isin([0, 1])].copy()
    trainable["label"] = pd.to_numeric(trainable["label"], errors="coerce").astype(int)
    split_idx = int(len(trainable) * 0.8)
    holdout = trainable.iloc[split_idx:].copy()
    holdout["probability"] = pd.to_numeric(holdout.get("swipe_probability", ""), errors="coerce")
    eval_df = holdout.dropna(subset=["probability"]).copy()
    eval_df["probability"] = eval_df["probability"].clip(0.0, 1.0)

    cfg = policy_config(config or {})
    precision_floor = float(cfg.get("precision_floor", 0.55))
    target_recall = float(cfg.get("target_like_recall", 0.88))

    if len(eval_df) > 0:
        y = eval_df["label"].astype(int).to_numpy()
        p = eval_df["probability"].astype(float).to_numpy()
        threshold = optimize_like_threshold(
            y,
            p,
            precision_floor=precision_floor,
            target_like_recall=target_recall,
        )
        chosen_threshold = float(threshold.get("like_threshold", cfg.get("fallback_like_threshold", 0.48)))
        metrics = _confusion(y, p, chosen_threshold)
        brier = float(np.mean((p - y) ** 2))
        calibration = _calibration_bins(y, p)
        ranges = _range_metrics(y, p, chosen_threshold)
    else:
        threshold = {
            "like_threshold": float(cfg.get("fallback_like_threshold", 0.48)),
            "source": "fallback_no_probability_rows",
            "candidates": 0,
        }
        metrics = {}
        brier = None
        calibration = []
        ranges = []

    superlike_count = 0
    for raw in df.get("feedback_details", pd.Series(dtype=str)).fillna(""):
        if _details(raw).get("target_action") == "super_like":
            superlike_count += 1

    manual_corrected = df.get("manual_corrected", pd.Series(dtype=str)).fillna("")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "dataset": {
            "real_rows": int(len(df)),
            "trainable_real_rows": int(len(trainable)),
            "holdout_rows": int(len(holdout)),
            "evaluated_rows": int(len(eval_df)),
            "probability_coverage": round(len(eval_df) / len(holdout), 4) if len(holdout) else 0.0,
            "like_rate_trainable": round(float(trainable["label"].mean()), 4) if len(trainable) else 0.0,
            "synthetic_rows": 0,
        },
        "thresholds": threshold,
        "metrics": metrics,
        "brier_score": round(brier, 4) if brier is not None else None,
        "calibration_bins": calibration,
        "probability_ranges": ranges,
        "corrections": {
            "manual_corrected": int(manual_corrected.map(_truthy_token).sum()),
            "by_domain": _counts(df.get("feedback_domain", pd.Series(dtype=str))),
            "by_correction_type": _counts(df.get("correction_type", pd.Series(dtype=str))),
        },
        "super_like": {
            "strong_like_rows": int((df.get("preference_tier", pd.Series(dtype=str)).fillna("") == "strong_like").sum()),
            "target_action_rows": int(superlike_count),
        },
    }
    if write_reports:
        write_report_files(report)
    return report


def write_report_files(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    LATEST_MD.write_text(format_markdown_report(report), encoding="utf-8")


def format_markdown_report(report: dict) -> str:
    dataset = report.get("dataset", {})
    metrics = report.get("metrics", {})
    thresholds = report.get("thresholds", {})
    lines = [
        "# Avaliação do Modelo",
        "",
        f"Gerado em: {report.get('generated_at', '')}",
        "",
        "## Dataset",
        f"- Reais: {dataset.get('real_rows', 0)}",
        f"- Treináveis: {dataset.get('trainable_real_rows', 0)}",
        f"- Holdout: {dataset.get('holdout_rows', 0)}",
        f"- Avaliados com probabilidade: {dataset.get('evaluated_rows', 0)}",
        "",
        "## Threshold",
        f"- Curtir: {thresholds.get('like_threshold', 0.48)}",
        f"- Origem: {thresholds.get('source', '')}",
        "",
        "## Métricas",
        f"- Precision: {metrics.get('precision', '')}",
        f"- Recall curtidas: {metrics.get('recall', '')}",
        f"- F1: {metrics.get('f1', '')}",
        f"- Falsos negativos: {metrics.get('fn', '')}",
        f"- Brier score: {report.get('brier_score', '')}",
        "",
        "## Calibração",
        "| Faixa | N | Prob. média | Like observado |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in report.get("calibration_bins", []):
        lo, hi = item.get("range", ["", ""])
        lines.append(
            f"| {lo}-{hi} | {item.get('count', 0)} | "
            f"{item.get('avg_probability', '')} | {item.get('observed_like_rate', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = evaluate_profiles(write_reports=True)
    print(format_markdown_report(report))
    print(f"\nRelatórios salvos em {LATEST_JSON} e {LATEST_MD}")


if __name__ == "__main__":
    main()
