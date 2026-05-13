"""
Importa perfis diretamente do JSON da API do Tinder.

Como usar:
  1. Abra o Tinder no browser
  2. F12 → aba Network → filtre por "recs" ou "v2/recs"
  3. Clique na requisição → aba Response → clique com botão direito → Copy Response
  4. Cole em um arquivo (ex: tinder_response.json)
  5. Execute: python3 src/import_json.py tinder_response.json
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import model as mdl
from synthetic import ensure_synthetic_exists
from explainer import explain, format_reason_summary
from filters import apply_hard_filters, apply_photo_filters
from decision_policy import apply_decision_policy
from photo_features import analyze_photos
from photo_storage import (
    download_profile_photos_async,
    eligible_photo_eviction_candidates as _eligible_photo_eviction_candidates,
)
from profile_parser import parse_profile
from config import get_photos_config
from logging_config import get_logger, setup_logging
from resource_guard import is_memory_pressure, memory_pressure_photo_features
import state


_load_photos_config = get_photos_config
logger = get_logger(__name__)


def _format_ml_reason(result: dict) -> str:
    """Formata o motivo da decisão ML em texto legível para salvar com a foto."""
    decision = result.get("decision", "?")
    prob = result.get("probability", 0.0)
    conf = prob * 100 if decision == "CURTIR" else (1 - prob) * 100
    model_type = result.get("model_type", "?")
    n_samples = result.get("n_samples", 0)

    lines = [
        f"Decisão: {decision}",
        f"Confiança: {conf:.0f}%",
        f"Modelo: {model_type} (treinado com {n_samples} amostras)",
        f"Photo score: {result.get('photo_score', 0.5):.2f}",
        f"Text score: {result.get('text_score', 0.5):.2f}",
        "",
        "Resumo dos motivos:",
        format_reason_summary(result, max_lines=8),
        "",
        "Features mais relevantes:",
    ]

    importances = result.get("importances", {})
    features = result.get("features", {})
    if importances:
        top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in top:
            val = features.get(feat, "?")
            lines.append(f"  {feat}: {val}  (importância: {imp:.3f})")

    return "\n".join(lines)


def _photo_memory_pressure_features(profile_name: str, stage: str) -> dict | None:
    pressure, reason, snapshot = is_memory_pressure()
    if not pressure:
        return None

    logger.warning(
        "Analise visual pulada por memoria critica: name=%r stage=%s reason=%s mem=%.1f%% avail=%.1fMB",
        profile_name,
        stage,
        reason,
        snapshot.get("mem_used_pct", 0.0),
        snapshot.get("mem_avail_mb", 0.0),
    )
    return memory_pressure_photo_features(reason)


def process_response(
    json_data: dict,
    show_explanation: bool = True,
    on_profile_ready=None,
    interactive_mode: bool = False,
    should_cancel=None,
) -> list[dict]:
    """Processa o JSON completo da API e retorna lista de resultados."""
    import time

    started_at = time.perf_counter()
    results_raw = json_data.get("data", {}).get("results", [])
    profiles_only = [r for r in results_raw if r.get("type") == "user"]
    logger.info("process_response iniciado: raw=%s profiles=%s interactive=%s", len(results_raw), len(profiles_only), interactive_mode)

    if not profiles_only:
        logger.warning("process_response sem perfis no JSON")
        print("  Nenhum perfil encontrado no JSON.")
        return []

    ensure_synthetic_exists()
    model_data = mdl.load_model()
    if model_data is None:
        print("  Treinando modelo...")
        mdl.train_model()
        model_data = mdl.load_model()

    outputs = []
    curtir_count = 0
    filtered_count = 0

    if not interactive_mode:
        print()
        print(f"  {len(profiles_only)} perfil(is) encontrado(s)\n")
        print("  " + "=" * 60)

    for i, raw in enumerate(profiles_only, 1):
        if should_cancel and should_cancel():
            logger.warning("process_response cancelado antes do perfil %s/%s", i, len(profiles_only))
            break

        profile_started_at = time.perf_counter()
        profile = parse_profile(raw)
        logger.info(
            "Perfil %s/%s parseado: name=%r age=%r id=%r photos=%s",
            i,
            len(profiles_only),
            profile.get("name"),
            profile.get("age"),
            profile.get("_tinder_id"),
            len(profile.get("_photo_urls_analysis") or []),
        )

        # ── Bloco 1: cabeçalho + filtros absolutos (lock breve) ─────────────
        rejected = False
        with state.terminal_lock:
            if not interactive_mode:
                print(f"\n  Perfil {i}/{len(profiles_only)}: {profile['name']}, {profile['age']} anos")
                if profile["_distance_km"]:
                    print(f"  Distância: {profile['_distance_km']} km")
                if profile["bio"]:
                    bio_preview = profile["bio"][:80].replace("\n", " ")
                    print(f"  Bio: {bio_preview}{'...' if len(profile['bio']) > 80 else ''}")
                if profile["interests"]:
                    print(f"  Interesses: {', '.join(profile['interests'])}")
                if profile["_descriptors"]:
                    desc_str = " | ".join(f"{k}: {v}" for k, v in profile["_descriptors"].items())
                    print(f"  Descritores: {desc_str}")
                print()

            rejected, filter_reason = apply_hard_filters(profile)
            if rejected:
                logger.info("Perfil bloqueado por hard_filter: name=%r reason=%s", profile.get("name"), filter_reason)
                profile["_photo_features"] = {}
                policy = apply_decision_policy(0.0, forced_pass=True, filter_reason=filter_reason)
                result = {"decision": "NÃO CURTIR", "probability": 0.0,
                          "features": {}, "importances": {}, "model_type": "filtro",
                          "n_samples": 0, "filter_reason": filter_reason,
                          "decision_policy": policy,
                          "preference_tier": policy["preference_tier"],
                          "correction_type": policy["correction_type"],
                          "ranking_score": policy["ranking_score"],
                          "review_priority": policy["review_priority"]}
                profile["_ml_result"] = result
                filtered_count += 1
                outputs.append({"profile": profile, "result": result})
                if not interactive_mode:
                    print(f"  ✗ NÃO CURTIR — {filter_reason}")
                    print("  " + "-" * 60)

        if rejected:
            if should_cancel and should_cancel():
                logger.warning("process_response cancelado apos hard_filter: name=%r", profile.get("name"))
                break
            download_profile_photos_async(
                profile,
                "NÃO CURTIR",
                f"Decisão: NÃO CURTIR\nMotivo: {filter_reason}",
            )
            if on_profile_ready:
                if not should_cancel or not should_cancel():
                    on_profile_ready(profile, result)
            continue

        # ── Análise de fotos fora do lock (operação lenta) ──────────────────
        analysis_urls = profile.get("_photo_urls_analysis") or []
        cfg_photos = _load_photos_config()
        normal_max = int(cfg_photos.get("max_analysis_photos", 2))
        semantic_cfg = cfg_photos.get("semantic_embedding", {}) or {}
        if isinstance(semantic_cfg, dict) and bool(semantic_cfg.get("enabled", False)):
            try:
                normal_max = max(normal_max, int(semantic_cfg.get("max_photos_per_profile", normal_max)))
            except Exception:
                pass
        training_initial_max = int(cfg_photos.get("training_initial_max_photos", 1))
        suspicious_threshold = float(cfg_photos.get("reanalyze_all_if_woman_confidence_below", 0.50))
        suspicious_max = int(cfg_photos.get("suspicious_gender_max_photos", 6))
        initial_max = training_initial_max if interactive_mode else normal_max

        if analysis_urls:
            n_total = min(len(analysis_urls), initial_max)
            logger.info(
                "Analise de fotos iniciada: name=%r initial_max=%s total_urls=%s",
                profile.get("name"),
                initial_max,
                len(analysis_urls),
            )
            pressure_features = _photo_memory_pressure_features(profile.get("name"), "initial")
            if pressure_features is not None:
                photo_feat = pressure_features
                if not interactive_mode:
                    with state.terminal_lock:
                        print("  Análise de fotos pulada por memória crítica — usando features neutras")
            else:
                if not interactive_mode:
                    with state.terminal_lock:
                        print(f"  Analisando {n_total} foto(s)...")
                photo_started_at = time.perf_counter()
                photo_feat = analyze_photos(analysis_urls, profile["age"], initial_max)
                logger.info(
                    "Analise de fotos concluida: name=%r faces=%s/%s woman=%.3f elapsed=%.2fs",
                    profile.get("name"),
                    photo_feat.get("_faces_found", 0),
                    photo_feat.get("_photos_analyzed", 0),
                    float(photo_feat.get("photo_woman_confidence", 0.5)),
                    time.perf_counter() - photo_started_at,
                )
            if should_cancel and should_cancel():
                logger.warning("process_response cancelado apos analise de fotos: name=%r", profile.get("name"))
                break

            should_reanalyze = (
                len(analysis_urls) > initial_max
                and photo_feat.get("photo_has_face")
                and photo_feat.get("photo_woman_confidence", 0.5) < suspicious_threshold
            )
            if should_reanalyze:
                pressure_features = _photo_memory_pressure_features(profile.get("name"), "gender_reanalysis")
                if pressure_features is not None:
                    photo_feat.update({
                        "_reanalysis_skipped": True,
                        "_reanalysis_skip_reason": pressure_features.get("_failure_reason", "memory_pressure"),
                    })
                    profile["_photo_features"] = photo_feat
                else:
                    logger.info(
                        "Reanalise de genero iniciada: name=%r woman=%.3f max=%s",
                        profile.get("name"),
                        float(photo_feat.get("photo_woman_confidence", 0.5)),
                        suspicious_max,
                    )
                    if not interactive_mode:
                        with state.terminal_lock:
                            pct = int(photo_feat.get("photo_woman_confidence", 0.5) * 100)
                            print(
                                f"  Reanalisando todas as fotos por suspeita de gênero "
                                f"({pct}% de confiança feminina)..."
                            )
                    photo_feat = analyze_photos(
                        analysis_urls,
                        profile["age"],
                        min(len(analysis_urls), suspicious_max),
                    )
                    if should_cancel and should_cancel():
                        logger.warning("process_response cancelado apos reanalise de fotos: name=%r", profile.get("name"))
                        break

                    photo_feat["_reanalyzed_all_photos"] = True
                    logger.info(
                        "Reanalise de genero concluida: name=%r faces=%s/%s woman=%.3f",
                        profile.get("name"),
                        photo_feat.get("_faces_found", 0),
                        photo_feat.get("_photos_analyzed", 0),
                        float(photo_feat.get("photo_woman_confidence", 0.5)),
                    )
            profile["_photo_features"] = photo_feat
        else:
            photo_feat = {}
            profile["_photo_features"] = {}
            logger.info("Perfil sem URL de foto para analise: name=%r", profile.get("name"))

        # ── Bloco 2: resultado das fotos + ML (lock breve) ──────────────────
        photo_rejected = False
        with state.terminal_lock:
            if analysis_urls and not interactive_mode:
                n_faces = photo_feat.get("_faces_found", 0)
                n_analyzed = photo_feat.get("_photos_analyzed", n_total)
                if photo_feat.get("_analysis_failed"):
                    failed = photo_feat.get("_photos_failed", 0)
                    timed_out = photo_feat.get("_photos_timed_out", 0)
                    reason = photo_feat.get("_failure_reason", "indisponível")
                    print(
                        f"  Foto indisponível/parcial ({failed}/{n_analyzed} falharam, "
                        f"{timed_out} timeout) — usando features neutras [{reason}]"
                    )
                elif photo_feat.get("photo_has_face"):
                    woman_pct = int(photo_feat["photo_woman_confidence"] * 100)
                    lightness = int(photo_feat["photo_skin_lightness"] * 100)
                    race = photo_feat.get("_dominant_race", "?")
                    print(f"  {n_faces}/{n_analyzed} com rosto  mulher={woman_pct}%  tom={lightness}%  etnia={race}")
                    body_quality = float(photo_feat.get("photo_body_signal_quality", 0.0) or 0.0)
                    if body_quality > 0:
                        body_visible = int(float(photo_feat.get("photo_body_visible", 0.0) or 0.0) * 100)
                        body_width = int(float(photo_feat.get("photo_body_width_ratio", 0.5) or 0.5) * 100)
                        print(f"  corpo_visível={body_visible}%  largura_visual={body_width}%  sinal={int(body_quality * 100)}%")

                else:
                    print(f"  0/{n_analyzed} com rosto detectado")

            photo_rejected, photo_reason = apply_photo_filters(photo_feat)
            if photo_rejected:
                logger.info("Perfil bloqueado por filtro de foto: name=%r reason=%s", profile.get("name"), photo_reason)
                policy = apply_decision_policy(0.0, forced_pass=True, filter_reason=photo_reason)
                result = {"decision": "NÃO CURTIR", "probability": 0.0,
                          "features": {}, "importances": {}, "model_type": "filtro",
                          "n_samples": 0, "filter_reason": photo_reason,
                          "decision_policy": policy,
                          "preference_tier": policy["preference_tier"],
                          "correction_type": policy["correction_type"],
                          "ranking_score": policy["ranking_score"],
                          "review_priority": policy["review_priority"]}
                profile["_ml_result"] = result
                filtered_count += 1
                outputs.append({"profile": profile, "result": result})
                if not interactive_mode:
                    print(f"  ✗ NÃO CURTIR — Filtro de foto — {photo_reason}")
                    print("  " + "-" * 60)

        if photo_rejected:
            if should_cancel and should_cancel():
                logger.warning("process_response cancelado apos filtro de foto: name=%r", profile.get("name"))
                break
            download_profile_photos_async(
                profile,
                "NÃO CURTIR",
                f"Decisão: NÃO CURTIR\nMotivo: Filtro de foto — {photo_reason}",
            )
            if on_profile_ready:
                if not should_cancel or not should_cancel():
                    on_profile_ready(profile, result)
            continue

        # ── Predição ML ─────────────────────────────────────────────────────
        model_profile = {
            "name": profile["name"],
            "age": profile["age"],
            "bio": profile["bio"],
            "interests": profile["interests"],
            "_descriptors": profile.get("_descriptors", {}),
            "_photo_features": profile.get("_photo_features", {}),
        }
        result = mdl.predict(model_profile, model_data)
        profile["_ml_result"] = result
        logger.info(
            "Predicao ML: name=%r decision=%s prob=%.4f photo_score=%.4f text_score=%.4f elapsed=%.2fs",
            profile.get("name"),
            result.get("decision"),
            float(result.get("probability", 0.0)),
            float(result.get("photo_score", 0.5)),
            float(result.get("text_score", 0.5)),
            time.perf_counter() - profile_started_at,
        )

        # ── Reanalisa com mais fotos se o modelo está incerto ────────────────
        uncertain_max = int(cfg_photos.get("uncertain_max_photos", 6))
        uncertain_threshold = float(cfg_photos.get("uncertain_probability_threshold", 0.15))
        _prob = float(result.get("probability", 0.5))
        _is_uncertain = abs(_prob - 0.5) <= uncertain_threshold
        if (
            _is_uncertain
            and not interactive_mode
            and analysis_urls
            and uncertain_max > initial_max
            and len(analysis_urls) > initial_max
            and not photo_feat.get("_reanalyzed_all_photos")
        ):
            _conf_pct = int(max(_prob, 1 - _prob) * 100)
            pressure_features = _photo_memory_pressure_features(profile.get("name"), "uncertain_reanalysis")
            if pressure_features is not None:
                photo_feat.update({
                    "_uncertain_reanalysis_skipped": True,
                    "_uncertain_reanalysis_skip_reason": pressure_features.get("_failure_reason", "memory_pressure"),
                })
                profile["_photo_features"] = photo_feat
                model_profile["_photo_features"] = photo_feat
                logger.warning(
                    "Reanalise incerto pulada por memoria critica: name=%r prob=%.4f",
                    profile.get("name"),
                    _prob,
                )
            else:
                logger.info(
                    "Perfil incerto: reanalise com mais fotos: name=%r prob=%.4f max=%s",
                    profile.get("name"), _prob, uncertain_max,
                )
                with state.terminal_lock:
                    print(f"  Incerto ({_conf_pct}% confiança) — analisando fotos extras...")
                photo_feat = analyze_photos(
                    analysis_urls, profile["age"], min(len(analysis_urls), uncertain_max)
                )
                photo_feat["_uncertain_reanalyzed"] = True
                profile["_photo_features"] = photo_feat
                model_profile["_photo_features"] = photo_feat
                result = mdl.predict(model_profile, model_data)
                profile["_ml_result"] = result
                _new_prob = float(result.get("probability", 0.5))
                logger.info(
                    "Reanalise incerto concluida: name=%r prob_before=%.4f prob_after=%.4f decision=%s",
                    profile.get("name"), _prob, _new_prob, result.get("decision"),
                )

        with state.terminal_lock:
            if not interactive_mode:
                if show_explanation:
                    print(explain(result))
                else:
                    decision = result["decision"]
                    prob = result["probability"]
                    conf = prob * 100 if decision == "CURTIR" else (1 - prob) * 100
                    status = "✓" if decision == "CURTIR" else "✗"
                    print(f"  [{status}] {decision} ({conf:.0f}% confiança)")
                print("  " + "-" * 60)

        if result["decision"] == "CURTIR":
            curtir_count += 1

        download_profile_photos_async(profile, result["decision"], _format_ml_reason(result))

        if on_profile_ready and (not should_cancel or not should_cancel()):
            on_profile_ready(profile, result)

        outputs.append({"profile": profile, "result": result})

    total_processed = len(outputs)
    ml_processed = total_processed - filtered_count
    if not interactive_mode:
        with state.terminal_lock:
            print(
                f"\n  Resumo: {curtir_count} CURTIR / "
                f"{ml_processed - curtir_count} NÃO CURTIR (ML) / "
                f"{filtered_count} bloqueado(s) por filtro"
                f"  ({total_processed}/{len(profiles_only)} processado(s))\n"
            )
    logger.info(
        "process_response concluido: profiles=%s curtir=%s filtered=%s elapsed=%.2fs",
        len(profiles_only),
        curtir_count,
        filtered_count,
        time.perf_counter() - started_at,
    )
    return outputs


def main():
    setup_logging()
    logger.info("import_json CLI iniciado")
    parser = argparse.ArgumentParser(
        description="Processa JSON da API do Tinder e decide curtir ou não"
    )
    parser.add_argument("arquivo", nargs="?",
                        help="Arquivo JSON com a resposta da API do Tinder")
    parser.add_argument("--resumido", action="store_true",
                        help="Exibe apenas a decisão final sem explicação detalhada")
    args = parser.parse_args()

    if not args.arquivo:
        print("\n  Como usar:")
        print("    python3 src/import_json.py tinder_response.json")
        print("    python3 src/import_json.py tinder_response.json --resumido\n")
        sys.exit(0)

    path = Path(args.arquivo)
    if not path.exists():
        logger.error("Arquivo JSON nao encontrado: %s", path)
        print(f"\n  Arquivo não encontrado: {path}\n")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logger.exception("Erro ao ler JSON: %s", path)
            print(f"\n  Erro ao ler JSON: {e}\n")
            sys.exit(1)

    process_response(data, show_explanation=not args.resumido)


if __name__ == "__main__":
    main()
