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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
from resource_guard import get_system_resource_snapshot, is_memory_pressure, wait_for_memory_relief
import state


_load_photos_config = get_photos_config
logger = get_logger(__name__)


def _photo_unavailable_features(reason: str) -> dict:
    """Features neutras quando evitamos trabalho visual duplicado ou caro demais."""
    clean_reason = str(reason or "photo_unavailable").strip()
    return {
        "_analysis_failed": True,
        "_analysis_skipped": True,
        "_failure_reason": clean_reason,
        "_photo_analysis_incomplete": True,
        "_photos_analyzed": 0,
        "_photos_failed": 0,
        "_photos_timed_out": 0,
        "photo_has_face": 1.0,
        "photo_faces_ratio": 0.0,
        "photo_failure_ratio": 0.0,
        "photo_woman_confidence": 0.5,
        "photo_skin_lightness": 0.5,
        "photo_face_similarity": 0.5,
        "photo_gender_certainty": 0.0,
    }


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
        "Analise visual aguardando alivio de memoria: name=%r stage=%s reason=%s mem=%.1f%% avail=%.1fMB",
        profile_name,
        stage,
        reason,
        snapshot.get("mem_used_pct", 0.0),
        snapshot.get("mem_avail_mb", 0.0),
    )
    cleared, final_reason, final_snapshot, waited = wait_for_memory_relief()
    if cleared:
        logger.info(
            "Memoria aliviou; analise visual continuara: name=%r stage=%s waited=%.1fs mem=%.1f%% avail=%.1fMB",
            profile_name,
            stage,
            waited,
            final_snapshot.get("mem_used_pct", 0.0),
            final_snapshot.get("mem_avail_mb", 0.0),
        )
        return None
    logger.warning(
        "Memoria ainda pressionada apos espera; adiando analise visual: name=%r stage=%s waited=%.1fs reason=%s mem=%.1f%% avail=%.1fMB swap=%.1f%%",
        profile_name,
        stage,
        waited,
        final_reason,
        final_snapshot.get("mem_used_pct", 0.0),
        final_snapshot.get("mem_avail_mb", 0.0),
        final_snapshot.get("swap_used_pct", 0.0),
    )
    return {
        "_defer_due_memory_pressure": True,
        "_failure_reason": f"memory_pressure:{final_reason or reason}",
    }


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
    uncertain_max = int(cfg_photos.get("uncertain_max_photos", 6))
    uncertain_threshold = float(cfg_photos.get("uncertain_probability_threshold", 0.15))
    initial_max = training_initial_max if interactive_mode else normal_max
    prefetch_window = 0 if interactive_mode else int(cfg_photos.get("prefetch_next_profiles", 0) or 0)
    prefetch_workers = max(1, int(cfg_photos.get("prefetch_parallel_workers", 1) or 1))
    prefetch_enabled = bool(cfg_photos.get("prefetch_enabled", prefetch_window > 0)) and prefetch_window > 0
    prefetch_wait_timeout = max(0.0, float(cfg_photos.get("prefetch_wait_timeout_seconds", 2.5) or 0.0))
    prefetch_current_max_wait = max(
        prefetch_wait_timeout,
        float(cfg_photos.get("prefetch_current_max_wait_seconds", 18) or 0.0),
    )
    prefetch_avoid_direct_duplicate = bool(cfg_photos.get("prefetch_avoid_direct_duplicate", True))
    prefetch_futures = {}
    parsed_cache = {}

    def _profile_at(index: int) -> dict:
        if index not in parsed_cache:
            parsed_cache[index] = parse_profile(profiles_only[index - 1])
        return parsed_cache[index]

    if not interactive_mode:
        current_name, current_id, current_age_visible, current_report_age = state.get_current_meta()
        current_idx = 0
        if current_report_age <= 10.0 and (current_name or current_id):
            for idx in range(1, len(profiles_only) + 1):
                candidate = _profile_at(idx)
                id_match = current_id and str(candidate.get("_tinder_id") or "") == current_id
                name_match = (
                    current_name
                    and str(candidate.get("name") or "").strip().casefold() == current_name.strip().casefold()
                    and (not current_age_visible or int(candidate.get("age") or 0) == int(current_age_visible))
                )
                if id_match or name_match:
                    current_idx = idx
                    break
        if current_idx > 1:
            current_raw = profiles_only[current_idx - 1]
            profiles_only = [current_raw] + profiles_only[: current_idx - 1] + profiles_only[current_idx:]
            parsed_cache = {1: parse_profile(current_raw)}
            logger.info(
                "Perfil visivel priorizado no processamento: original_idx=%s name=%r id=%r age=%s",
                current_idx,
                current_name,
                current_id,
                current_age_visible,
            )

    prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="photo-prefetch") if prefetch_enabled else None

    def _prefetch_key(profile: dict) -> str:
        return str(profile.get("_tinder_id") or f"{profile.get('name','')}:{profile.get('age','')}")

    def _prefetch_window_for_current_resources() -> int:
        if prefetch_window <= 0:
            return 0
        try:
            soft_limit = float(cfg_photos.get("prefetch_memory_soft_limit_percent", 0) or 0)
            soft_window = int(cfg_photos.get("prefetch_memory_soft_window", 1) or 1)
            if soft_limit <= 0:
                return prefetch_window
            snap = get_system_resource_snapshot()
            mem_used = float(snap.get("mem_used_pct", 0.0) or 0.0)
            if mem_used >= soft_limit:
                window = max(0, min(prefetch_window, soft_window))
                logger.info(
                    "Prefetch visual reduzido por memoria: window=%s->%s mem=%.1f%% avail=%.1fMB swap=%.1f%%",
                    prefetch_window,
                    window,
                    mem_used,
                    float(snap.get("mem_avail_mb", 0.0) or 0.0),
                    float(snap.get("swap_used_pct", 0.0) or 0.0),
                )
                return window
        except Exception:
            logger.debug("Falha ao avaliar janela dinamica de prefetch", exc_info=True)
        return prefetch_window

    def _schedule_photo_prefetches(after_index: int) -> None:
        if not prefetch_executor:
            return
        pressure, reason, snap = is_memory_pressure()
        if pressure:
            logger.info(
                "Prefetch visual pausado por memoria: reason=%s mem=%.1f%% avail=%.1fMB swap=%.1f%%",
                reason,
                snap.get("mem_used_pct", 0.0),
                snap.get("mem_avail_mb", 0.0),
                snap.get("swap_used_pct", 0.0),
            )
            return
        current_prefetch_window = _prefetch_window_for_current_resources()
        if current_prefetch_window <= 0:
            return
        last_index = min(len(profiles_only), after_index + current_prefetch_window)
        for idx in range(after_index + 1, last_index + 1):
            if idx in prefetch_futures:
                continue
            profile = _profile_at(idx)
            rejected, filter_reason = apply_hard_filters(profile)
            if rejected:
                logger.info("Prefetch visual pulado por hard_filter: idx=%s name=%r reason=%s", idx, profile.get("name"), filter_reason)
                continue
            analysis_urls = profile.get("_photo_urls_analysis") or []
            if not analysis_urls:
                continue
            max_photos = min(len(analysis_urls), initial_max)
            key = _prefetch_key(profile)
            logger.info(
                "Prefetch visual agendado: idx=%s/%s name=%r key=%s photos=%s max=%s",
                idx,
                len(profiles_only),
                profile.get("name"),
                key,
                len(analysis_urls),
                max_photos,
            )
            prefetch_futures[idx] = prefetch_executor.submit(analyze_photos, analysis_urls, profile["age"], max_photos)

    def _consume_prefetched_photo(index: int, profile: dict):
        future = prefetch_futures.pop(index, None)
        if future is None:
            return None
        key = _prefetch_key(profile)
        started = time.perf_counter()
        try:
            logger.info("Aguardando prefetch visual: idx=%s name=%r key=%s done=%s", index, profile.get("name"), key, future.done())
            result = future.result(timeout=prefetch_wait_timeout if prefetch_wait_timeout > 0 else None)
            logger.info(
                "Prefetch visual usado: idx=%s name=%r key=%s faces=%s/%s woman=%.3f wait=%.2fs",
                index,
                profile.get("name"),
                key,
                result.get("_faces_found", 0),
                result.get("_photos_analyzed", 0),
                float(result.get("photo_woman_confidence", 0.5)),
                time.perf_counter() - started,
            )
            return result
        except FutureTimeoutError:
            logger.warning(
                "Prefetch visual atrasado: idx=%s name=%r key=%s timeout=%.2fs running=%s",
                index,
                profile.get("name"),
                key,
                prefetch_wait_timeout,
                future.running(),
            )
            if prefetch_avoid_direct_duplicate and future.running():
                extra_wait = max(0.0, prefetch_current_max_wait - prefetch_wait_timeout)
                if extra_wait > 0:
                    try:
                        logger.info(
                            "Aguardando prefetch visual para evitar analise duplicada: idx=%s name=%r key=%s extra_timeout=%.2fs",
                            index,
                            profile.get("name"),
                            key,
                            extra_wait,
                        )
                        result = future.result(timeout=extra_wait)
                        logger.info(
                            "Prefetch visual usado apos espera extra: idx=%s name=%r key=%s faces=%s/%s woman=%.3f wait=%.2fs",
                            index,
                            profile.get("name"),
                            key,
                            result.get("_faces_found", 0),
                            result.get("_photos_analyzed", 0),
                            float(result.get("photo_woman_confidence", 0.5)),
                            time.perf_counter() - started,
                        )
                        return result
                    except FutureTimeoutError:
                        pass
                logger.warning(
                    "Prefetch visual ainda em execucao; evitando analise direta duplicada e usando features neutras: idx=%s name=%r key=%s max_wait=%.2fs",
                    index,
                    profile.get("name"),
                    key,
                    prefetch_current_max_wait,
                )
                return _photo_unavailable_features("prefetch_visual_timeout_sem_duplicar")
            cancelled = future.cancel()
            logger.warning(
                "Prefetch visual cancelado=%s; seguindo com analise direta: idx=%s name=%r key=%s",
                cancelled,
                index,
                profile.get("name"),
                key,
            )
            return None
        except Exception:
            logger.exception("Prefetch visual falhou; analisando direto: idx=%s name=%r key=%s", index, profile.get("name"), key)
            return None

    if not interactive_mode:
        print()
        print(f"  {len(profiles_only)} perfil(is) encontrado(s)\n")
        print("  " + "=" * 60)

    for i, raw in enumerate(profiles_only, 1):
        if should_cancel and should_cancel():
            logger.warning("process_response cancelado antes do perfil %s/%s", i, len(profiles_only))
            break

        profile_started_at = time.perf_counter()
        profile = _profile_at(i)
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
            _schedule_photo_prefetches(i)
            continue

        # ── Análise de fotos fora do lock (operação lenta) ──────────────────
        analysis_urls = profile.get("_photo_urls_analysis") or []
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
                if pressure_features.get("_defer_due_memory_pressure"):
                    logger.warning(
                        "process_response adiado antes da analise de fotos: name=%r reason=%s",
                        profile.get("name"),
                        pressure_features.get("_failure_reason", "memory_pressure"),
                    )
                    break
                photo_feat = pressure_features
                if not interactive_mode:
                    with state.terminal_lock:
                        print("  Análise de fotos pulada por memória crítica — usando features neutras")
            else:
                if not interactive_mode:
                    with state.terminal_lock:
                        print(f"  Analisando {n_total} foto(s)...")
                photo_started_at = time.perf_counter()
                photo_feat = _consume_prefetched_photo(i, profile)
                if photo_feat is None:
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
                    if pressure_features.get("_defer_due_memory_pressure"):
                        logger.warning(
                            "Reanalise de genero adiada por memoria: name=%r reason=%s",
                            profile.get("name"),
                            pressure_features.get("_failure_reason", "memory_pressure"),
                        )
                        break
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
            _schedule_photo_prefetches(i)
        else:
            photo_feat = {}
            profile["_photo_features"] = {}
            logger.info("Perfil sem URL de foto para analise: name=%r", profile.get("name"))
            _schedule_photo_prefetches(i)

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
            _schedule_photo_prefetches(i)
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
                if pressure_features.get("_defer_due_memory_pressure"):
                    logger.warning(
                        "Reanalise incerto adiada por memoria: name=%r prob=%.4f reason=%s",
                        profile.get("name"),
                        _prob,
                        pressure_features.get("_failure_reason", "memory_pressure"),
                    )
                    break
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
        _schedule_photo_prefetches(i)

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
    if prefetch_executor:
        prefetch_executor.shutdown(wait=False, cancel_futures=True)
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
