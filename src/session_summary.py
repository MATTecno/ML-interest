"""Resumo em memoria da sessao atual de swipes."""

from __future__ import annotations

import time
from collections import Counter
from threading import Lock


class SwipeSessionStats:
    def __init__(self) -> None:
        self._lock = Lock()
        self.started_at = time.time()
        self.swipes = 0
        self.likes = 0
        self.super_likes = 0
        self.passes = 0
        self.ai_likes = 0
        self.ai_passes = 0
        self.manual_prompts = 0
        self.manual_corrections = 0
        self.corrections_to_like = 0
        self.corrections_to_pass = 0
        self.auto_filtered = 0
        self.errors = 0
        self.pending_cleared = 0
        self.feedback_domains: Counter[str] = Counter()
        self.filter_reasons: Counter[str] = Counter()
        self.interests_liked: Counter[str] = Counter()
        self.interests_passed: Counter[str] = Counter()
        self.bio_positive_liked: Counter[str] = Counter()
        self.bio_negative_passed: Counter[str] = Counter()
        self.races_liked: Counter[str] = Counter()
        self.races_passed: Counter[str] = Counter()
        self.photo_scores_liked: list[float] = []
        self.photo_scores_passed: list[float] = []
        self.text_scores_liked: list[float] = []
        self.text_scores_passed: list[float] = []
        self.similarity_liked: list[float] = []
        self.similarity_passed: list[float] = []

    def record_auto_filtered(self, profile: dict, reason: str) -> None:
        with self._lock:
            self.auto_filtered += 1
            self.filter_reasons[(reason or "Filtro automático").split(" — ")[0]] += 1

    def record_swipe(
        self,
        profile: dict,
        ai_decision: str,
        final_decision: str,
        interactive: bool,
        feedback: dict | None = None,
    ) -> None:
        feedback = feedback or {}
        with self._lock:
            self.swipes += 1
            is_like = str(final_decision or "").strip().upper() in {"CURTIR", "SUPER_LIKE", "SUPER LIKE"}
            if is_like:
                self.likes += 1
                if str(final_decision or "").strip().upper() in {"SUPER_LIKE", "SUPER LIKE"}:
                    self.super_likes += 1
            else:
                self.passes += 1

            if ai_decision == "CURTIR":
                self.ai_likes += 1
            else:
                self.ai_passes += 1

            if interactive:
                self.manual_prompts += 1
            if final_decision != ai_decision:
                self.manual_corrections += 1
                if is_like:
                    self.corrections_to_like += 1
                else:
                    self.corrections_to_pass += 1

            domain = feedback.get("feedback_domain") or ""
            if domain:
                self.feedback_domains[domain] += 1

            self._record_common(profile, final_decision)

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def record_pending_cleared(self, count: int) -> None:
        with self._lock:
            self.pending_cleared += max(0, int(count or 0))

    def _record_common(self, profile: dict, decision: str) -> None:
        is_like = str(decision or "").strip().upper() in {"CURTIR", "SUPER_LIKE", "SUPER LIKE"}
        interests_counter = self.interests_liked if is_like else self.interests_passed
        for interest in profile.get("interests") or []:
            if interest:
                interests_counter[str(interest)] += 1

        ml = profile.get("_ml_result") or {}
        photo_score = ml.get("photo_score")
        text_score = ml.get("text_score")
        photo = profile.get("_photo_features") or {}
        similarity = photo.get("photo_face_similarity")
        race = photo.get("_dominant_race")

        if is_like:
            self._append_float(self.photo_scores_liked, photo_score)
            self._append_float(self.text_scores_liked, text_score)
            self._append_float(self.similarity_liked, similarity)
            if race:
                self.races_liked[str(race)] += 1
        else:
            self._append_float(self.photo_scores_passed, photo_score)
            self._append_float(self.text_scores_passed, text_score)
            self._append_float(self.similarity_passed, similarity)
            if race:
                self.races_passed[str(race)] += 1

    @staticmethod
    def _append_float(values: list[float], raw) -> None:
        try:
            values.append(float(raw))
        except Exception:
            return

    @staticmethod
    def _avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _fmt_pct(value: float | None) -> str:
        return "--" if value is None else f"{int(value * 100)}%"

    @staticmethod
    def _top(counter: Counter[str], n: int = 5) -> str:
        if not counter:
            return "--"
        return ", ".join(f"{key} ({count})" for key, count in counter.most_common(n))

    def format_summary(self, reason: str = "") -> str:
        with self._lock:
            elapsed = max(1, int(time.time() - self.started_at))
            minutes = elapsed // 60
            seconds = elapsed % 60
            correction_rate = (
                self.manual_corrections / self.manual_prompts
                if self.manual_prompts
                else 0.0
            )
            like_rate = self.likes / self.swipes if self.swipes else 0.0

            photo_like = self._avg(self.photo_scores_liked)
            photo_pass = self._avg(self.photo_scores_passed)
            text_like = self._avg(self.text_scores_liked)
            text_pass = self._avg(self.text_scores_passed)
            sim_like = self._avg(self.similarity_liked)
            sim_pass = self._avg(self.similarity_passed)

            lines = [
                "",
                "  ============================================================",
                "  Resumo da sessão de swipes",
                "  ============================================================",
                f"  Motivo do encerramento : {reason or 'hotkey'}",
                f"  Duração                : {minutes}m{seconds:02d}s",
                f"  Swipes processados     : {self.swipes} ({self.auto_filtered} por filtro automático)",
                f"  Resultado dos swipes   : {self.likes} curtidas ({self.super_likes} super) / {self.passes} passadas ({self._fmt_pct(like_rate)} like rate)",
                f"  IA originalmente       : {self.ai_likes} curtir / {self.ai_passes} passar",
                f"  Correções manuais      : {self.manual_corrections}/{self.manual_prompts} ({self._fmt_pct(correction_rate)})",
                f"  Correções para curtir  : {self.corrections_to_like}",
                f"  Correções para passar  : {self.corrections_to_pass}",
                f"  Fila descartada        : {self.pending_cleared} perfil(is)",
                "",
                "  O que pareceu pesar hoje",
                f"  Foto média curtidas    : {self._fmt_pct(photo_like)} | passadas: {self._fmt_pct(photo_pass)}",
                f"  Texto médio curtidas   : {self._fmt_pct(text_like)} | passadas: {self._fmt_pct(text_pass)}",
                f"  Similaridade visual    : curtidas {self._fmt_pct(sim_like)} | passadas {self._fmt_pct(sim_pass)}",
                f"  Etnias nas curtidas    : {self._top(self.races_liked)}",
                f"  Etnias nas passadas    : {self._top(self.races_passed)}",
                "",
                "  Padrões textuais",
                f"  Interesses curtidos    : {self._top(self.interests_liked)}",
                f"  Interesses passados    : {self._top(self.interests_passed)}",
                f"  Motivos informados     : {self._top(self.feedback_domains)}",
                f"  Filtros automáticos    : {self._top(self.filter_reasons)}",
            ]
            if self.errors:
                lines.append(f"  Erros no swipe         : {self.errors} (ver data/logs/tinder_ia.log)")

            if self.manual_prompts >= 10 and correction_rate >= 0.35:
                lines.append("")
                lines.append("  Observação: muita correção manual; vale treinar mais antes do modo automático.")
            elif self.manual_prompts >= 10 and correction_rate <= 0.15:
                lines.append("")
                lines.append("  Observação: baixa correção manual; o modelo já está ficando bem alinhado.")

            lines.append("  ============================================================\n")
            return "\n".join(lines)


session_stats = SwipeSessionStats()
