"""
Monitor de logs do Tinder-IA.

Fica observando data/logs/tinder_ia.log e cria prompts de investigação em
data/alerts/ quando encontra padrões prováveis de bug.

Uso:
  python3 src/log_monitor.py
  python3 src/log_monitor.py --once
  python3 src/log_monitor.py --log data/logs/tinder_ia.log --tail-lines 500
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
DEFAULT_LOG = ROOT_DIR / "data" / "logs" / "tinder_ia.log"
ALERTS_DIR = ROOT_DIR / "data" / "alerts"


@dataclass(frozen=True)
class Rule:
    key: str
    title: str
    pattern: re.Pattern
    threshold: int
    window_seconds: int
    severity: str
    hypothesis: str
    suggested_task: str


RULES = [
    Rule(
        key="fatal_error",
        title="Erro crítico ou traceback",
        pattern=re.compile(r"\b(ERROR|CRITICAL)\b|Traceback|Excecao nao capturada", re.I),
        threshold=1,
        window_seconds=60,
        severity="alta",
        hypothesis="Alguma exceção chegou ao log e pode ter interrompido uma thread ou o servidor.",
        suggested_task="Investigue a exceção, identifique a causa raiz e proponha/implemente a correção mais segura.",
    ),
    Rule(
        key="reload_loop",
        title="Reload/sincronização em loop",
        pattern=re.compile(r"Reload solicitado|Falha persistente de sincronização", re.I),
        threshold=3,
        window_seconds=10 * 60,
        severity="alta",
        hypothesis="O swiper está perdendo sincronização repetidamente e pedindo reloads em sequência.",
        suggested_task="Investigue a sincronização entre extensão, /current, fila do swiper e geração de reload.",
    ),
    Rule(
        key="photo_timeout_storm",
        title="Tempestade de timeouts na análise de foto",
        pattern=re.compile(r"Timeout na analise de foto|Analise de conjunto indisponivel", re.I),
        threshold=4,
        window_seconds=5 * 60,
        severity="média",
        hypothesis="A análise visual está lenta ou indisponível em sequência; pode ser DeepFace, cache, qualidade ou concorrência.",
        suggested_task="Investigue photo_features.py, configuração photos.* e cache de features; evite que falha visual vire decisão ruim.",
    ),
    Rule(
        key="old_batch_discard_loop",
        title="Muitas levas descartadas por geração antiga",
        pattern=re.compile(r"Leva descartada por geracao antiga|Perfil pronto ignorado por geracao antiga", re.I),
        threshold=4,
        window_seconds=10 * 60,
        severity="média",
        hypothesis="A proteção de geração está funcionando, mas talvez haja reloads demais ou backlog de levas antigas.",
        suggested_task="Verifique se /profiles está chegando em excesso, se a fila está acumulando e se reloads estão sendo pedidos cedo demais.",
    ),
    Rule(
        key="photo_retention_stuck",
        title="Retenção de fotos não consegue limpar arquivos antigos",
        pattern=re.compile(r"Retencao de fotos: .*excess=[1-9]\d* .*candidates=0", re.I),
        threshold=5,
        window_seconds=20 * 60,
        severity="baixa",
        hypothesis="Há mais fotos que o limite, mas nenhuma é considerada consolidada para remoção.",
        suggested_task="Investigue photo_storage.py e profiles.csv para entender por que candidates=0 mesmo com files > max.",
    ),
    Rule(
        key="unexpected_error",
        title="Erro inesperado em runtime",
        pattern=re.compile(r"Erro inesperado|Erro ao processar lote|Falha ao salvar|Falha ao carregar", re.I),
        threshold=1,
        window_seconds=2 * 60,
        severity="média",
        hypothesis="Uma operação esperada falhou e pode ter deixado estado parcial.",
        suggested_task="Investigue o trecho do log e corrija o tratamento de erro ou a operação que falhou.",
    ),
    Rule(
        key="high_resource_usage",
        title="Uso alto de CPU ou memória",
        pattern=re.compile(r"Resource high usage", re.I),
        threshold=1,
        window_seconds=10 * 60,
        severity="alta",
        hypothesis="O sistema está usando muitos recursos e pode não conseguir processar mais batches sem travar.",
        suggested_task="Reduza o número de batches pendentes, ajuste os limites de batch ou a configuração de concorrência e monitore o uso de CPU/memória.",
    ),
]


def _parse_time(line: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})", line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
            microsecond=int(match.group(2)) * 1000
        )
    except ValueError:
        return None


def _tail_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-count:]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", text.lower()).strip("_")


def _write_alert(rule: Rule, events: list[tuple[datetime, str]], log_path: Path) -> Path:
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now()
    filename = f"{created_at.strftime('%Y%m%d_%H%M%S_%f')}_{rule.key}.md"
    path = ALERTS_DIR / filename

    recent_lines = "\n".join(f"- `{ts}` {line}" for ts, line in events[-12:])
    prompt = f"""# Alerta Tinder-IA: {rule.title}

Gerado em: {created_at.strftime('%Y-%m-%d %H:%M:%S')}
Severidade: {rule.severity}
Log monitorado: `{log_path}`

## Hipótese
{rule.hypothesis}

## Evidências recentes
{recent_lines}

## Prompt sugerido para Codex
Investigue o projeto Tinder-IA a partir deste alerta.

Problema observado: {rule.title}

Hipótese inicial: {rule.hypothesis}

Tarefa:
{rule.suggested_task}

Use o log `{log_path}` e os arquivos relevantes do projeto. Se encontrar a causa, implemente a correção com cuidado, valide com comandos locais e explique o resultado.
"""
    path.write_text(prompt, encoding="utf-8")
    return path


class LogMonitor:
    def __init__(self, log_path: Path, cooldown_seconds: int = 10 * 60) -> None:
        self.log_path = log_path
        self.cooldown_seconds = cooldown_seconds
        self.events: dict[str, deque[tuple[datetime, str]]] = defaultdict(deque)
        self.last_alert_at: dict[str, datetime] = {}

    def process_line(self, line: str) -> list[Path]:
        now = _parse_time(line) or datetime.now()
        created = []

        for rule in RULES:
            if not rule.pattern.search(line):
                continue

            bucket = self.events[rule.key]
            bucket.append((now, line))
            cutoff = now - timedelta(seconds=rule.window_seconds)
            while bucket and bucket[0][0] < cutoff:
                bucket.popleft()

            if len(bucket) < rule.threshold:
                continue

            wall_now = datetime.now()
            last_alert = self.last_alert_at.get(rule.key)
            if last_alert and (wall_now - last_alert).total_seconds() < self.cooldown_seconds:
                continue

            path = _write_alert(rule, list(bucket), self.log_path)
            self.last_alert_at[rule.key] = wall_now
            created.append(path)

        return created


def follow_file(path: Path, monitor: LogMonitor, tail_lines: int) -> None:
    for line in _tail_lines(path, tail_lines):
        for alert in monitor.process_line(line):
            print(f"[alerta] {alert}")

    print(f"[monitor] Observando {path}")
    print(f"[monitor] Alertas serão salvos em {ALERTS_DIR}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.4)
                continue
            for alert in monitor.process_line(line.rstrip("\n")):
                print(f"[alerta] {alert}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitora logs do Tinder-IA e gera prompts de bug.")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="Caminho do arquivo de log")
    parser.add_argument("--tail-lines", type=int, default=250, help="Linhas recentes analisadas ao iniciar")
    parser.add_argument("--cooldown-minutes", type=int, default=10, help="Cooldown por tipo de alerta")
    parser.add_argument("--once", action="store_true", help="Analisa apenas as últimas linhas e sai")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log não encontrado: {log_path}")

    monitor = LogMonitor(log_path, cooldown_seconds=max(1, args.cooldown_minutes) * 60)
    lines = _tail_lines(log_path, args.tail_lines)

    if args.once:
        alerts = []
        for line in lines:
            alerts.extend(monitor.process_line(line))
        if alerts:
            for alert in alerts:
                print(f"[alerta] {alert}")
        else:
            print("[monitor] Nenhum padrão crítico detectado nas linhas analisadas.")
        return

    follow_file(log_path, monitor, args.tail_lines)


if __name__ == "__main__":
    main()
