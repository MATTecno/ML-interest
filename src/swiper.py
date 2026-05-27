"""
Swiper humanizado: move o mouse no nível do OS para realizar swipes no Tinder.

Por que PyAutoGUI e não JavaScript?
  - Eventos JS gerados por código têm isTrusted=false (detectável)
  - PyAutoGUI move o cursor do sistema operacional → browser recebe o evento
    como se fosse do usuário real → isTrusted=true

Técnicas anti-detecção usadas:
  - Curvas de Bézier no movimento (trajetória não-linear)
  - Ease-in-out na velocidade (acelera/desacelera como humano)
  - Micro-tremores aleatórios durante o movimento
  - Posição de início do swipe varia a cada vez
  - Tempo olhando o perfil antes de swipe é aleatório
  - Humanos olham mais para perfis que vão curtir — simulamos isso

Failsafe de emergência:
  - Mova o mouse para o canto SUPERIOR ESQUERDO da tela para parar tudo
"""

import random
import subprocess
import time
import math
import threading
import unicodedata
from collections import deque
from threading import Thread, Lock

import pyautogui
from config import get_swiper_config, load_config
from feedback import (
    DOMAIN_LABELS,
    KEY_TO_DOMAIN,
    PHOTO_KEY_TO_REASON,
    PHOTO_REASON_LABELS,
    normalize_feedback,
)
from logging_config import get_logger
from prompt_formatter import format_interactive_prompt
from resource_guard import is_memory_pressure, memory_relief_reached
from reload_controller import request_tinder_reload
from session_summary import session_stats
from desktop_notify import notify as desktop_notify

logger = get_logger(__name__)

pyautogui.FAILSAFE = True  # canto superior esquerdo = parada de emergência
pyautogui.PAUSE = 0        # controlamos os delays manualmente
CURRENT_STALE_AFTER_SECONDS = 600.0
_HOTKEY_LISTENER_STARTED = False
_hotkey_manager: "_HotkeyManager | None" = None
_BLOCKING_MODAL_DISMISS_COOLDOWN_SECONDS = 0.75
_last_blocking_modal_dismiss_at = 0.0


def _decision_is_like(decision: str) -> bool:
    return str(decision or "").strip().upper() in {"CURTIR", "SUPER_LIKE", "SUPER LIKE"}


def _decision_is_super_like(decision: str) -> bool:
    return str(decision or "").strip().upper() in {"SUPER_LIKE", "SUPER LIKE"}


class _HotkeyManager:
    """
    Gerencia o ciclo de vida do listener de teclado.
    Mantém o listener vivo para que a hotkey de emergência continue funcionando.
    A filtragem por foco acontece dentro do callback: F8 só vale no Tinder/terminal,
    F10 vale globalmente para conseguir parar swipes desalinhados.
    """

    def __init__(self, on_press_fn):
        self._on_press = on_press_fn
        self._listener = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self._listener is not None and getattr(self._listener, "running", True):
                return True
            try:
                from pynput import keyboard as pynput_kb
                self._listener = pynput_kb.Listener(on_press=self._on_press)
                self._listener.daemon = True
                self._listener.start()
                logger.warning("Hotkey listener iniciado")
                return True
            except Exception:
                logger.exception("Falha ao iniciar hotkey listener")
                self._listener = None
                return False

    def stop(self) -> None:
        with self._lock:
            if self._listener is None:
                return
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
            logger.debug("Hotkey listener parado")

    def start_focus_monitor(self) -> None:
        """Inicia o listener e mantém um watchdog simples caso ele morra."""
        self.start()

        def _monitor():
            while True:
                time.sleep(5.0)
                with self._lock:
                    alive = self._listener is not None and getattr(self._listener, "running", True)
                if not alive:
                    if self.start():
                        logger.warning("Hotkey listener reiniciado pelo watchdog")

        t = threading.Thread(target=_monitor, daemon=True, name="hotkey-focus-monitor")
        t.start()


class SwipePaused(RuntimeError):
    """Sinal interno: pausa manual acionada durante uma ação de mouse."""


class SwipeStopped(RuntimeError):
    """Sinal interno: encerramento manual acionado durante uma ação de mouse."""


def _active_window_title() -> str:
    """Titulo da janela ativa, usado só para diagnóstico/foco."""
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=0.5,
        )
        return (result.stdout or "").strip()
    except Exception:
        return ""


def _is_tinder_focused() -> bool:
    """Retorna True se o Chrome estiver com o Tinder em foco (checa título da janela)."""
    title = _active_window_title()
    if not title:
        return True  # se não conseguir checar, assume que tá focado
    return "tinder" in title.lower()


_TERMINAL_KEYWORDS = ("terminal", "konsole", "xterm", "alacritty", "kitty", "bash", "python")


def _is_hotkey_window_focused() -> bool:
    """
    Retorna True se F8/F10 devem ser aceitos na janela ativa.
    Aceita quando o Tinder ou o terminal do swiper estiver em foco.
    Impede que hotkeys disparem acidentalmente em outros apps (WhatsApp, jogos, etc.).
    """
    try:
        title = _active_window_title().lower()
        return "tinder" in title or any(k in title for k in _TERMINAL_KEYWORDS)
    except Exception:
        return True


def _wait_for_tinder_focus() -> None:
    """Bloqueia até o Tinder ter foco. Não faz nada se auto_pause_on_focus_loss=false."""
    import state

    cfg = _load_swiper_config()
    if not cfg.get("auto_pause_on_focus_loss", True):
        return

    if _is_tinder_focused():
        return

    announced = False
    while not _is_tinder_focused():
        if state.is_swipe_stop_requested():
            return
        if state.is_swipe_paused():
            _wait_while_paused()
            if state.is_swipe_stop_requested():
                return
        if not announced:
            announced = True
            with state.terminal_lock:
                print("\n  [foco] Tinder sem foco — mouse pausado até você voltar para a aba\n")
        time.sleep(0.5)

    if announced:
        with state.terminal_lock:
            print("\n  [foco] Tinder em foco — retomando\n")


def _load_swiper_config() -> dict:
    return get_swiper_config()


def _sleep_with_pause(duration: float) -> None:
    """Dorme respeitando pausa manual dos swipes."""
    import state

    deadline = time.time() + max(0.0, duration)
    pause_announced = False
    while True:
        if state.is_swipe_stop_requested():
            break

        try:
            if not state.is_swipe_paused() and _dismiss_blocking_modal_if_present():
                deadline += 0.20
        except SwipeStopped:
            break
        except SwipePaused:
            pass

        if state.is_swipe_paused():
            if not pause_announced:
                pause_announced = True
                with state.terminal_lock:
                    print("\n  [pause] Swipes pausados — aperte F8 para retomar\n")
            paused_at = time.time()
            while state.is_swipe_paused() and not state.is_swipe_stop_requested():
                time.sleep(0.25)
            if state.is_swipe_stop_requested():
                break
            deadline += time.time() - paused_at
            with state.terminal_lock:
                print("\n  [pause] Swipes retomados\n")

        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(0.15, remaining))


def _wait_while_paused() -> None:
    """Bloqueia antes de qualquer ação de mouse enquanto a pausa estiver ativa."""
    import state

    if state.is_swipe_stop_requested() or not state.is_swipe_paused():
        return

    with state.terminal_lock:
        print("\n  [pause] Swipes pausados — aperte F8 para retomar\n")
    while state.is_swipe_paused() and not state.is_swipe_stop_requested():
        time.sleep(0.25)
    if state.is_swipe_stop_requested():
        return
    with state.terminal_lock:
        print("\n  [pause] Swipes retomados\n")


def _key_matches_hotkey(key, hotkey: str) -> bool:
    try:
        from pynput import keyboard as pynput_kb
    except Exception:
        return False

    wanted = (hotkey or "f8").strip().lower()
    if not wanted:
        return False

    try:
        char = getattr(key, "char", None)
        if char and char.lower() == wanted:
            return True
    except Exception:
        pass

    try:
        name = str(key).lower()
        if wanted and (name == wanted or name.endswith(f".{wanted}") or wanted in name):
            return True
    except Exception:
        pass

    special_map = {
        "f8": pynput_kb.Key.f8,
        "f9": pynput_kb.Key.f9,
        "f10": pynput_kb.Key.f10,
        "f11": pynput_kb.Key.f11,
        "f12": pynput_kb.Key.f12,
        "pause": pynput_kb.Key.pause,
        "space": pynput_kb.Key.space,
    }
    return wanted in special_map and key == special_map[wanted]


def handle_hotkey_action(action: str, source: str = "local") -> bool:
    """Executa ação de hotkey. Usado pelo listener global e pela extensão."""
    action = (action or "").strip().lower()
    cfg = _load_swiper_config()
    pause_hotkey = (cfg.get("pause_hotkey") or "f8").strip().lower()
    stop_hotkey = (cfg.get("stop_hotkey") or "f10").strip().lower()

    try:
        import state
    except Exception:
        logger.exception("Nao foi possivel processar hotkey %s", action)
        return False

    if action in {"stop", stop_hotkey}:
        logger.warning("Hotkey de parada recebida: action=%s source=%s", action, source)
        state.request_swipe_stop(f"{source} {stop_hotkey.upper()}")
        pending = 0
        queue = globals().get("swiper_queue")
        if queue is not None:
            pending = queue.clear_pending()
            session_stats.record_pending_cleared(pending)
            queue.print_session_summary_once(f"{source} {stop_hotkey.upper()}")
        with state.terminal_lock:
            print(f"\n  [hotkey] Encerrando swipes ({stop_hotkey.upper()}, {source}) — fila descartada: {pending}\n")
        return True

    if action in {"pause", pause_hotkey}:
        if source == "local" and not _is_hotkey_window_focused():
            logger.debug("Hotkey de pausa ignorada fora do Tinder/terminal")
            return False
        paused = state.toggle_swipe_pause()
        logger.warning("Hotkey de pausa recebida: paused=%s source=%s", paused, source)
        with state.terminal_lock:
            status = "PAUSADOS" if paused else "RETOMADOS"
            print(f"\n  [hotkey] Swipes {status} ({pause_hotkey.upper()}, {source})\n")
        return True

    return False


def start_pause_hotkey_listener() -> None:
    """Inicia hotkeys globais para pausar/retomar e encerrar swipes."""
    global _HOTKEY_LISTENER_STARTED, _hotkey_manager
    if _HOTKEY_LISTENER_STARTED:
        return

    cfg = _load_swiper_config()
    pause_hotkey = (cfg.get("pause_hotkey") or "f8").strip().lower()
    stop_hotkey = (cfg.get("stop_hotkey") or "f10").strip().lower()

    try:
        import state
    except Exception:
        logger.exception("Nao foi possivel iniciar hotkey de pausa")
        return

    def on_press(key):
        if _key_matches_hotkey(key, stop_hotkey):
            handle_hotkey_action("stop", "local")
            return

        if not _key_matches_hotkey(key, pause_hotkey):
            return

        handle_hotkey_action("pause", "local")

    _hotkey_manager = _HotkeyManager(on_press)
    _hotkey_manager.start_focus_monitor()
    _HOTKEY_LISTENER_STARTED = True
    logger.info("Hotkeys dos swipes iniciadas: pause=%s stop=%s (com monitor de foco)", pause_hotkey, stop_hotkey)


# ─── Funções de movimento ────────────────────────────────────────────────────

def _ease_in_out(t: float) -> float:
    """Velocidade em S: começa devagar, acelera no meio, desacelera no fim."""
    return t * t * (3.0 - 2.0 * t)


def _stop_requested() -> bool:
    try:
        import state
        return state.is_swipe_stop_requested()
    except Exception:
        return False


def _pause_requested() -> bool:
    try:
        import state
        return state.is_swipe_paused()
    except Exception:
        return False


def _raise_if_mouse_blocked() -> None:
    if _stop_requested():
        raise SwipeStopped()
    if _pause_requested():
        raise SwipePaused()


def _native_fallback_swipe(direction: str, cfg: dict) -> bool:
    """Tenta um comando nativo do Tinder quando o drag não confirma."""
    fallback_cfg = cfg.get("native_fallback", {}) or {}
    if not bool(fallback_cfg.get("enabled", True)):
        return False

    method = str(fallback_cfg.get("method", "keyboard") or "keyboard").strip().lower()
    wait_after = max(0.1, float(fallback_cfg.get("wait_after_seconds", 0.8) or 0.8))
    if direction == "up":
        key = "up"
    elif direction == "right":
        key = "right"
    else:
        key = "left"

    _wait_for_tinder_focus()
    _wait_while_paused()
    _raise_if_mouse_blocked()
    _dismiss_blocking_modal_if_present()

    title = _active_window_title()
    logger.warning(
        "Tentando fallback nativo de swipe: method=%s key=%s active_window=%r",
        method,
        key,
        title,
    )
    if method not in {"keyboard", "arrow", "keys"}:
        return False

    pyautogui.press(key)
    _sleep_with_pause(wait_after)
    return True


def _bezier(p0, cp1, cp2, p1, n: int = 60) -> list[tuple[int, int]]:
    """Curva de Bézier cúbica entre p0 e p1 com pontos de controle cp1, cp2."""
    points = []
    for i in range(n + 1):
        t = i / n
        x = ((1 - t) ** 3 * p0[0]
             + 3 * (1 - t) ** 2 * t * cp1[0]
             + 3 * (1 - t) * t ** 2 * cp2[0]
             + t ** 3 * p1[0])
        y = ((1 - t) ** 3 * p0[1]
             + 3 * (1 - t) ** 2 * t * cp1[1]
             + 3 * (1 - t) * t ** 2 * cp2[1]
             + t ** 3 * p1[1])
        points.append((int(x), int(y)))
    return points


def _human_move(dest_x: int, dest_y: int, duration: float | None = None) -> None:
    """Move o mouse até (dest_x, dest_y) com trajetória e velocidade humanizadas."""
    _raise_if_mouse_blocked()
    if duration is None:
        duration = random.uniform(0.25, 0.55)

    orig_x, orig_y = pyautogui.position()
    dist = math.hypot(dest_x - orig_x, dest_y - orig_y)
    if dist < 5:
        _raise_if_mouse_blocked()
        pyautogui.moveTo(dest_x, dest_y)
        return

    # Pontos de controle do Bézier — cria um leve arco no caminho
    arc = random.uniform(0.05, 0.2) * dist * random.choice([-1, 1])
    mid_x = (orig_x + dest_x) / 2
    mid_y = (orig_y + dest_y) / 2
    perp_x = -(dest_y - orig_y) / dist  # vetor perpendicular normalizado
    perp_y = (dest_x - orig_x) / dist

    cp1 = (
        orig_x + (dest_x - orig_x) * 0.3 + perp_x * arc * 0.6,
        orig_y + (dest_y - orig_y) * 0.3 + perp_y * arc * 0.6,
    )
    cp2 = (
        orig_x + (dest_x - orig_x) * 0.7 + perp_x * arc * 0.4,
        orig_y + (dest_y - orig_y) * 0.7 + perp_y * arc * 0.4,
    )

    n_points = max(30, int(dist / 8))
    points = _bezier((orig_x, orig_y), cp1, cp2, (dest_x, dest_y), n=n_points)

    t_start = time.perf_counter()
    for i, (px, py) in enumerate(points):
        _raise_if_mouse_blocked()

        # Timing com ease-in-out
        t_target = _ease_in_out((i + 1) / len(points)) * duration
        elapsed = time.perf_counter() - t_start
        while t_target > elapsed:
            time.sleep(min(t_target - elapsed, 0.03))
            _raise_if_mouse_blocked()
            elapsed = time.perf_counter() - t_start

        # Micro-tremor humano (30% de chance em cada ponto)
        jx = random.randint(-1, 1) if random.random() < 0.3 else 0
        jy = random.randint(-1, 1) if random.random() < 0.3 else 0
        pyautogui.moveTo(px + jx, py + jy)


def _human_drag(
    start: tuple[int, int],
    end: tuple[int, int],
    duration: float | None = None,
) -> None:
    """Arrasta do ponto start ao end com movimento humanizado."""
    _raise_if_mouse_blocked()
    if duration is None:
        duration = random.uniform(0.18, 0.38)

    # Move até o ponto de início
    _human_move(start[0], start[1])
    time.sleep(random.uniform(0.04, 0.12))
    _raise_if_mouse_blocked()

    mouse_is_down = False
    try:
        pyautogui.mouseDown()
        mouse_is_down = True
        time.sleep(random.uniform(0.03, 0.07))
        _raise_if_mouse_blocked()

        # Arrasto com Bézier e ease-in-out
        n_steps = random.randint(20, 35)
        arc_y = random.randint(-15, 15)  # leve curvatura vertical no arrasto
        t_start = time.perf_counter()

        for i in range(1, n_steps + 1):
            _raise_if_mouse_blocked()

            t = _ease_in_out(i / n_steps)
            ix = int(start[0] + (end[0] - start[0]) * t)
            iy = int(start[1] + (end[1] - start[1]) * t + arc_y * math.sin(t * math.pi))
            jx = random.randint(-2, 2) if random.random() < 0.25 else 0
            jy = random.randint(-1, 1) if random.random() < 0.25 else 0

            t_target = (i / n_steps) * duration
            elapsed = time.perf_counter() - t_start
            while t_target > elapsed:
                time.sleep(min(t_target - elapsed, 0.03))
                _raise_if_mouse_blocked()
                elapsed = time.perf_counter() - t_start

            pyautogui.moveTo(ix + jx, iy + jy)

        time.sleep(random.uniform(0.03, 0.08))
    finally:
        if mouse_is_down:
            pyautogui.mouseUp()


# ─── Swipe ───────────────────────────────────────────────────────────────────

def swipe(direction: str, card_center: tuple[int, int] | None = None) -> None:
    """
    Realiza um swipe humanizado.

    direction: "right" (curtir), "left" (não curtir) ou "up" (super like)
    card_center: (x, y) do centro do card na tela. None = centro da tela.
    """
    cfg = _load_swiper_config()

    if card_center is None:
        raw = cfg.get("card_center")
        if raw:
            card_center = tuple(raw)
        else:
            sw, sh = pyautogui.size()
            card_center = (sw // 2, sh // 2)

    cx, cy = card_center
    full_cfg = load_config()
    super_cfg = full_cfg.get("super_like", {}) or {}
    if direction == "up":
        drag_min, drag_max = super_cfg.get("drag_distance_range") or cfg.get("drag_distance_range", [280, 420])
    else:
        drag_min, drag_max = cfg.get("drag_distance_range", [280, 420])

    # Posição de início com variação aleatória em torno do centro do card
    start_x = cx + random.randint(-45, 45)
    start_y = cy + random.randint(-70, 70)

    drag_dist = random.randint(drag_min, drag_max)
    if direction == "up":
        end_x = start_x + random.randint(-35, 35)
        end_y = start_y - drag_dist
    else:
        sign = 1 if direction == "right" else -1
        end_x = start_x + sign * drag_dist
        end_y = start_y + random.randint(-30, 30)

    _human_drag((start_x, start_y), (end_x, end_y))


def _configured_card_center(cfg: dict) -> tuple[int, int]:
    raw = cfg.get("card_center")
    if raw:
        return tuple(raw)
    sw, sh = pyautogui.size()
    return (sw // 2, sh // 2)


def _human_click(x: int, y: int) -> None:
    """Clica em um ponto com pequena variação humana de movimento e duração."""
    _raise_if_mouse_blocked()
    _human_move(
        x + random.randint(-8, 8),
        y + random.randint(-8, 8),
        duration=random.uniform(0.18, 0.42),
    )
    time.sleep(random.uniform(0.04, 0.12))
    _raise_if_mouse_blocked()
    pyautogui.click()


def _dismiss_blocking_modal_if_present(max_age_seconds: float = 2.5) -> bool:
    """Fecha modal bloqueante com clique real do PyAutoGUI, se a extensão reportou um alvo."""
    global _last_blocking_modal_dismiss_at

    try:
        import state
    except Exception:
        return False

    if state.is_swipe_stop_requested() or state.is_swipe_paused():
        return False

    cfg = _load_swiper_config()
    if cfg.get("auto_pause_on_focus_loss", True) and not _is_tinder_focused():
        return False

    target = state.get_blocking_modal_target(max_age_seconds)
    if not target:
        return False

    now = time.time()
    if now - _last_blocking_modal_dismiss_at < _BLOCKING_MODAL_DISMISS_COOLDOWN_SECONDS:
        return False

    x = int(target.get("x") or 0)
    y = int(target.get("y") or 0)
    kind = str(target.get("kind") or "")
    label = str(target.get("target") or "modal")
    if x == 0 and y == 0:
        state.clear_blocking_modal_target(kind)
        return False

    _last_blocking_modal_dismiss_at = now
    logger.info("Fechando modal bloqueante via PyAutoGUI: kind=%s target=%s x=%s y=%s", kind, label, x, y)
    with state.terminal_lock:
        print(f"  [modal] Fechando popup do Tinder via PyAutoGUI ({label})")
    desktop_notify(
        "modal_dismissed",
        "Tinder IA fechou um popup",
        "Popup bloqueante detectado durante o swipe.",
        urgency="normal",
    )

    _human_click(x, y)
    if kind == "super_like_upsell":
        state.mark_super_likes_depleted("modal_super_like_upsell")
    state.clear_blocking_modal_target(kind)
    time.sleep(0.20)
    return True


def _browse_card_photos(duration: float, cfg: dict) -> None:
    """
    Simula cliques laterais no card para navegar entre fotos.

    Só deve ser usado no modo automático. No modo interativo, o usuário já está
    olhando e decidindo manualmente, então não mexemos no card.
    """
    browse_cfg = cfg.get("browse_photos", {})
    if not browse_cfg.get("enabled", False):
        _sleep_with_pause(duration)
        return

    if random.random() > float(browse_cfg.get("chance", 0.75)):
        _sleep_with_pause(duration)
        return

    cx, cy = _configured_card_center(cfg)
    x_offset = int(browse_cfg.get("click_x_offset", 130))
    y_min, y_max = browse_cfg.get("click_y_offset_range", [-170, 60])
    delay_min, delay_max = browse_cfg.get("delay_between_clicks", [0.7, 1.8])
    max_clicks = max(1, int(browse_cfg.get("max_clicks", 3)))
    next_probability = float(browse_cfg.get("next_probability", 0.80))

    click_budget = max(1, min(max_clicks, int(duration // max(delay_min, 0.4)) + 1))
    clicks = random.randint(1, click_budget)
    elapsed = 0.0

    logger.debug("Navegando fotos do card: duration=%.2f clicks=%s", duration, clicks)
    for _ in range(clicks):
        wait = random.uniform(delay_min, delay_max)
        if elapsed + wait >= duration:
            break
        _sleep_with_pause(wait)
        elapsed += wait

        side = 1 if random.random() < next_probability else -1
        target_x = cx + side * x_offset
        target_y = cy + random.randint(int(y_min), int(y_max))
        _wait_while_paused()
        _raise_if_mouse_blocked()
        _human_click(target_x, target_y)

    remaining = duration - elapsed
    if remaining > 0:
        _sleep_with_pause(remaining)


def _match_visible_profile(
    profile: dict,
    current_name: str,
    current_id: str,
    current_visible_age: int = 0,
) -> tuple[bool, str]:
    """Compara um perfil esperado com o que a extensão reportou como visível."""
    expected_name = (profile.get("name") or "").strip()
    expected_id = (profile.get("_tinder_id") or "").strip()
    expected_age = int(profile.get("age") or 0)

    if current_id and expected_id:
        match = current_id == expected_id
        detail = f"id esperado: '{expected_id}' | visível: '{current_id}'"
        return match, detail

    def norm(s: str) -> str:
        base = unicodedata.normalize("NFD", (s or "").strip().lower())
        return "".join(ch for ch in base if unicodedata.category(ch) != "Mn")

    if expected_name and current_name and expected_age and current_visible_age:
        match = norm(expected_name) == norm(current_name) and expected_age == current_visible_age
        detail = (
            f"nome/idade esperados: '{expected_name}'/{expected_age} | "
            f"visível: '{current_name}'/{current_visible_age}"
        )
        return match, detail

    match = bool(expected_name and current_name) and norm(expected_name) == norm(current_name)
    detail = f"nome esperado: '{expected_name}' | visível: '{current_name}'"
    return match, detail


# ─── Fila de swipes ──────────────────────────────────────────────────────────

def _queue_profile_key(profile: dict) -> tuple | None:
    """Chave estável para não enfileirar o mesmo perfil várias vezes."""
    tinder_id = str(profile.get("_tinder_id") or "").strip()
    if tinder_id:
        return ("id", tinder_id)

    content_hash = str(profile.get("_content_hash") or "").strip()
    if content_hash:
        return ("content_hash", content_hash)

    name = str(profile.get("name") or "").strip()
    age = int(profile.get("age") or 0)
    if name and age:
        base = unicodedata.normalize("NFD", name.lower())
        normalized = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
        return ("name_age", normalized, age)

    return None


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _super_like_range(full_cfg: dict, super_cfg: dict) -> tuple[float, float]:
    raw = super_cfg.get("distance_range_km") or (full_cfg.get("preferences", {}) or {}).get("distance_range_km") or [0, 35]
    try:
        min_km = float(raw[0])
        max_km = float(raw[1])
    except Exception:
        min_km, max_km = 0.0, 35.0
    if max_km < min_km:
        min_km, max_km = max_km, min_km
    return min_km, max_km


def _profile_distance(profile: dict, result: dict) -> tuple[float | None, bool]:
    features = result.get("features", {}) or {}
    raw = profile.get("distance_km", profile.get("_distance_km", ""))
    if raw in ("", None):
        raw = features.get("distance_km", "")
    missing = raw in ("", None)
    if "distance_missing" in features:
        missing = missing or _as_float(features.get("distance_missing", 1.0), 1.0) >= 1.0
    if raw in ("", None) or missing:
        return None, True
    try:
        return max(0.0, float(raw)), False
    except Exception:
        return None, True


def _super_like_balance_available(super_cfg: dict) -> tuple[bool | None, str]:
    """Usa o saldo real do /v2/profile; retorna None quando ainda nao ha leitura."""
    try:
        import state

        max_age = float(super_cfg.get("balance_max_age_seconds", 900) or 900)
        balance, age = state.get_super_like_balance(max_age)
    except Exception:
        logger.exception("Falha ao ler saldo de super likes")
        return None, "erro ao ler saldo"

    if balance is None:
        age_txt = "sem leitura" if age == float("inf") else f"saldo velho ({age:.0f}s)"
        return None, age_txt

    available = balance.get("available")
    if available is None:
        total = 0
        for key in ("remaining", "alc_remaining", "new_alc_remaining"):
            try:
                total += max(0, int(balance.get(key) or 0))
            except Exception:
                pass
        available = total > 0
    total = balance.get("total_available")
    if total in ("", None):
        total = sum(
            max(0, int(balance.get(key) or 0))
            for key in ("remaining", "alc_remaining", "new_alc_remaining")
            if str(balance.get(key, "")).strip() not in {"", "None"}
        )
    source = balance.get("source") or "network"
    if available:
        return True, f"saldo network ok ({total} disponível, {source})"
    reset = balance.get("resets_at") or ""
    suffix = f", recarrega em {reset}" if reset else ""
    return False, f"saldo network zerado ({source}{suffix})"


def _super_like_gate(profile: dict, final_decision: str) -> tuple[bool, str]:
    """Decide se uma curtida deve virar super like antes do swipe real."""
    if not _decision_is_like(final_decision) or _decision_is_super_like(final_decision):
        return False, "decisão não é curtida comum"

    full_cfg = load_config()
    super_cfg = full_cfg.get("super_like", {}) or {}
    if not bool(super_cfg.get("enabled", False)):
        return False, "super like desativado no config.yaml"

    result = profile.get("_ml_result") or {}
    if not result or result.get("model_type") == "filtro":
        return False, "sem score de modelo para super like"

    prob = _as_float(result.get("probability"), 0.0)
    text_score = _as_float(result.get("text_score"), 0.0)
    photo_score = _as_float(result.get("photo_score"), 0.0)
    prob_min = float(super_cfg.get("probability_threshold", 0.80) or 0.80)
    text_min = float(super_cfg.get("text_score_min", 0.70) or 0.70)
    photo_min = float(super_cfg.get("photo_score_min", 0.70) or 0.70)

    if prob < prob_min:
        return False, f"probabilidade {prob * 100:.0f}% abaixo de {prob_min * 100:.0f}%"
    if text_score < text_min:
        return False, f"texto {text_score * 100:.0f}% abaixo de {text_min * 100:.0f}%"
    if photo_score < photo_min:
        return False, f"foto {photo_score * 100:.0f}% abaixo de {photo_min * 100:.0f}%"

    superlike_prob = result.get("superlike_probability")
    superlike_min = float(super_cfg.get("superlike_probability_min", 0.0) or 0.0)
    if superlike_prob not in ("", None) and superlike_min > 0:
        sp = _as_float(superlike_prob, 0.0)
        if sp < superlike_min:
            return False, f"modelo superlike {sp * 100:.0f}% abaixo de {superlike_min * 100:.0f}%"

    distance_km, distance_missing = _profile_distance(profile, result)
    require_distance = bool(super_cfg.get("require_distance_known", True))
    if distance_missing:
        if require_distance:
            return False, "distância não informada"
    else:
        min_km, max_km = _super_like_range(full_cfg, super_cfg)
        if distance_km is None or not (min_km <= distance_km <= max_km):
            shown = "?" if distance_km is None else f"{distance_km:.0f} km"
            return False, f"distância {shown} fora da faixa {min_km:.0f}-{max_km:.0f} km"

    if bool(super_cfg.get("require_balance_available", True)):
        balance_ok, balance_reason = _super_like_balance_available(super_cfg)
        if balance_ok is False:
            return False, f"super like indisponível: {balance_reason}"
        if balance_ok is True:
            logger.debug("Super like saldo confirmado: %s", balance_reason)
        elif bool(super_cfg.get("network_balance_required", False)):
            return False, f"saldo de super like sem leitura recente ({balance_reason})"

    if bool(super_cfg.get("require_tinder_available", True)):
        try:
            import state

            max_age = float(super_cfg.get("availability_max_age_seconds", 8) or 8)
            available, reason, age = state.get_current_super_like_state(max_age)
        except Exception:
            logger.exception("Falha ao ler disponibilidade de super like")
            available, reason, age = None, "erro_ao_ler_estado", float("inf")
        if available is not True:
            age_txt = "sem leitura recente" if age == float("inf") else f"leitura {age:.1f}s atrás"
            return False, f"super like indisponível no Tinder ({reason}, {age_txt})"

    bits = [
        f"prob {prob * 100:.0f}%",
        f"texto {text_score * 100:.0f}%",
        f"foto {photo_score * 100:.0f}%",
    ]
    if superlike_prob not in ("", None):
        bits.append(f"modelo superlike {_as_float(superlike_prob) * 100:.0f}%")
    if distance_km is not None:
        bits.append(f"{distance_km:.0f} km")
    return True, ", ".join(bits)


def _wait_for_memory_recovery(initial_reason: str, initial_snapshot: dict) -> bool:
    """Pausa o worker por memoria critica e retoma quando o guard liberar."""
    try:
        import state
    except Exception:
        state = None

    cfg = load_config()
    resume_cfg = ((cfg.get("swiper", {}) or {}).get("memory_resume", {}) or {})
    if not bool(resume_cfg.get("enabled", True)):
        return False

    interval = max(1.0, float(resume_cfg.get("check_interval_seconds", 10) or 10))
    request_reload = bool(resume_cfg.get("reload_on_resume", False))
    announced_at = time.time()
    terminal_lock = state.terminal_lock if state else threading.Lock()
    with terminal_lock:
        print(
            "\n  [swipe] Memória crítica "
            f"({initial_snapshot.get('mem_used_pct', 0.0):.1f}% usada, "
            f"{initial_snapshot.get('mem_avail_mb', 0.0):.0f}MB livres) — "
            "pausando e aguardando liberar para retomar\n"
        )
    logger.warning("Worker do swiper aguardando memoria liberar: reason=%s", initial_reason)
    desktop_notify(
        "memory_pause",
        "Tinder IA pausou por memoria",
        f"{initial_snapshot.get('mem_used_pct', 0.0):.1f}% usada · aguardando liberar",
        urgency="critical",
    )

    while True:
        if state and state.is_swipe_stop_requested():
            return False
        time.sleep(interval)
        cfg_now = load_config()
        pressure, pressure_reason, snapshot = is_memory_pressure(cfg_now)
        relief, relief_reason, snapshot = memory_relief_reached(cfg_now, snapshot)
        if not pressure and relief:
            logger.warning(
                "Memoria liberada; retomando swiper: mem=%.1f%% avail=%.1fMB swap=%.1f%% waited=%.0fs",
                snapshot.get("mem_used_pct", 0.0),
                snapshot.get("mem_avail_mb", 0.0),
                snapshot.get("swap_used_pct", 0.0),
                time.time() - announced_at,
            )
            with terminal_lock:
                print(
                    "  [swipe] Memória liberada "
                    f"({snapshot.get('mem_used_pct', 0.0):.1f}% usada, "
                    f"{snapshot.get('mem_avail_mb', 0.0):.0f}MB livres) — retomando\n"
                )
            desktop_notify(
                "memory_resume",
                "Tinder IA retomou",
                "Memoria voltou a um nivel seguro.",
                urgency="normal",
            )
            if request_reload and state:
                request_tinder_reload(
                    "Memoria liberada apos pausa; recarregando para ressincronizar",
                    source="swiper_memory_resume",
                    navigate_to_recs=False,
                )
            return True
        logger.info(
            "Memoria ainda critica: %s mem=%.1f%% avail=%.1fMB swap=%.1f%%",
            pressure_reason or relief_reason,
            snapshot.get("mem_used_pct", 0.0),
            snapshot.get("mem_avail_mb", 0.0),
            snapshot.get("swap_used_pct", 0.0),
        )


class SwiperQueue:
    """
    Fila thread-safe de decisões de swipe.

    Fluxo:
      1. server.py chama swiper_queue.add([(profile, decision), ...])
      2. Uma thread worker processa a fila com delays humanizados
      3. Cada swipe espera um tempo de "visualização" antes de agir
    """

    def __init__(self) -> None:
        self._queue: deque[tuple[dict, str]] = deque()
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None
        self._thread_generation: int = 0
        self._generation = 0
        self._last_sync_warning_at = 0.0
        self._summary_printed = False
        self._last_reload_at: float = 0.0
        self._consecutive_sync_skips: int = 0
        self._current_key: tuple | None = None

    def add(self, decisions: list[tuple[dict, str]]) -> None:
        """Adiciona lista de (profile_dict, decision_str) à fila."""
        import state

        if state.is_swipe_stop_requested():
            logger.warning("SwipeQueue.add ignorado: encerramento solicitado")
            return

        cfg = _load_swiper_config()
        max_pending = int(cfg.get("max_pending_queue", 40) or 0)
        drop_duplicates = bool(cfg.get("drop_duplicate_profiles", True))
        recent_ttl = float(cfg.get("recent_swipe_ttl_seconds", 600) or 0)

        added_pairs: list[tuple[dict, str]] = []
        skipped_duplicates = 0
        skipped_recent = 0
        dropped_old = 0
        pending = 0
        should_start = False
        started_generation = 0

        with self._lock:
            existing_keys = {
                key for queued_profile, _ in self._queue
                if (key := _queue_profile_key(queued_profile)) is not None
            }

            if self._current_key is not None:
                existing_keys.add(self._current_key)

            for profile, decision in decisions:
                key = _queue_profile_key(profile)

                if drop_duplicates and recent_ttl > 0:
                    already_swiped, _ = state.is_recently_swiped(
                        profile.get("_tinder_id", ""),
                        profile.get("name", ""),
                        int(profile.get("age") or 0),
                        recent_ttl,
                    )
                    if already_swiped:
                        skipped_recent += 1
                        continue

                if drop_duplicates and key is not None and key in existing_keys:
                    skipped_duplicates += 1
                    continue

                added_pairs.append((profile, decision))

                if key is not None:
                    existing_keys.add(key)

            self._queue.extend(added_pairs)

            if max_pending > 0:
                while len(self._queue) > max_pending:
                    self._queue.pop()
                    dropped_old += 1

            pending = len(self._queue)

            thread_alive = self._thread is not None and self._thread.is_alive()
            thread_stale = thread_alive and self._thread_generation != self._generation
            should_start = pending > 0 and (not thread_alive or thread_stale)

            if should_start:
                self._running = True
                started_generation = self._generation
                self._thread_generation = started_generation
                self._thread = Thread(
                    target=self._worker,
                    args=(started_generation,),
                    daemon=True,
                    name=f"swiper-worker-gen-{started_generation}",
                )
                self._thread.start()

        logger.info(
            "SwipeQueue.add: received=%s added=%s skipped_duplicates=%s skipped_recent=%s dropped_old=%s pending=%s thread_started=%s generation=%s",
            len(decisions),
            len(added_pairs),
            skipped_duplicates,
            skipped_recent,
            dropped_old,
            pending,
            should_start,
            started_generation if should_start else self._generation,
        )

        if skipped_duplicates or skipped_recent or dropped_old:
            with state.terminal_lock:
                print(
                    "  [swipe] Fila ajustada: "
                    f"{len(added_pairs)} novo(s), {skipped_duplicates} duplicado(s), "
                    f"{skipped_recent} já swipado(s), "
                    f"{dropped_old} antigo(s) descartado(s), pendentes={pending}"
                )

        if should_start:
            logger.info("Thread do swiper iniciada generation=%s", started_generation)

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear_pending(self) -> int:
        with self._lock:
            pending = len(self._queue)
            self._queue.clear()
            self._current_key = None
            self._generation += 1
            generation = self._generation

        logger.warning("Fila de swipes limpa: pending=%s generation=%s", pending, generation)
        return pending

    def acknowledge_network_swipe(self, tinder_id: str, action: str = "", status: int | None = None) -> int:
        """Remove da fila entradas do perfil que o Tinder já confirmou via rede."""
        clean_id = str(tinder_id or "").strip()
        if not clean_id:
            return 0

        removed = 0
        with self._lock:
            kept: deque[tuple[dict, str]] = deque()
            for profile, decision in self._queue:
                if str(profile.get("_tinder_id") or "").strip() == clean_id:
                    removed += 1
                    continue
                kept.append((profile, decision))
            self._queue = kept
            if self._current_key == ("id", clean_id):
                self._current_key = None

        if removed:
            logger.info(
                "Perfis removidos da fila apos confirmacao de rede: id=%r action=%r status=%r removed=%s",
                clean_id,
                action,
                status,
                removed,
            )
        return removed

    def _model_profile_for_prediction(self, profile: dict) -> dict:
        model_profile = {
            "name": profile.get("name", ""),
            "age": profile.get("age", 0),
            "distance_km": profile.get("distance_km", profile.get("_distance_km", "")),
            "bio": profile.get("bio", ""),
            "interests": profile.get("interests", []),
            "_photo_features": profile.get("_photo_features", {}),
        }
        if profile.get("_descriptors"):
            model_profile["_descriptors"] = profile.get("_descriptors")
        if profile.get("descriptors"):
            model_profile["descriptors"] = profile.get("descriptors")
        return model_profile

    def _prediction_stamp(self, result: dict | None) -> tuple[object, object, object]:
        result = result or {}
        return (
            result.get("training_version"),
            result.get("n_samples"),
            result.get("photo_n_samples"),
        )

    def _model_stamp(self, model_data: dict | None) -> tuple[object, object, object]:
        model_data = model_data or {}
        return (
            model_data.get("training_version"),
            model_data.get("n_samples"),
            model_data.get("photo_n_samples"),
        )

    def _refresh_prediction(
        self,
        profile: dict,
        decision: str,
        model_data: dict,
    ) -> tuple[dict, str, bool]:
        """Recalcula a recomendacao de um perfil ja analisado sem refazer fotos."""
        if profile.get("_skip_prompt") or profile.get("_filter_reason"):
            return profile, decision, False

        try:
            import model as mdl

            result = mdl.predict(self._model_profile_for_prediction(profile), model_data)
            new_decision = result.get("decision", decision)
            changed = new_decision != decision or result.get("probability") != profile.get("_ml_result", {}).get("probability")
            profile["_ml_result"] = result
            return profile, new_decision, changed
        except Exception:
            logger.exception("Falha ao recalcular predicao pendente: name=%r", profile.get("name"))
            return profile, decision, False

    def _refresh_if_model_changed(self, profile: dict, decision: str) -> tuple[dict, str]:
        """Antes do prompt/swipe, evita usar previsao feita por modelo antigo."""
        if profile.get("_skip_prompt") or profile.get("_filter_reason"):
            return profile, decision

        try:
            import model as mdl
            import state

            now = time.time()
            last_checked = float(profile.get("_model_refresh_checked_at", 0.0) or 0.0)
            if now - last_checked < 30.0:
                return profile, decision
            profile["_model_refresh_checked_at"] = now

            model_data = mdl.load_model()
            if model_data is None:
                return profile, decision

            old_stamp = self._prediction_stamp(profile.get("_ml_result"))
            new_stamp = self._model_stamp(model_data)
            if old_stamp == new_stamp:
                return profile, decision

            old_decision = decision
            old_prob = profile.get("_ml_result", {}).get("probability")
            profile, decision, changed = self._refresh_prediction(profile, decision, model_data)
            if changed:
                with state.terminal_lock:
                    print(
                        "\n  [model] Recomendação recalculada com o modelo novo: "
                        f"{old_decision} ({float(old_prob or 0.5) * 100:.0f}%) "
                        f"→ {decision} ({float(profile.get('_ml_result', {}).get('probability', 0.5)) * 100:.0f}%)\n"
                    )
            return profile, decision
        except Exception:
            logger.exception("Falha ao checar versao do modelo antes do prompt: name=%r", profile.get("name"))
            return profile, decision

    def recalculate_pending(self, model_data: dict | None = None) -> dict:
        """
        Recalcula todos os perfis ainda pendentes depois de um retreino.

        Isso evita a situação de a fila continuar mostrando 99%/95% calculados
        por um modelo anterior enquanto o modelo salvo já foi atualizado.
        """
        try:
            import model as mdl

            if model_data is None:
                model_data = mdl.load_model()
            if model_data is None:
                return {"checked": 0, "updated": 0, "changed": 0}
        except Exception:
            logger.exception("Nao foi possivel carregar modelo para recalcular fila")
            return {"checked": 0, "updated": 0, "changed": 0}

        new_pairs: list[tuple[dict, str]] = []
        checked = 0
        updated = 0
        changed = 0

        with self._lock:
            pairs = list(self._queue)
            for profile, decision in pairs:
                checked += 1
                old_decision = decision
                old_result = profile.get("_ml_result", {})
                old_stamp = self._prediction_stamp(old_result)
                if old_stamp == self._model_stamp(model_data):
                    new_pairs.append((profile, decision))
                    continue

                profile, decision, did_update = self._refresh_prediction(profile, decision, model_data)
                if did_update:
                    updated += 1
                if decision != old_decision:
                    changed += 1
                new_pairs.append((profile, decision))

            self._queue = deque(new_pairs)

        logger.info(
            "Fila recalculada apos retreino: checked=%s updated=%s changed=%s",
            checked,
            updated,
            changed,
        )
        return {"checked": checked, "updated": updated, "changed": changed}

    def print_session_summary_once(self, reason: str = "") -> None:
        import state

        with self._lock:
            if self._summary_printed:
                return
            self._summary_printed = True

        with state.terminal_lock:
            print(session_stats.format_summary(reason), flush=True)

    def _push_front(self, pair: tuple[dict, str]) -> None:
        with self._lock:
            self._queue.appendleft(pair)

    def _clear_current_key(self, profile: dict | None = None) -> None:
        key = _queue_profile_key(profile or {}) if profile is not None else None
        with self._lock:
            if profile is None or self._current_key == key:
                self._current_key = None

    def _promote_visible_profile(
        self,
        current_name: str,
        current_id: str,
        current_visible_age: int,
        current_pair: tuple[dict, str],
    ) -> bool:
        """
        Se o perfil visível já estiver na fila, move-o para a frente e descarta
        o current antigo. Recolocar o antigo no topo fazia cards stale voltarem
        antes do perfil que a tela de fato mostra.
        """
        with self._lock:
            if not self._queue:
                logger.debug("Promote visivel abortado: fila vazia")
                return False

            visible_idx = None
            for idx, queued_pair in enumerate(self._queue):
                queued_profile, _ = queued_pair
                match, _ = _match_visible_profile(
                    queued_profile,
                    current_name,
                    current_id,
                    current_visible_age,
                )
                if match:
                    visible_idx = idx
                    break

            if visible_idx is None:
                logger.debug("Perfil visivel nao encontrado na fila para promover: name=%r id=%r age=%s", current_name, current_id, current_visible_age)
                return False

            visible_pair = self._queue[visible_idx]
            del self._queue[visible_idx]
            stale_profile = current_pair[0] if current_pair else {}
            stale_key = _queue_profile_key(stale_profile)
            if stale_key is not None:
                self._queue = deque(
                    pair for pair in self._queue
                    if _queue_profile_key(pair[0]) != stale_key
                )
            self._queue.appendleft(visible_pair)
            self._current_key = _queue_profile_key(visible_pair[0])
            try:
                import state

                state.remove_active_profile(
                    stale_profile.get("name", ""),
                    stale_profile.get("_tinder_id", ""),
                    int(stale_profile.get("age") or 0),
                )
            except Exception:
                logger.debug("Falha ao remover perfil stale dos ativos apos promocao", exc_info=True)
            logger.info(
                "Perfil visivel promovido para frente da fila: name=%r id=%r age=%s idx=%s stale_name=%r stale_id=%r",
                current_name,
                current_id,
                current_visible_age,
                visible_idx,
                stale_profile.get("name", ""),
                stale_profile.get("_tinder_id", ""),
            )
            return True

    def _wait_for_visible_match(
        self,
        profile: dict,
        decision: str,
        timeout_s: float = 0.8,
    ) -> tuple[str, str]:
        """
        Aguarda brevemente o perfil da tela bater com o esperado.
        Se o perfil visível já estiver em outra posição da fila, promove-o.
        """
        import state

        deadline = time.time() + timeout_s
        last_detail = ""
        saw_visible_profile = False

        def _poll_once() -> tuple[str | None, str]:
            nonlocal saw_visible_profile, last_detail
            _dismiss_blocking_modal_if_present()
            current_name, current_id, current_visible_age, current_age = state.get_current_meta()
            if current_age > CURRENT_STALE_AFTER_SECONDS:
                return None, last_detail
            if not (current_name or current_id):
                return None, last_detail

            saw_visible_profile = True

            match, detail = _match_visible_profile(
                profile,
                current_name,
                current_id,
                current_visible_age,
            )
            last_detail = detail
            if match:
                return "match", detail

            promoted = self._promote_visible_profile(
                current_name,
                current_id,
                current_visible_age,
                (profile, decision),
            )
            if promoted:
                return "reordered", f"reordenado para perfil visível | {detail}"

            return None, detail

        while time.time() < deadline:
            status, detail = _poll_once()
            if status:
                logger.debug("Sync status=%s detail=%s", status, detail)
                return status, detail

            time.sleep(0.12)

        # Segunda chance curta de redetecção quando ainda não vimos nenhum perfil
        # visível. Isso ajuda quando o card acabou de trocar e o content script
        # ainda está terminando a leitura do DOM.
        if not saw_visible_profile:
            retry_deadline = time.time() + 0.9
            while time.time() < retry_deadline:
                status, detail = _poll_once()
                if status:
                    logger.debug("Sync retry status=%s detail=%s", status, detail)
                    return status, detail
                time.sleep(0.10)

        if saw_visible_profile:
            logger.debug("Sync out_of_queue detail=%s", last_detail)
            return "out_of_queue", last_detail
        logger.debug("Sync no_detection detail=%s", last_detail)
        return "no_detection", last_detail

    def _wait_for_swipe_effect(
        self,
        profile: dict,
        timeout_s: float,
        direction: str = "",
    ) -> tuple[bool, str]:
        """
        Confirma que o card atual mudou depois do swipe.

        A extensão reporta o perfil visível via /current. Se após o swipe o
        perfil visível não bater mais com o perfil que acabamos de arrastar,
        consideramos que o Tinder aceitou a ação.

        Caso especial — popup de match (swipe direita):
        Quando ocorre um match, o Tinder exibe um popup que cobre o card.
        A extensão não consegue detectar o card abaixo e para de enviar /current.
        Se a última atualização de /current ficar estagnada por >=2s após o swipe
        E o swipe foi para a direita (curtir), tratamos como sucesso — o swipe
        funcionou e o silêncio indica o popup de match, não uma falha.
        """
        import state

        deadline = time.time() + max(0.5, timeout_s)
        last_detail = ""
        # Armazena a idade do /current no momento do swipe para detectar silêncio
        _, _, _, initial_current_age = state.get_current_meta()
        MATCH_POPUP_SILENCE_THRESHOLD = 2.0  # segundos sem /current novo = provável popup

        while time.time() < deadline:
            if state.is_swipe_stop_requested():
                raise SwipeStopped()
            if state.is_swipe_paused():
                paused_at = time.time()
                _wait_while_paused()
                if state.is_swipe_stop_requested():
                    raise SwipeStopped()
                deadline += time.time() - paused_at

            _dismiss_blocking_modal_if_present()
            current_name, current_id, current_visible_age, current_age = state.get_current_meta()

            # Detecção de popup de match: extensão parou de enviar /current
            # (o popup cobre o card e a extensão não consegue detectá-lo)
            # Medimos o CRESCIMENTO da idade desde o início da verificação para evitar
            # falsos positivos quando o /current já estava atrasado antes do swipe.
            if direction in {"right", "up"} and (current_age - initial_current_age) >= MATCH_POPUP_SILENCE_THRESHOLD:
                logger.info(
                    "Popup de match detectado (silêncio de /current >= %.1fs): name=%r",
                    current_age,
                    profile.get("name"),
                )
                return True, f"popup de match detectado (sem /current por {current_age:.1f}s)"

            if current_age <= CURRENT_STALE_AFTER_SECONDS and (current_name or current_id):
                match, detail = _match_visible_profile(
                    profile,
                    current_name,
                    current_id,
                    current_visible_age,
                )
                last_detail = detail
                if not match:
                    visible = current_name or current_id or "novo perfil"
                    if current_visible_age:
                        visible = f"{visible}, {current_visible_age} anos"
                    return True, f"novo perfil detectado: {visible}"

            time.sleep(0.15)

        if last_detail:
            return False, f"perfil ainda parece o mesmo após o swipe | {last_detail}"
        return False, "não recebi confirmação de troca de perfil após o swipe"

    def _perform_verified_swipe(
        self,
        profile: dict,
        direction: str,
        cfg: dict,
    ) -> bool:
        """Executa o swipe e só libera a fila depois de confirmar troca de card."""
        import state

        verify_timeout = max(0.5, float(cfg.get("swipe_verify_timeout_seconds", 6.0)))
        max_attempts = max(1, int(cfg.get("swipe_verify_max_attempts", 2)))
        reload_on_swipe_failure = bool(cfg.get("reload_on_swipe_failure", False))
        converted_superlike_to_like = False

        while True:
            last_detail = ""
            for attempt in range(1, max_attempts + 1):
                _wait_for_tinder_focus()
                _wait_while_paused()
                if state.is_swipe_stop_requested():
                    raise SwipeStopped()
                _dismiss_blocking_modal_if_present()

                if attempt > 1:
                    with state.terminal_lock:
                        print(f"  [swipe] Tentando novamente o mesmo swipe ({attempt}/{max_attempts})")

                logger.info(
                    "Executando swipe: name=%r direction=%s attempt=%s/%s",
                    profile.get("name"),
                    direction,
                    attempt,
                    max_attempts,
                )
                try:
                    swipe(direction)
                    logger.info("Swipe enviado: name=%r direction=%s attempt=%s", profile.get("name"), direction, attempt)
                except SwipePaused:
                    with state.terminal_lock:
                        print("\n  [pause] Movimento interrompido — mouse liberado. Aperte F8 para retomar.\n")
                    _wait_while_paused()
                    if state.is_swipe_stop_requested():
                        raise SwipeStopped()
                    ok, detail = self._wait_for_swipe_effect(profile, verify_timeout, direction)
                    if ok:
                        logger.info("Swipe confirmado apos pausa: name=%r detail=%s", profile.get("name"), detail)
                        return True
                    continue

                ok, detail = self._wait_for_swipe_effect(profile, verify_timeout, direction)
                if ok:
                    logger.info("Swipe confirmado: name=%r direction=%s detail=%s", profile.get("name"), direction, detail)
                    return True
                last_detail = detail

                logger.warning(
                    "Swipe nao confirmado: name=%r direction=%s attempt=%s/%s detail=%s",
                    profile.get("name"),
                    direction,
                    attempt,
                    max_attempts,
                    detail,
                )
                with state.terminal_lock:
                    print(f"  [swipe] ⚠ Não consegui confirmar se o swipe funcionou.")
                    print(f"          {detail}")

            if direction == "up" and not converted_superlike_to_like:
                converted_superlike_to_like = True
                state.mark_super_likes_depleted("super_like_nao_confirmado")
                direction = "right"
                with state.terminal_lock:
                    print("  [super like] Super Like não confirmou — fechando popup se houver e tentando curtir normal")
                desktop_notify(
                    "super_like_to_like",
                    "Tinder IA mudou Super Like para curtida",
                    "Super Like não confirmou ou abriu popup; tentando like normal no mesmo perfil.",
                    urgency="normal",
                )
                _dismiss_blocking_modal_if_present(max_age_seconds=6.0)
                continue

            try:
                if _native_fallback_swipe(direction, cfg):
                    with state.terminal_lock:
                        print("  [swipe] Tentando fallback nativo do Tinder (tecla seta)")
                    ok, detail = self._wait_for_swipe_effect(profile, verify_timeout, direction)
                    if ok:
                        logger.warning(
                            "Swipe confirmado apos fallback nativo: name=%r direction=%s detail=%s",
                            profile.get("name"),
                            direction,
                            detail,
                        )
                        return True
                    last_detail = f"fallback nativo também não confirmou | {detail}"
                    logger.warning(
                        "Fallback nativo nao confirmou swipe: name=%r direction=%s detail=%s",
                        profile.get("name"),
                        direction,
                        detail,
                    )
            except SwipePaused:
                raise
            except SwipeStopped:
                raise
            except Exception:
                logger.exception("Erro no fallback nativo de swipe: name=%r direction=%s", profile.get("name"), direction)

            if reload_on_swipe_failure:
                now = time.time()
                min_reload_interval = max(0.0, float(cfg.get("min_reload_interval_seconds", 90)))
                cooldown_ok = (now - self._last_reload_at) >= min_reload_interval
                if cooldown_ok:
                    reason = (
                        f"Falha persistente ao confirmar swipe: {profile.get('name', '?')} "
                        f"({profile.get('age', '?')} anos)"
                    )
                    if last_detail:
                        reason = f"{reason} | {last_detail}"
                    self._last_reload_at = now
                    self._consecutive_sync_skips = 0
                    request_tinder_reload(
                        reason,
                        source="swiper_swipe_failure",
                        notify_event="swipe_failure_reload",
                        notify_title="Tinder IA vai recarregar a página",
                        notify_message="Swipe não confirmou depois das tentativas.",
                        urgency="critical",
                    )
                    state.clear_active_profiles()
                    with self._lock:
                        pending = len(self._queue)
                        self._queue.clear()
                        self._current_key = None
                        self._generation += 1
                    session_stats.record_error()
                    session_stats.record_pending_cleared(pending)
                    with state.terminal_lock:
                        print("  [swipe] Falha persistente ao confirmar swipe — solicitando reload da página")
                        if last_detail:
                            print(f"          {last_detail}")
                        if pending:
                            print(f"          Fila descartada para evitar swipes desalinhados: {pending}")
                    logger.warning(
                        "Worker abortado por falha persistente de swipe: name=%r direction=%s detail=%s pending_cleared=%s",
                        profile.get("name"),
                        direction,
                        last_detail,
                        pending,
                    )
                    return False
                else:
                    # Cooldown ativo: não pode recarregar, então apenas pula o perfil.
                    # NÃO pausa — continua o próximo da fila automaticamente.
                    remaining = int(min_reload_interval - (now - self._last_reload_at))
                    with state.terminal_lock:
                        print(
                            f"  [swipe] Falha ao confirmar swipe — reload em cooldown (~{remaining}s), pulando perfil"
                        )
                    desktop_notify(
                        "swipe_failure_skip",
                        "Tinder IA pulou um perfil",
                        "Swipe não confirmou e o reload ainda está em cooldown.",
                        urgency="normal",
                    )
                    logger.warning(
                        "Reload de swipe suprimido por cooldown, perfil pulado: name=%r elapsed=%.0fs",
                        profile.get("name"),
                        now - self._last_reload_at,
                    )
                    session_stats.record_error()
                    return None  # pula perfil, worker continua com a fila

            # reload desabilitado: pausa para intervenção manual
            state.set_swipe_paused(True)
            with state.terminal_lock:
                print("  [swipe] Pausado para evitar avançar desalinhado.")
                print("          Confira a tela; se precisar, faça o swipe manualmente e aperte F8 para tentar/continuar.\n")
            _wait_while_paused()
            if state.is_swipe_stop_requested():
                raise SwipeStopped()

            ok, detail = self._wait_for_swipe_effect(profile, verify_timeout, direction)
            if ok:
                logger.info("Swipe confirmado apos intervencao manual: name=%r detail=%s", profile.get("name"), detail)
                return True

    def _worker(self, generation: int) -> None:
        import state

        logger.info("Worker do swiper entrou em execucao generation=%s", generation)
        cfg = _load_swiper_config()
        view_min, view_max = cfg.get("view_time_range", [1.5, 4.0])
        pause_min, pause_max = cfg.get("pause_between_swipes", [0.8, 2.0])
        interactive = cfg.get("interactive_mode", False)
        reload_on_sync_failure = bool(cfg.get("reload_on_sync_failure", False))
        sync_max_wait_count = max(1, int(cfg.get("sync_max_wait_count", 20)))
        sync_retry_sleep = max(0.15, float(cfg.get("sync_retry_sleep", 0.5)))
        min_reload_interval = max(0.0, float(cfg.get("min_reload_interval_seconds", 90)))
        max_consecutive_sync_skips = max(1, int(cfg.get("max_consecutive_sync_skips", 3)))
        recent_ttl = float(cfg.get("recent_swipe_ttl_seconds", 600) or 0)

        while True:
            with self._lock:
                if generation != self._generation:
                    if self._thread is threading.current_thread():
                        self._running = False
                    logger.warning(
                        "Worker antigo finalizado por mudança de geração: worker_generation=%s current_generation=%s",
                        generation,
                        self._generation,
                    )
                    return

            if state.is_swipe_stop_requested():
                pending = self.clear_pending()
                session_stats.record_pending_cleared(pending)
                self.print_session_summary_once("hotkey")
                logger.warning("Worker do swiper finalizado por hotkey antes do proximo perfil")
                return

            if not state.is_scheduler_active():
                _sleep_with_pause(30)
                continue

            _wait_while_paused()

            if state.is_swipe_stop_requested():
                pending = self.clear_pending()
                session_stats.record_pending_cleared(pending)
                self.print_session_summary_once("hotkey")
                logger.warning("Worker do swiper finalizado por hotkey apos pausa")
                return

            pressure, reason, snapshot = is_memory_pressure(load_config())
            if pressure:
                resumed = _wait_for_memory_recovery(reason, snapshot)
                if resumed:
                    continue

                pending = self.clear_pending()
                session_stats.record_pending_cleared(pending)

                with state.terminal_lock:
                    print(
                        "\n  [swipe] Memória crítica "
                        f"({snapshot.get('mem_used_pct', 0.0):.1f}% usada, "
                        f"{snapshot.get('mem_avail_mb', 0.0):.0f}MB livres) — "
                        f"descartando {pending} swipe(s) pendente(s)\n"
                    )

                logger.warning(
                    "Worker do swiper finalizado por memoria critica: reason=%s pending_cleared=%s",
                    reason,
                    pending,
                )

                desktop_notify(
                    "memory_stop",
                    "Tinder IA parou por memoria",
                    f"Fila descartada: {pending} swipe(s).",
                    urgency="critical",
                )
                return

            with self._lock:
                if generation != self._generation:
                    if self._thread is threading.current_thread():
                        self._running = False
                    logger.warning(
                        "Worker antigo finalizado antes de pegar item: worker_generation=%s current_generation=%s",
                        generation,
                        self._generation,
                    )
                    return

                if not self._queue:
                    if self._thread is threading.current_thread():
                        self._running = False
                    logger.info("Worker do swiper finalizado: fila vazia generation=%s", generation)
                    return

                profile, decision = self._queue.popleft()
                pending_after_pop = len(self._queue)
                self._current_key = _queue_profile_key(profile)

            if recent_ttl > 0:
                already_swiped, recent_record = state.is_recently_swiped(
                    profile.get("_tinder_id", ""),
                    profile.get("name", ""),
                    int(profile.get("age") or 0),
                    recent_ttl,
                )
                if already_swiped:
                    state.remove_active_profile(
                        profile.get("name", ""),
                        profile.get("_tinder_id", ""),
                        int(profile.get("age") or 0),
                    )
                    self._clear_current_key(profile)
                    action = (recent_record or {}).get("action", "swipe")
                    source = (recent_record or {}).get("source", "cache")
                    with state.terminal_lock:
                        print(
                            f"  [swipe] Pulando {profile.get('name', '?')} — "
                            f"já confirmado recentemente ({action}, {source})"
                        )
                    logger.info(
                        "Perfil pulado por cache de swipe recente: name=%r id=%r action=%r source=%r",
                        profile.get("name"),
                        profile.get("_tinder_id"),
                        action,
                        source,
                    )
                    continue
            profile, decision = self._refresh_if_model_changed(profile, decision)
            logger.info(
                "Worker processando perfil: generation=%s name=%r age=%r decision=%s pending=%s sync_wait=%s",
                generation,
                profile.get("name"),
                profile.get("age"),
                decision,
                pending_after_pop,
                int(profile.get("_sync_wait_count", 0) or 0),
            )

            if profile.pop("_skip_view_once", False):
                view_time = 0.0
            else:
                # Humanos olham mais para perfis que vão curtir
                if _decision_is_like(decision):
                    view_time = random.uniform(view_min * 1.2, view_max * 1.4)
                else:
                    view_time = random.uniform(view_min * 0.7, view_max)

                # Pequena chance de "hesitar" (~10%)
                if random.random() < 0.10:
                    view_time += random.uniform(2.0, 5.0)

            if view_time > 0 and interactive:
                logger.debug("Aguardando tempo de visualizacao: name=%r seconds=%.2f", profile.get("name"), view_time)
                _sleep_with_pause(view_time)
                if state.is_swipe_stop_requested():
                    pending = self.clear_pending()
                    session_stats.record_pending_cleared(pending)
                    self.print_session_summary_once("hotkey")
                    logger.warning("Worker do swiper finalizado por hotkey durante visualizacao")
                    return

            # ── Sincronização com o perfil visível ────────────────────────
            sync_status, sync_detail = self._wait_for_visible_match(profile, decision)
            if state.is_swipe_stop_requested():
                pending = self.clear_pending()
                session_stats.record_pending_cleared(pending)
                self.print_session_summary_once("hotkey")
                logger.warning("Worker do swiper finalizado por hotkey durante sync")
                return
            if sync_status == "reordered":
                logger.info("Sync reordenou fila: %s", sync_detail)
                with state.terminal_lock:
                    print(f"  [sync] Reordenando fila para o perfil visível na tela")
                    print(f"         {sync_detail}")
                continue

            if sync_status in {"no_detection", "out_of_queue"}:
                wait_count = int(profile.get("_sync_wait_count", 0)) + 1
                profile["_sync_wait_count"] = wait_count

                effective_sync_max_wait_count = sync_max_wait_count
                if sync_status == "out_of_queue":
                    effective_sync_max_wait_count = min(sync_max_wait_count, 3)

                if wait_count >= effective_sync_max_wait_count:
                    now = time.time()
                    cooldown_ok = (now - self._last_reload_at) >= min_reload_interval
                    # Tenta pular o perfil antes de recarregar a página.
                    # Só recarrega após max_consecutive_sync_skips pulos consecutivos
                    # (indica problema estrutural, não só um perfil ruim) e com cooldown.
                    if self._consecutive_sync_skips < max_consecutive_sync_skips:
                        self._consecutive_sync_skips += 1
                        state.remove_active_profile(
                            profile.get("name", ""),
                            profile.get("_tinder_id", ""),
                            int(profile.get("age") or 0),
                        )
                        self._clear_current_key(profile)
                        with state.terminal_lock:
                            print(
                                f"  [sync] Perfil não sincronizou após {effective_sync_max_wait_count} tentativas"
                                f" — pulando ({self._consecutive_sync_skips}/{max_consecutive_sync_skips})"
                            )
                            if sync_detail:
                                print(f"         {sync_detail}")
                        logger.warning(
                            "Perfil pulado por sync persistente: name=%r age=%r status=%s skip=%s/%s",
                            profile.get("name"),
                            profile.get("age"),
                            sync_status,
                            self._consecutive_sync_skips,
                            max_consecutive_sync_skips,
                        )
                        session_stats.record_error()
                        continue

                    # Atingiu o limite de pulos consecutivos
                    if reload_on_sync_failure and cooldown_ok:
                        reason = (
                            "Falha persistente de sincronização: perfil visível não foi detectado"
                            if sync_status == "no_detection"
                            else "Falha persistente de sincronização: tela fora da fila atual"
                        )
                        self._last_reload_at = now
                        self._consecutive_sync_skips = 0
                        request_tinder_reload(
                            reason,
                            source="swiper_sync_failure",
                            notify_event="sync_failure_reload",
                            notify_title="Tinder IA vai ressincronizar o Tinder",
                            notify_message=str(reason or sync_detail or "")[:180],
                            urgency="critical",
                        )
                        state.clear_active_profiles()
                        with self._lock:
                            self._queue.clear()
                            self._current_key = None
                            self._generation += 1
                        with state.terminal_lock:
                            print(
                                f"  [sync] {max_consecutive_sync_skips} perfis pulados consecutivamente"
                                " — solicitando reload da página"
                            )
                            if sync_detail:
                                print(f"         {sync_detail}")
                        logger.warning(
                            "Worker abortado por falha persistente de sync: status=%s detail=%s",
                            sync_status,
                            sync_detail,
                        )
                        return

                    # Cooldown ativo ou reload desabilitado: pula sem recarregar
                    if not cooldown_ok:
                        remaining = int(min_reload_interval - (now - self._last_reload_at))
                        with state.terminal_lock:
                            print(f"  [sync] Reload em cooldown (~{remaining}s restantes) — pulando perfil")
                        logger.warning(
                            "Reload suprimido por cooldown: name=%r elapsed=%.0fs min_interval=%.0fs",
                            profile.get("name"),
                            now - self._last_reload_at,
                            min_reload_interval,
                        )
                    else:
                        with state.terminal_lock:
                            print("  [sync] Perfil não sincronizou; pulando sem swipe")
                            if sync_detail:
                                print(f"         {sync_detail}")
                    self._consecutive_sync_skips = 0
                    state.remove_active_profile(
                        profile.get("name", ""),
                        profile.get("_tinder_id", ""),
                        int(profile.get("age") or 0),
                    )
                    self._clear_current_key(profile)
                    session_stats.record_error()
                    continue

                profile["_skip_view_once"] = True
                with state.terminal_lock:
                    print(
                        f"  [sync] Tentativa {wait_count}/{effective_sync_max_wait_count}: "
                        f"perfil esperado não bate com a tela — {profile.get('name', '?')}"
                    )
                    if sync_detail:
                        print(f"         {sync_detail}")
                self._push_front((profile, decision))
                logger.debug(
                    "Perfil recolocado na frente aguardando sync: name=%r wait_count=%s/%s status=%s detail=%s",
                    profile.get("name"),
                    wait_count,
                    effective_sync_max_wait_count,
                    sync_status,
                    sync_detail,
                )
                now = time.time()
                if now - self._last_sync_warning_at > 5.0:
                    self._last_sync_warning_at = now
                    with state.terminal_lock:
                        if sync_status == "no_detection":
                            print("  [sync] Aguardando detectar o perfil visível da tela antes de abrir o prompt")
                        else:
                            print("  [sync] Perfil visível não pertence à fila atual; aguardando sincronizar")
                            print(f"         {sync_detail}")
                _sleep_with_pause(sync_retry_sleep)
                continue
            else:
                profile["_sync_wait_count"] = 0
                logger.debug("Sync ok para perfil: name=%r detail=%s", profile.get("name"), sync_detail)

            # ── Modo interativo: pergunta ao usuário antes de swipear ─────────
            skip_prompt = profile.get("_skip_prompt", False)

            if interactive and not skip_prompt:
                logger.info("Abrindo prompt interativo: name=%r ai_decision=%s", profile.get("name"), decision)
                final_decision, feedback = _interactive_prompt(profile, decision)
                if state.is_swipe_stop_requested():
                    pending = self.clear_pending()
                    session_stats.record_pending_cleared(pending)
                    self.print_session_summary_once("hotkey")
                    logger.warning("Worker do swiper finalizado por hotkey apos prompt")
                    return
                logger.info("Prompt concluido: name=%r final=%s feedback=%s", profile.get("name"), final_decision, feedback)

                # Imports locais para evitar circular na inicialização
                import model as mdl
                import threading as _th

                saveable = {k: v for k, v in profile.items() if not k.startswith("_")}
                if profile.get("_photo_features"):
                    saveable["_photo_features"] = profile.get("_photo_features")
                if profile.get("_descriptors"):
                    saveable["_descriptors"] = profile.get("_descriptors")
                if profile.get("_distance_km") not in ("", None) and not saveable.get("distance_km"):
                    saveable["distance_km"] = profile.get("_distance_km")
                saveable.update(feedback)
                ml_prob = profile.get("_ml_result", {}).get("probability")
                if ml_prob is not None:
                    saveable["swipe_probability"] = round(float(ml_prob), 4)
                n_before = mdl.count_trainable_real_profiles()
                mdl.save_labeled_profile(saveable, 1 if _decision_is_like(final_decision) else 0)
                n_after = mdl.count_trainable_real_profiles()
                logger.info("Perfil salvo no dataset: name=%r count_before=%s count_after=%s", profile.get("name"), n_before, n_after)

                retrain_every = mdl.load_config().get("model", {}).get("retrain_every", 10)
                if mdl.should_retrain(n_before, n_after, retrain_every):
                    def _retrain():
                        import state as _state
                        logger.info("Retreino automatico iniciado")
                        with _state.terminal_lock:
                            print("\n  [model] Retreinando com novos dados reais...")
                        desktop_notify(
                            "retrain_start",
                            "Tinder IA iniciou retreino",
                            "Atualizando o modelo com suas correcoes.",
                            urgency="normal",
                        )
                        try:
                            mdl.train_model()
                            model_data = mdl.load_model()
                            recalc = self.recalculate_pending(model_data)
                            logger.info("Retreino automatico concluido")
                            with _state.terminal_lock:
                                print(
                                    "  [model] ✓ Modelo atualizado. "
                                    f"Fila recalculada: {recalc['updated']}/{recalc['checked']} "
                                    f"pendentes, {recalc['changed']} mudança(s) de decisão.\n"
                                )
                            desktop_notify(
                                "retrain_done",
                                "Tinder IA concluiu retreino",
                                f"Fila recalculada: {recalc['updated']}/{recalc['checked']} pendentes.",
                                urgency="normal",
                            )
                        except Exception:
                            logger.exception("Erro no retreino automatico")
                            with _state.terminal_lock:
                                print("  [model] ⚠ Erro ao retreinar. Veja data/logs/tinder_ia.log\n")
                            desktop_notify(
                                "retrain_failed",
                                "Tinder IA: erro no retreino",
                                "Veja data/logs/tinder_ia.log.",
                                urgency="critical",
                            )
                    _th.Thread(target=_retrain, daemon=True).start()
            else:
                if view_time > 0:
                    logger.debug(
                        "Visualizacao automatica do perfil: name=%r seconds=%.2f",
                        profile.get("name"),
                        view_time,
                    )
                    try:
                        _wait_for_tinder_focus()
                        _browse_card_photos(view_time, cfg)
                    except SwipePaused:
                        with state.terminal_lock:
                            print("\n  [pause] Navegação automática interrompida. Aperte F8 para retomar.\n")
                        profile["_skip_view_once"] = True
                        self._push_front((profile, decision))
                        _wait_while_paused()
                        if state.is_swipe_stop_requested():
                            pending = self.clear_pending()
                            session_stats.record_pending_cleared(pending)
                            self.print_session_summary_once("hotkey")
                            logger.warning("Worker do swiper finalizado por hotkey durante pausa")
                            return
                        continue
                    except SwipeStopped:
                        pending = self.clear_pending()
                        session_stats.record_pending_cleared(pending)
                        self.print_session_summary_once("hotkey")
                        logger.warning("Worker do swiper finalizado por hotkey durante navegacao de fotos")
                        return

                final_decision = decision
                if skip_prompt:
                    reason = profile.get("_filter_reason") or "Filtro automático"
                    session_stats.record_auto_filtered(profile, reason)
                    with state.terminal_lock:
                        print(
                            f"\n  [auto] {profile.get('name', '?')} ({profile.get('age', '?')} anos) "
                            f"→ ◄ PASSOU  |  {reason}"
                        )

            if not interactive and _decision_is_like(final_decision) and not _decision_is_super_like(final_decision):
                super_ok, super_reason = _super_like_gate(profile, final_decision)
                if super_ok:
                    final_decision = "SUPER_LIKE"
                    profile["_super_like_applied"] = True
                    profile["_super_like_reason"] = super_reason
                    logger.info(
                        "Curtida promovida para super like: name=%r reason=%s",
                        profile.get("name"),
                        super_reason,
                    )
                    with state.terminal_lock:
                        print(
                            f"  [super like] {profile.get('name', '?')} qualifica: {super_reason}"
                        )
                else:
                    logger.debug(
                        "Super like nao aplicado: name=%r reason=%s",
                        profile.get("name"),
                        super_reason,
                    )

            # Atualiza vetor de preferência com a decisão final (humana ou da IA)
            embedding = profile.get("_photo_features", {}).get("_embedding")
            if embedding is not None:
                from face_embeddings import update_preference
                try:
                    update_preference(embedding, final_decision)
                    logger.debug("Preferencia visual atualizada: name=%r decision=%s", profile.get("name"), final_decision)
                except Exception:
                    logger.exception("Erro ao atualizar preferencia visual: name=%r", profile.get("name"))

            if _decision_is_super_like(final_decision):
                direction = "up"
            else:
                direction = "right" if _decision_is_like(final_decision) else "left"
            _wait_for_tinder_focus()
            _wait_while_paused()
            if state.is_swipe_stop_requested():
                pending = self.clear_pending()
                session_stats.record_pending_cleared(pending)
                self.print_session_summary_once("hotkey")
                logger.warning("Worker do swiper finalizado por hotkey antes do swipe")
                return
            with state.terminal_lock:
                action_label = (
                    "SUPER LIKE ▲"
                    if direction == "up"
                    else ("CURTIU ►" if direction == "right" else "◄ PASSOU")
                )
                print(
                    f"  [swipe] {profile.get('name', '?')} ({profile.get('age', '?')} anos) "
                    f"→ {action_label}"
                )

            try:
                swipe_ok = self._perform_verified_swipe(profile, direction, cfg)
                if swipe_ok is None:
                    # Cooldown ativo: perfil pulado, mas fila continua normalmente
                    state.remove_active_profile(
                        profile.get("name", ""),
                        profile.get("_tinder_id", ""),
                        int(profile.get("age") or 0),
                    )
                    self._clear_current_key(profile)
                    continue
                if not swipe_ok:
                    self.print_session_summary_once("reload")
                    return
                self._consecutive_sync_skips = 0  # swipe bem-sucedido limpa histórico de falhas
                state.mark_recent_swipe(
                    tinder_id=profile.get("_tinder_id", ""),
                    name=profile.get("name", ""),
                    age=int(profile.get("age") or 0),
                    action=final_decision,
                    source="autogui",
                    status=200,
                )
                session_stats.record_swipe(
                    profile=profile,
                    ai_decision=decision,
                    final_decision=final_decision,
                    interactive=bool(interactive and not skip_prompt),
                    feedback=feedback if interactive and not skip_prompt else {},
                )
                if not interactive:
                    try:
                        from review_queue import enqueue_auto_decision

                        enqueue_auto_decision(profile, decision, final_decision)
                    except Exception:
                        logger.exception("Falha ao enfileirar perfil para revisao: name=%r", profile.get("name"))
            except pyautogui.FailSafeException:
                logger.warning("Failsafe do PyAutoGUI acionado")
                with state.terminal_lock:
                    print("\n  [swipe] PARADO — mouse no canto superior esquerdo (failsafe)\n")

                with self._lock:
                    self._queue.clear()
                    self._current_key = None
                    self._generation += 1
                    if self._thread is threading.current_thread():
                        self._running = False

                state.clear_active_profiles()
                return

            except SwipePaused:
                with state.terminal_lock:
                    print("\n  [pause] Swipe pausado antes de confirmar. Aperte F8 para retomar.\n")
                _wait_while_paused()
                if state.is_swipe_stop_requested():
                    pending = self.clear_pending()
                    session_stats.record_pending_cleared(pending)
                    self.print_session_summary_once("hotkey")
                    logger.warning("Worker do swiper finalizado por hotkey apos pausa de swipe")
                    return
                profile["_skip_view_once"] = True
                self._push_front((profile, decision))
                continue
            except SwipeStopped:
                pending = self.clear_pending()
                session_stats.record_pending_cleared(pending)
                self.print_session_summary_once("hotkey")
                logger.warning("Worker do swiper finalizado por hotkey durante swipe")
                return
            except Exception as e:
                session_stats.record_error()
                logger.exception("Erro inesperado no swipe: name=%r", profile.get("name"))
                with state.terminal_lock:
                    print(f"\n  [swipe] ⚠ Erro inesperado: {type(e).__name__}: {e}\n")

            state.remove_active_profile(
                profile.get("name", ""),
                profile.get("_tinder_id", ""),
                int(profile.get("age") or 0),
            )
            self._clear_current_key(profile)
            _sleep_with_pause(random.uniform(pause_min, pause_max))


def _wait_for_key(valid_chars: set[str] | None = None, allow_enter: bool = True) -> str:
    """
    Aguarda uma tecla independente do foco do teclado.
    Usa pynput para captura global — funciona mesmo com VSCode ou browser em foco.
    Retorna '' para Enter quando allow_enter=True.
    """
    try:
        from pynput import keyboard as pynput_kb
    except ImportError:
        logger.warning("pynput nao instalado; usando input() no terminal")
        raw = input().strip().lower()
        if raw == "" and allow_enter:
            return ""
        if valid_chars is None or raw in valid_chars:
            return raw
        return ""

    result = ['']
    done = threading.Event()
    valid = {c.lower() for c in valid_chars} if valid_chars else None
    try:
        import state as prompt_state
    except Exception:
        prompt_state = None

    def on_press(key):
        if prompt_state is not None and prompt_state.is_swipe_stop_requested():
            result[0] = "__stop__"
            done.set()
            return False

        if prompt_state is not None and prompt_state.is_swipe_paused():
            return None

        try:
            char = key.char
        except AttributeError:
            char = None

        if char:
            lowered = char.lower()
            if valid is None or lowered in valid:
                result[0] = lowered
                done.set()
                return False  # para o listener

        if allow_enter and key == pynput_kb.Key.enter:
            done.set()
            return False

    listener = pynput_kb.Listener(on_press=on_press)
    listener.start()
    try:
        import state
        while not done.wait(0.10):
            if state.is_swipe_stop_requested():
                result[0] = "__stop__"
                done.set()
                break
    except Exception:
        done.wait()
    listener.stop()
    logger.debug("Tecla capturada no prompt: %r", result[0] or "ENTER")
    return result[0]


def _ask_photo_detail(final_decision: str) -> str:
    """Quando o motivo é foto, pergunta qual parte da foto pesou."""
    import state

    sentiment = "gostou" if _decision_is_like(final_decision) else "não gostou"
    with state.terminal_lock:
        print(f"  O que na foto você {sentiment}?")
        print("    [r] rosto")
        print("    [c] corpo")
        print("    [a] contexto/cenário/fundo")
        print("    [e] estilo/pose/qualidade da foto")
        print("    [o/Enter] foto geral / sem certeza")
        print("  > ", end="", flush=True)

    try:
        chosen = _wait_for_key({"r", "c", "a", "e", "o"}, allow_enter=True)
    except Exception:
        logger.exception("Erro ao aguardar detalhe de foto")
        chosen = ""

    if chosen == "__stop__":
        chosen = ""

    reason = PHOTO_KEY_TO_REASON.get(chosen, "photo_general")
    with state.terminal_lock:
        print(f"\n  → Detalhe da foto: {PHOTO_REASON_LABELS.get(reason, reason)}")
    return reason


def _ask_descriptor_detail(profile: dict, final_decision: str) -> str:
    """Quando o motivo é descritor, pergunta qual descritor pesou."""
    import state

    descriptors = profile.get("_descriptors") or {}
    if not descriptors:
        return "descritores"

    items = list(descriptors.items())[:9]
    sentiment = "gostou" if _decision_is_like(final_decision) else "não gostou"
    with state.terminal_lock:
        print(f"  Qual descritor você {sentiment}?")
        for idx, (key, value) in enumerate(items, 1):
            print(f"    [{idx}] {key}: {value}")
        print("  Número do descritor ou Enter para pular: ", end="", flush=True)

    valid = {str(i) for i in range(1, len(items) + 1)}
    try:
        chosen = _wait_for_key(valid, allow_enter=True)
    except Exception:
        logger.exception("Erro ao aguardar detalhe de descritor")
        chosen = ""

    if not chosen or chosen == "__stop__":
        return "descritores"

    key, value = items[int(chosen) - 1]
    detail = f"{key}: {value}"
    with state.terminal_lock:
        print(f"\n  → Descritor registrado: {detail}")
    return detail


def _ask_feedback_intensity(default: int = 2) -> int:
    """Pergunta o quanto aquele motivo pesou na decisão."""
    import state

    default = default if default in {1, 2, 3} else 2
    with state.terminal_lock:
        print(
            f"  Intensidade do motivo [1=pouco | 2=médio | 3=muito | Enter={default}]: ",
            end="",
            flush=True,
        )

    try:
        chosen = _wait_for_key({"1", "2", "3"}, allow_enter=True)
    except Exception:
        logger.exception("Erro ao aguardar intensidade do feedback")
        chosen = ""

    if chosen == "__stop__":
        return 2
    if chosen in {"1", "2", "3"}:
        intensity = int(chosen)
    else:
        intensity = default

    with state.terminal_lock:
        label = {1: "pouco", 2: "médio", 3: "muito"}[intensity]
        print(f"\n  → Intensidade registrada: {intensity} ({label})")
    return intensity


def _interactive_prompt(profile: dict, ai_decision: str) -> tuple[str, dict]:
    """
    Exibe todas as infos do perfil + recomendação da IA e aguarda input do usuário.
    Adquire terminal_lock para que o servidor não imprima por cima do prompt.
    """
    import state

    ml = profile.get("_ml_result", {})
    prob = ml.get("probability", 0.5)
    conf = int(prob * 100) if ai_decision == "CURTIR" else int((1 - prob) * 100)
    rec = "✓ CURTIR" if ai_decision == "CURTIR" else "✗ NÃO CURTIR"
    current_name, current_id, current_visible_age, current_age = state.get_current_meta()
    if current_age > CURRENT_STALE_AFTER_SECONDS:
        current_name, current_id, current_visible_age = "", "", 0
    sync_ok, _ = _match_visible_profile(profile, current_name, current_id, current_visible_age)

    final = ai_decision
    feedback = {
        "ai_decision": ai_decision,
        "final_decision": ai_decision,
        "manual_corrected": 0,
        "feedback_domain": "",
        "feedback_reason": "",
        "feedback_intensity": "",
        "feedback_sentiment": "",
    }
    state.set_prompt_active(True)
    try:
        with state.terminal_lock:
            current_label = current_name or "(aguardando detecção)"
            if current_name and current_visible_age:
                current_label = f"{current_name}, {current_visible_age} anos"
            print(
                format_interactive_prompt(
                    profile=profile,
                    ai_decision=ai_decision,
                    current_label=current_label,
                    sync_ok=sync_ok,
                ),
                end="",
                flush=True,
            )

        try:
            raw = _wait_for_key({"c", "p", "n"}, allow_enter=True)
        except Exception:
            logger.exception("Erro ao aguardar tecla de decisao no prompt")
            raw = ""

        if raw == "n":
            final = "NÃO CURTIR" if ai_decision == "CURTIR" else "CURTIR"
        elif raw == "c":
            final = "CURTIR"
        elif raw == "p":
            final = "NÃO CURTIR"

        feedback["final_decision"] = final
        feedback["manual_corrected"] = 1 if final != ai_decision else 0

        with state.terminal_lock:
            print()
            if raw:
                print(f"  → Decisão: {final}")
            if final != ai_decision:
                print(f"  → Corrigido de {ai_decision} para: {final}")

            print("  Motivo principal [f=foto | i=interesses | b=bio | d=descritores | o/Enter=sem certeza]: ", end="", flush=True)

        try:
            reason_key = _wait_for_key({"f", "i", "b", "d", "o"}, allow_enter=True)
        except Exception:
            logger.exception("Erro ao aguardar tecla de motivo no prompt")
            reason_key = ""

        if reason_key == "__stop__":
            reason_key = "o"

        domain = KEY_TO_DOMAIN.get(reason_key, "other")
        reason = ""
        if domain == "photo":
            reason = _ask_photo_detail(final)
        elif domain == "descriptors":
            reason = _ask_descriptor_detail(profile, final)

        intensity = _ask_feedback_intensity(default=1 if domain == "other" else 2)
        feedback.update(
            normalize_feedback(
                profile,
                final,
                feedback_domain=domain,
                feedback_reason=reason,
                feedback_intensity=intensity,
                ai_decision=ai_decision,
            )
        )
        feedback["final_decision"] = final

        with state.terminal_lock:
            label = DOMAIN_LABELS.get(feedback["feedback_domain"], feedback["feedback_domain"])
            reason_label = (
                PHOTO_REASON_LABELS.get(feedback["feedback_reason"], feedback["feedback_reason"])
                if feedback["feedback_domain"] == "photo"
                else feedback["feedback_reason"]
            )
            print()
            print(
                f"  → Motivo registrado: {label} "
                f"/ {reason_label} "
                f"(intensidade {feedback['feedback_intensity']}, {feedback['feedback_sentiment']})"
            )
        logger.info("Feedback interativo registrado: name=%r final=%s feedback=%s", profile.get("name"), final, feedback)
    finally:
        state.set_prompt_active(False)

    return final, feedback


# Instância global — importada pelo server.py
swiper_queue = SwiperQueue()
