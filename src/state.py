"""
Estado compartilhado entre server.py e swiper.py.
Ambos rodam no mesmo processo, então variáveis de módulo são suficientes.
"""

import time
import unicodedata
from threading import Lock
from logging_config import get_logger

logger = get_logger(__name__)

_lock = Lock()

# Lock que protege o terminal: quem o tem pode imprimir e aguardar input.
# O servidor e o swiper o disputam, nunca imprimem ao mesmo tempo.
terminal_lock = Lock()

# Nome do perfil atualmente visível na tela (atualizado pelo /current endpoint)
current_visible_name: str = ""

# ID Tinder do perfil atualmente visível
current_visible_id: str = ""

# Idade do perfil atualmente visível
current_visible_age: int = 0

# Timestamp da última atualização via /current
current_updated_at: float = 0.0
current_previous_visible_name: str = ""
current_previous_visible_id: str = ""
current_previous_visible_age: int = 0
current_previous_updated_at: float = 0.0
CURRENT_FLIPFLOP_IGNORE_SECONDS = 1.2

# Estado do botão de Super Like reportado pela extensão no card visível.
current_super_like_available: bool | None = None
current_super_like_reason: str = "unknown"
current_super_like_updated_at: float = 0.0

# Saldo real de Super Likes reportado pelo endpoint /v2/profile.
super_like_balance: dict = {}
super_like_balance_updated_at: float = 0.0

# Scheduler: True = dentro do horário ativo, False = fora
scheduler_active: bool = True

# True enquanto o prompt interativo está visível/aguardando tecla.
prompt_active: bool = False

# Pausa manual dos swipes via hotkey global.
swipe_paused: bool = False

# Pedido manual para encerrar a fila de swipes da sessão atual.
swipe_stop_requested: bool = False
swipe_stop_reason: str = ""
swipe_stop_requested_at: float = 0.0

# Sinaliza que a página deve ser recarregada por perda de sincronização.
reload_requested: bool = False
reload_reason: str = ""
reload_requested_at: float = 0.0
reload_generation: int = 0

# Sinaliza que a extensão deve navegar para uma URL específica. Usado quando o
# Tinder cai em compra/paywall e precisa voltar para a tela de swipes.
navigation_requested: bool = False
navigation_target_url: str = ""
navigation_reason: str = ""
navigation_requested_at: float = 0.0
navigation_generation: int = 0

# Estado reportado pela extensão: True quando a aba do Tinder está visível e
# focada. Usado para ignorar capturas enquanto o usuário está em outra tela.
tinder_capture_active: bool = True
tinder_capture_reason: str = "initial"
tinder_capture_url: str = ""
tinder_capture_updated_at: float = 0.0

# Alvo de clique para fechar modal bloqueante reportado pela extensão.
blocking_modal_kind: str = ""
blocking_modal_target: str = ""
blocking_modal_x: int = 0
blocking_modal_y: int = 0
blocking_modal_updated_at: float = 0.0

# Conjunto dos perfis ainda pendentes de swipe/processamento.
active_profile_ids: set[str] = set()
active_profile_names: set[str] = set()
active_profile_keys: set[str] = set()

# Swipes confirmados recentemente. Usado para evitar clicar de novo quando o
# Tinder devolve o mesmo perfil em outra leva/retry.
recent_swipes: dict[tuple[str, str], dict] = {}


def _norm_name(name: str) -> str:
    base = unicodedata.normalize("NFD", (name or "").strip().lower())
    return "".join(ch for ch in base if unicodedata.category(ch) != "Mn")


def _profile_key(name: str, age: int = 0) -> str:
    norm_name = _norm_name(name)
    return f"{norm_name}::{int(age or 0)}" if norm_name else ""


def _prune_recent_swipes_locked(now: float, max_age_seconds: float) -> None:
    if max_age_seconds <= 0:
        recent_swipes.clear()
        return
    expired = [
        key for key, record in recent_swipes.items()
        if now - float(record.get("swiped_at", 0.0) or 0.0) > max_age_seconds
    ]
    for key in expired:
        recent_swipes.pop(key, None)


def mark_recent_swipe(
    tinder_id: str = "",
    name: str = "",
    age: int = 0,
    action: str = "",
    source: str = "",
    status: int | None = None,
    match: bool | None = None,
    likes_remaining: int | None = None,
) -> dict:
    clean_id = (tinder_id or "").strip()
    clean_name = (name or "").strip()
    clean_age = int(age or 0)
    now = time.time()
    record = {
        "tinder_id": clean_id,
        "name": clean_name,
        "age": clean_age,
        "action": (action or "").strip(),
        "source": (source or "").strip(),
        "status": status,
        "match": match,
        "likes_remaining": likes_remaining,
        "swiped_at": now,
    }
    with _lock:
        _prune_recent_swipes_locked(now, 60 * 60)
        if clean_id:
            recent_swipes[("id", clean_id)] = record
        key = _profile_key(clean_name, clean_age)
        if key:
            recent_swipes[("name_age", key)] = record
    logger.info(
        "Swipe recente marcado: id=%r name=%r age=%s action=%r source=%r status=%r",
        clean_id,
        clean_name,
        clean_age,
        record["action"],
        record["source"],
        status,
    )
    return record


def is_recently_swiped(
    tinder_id: str = "",
    name: str = "",
    age: int = 0,
    max_age_seconds: float = 600.0,
) -> tuple[bool, dict | None]:
    clean_id = (tinder_id or "").strip()
    key = _profile_key(name, age)
    now = time.time()
    with _lock:
        _prune_recent_swipes_locked(now, max_age_seconds)
        if clean_id:
            record = recent_swipes.get(("id", clean_id))
            if record:
                return True, dict(record)
        if key:
            record = recent_swipes.get(("name_age", key))
            if record:
                return True, dict(record)
    return False, None


def clear_recent_swipes() -> int:
    with _lock:
        count = len(recent_swipes)
        recent_swipes.clear()
    logger.info("Cache de swipes recentes limpo: %s entrada(s)", count)
    return count


def set_current(
    name: str,
    tinder_id: str = "",
    age: int = 0,
    super_like_available: bool | None = None,
    super_like_reason: str = "",
) -> None:
    global current_visible_name, current_visible_id, current_visible_age, current_updated_at
    global current_previous_visible_name, current_previous_visible_id, current_previous_visible_age, current_previous_updated_at
    global current_super_like_available, current_super_like_reason, current_super_like_updated_at
    clean_name = name.strip()
    clean_id = tinder_id.strip()
    clean_age = int(age or 0)
    clean_super_reason = (super_like_reason or "").strip() or "unknown"
    changed = False
    ignored_flipflop = False
    with _lock:
        now = time.time()
        if (
            clean_id
            and current_visible_id
            and clean_id != current_visible_id
            and clean_id == current_previous_visible_id
            and _norm_name(clean_name) == _norm_name(current_visible_name)
            and _norm_name(clean_name) == _norm_name(current_previous_visible_name)
            and clean_age == current_visible_age
            and clean_age == current_previous_visible_age
            and now - current_previous_updated_at <= CURRENT_FLIPFLOP_IGNORE_SECONDS
        ):
            ignored_flipflop = True
        if ignored_flipflop:
            if super_like_available is not None:
                current_super_like_available = bool(super_like_available)
                current_super_like_reason = clean_super_reason
                current_super_like_updated_at = now
        else:
            old_name = current_visible_name
            old_id = current_visible_id
            old_age = current_visible_age
            changed = (
                current_visible_name != clean_name
                or current_visible_id != clean_id
                or current_visible_age != clean_age
            )
            if changed:
                current_previous_visible_name = old_name
                current_previous_visible_id = old_id
                current_previous_visible_age = old_age
                current_previous_updated_at = now
            current_visible_name = clean_name
            current_visible_id = clean_id
            current_visible_age = clean_age
            current_updated_at = now
            if super_like_available is not None:
                current_super_like_available = bool(super_like_available)
                current_super_like_reason = clean_super_reason
                current_super_like_updated_at = current_updated_at
            elif changed:
                current_super_like_available = None
                current_super_like_reason = "unknown"
                current_super_like_updated_at = 0.0
    if ignored_flipflop:
        logger.info(
            "Perfil visivel ignorado por flip-flop curto: name=%r age=%s id=%r current_id=%r previous_id=%r",
            clean_name,
            clean_age,
            clean_id,
            current_visible_id,
            current_previous_visible_id,
        )
        return
    if changed:
        logger.info("Perfil visivel atualizado: name=%r age=%s id=%r", clean_name, clean_age, clean_id)


def get_current() -> tuple[str, str]:
    with _lock:
        return current_visible_name, current_visible_id


def get_current_meta() -> tuple[str, str, int, float]:
    with _lock:
        age_seconds = time.time() - current_updated_at if current_updated_at else float("inf")
        return current_visible_name, current_visible_id, current_visible_age, age_seconds


def get_current_super_like_state(max_age_seconds: float = 8.0) -> tuple[bool | None, str, float]:
    with _lock:
        age_seconds = (
            time.time() - current_super_like_updated_at
            if current_super_like_updated_at else float("inf")
        )
        if max_age_seconds > 0 and age_seconds > max_age_seconds:
            return None, "stale", age_seconds
        return current_super_like_available, current_super_like_reason, age_seconds


def set_super_like_balance(balance: dict) -> None:
    """Atualiza o saldo de Super Likes observado no network do Tinder."""
    global super_like_balance, super_like_balance_updated_at
    clean = dict(balance or {})
    with _lock:
        super_like_balance = clean
        super_like_balance_updated_at = time.time()
    logger.info(
        "Saldo de Super Likes atualizado: remaining=%r alc=%r new_alc=%r resets_at=%r",
        clean.get("remaining"),
        clean.get("alc_remaining"),
        clean.get("new_alc_remaining"),
        clean.get("resets_at"),
    )


def get_super_like_balance(max_age_seconds: float = 300.0) -> tuple[dict | None, float]:
    """Retorna (saldo, idade_em_segundos), ou (None, idade) se estiver velho."""
    with _lock:
        age_seconds = (
            time.time() - super_like_balance_updated_at
            if super_like_balance_updated_at else float("inf")
        )
        if max_age_seconds > 0 and age_seconds > max_age_seconds:
            return None, age_seconds
        return dict(super_like_balance), age_seconds


def mark_super_likes_depleted(reason: str = "") -> None:
    """Marca saldo zero quando o Tinder mostra modal de falta de Super Likes."""
    set_super_like_balance({
        "remaining": 0,
        "alc_remaining": 0,
        "new_alc_remaining": 0,
        "available": False,
        "source": reason or "modal",
    })


def add_active_profiles(ids: list[str], names: list[str], name_ages: list[tuple[str, int]] | None = None) -> None:
    global active_profile_ids, active_profile_names, active_profile_keys

    ids_set = {x.strip() for x in ids if x and x.strip()}
    names_set = {_norm_name(x) for x in names if x and x.strip()}
    keys_set = {
        _profile_key(name, age)
        for name, age in (name_ages or [])
        if _profile_key(name, age)
    }

    with _lock:
        active_profile_ids |= ids_set
        active_profile_names |= names_set
        active_profile_keys |= keys_set
    logger.info("Perfis ativos adicionados: ids=%s names=%s keys=%s", len(ids_set), len(names_set), len(keys_set))


def remove_active_profile(name: str = "", tinder_id: str = "", age: int = 0) -> None:
    with _lock:
        if tinder_id and tinder_id.strip():
            active_profile_ids.discard(tinder_id.strip())

        if name and name.strip():
            active_profile_names.discard(_norm_name(name))

        key = _profile_key(name, age)
        if key:
            active_profile_keys.discard(key)


def clear_active_profiles() -> None:
    global active_profile_ids, active_profile_names, active_profile_keys
    global current_visible_name, current_visible_id, current_visible_age, current_updated_at
    global current_previous_visible_name, current_previous_visible_id, current_previous_visible_age, current_previous_updated_at
    global current_super_like_available, current_super_like_reason, current_super_like_updated_at
    global super_like_balance, super_like_balance_updated_at
    global blocking_modal_kind, blocking_modal_target, blocking_modal_x, blocking_modal_y, blocking_modal_updated_at

    with _lock:
        active_profile_ids.clear()
        active_profile_names.clear()
        active_profile_keys.clear()
        current_visible_name = ""
        current_visible_id = ""
        current_visible_age = 0
        current_updated_at = 0.0
        current_previous_visible_name = ""
        current_previous_visible_id = ""
        current_previous_visible_age = 0
        current_previous_updated_at = 0.0
        current_super_like_available = None
        current_super_like_reason = "unknown"
        current_super_like_updated_at = 0.0
        super_like_balance = {}
        super_like_balance_updated_at = 0.0
        blocking_modal_kind = ""
        blocking_modal_target = ""
        blocking_modal_x = 0
        blocking_modal_y = 0
        blocking_modal_updated_at = 0.0
    logger.info("Perfis ativos e current visivel limpos")


def belongs_to_active_profiles(name: str, tinder_id: str = "", age: int = 0) -> bool:
    with _lock:
        if not active_profile_ids and not active_profile_names and not active_profile_keys:
            return True

        if tinder_id and tinder_id.strip() and tinder_id.strip() in active_profile_ids:
            return True

        if name and age:
            return _profile_key(name, age) in active_profile_keys

        if name and _norm_name(name) in active_profile_names:
            return True

        return False


def set_scheduler_active(active: bool) -> None:
    global scheduler_active
    with _lock:
        scheduler_active = active


def is_scheduler_active() -> bool:
    with _lock:
        return scheduler_active


def set_prompt_active(active: bool) -> None:
    global prompt_active
    with _lock:
        prompt_active = active


def is_prompt_active() -> bool:
    with _lock:
        return prompt_active


def set_swipe_paused(paused: bool) -> None:
    global swipe_paused
    with _lock:
        swipe_paused = bool(paused)
    logger.warning("Swipes %s manualmente", "pausados" if paused else "retomados")


def toggle_swipe_pause() -> bool:
    global swipe_paused
    with _lock:
        swipe_paused = not swipe_paused
        paused = swipe_paused
    logger.warning("Swipes %s por hotkey", "pausados" if paused else "retomados")
    return paused


def is_swipe_paused() -> bool:
    with _lock:
        return swipe_paused


def request_swipe_stop(reason: str = "") -> None:
    global swipe_stop_requested, swipe_stop_reason, swipe_stop_requested_at, swipe_paused
    with _lock:
        swipe_stop_requested = True
        swipe_stop_reason = (reason or "").strip()
        swipe_stop_requested_at = time.time()
        swipe_paused = False
    logger.warning("Encerramento dos swipes solicitado: %s", swipe_stop_reason or "(sem motivo)")


def is_swipe_stop_requested() -> bool:
    with _lock:
        return swipe_stop_requested


def get_swipe_stop_request() -> tuple[bool, str, float]:
    with _lock:
        return swipe_stop_requested, swipe_stop_reason, swipe_stop_requested_at


def clear_swipe_stop_request() -> None:
    global swipe_stop_requested, swipe_stop_reason, swipe_stop_requested_at
    with _lock:
        swipe_stop_requested = False
        swipe_stop_reason = ""
        swipe_stop_requested_at = 0.0
    logger.info("Pedido de encerramento dos swipes limpo")


def request_reload(reason: str = "") -> int:
    global reload_requested, reload_reason, reload_requested_at, reload_generation
    with _lock:
        reload_requested = True
        reload_reason = (reason or "").strip()
        reload_requested_at = time.time()
        reload_generation += 1
        generation = reload_generation
    logger.warning("Reload solicitado: %s generation=%s", reload_reason or "(sem motivo)", generation)
    return generation


def peek_reload_request() -> tuple[bool, str, float]:
    with _lock:
        return reload_requested, reload_reason, reload_requested_at


def is_reload_pending() -> bool:
    with _lock:
        return reload_requested


def get_reload_generation() -> int:
    with _lock:
        return reload_generation


def clear_reload_request() -> None:
    global reload_requested, reload_reason, reload_requested_at
    with _lock:
        reload_requested = False
        reload_reason = ""
        reload_requested_at = 0.0
    logger.info("Pedido de reload limpo")


def request_navigation(url: str, reason: str = "") -> int:
    global navigation_requested, navigation_target_url, navigation_reason
    global navigation_requested_at, navigation_generation
    target = (url or "").strip()
    clean_reason = (reason or "").strip()
    if not target:
        return 0
    with _lock:
        navigation_requested = True
        navigation_target_url = target
        navigation_reason = clean_reason
        navigation_requested_at = time.time()
        navigation_generation += 1
        generation = navigation_generation
    logger.warning(
        "Navegacao solicitada: target=%s reason=%s generation=%s",
        target,
        clean_reason or "(sem motivo)",
        generation,
    )
    return generation


def peek_navigation_request() -> tuple[bool, str, str, float, int]:
    with _lock:
        return (
            navigation_requested,
            navigation_target_url,
            navigation_reason,
            navigation_requested_at,
            navigation_generation,
        )


def clear_navigation_request() -> None:
    global navigation_requested, navigation_target_url, navigation_reason, navigation_requested_at
    with _lock:
        navigation_requested = False
        navigation_target_url = ""
        navigation_reason = ""
        navigation_requested_at = 0.0
    logger.info("Pedido de navegacao limpo")


def set_tinder_capture_active(active: bool, reason: str = "", url: str = "") -> None:
    global tinder_capture_active, tinder_capture_reason, tinder_capture_url, tinder_capture_updated_at
    reason = (reason or "").strip()
    url = (url or "").strip()
    changed = False
    with _lock:
        changed = (
            tinder_capture_active != bool(active)
            or tinder_capture_reason != reason
            or tinder_capture_url != url
        )
        tinder_capture_active = bool(active)
        tinder_capture_reason = reason
        tinder_capture_url = url
        tinder_capture_updated_at = time.time()
    if changed:
        logger.info("Captura Tinder %s: reason=%r url=%r", "ativa" if active else "pausada", reason, url)


def is_tinder_capture_active(max_age_seconds: float = 30.0) -> bool:
    with _lock:
        if tinder_capture_updated_at <= 0:
            return True
        if max_age_seconds > 0 and time.time() - tinder_capture_updated_at > max_age_seconds:
            return True
        return tinder_capture_active


def get_tinder_capture_state() -> tuple[bool, str, str, float]:
    with _lock:
        age_seconds = time.time() - tinder_capture_updated_at if tinder_capture_updated_at else float("inf")
        return tinder_capture_active, tinder_capture_reason, tinder_capture_url, age_seconds


def set_blocking_modal_target(kind: str, x: int, y: int, target: str = "") -> None:
    global blocking_modal_kind, blocking_modal_target, blocking_modal_x, blocking_modal_y, blocking_modal_updated_at

    clean_kind = (kind or "").strip()
    clean_target = (target or "").strip()
    with _lock:
        blocking_modal_kind = clean_kind
        blocking_modal_target = clean_target
        blocking_modal_x = int(x or 0)
        blocking_modal_y = int(y or 0)
        blocking_modal_updated_at = time.time()
    logger.info(
        "Modal bloqueante reportado: kind=%r target=%r x=%s y=%s",
        clean_kind,
        clean_target,
        int(x or 0),
        int(y or 0),
    )


def get_blocking_modal_target(max_age_seconds: float = 2.5) -> dict | None:
    with _lock:
        if not blocking_modal_kind or (blocking_modal_x == 0 and blocking_modal_y == 0):
            return None

        age_seconds = (
            time.time() - blocking_modal_updated_at
            if blocking_modal_updated_at else float("inf")
        )
        if max_age_seconds > 0 and age_seconds > max_age_seconds:
            return None

        return {
            "kind": blocking_modal_kind,
            "target": blocking_modal_target,
            "x": blocking_modal_x,
            "y": blocking_modal_y,
            "age": age_seconds,
        }


def clear_blocking_modal_target(kind: str = "") -> None:
    global blocking_modal_kind, blocking_modal_target, blocking_modal_x, blocking_modal_y, blocking_modal_updated_at

    clean_kind = (kind or "").strip()
    with _lock:
        if clean_kind and blocking_modal_kind != clean_kind:
            return
        blocking_modal_kind = ""
        blocking_modal_target = ""
        blocking_modal_x = 0
        blocking_modal_y = 0
        blocking_modal_updated_at = 0.0
