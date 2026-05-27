"""Central Tkinter para controlar o servidor local do Tinder-IA."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, messagebox, ttk

from config import ROOT_DIR, load_config
from desktop_notify import notify as desktop_notify
from resource_guard import get_system_resource_snapshot


SERVER_PORT = 5043
REVIEW_PORT = 5055
SERVER_URL = f"http://localhost:{SERVER_PORT}/control"
REVIEW_URL = f"http://localhost:{REVIEW_PORT}"

SRC_DIR = ROOT_DIR / "src"
RUNTIME_DIR = ROOT_DIR / "data" / "runtime"
LOG_DIR = ROOT_DIR / "data" / "logs"
SETTINGS_PATH = RUNTIME_DIR / "control_ui_settings.json"
AUTOSTART_PATH = Path.home() / ".config" / "autostart" / "tinder-ia-control.desktop"
APPLICATION_LAUNCHER_PATH = Path.home() / ".local" / "share" / "applications" / "tinder-ia-control.desktop"
ICON_PATH = RUNTIME_DIR / "tinder_ia_control.png"

SERVER_PID = RUNTIME_DIR / "control_server.pid"
REVIEW_PID = RUNTIME_DIR / "control_review.pid"

SERVER_LOG = LOG_DIR / "control_ui_server.log"
REVIEW_LOG = LOG_DIR / "control_ui_review.log"
CHROME_OPEN_LOG = LOG_DIR / "control_ui_chrome.log"
TRAY_INSTALL_LOG = LOG_DIR / "control_ui_tray_install.log"
CONTROL_UI_LOG = LOG_DIR / "control_ui.log"

SERVER_SUPERVISOR_INTERVAL_MS = 5_000
SERVER_UNRESPONSIVE_RESTART_SECONDS = 25.0

TRAY_INSTALL_COMMAND = [
    "pkexec",
    "apt",
    "install",
    "-y",
    "python3-gi",
    "gir1.2-ayatanaappindicator3-0.1",
    "libayatana-appindicator3-1",
]


def ensure_dirs() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    AUTOSTART_PATH.parent.mkdir(parents=True, exist_ok=True)
    APPLICATION_LAUNCHER_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> tuple[dict, bool]:
    ensure_dirs()
    if not SETTINGS_PATH.exists():
        return {"autostart": True, "server_autostart": True}, True
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "autostart": bool(data.get("autostart", False)),
            "server_autostart": bool(data.get("server_autostart", True)),
        }, False
    except Exception:
        return {"autostart": False, "server_autostart": True}, False


def save_settings(settings: dict) -> None:
    ensure_dirs()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def control_log(message: str) -> None:
    ensure_dirs()
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    try:
        with open(CONTROL_UI_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def desktop_arg(value: str) -> str:
    """Cita um argumento para a chave Exec do .desktop."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def make_icon_image(size: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    pad = max(4, size // 12)
    radius = max(10, size // 5)
    draw.rounded_rectangle((pad, pad, size - pad, size - pad), radius=radius, fill=(24, 26, 32, 255))
    draw.ellipse((size * 0.22, size * 0.16, size * 0.78, size * 0.72), fill=(255, 68, 88, 255))
    draw.polygon(
        [
            (size * 0.50, size * 0.25),
            (size * 0.62, size * 0.50),
            (size * 0.50, size * 0.75),
            (size * 0.38, size * 0.50),
        ],
        fill=(255, 255, 255, 255),
    )
    return image


def ensure_app_icon() -> Path | None:
    try:
        ensure_dirs()
        make_icon_image(256).save(ICON_PATH)
        return ICON_PATH
    except Exception:
        return None


def desktop_dir() -> Path:
    try:
        result = subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True, timeout=1).strip()
        if result:
            path = Path(result).expanduser()
            if path.exists():
                return path
    except Exception:
        pass
    for candidate in (Path.home() / "Área de Trabalho", Path.home() / "Desktop"):
        if candidate.exists():
            return candidate
    return Path.home() / "Desktop"


def desktop_entry_content(minimized: bool) -> str:
    ensure_app_icon()
    args = " --minimized" if minimized else ""
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=Tinder-IA Control",
        "Comment=Central de controle do Tinder-IA",
        f"Exec={desktop_arg(sys.executable)} {desktop_arg(str(SRC_DIR / 'control_ui.py'))}{args}",
        f"Path={ROOT_DIR}",
        "Terminal=false",
        "Categories=Utility;",
    ]
    if ICON_PATH.exists():
        lines.append(f"Icon={ICON_PATH}")
    if minimized:
        lines.append("X-GNOME-Autostart-enabled=true")
    lines.append("")
    return "\n".join(lines)


def autostart_content() -> str:
    return desktop_entry_content(minimized=True)


def launcher_content() -> str:
    return desktop_entry_content(minimized=False)


def install_clickable_launchers() -> list[Path]:
    ensure_dirs()
    ensure_app_icon()
    created: list[Path] = []
    for path in (APPLICATION_LAUNCHER_PATH, desktop_dir() / "tinder-ia-control.desktop"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(launcher_content())
        path.chmod(0o755)
        created.append(path)
    return created


def ensure_clickable_launchers() -> None:
    desktop_launcher = desktop_dir() / "tinder-ia-control.desktop"
    if not APPLICATION_LAUNCHER_PATH.exists() or not desktop_launcher.exists():
        install_clickable_launchers()


def set_autostart(enabled: bool) -> None:
    ensure_dirs()
    if enabled:
        with open(AUTOSTART_PATH, "w", encoding="utf-8") as f:
            f.write(autostart_content())
        AUTOSTART_PATH.chmod(0o644)
    elif AUTOSTART_PATH.exists():
        AUTOSTART_PATH.unlink()


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def write_pid(path: Path, pid: int) -> None:
    ensure_dirs()
    path.write_text(str(pid), encoding="utf-8")


def remove_pid(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except Exception:
        return ""


def own_process_alive(pid_path: Path, expected_script: str) -> tuple[bool, int | None]:
    pid = read_pid(pid_path)
    if not pid_alive(pid):
        remove_pid(pid_path)
        return False, None
    cmdline = process_cmdline(pid or 0)
    if expected_script not in cmdline:
        return False, pid
    return True, pid


def tcp_port_open(port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def http_responsive(url: str, timeout: float = 0.7) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def start_process(command: list[str], log_path: Path, pid_path: Path) -> int:
    ensure_dirs()
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(command)}\n")
    log_file.flush()
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )
    except Exception:
        log_file.close()
        raise
    write_pid(pid_path, proc.pid)

    def _watch_exit() -> None:
        exitcode = proc.wait()
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] processo finalizado pid={proc.pid} exitcode={exitcode}\n"
        try:
            log_file.write(line)
            log_file.flush()
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass
        if read_pid(pid_path) == proc.pid:
            remove_pid(pid_path)
        control_log(f"Processo saiu: command={' '.join(command)} pid={proc.pid} exitcode={exitcode}")

    threading.Thread(target=_watch_exit, daemon=True, name=f"process-watch-{proc.pid}").start()
    return proc.pid


def find_google_chrome() -> str | None:
    for name in ("google-chrome", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            return path
    for path in (Path("/usr/bin/google-chrome"), Path("/usr/bin/google-chrome-stable")):
        if path.exists():
            return str(path)
    return None


def open_url_in_main_chrome(url: str) -> str:
    chrome = find_google_chrome()
    if not chrome:
        raise FileNotFoundError("Google Chrome nao encontrado em google-chrome/google-chrome-stable")

    ensure_dirs()
    log_file = open(CHROME_OPEN_LOG, "a", encoding="utf-8")
    log_file.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {chrome} --new-tab {url}\n")
    log_file.flush()
    try:
        subprocess.Popen(
            [chrome, "--new-tab", url],
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )
    finally:
        log_file.close()
    return chrome


def stop_process(pid_path: Path, expected_script: str, timeout: float = 5.0) -> bool:
    own_alive, pid = own_process_alive(pid_path, expected_script)
    if not own_alive or not pid:
        return False

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_pid(pid_path)
        return True
    except Exception:
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            remove_pid(pid_path)
            return True
        time.sleep(0.15)

    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        pass
    remove_pid(pid_path)
    return True


class ControlUI:
    def __init__(self, root: Tk, minimized: bool = False) -> None:
        self.root = root
        self.minimized = minimized
        self.settings, first_run = load_settings()
        self.tray_icon = None
        self.tray_thread: threading.Thread | None = None
        self.tray_available = False
        self.refresh_after_id: str | None = None
        self.supervisor_after_id: str | None = None
        self.tk_icon = None
        self.exiting = False
        self._server_unresponsive_since = 0.0

        if first_run:
            try:
                set_autostart(True)
            except Exception:
                self.settings["autostart"] = False
            save_settings(self.settings)
        try:
            ensure_clickable_launchers()
        except Exception:
            pass

        self.server_status = StringVar(value="Servidor: verificando...")
        self.review_status = StringVar(value="Review UI: verificando...")
        self.browser_status = StringVar(value="Chrome principal: verificando...")
        self.memory_status = StringVar(value="Memoria: verificando...")
        self.tray_status = StringVar(value="")
        self.info_status = StringVar(value="Pronto.")
        self.autostart_var = BooleanVar(value=bool(self.settings.get("autostart", False)))
        self.server_autostart_var = BooleanVar(value=bool(self.settings.get("server_autostart", True)))

        self.server_start_button: ttk.Button
        self.server_stop_button: ttk.Button
        self.review_open_button: ttk.Button
        self.review_stop_button: ttk.Button
        self.tinder_button: ttk.Button
        self.tray_install_button: ttk.Button
        self.launcher_button: ttk.Button

        self._build_ui()
        self._apply_window_icon()
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)
        self._start_tray()

        if self.minimized and self.tray_available:
            self.root.withdraw()
        elif self.minimized:
            self.root.after(250, self.hide_window)
            self.info_status.set("Tray indisponivel; painel ficara minimizado.")

        self.refresh_status()
        self._schedule_server_supervisor(500)

    def _build_ui(self) -> None:
        self.root.title("Tinder-IA Control")
        self.root.geometry("520x470")
        self.root.minsize(480, 430)

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        ttk.Label(outer, text="Tinder-IA Control", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(outer, textvariable=self.info_status).grid(row=1, column=0, sticky="w", pady=(4, 14))

        status = ttk.LabelFrame(outer, text="Status", padding=12)
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.server_status).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(status, textvariable=self.review_status).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(status, textvariable=self.browser_status).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(status, textvariable=self.memory_status).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(status, textvariable=self.tray_status).grid(row=4, column=0, sticky="w", pady=(8, 0))

        actions = ttk.LabelFrame(outer, text="Controles", padding=12)
        actions.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self.server_start_button = ttk.Button(actions, text="Iniciar servidor", command=self.start_server)
        self.server_start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=4)
        self.server_stop_button = ttk.Button(actions, text="Parar servidor", command=self.stop_server)
        self.server_stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=4)

        self.review_open_button = ttk.Button(actions, text="Abrir Review UI", command=self.open_review)
        self.review_open_button.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=4)
        self.review_stop_button = ttk.Button(actions, text="Parar Review UI", command=self.stop_review)
        self.review_stop_button.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=4)

        self.tinder_button = ttk.Button(actions, text="Abrir Tinder no Chrome", command=self.open_tinder)
        self.tinder_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)

        self.launcher_button = ttk.Button(actions, text="Criar icone clicavel", command=self.create_launcher)
        self.launcher_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)

        self.tray_install_button = ttk.Button(
            actions,
            text="Instalar suporte do tray",
            command=self.install_tray_support,
        )

        options = ttk.Frame(outer)
        options.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        options.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            options,
            text="Iniciar com o sistema",
            variable=self.autostart_var,
            command=self.toggle_autostart,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            options,
            text="Manter servidor ativo",
            variable=self.server_autostart_var,
            command=self.toggle_server_autostart,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        footer = ttk.Frame(outer)
        footer.grid(row=5, column=0, sticky="ew", pady=(18, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Button(footer, text="Ocultar", command=self.hide_window).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(footer, text="Sair da UI", command=self.exit_ui).grid(row=0, column=2)

    def _apply_window_icon(self) -> None:
        try:
            from PIL import ImageTk

            icon_path = ensure_app_icon()
            if icon_path is None:
                return
            self.tk_icon = ImageTk.PhotoImage(file=str(icon_path))
            self.root.iconphoto(True, self.tk_icon)
        except Exception:
            pass

    def _start_tray(self) -> None:
        try:
            import pystray

            menu = pystray.Menu(
                pystray.MenuItem("Abrir painel", lambda: self.root.after(0, self.show_window)),
                pystray.MenuItem("Iniciar servidor", lambda: self.root.after(0, self.start_server)),
                pystray.MenuItem("Abrir review", lambda: self.root.after(0, self.open_review)),
                pystray.MenuItem("Abrir Tinder no Chrome", lambda: self.root.after(0, self.open_tinder)),
                pystray.MenuItem("Sair da UI", lambda: self.root.after(0, self.exit_ui)),
            )
            self.tray_icon = pystray.Icon("tinder_ia_control", make_icon_image(64), "Tinder-IA Control", menu)
            self.tray_thread = threading.Thread(target=self._run_tray, daemon=True, name="control-ui-tray")
            self.tray_thread.start()
            self.tray_available = True
            self.tray_status.set("Tray: ativo.")
            self.tray_install_button.grid_remove()
        except Exception as exc:
            self.tray_available = False
            self.tray_status.set(f"Tray: indisponivel ({exc}).")
            self.tray_install_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _run_tray(self) -> None:
        try:
            if self.tray_icon is not None:
                self.tray_icon.run()
        except Exception as exc:
            self.root.after(0, lambda: self._tray_failed(exc))

    def _tray_failed(self, exc: Exception) -> None:
        self.tray_available = False
        self.tray_status.set(f"Tray: indisponivel ({exc}).")
        self.tray_install_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def show_window(self) -> None:
        if self.exiting:
            return
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide_window(self) -> None:
        if not self.tray_available:
            self.root.iconify()
            self.info_status.set("Tray indisponivel; painel minimizado na barra.")
            return
        self.root.withdraw()

    def exit_ui(self) -> None:
        self.exiting = True
        if self.refresh_after_id is not None:
            try:
                self.root.after_cancel(self.refresh_after_id)
            except Exception:
                pass
            self.refresh_after_id = None
        if self.supervisor_after_id is not None:
            try:
                self.root.after_cancel(self.supervisor_after_id)
            except Exception:
                pass
            self.supervisor_after_id = None
        if self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        self.root.destroy()

    def toggle_autostart(self) -> None:
        enabled = bool(self.autostart_var.get())
        self.settings["autostart"] = enabled
        try:
            set_autostart(enabled)
            save_settings(self.settings)
            self.info_status.set("Autostart ativado." if enabled else "Autostart desativado.")
        except Exception as exc:
            self.autostart_var.set(not enabled)
            self.settings["autostart"] = not enabled
            save_settings(self.settings)
            messagebox.showerror("Autostart", f"Nao consegui atualizar o autostart:\n{exc}")

    def toggle_server_autostart(self) -> None:
        enabled = bool(self.server_autostart_var.get())
        self.settings["server_autostart"] = enabled
        save_settings(self.settings)
        if enabled:
            self.info_status.set("Servidor sera mantido ativo.")
            self._schedule_server_supervisor(100)
        else:
            self.info_status.set("Supervisor do servidor desativado.")

    def server_state(self) -> tuple[str, bool, bool]:
        own_alive, _pid = own_process_alive(SERVER_PID, "server.py")
        port_open = tcp_port_open(SERVER_PORT)
        if http_responsive(SERVER_URL):
            return ("rodando" if own_alive else "rodando externo", own_alive, True)
        if port_open:
            return ("travado/sem resposta" if own_alive else "travado externo", own_alive, True)
        if own_alive:
            return "iniciando", True, True
        return "parado", False, False

    def _schedule_server_supervisor(self, delay_ms: int = SERVER_SUPERVISOR_INTERVAL_MS) -> None:
        if self.exiting:
            return
        if self.supervisor_after_id is not None:
            try:
                self.root.after_cancel(self.supervisor_after_id)
            except Exception:
                pass
            self.supervisor_after_id = None
        self.supervisor_after_id = self.root.after(max(100, int(delay_ms)), self._server_supervisor_tick)

    def _server_autostart_enabled(self) -> bool:
        return bool(self.settings.get("server_autostart", True)) and bool(self.server_autostart_var.get())

    def _server_supervisor_tick(self) -> None:
        self.supervisor_after_id = None
        try:
            self._check_server_supervisor()
        finally:
            if not self.exiting:
                self._schedule_server_supervisor()

    def _check_server_supervisor(self) -> None:
        if not self._server_autostart_enabled():
            self._server_unresponsive_since = 0.0
            return

        state, own_alive, busy = self.server_state()
        now = time.time()
        if state == "rodando" or state == "rodando externo":
            self._server_unresponsive_since = 0.0
            return
        if state == "iniciando":
            return
        if state == "travado/sem resposta" and own_alive:
            if self._server_unresponsive_since <= 0:
                self._server_unresponsive_since = now
                control_log("Servidor sem resposta; aguardando antes de reiniciar")
                return
            if now - self._server_unresponsive_since < SERVER_UNRESPONSIVE_RESTART_SECONDS:
                return
            control_log("Servidor sem resposta persistente; reiniciando")
            stop_process(SERVER_PID, "server.py", timeout=3.0)
            self._server_unresponsive_since = 0.0
            self._auto_start_server("servidor travado/sem resposta")
            return
        if busy:
            self._server_unresponsive_since = 0.0
            return

        self._server_unresponsive_since = 0.0
        self._auto_start_server("servidor parado ou caiu")

    def _auto_start_server(self, reason: str) -> None:
        state, _own, busy = self.server_state()
        if busy:
            return
        pid = start_process([sys.executable, str(SRC_DIR / "server.py")], SERVER_LOG, SERVER_PID)
        msg = f"Servidor reiniciado automaticamente ({reason}); pid {pid}"
        control_log(msg)
        self.info_status.set(msg)
        desktop_notify(
            "server_auto_restart",
            "Tinder IA religou o servidor",
            reason,
            urgency="critical" if "travado" in reason else "normal",
        )
        self.root.after(3500, lambda: self._request_reload_after_server_start(reason))
        self.refresh_status()

    def _request_reload_after_server_start(self, reason: str) -> None:
        if self.exiting:
            return
        if not http_responsive(SERVER_URL, timeout=1.0):
            control_log("Servidor ainda nao respondeu para solicitar reload apos restart")
            return
        payload = json.dumps(
            {"reason": f"servidor reiniciado automaticamente: {reason}; ressincronizando Tinder"}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"http://localhost:{SERVER_PORT}/reload-start",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0):
                pass
            control_log("Reload do Tinder solicitado apos restart automatico do servidor")
        except Exception as exc:
            control_log(f"Falha ao solicitar reload apos restart automatico: {exc}")

    def review_state(self) -> tuple[str, bool, bool]:
        own_alive, _pid = own_process_alive(REVIEW_PID, "review_ui.py")
        if tcp_port_open(REVIEW_PORT):
            return ("rodando" if own_alive else "rodando externo", own_alive, True)
        if own_alive:
            return "iniciando", True, True
        return "parado", False, False

    def refresh_status(self) -> None:
        if self.exiting:
            return
        if self.refresh_after_id is not None:
            try:
                self.root.after_cancel(self.refresh_after_id)
            except Exception:
                pass
            self.refresh_after_id = None

        server_state, server_own, server_busy = self.server_state()
        review_state, review_own, review_busy = self.review_state()
        chrome_path = find_google_chrome()
        self.server_status.set(f"Servidor: {server_state} ({SERVER_URL})")
        self.review_status.set(f"Review UI: {review_state} ({REVIEW_URL})")
        if chrome_path:
            self.browser_status.set(f"Chrome principal: pronto ({chrome_path})")
        else:
            self.browser_status.set("Chrome principal: nao encontrado")

        try:
            snap = get_system_resource_snapshot()
            mem_used = float(snap.get("mem_used_pct", 0.0))
            mem_avail = float(snap.get("mem_avail_mb", 0.0))
            swap_used = float(snap.get("swap_used_pct", 0.0))
            self.memory_status.set(
                f"Memoria: {mem_used:.1f}% usada, {mem_avail:.0f} MB livres | Swap: {swap_used:.1f}%"
            )
        except Exception as exc:
            self.memory_status.set(f"Memoria: indisponivel ({exc})")

        self.server_start_button.configure(state="disabled" if server_busy else "normal")
        self.server_stop_button.configure(state="normal" if server_own else "disabled")
        self.review_stop_button.configure(state="normal" if review_own else "disabled")
        self.review_open_button.configure(state="normal")
        self.tinder_button.configure(state="normal")

        self.refresh_after_id = self.root.after(1500, self.refresh_status)

    def start_server(self) -> None:
        state, _own, busy = self.server_state()
        if busy:
            self.info_status.set(f"Servidor ja esta {state}.")
            return
        self.settings["server_autostart"] = True
        self.server_autostart_var.set(True)
        save_settings(self.settings)
        pid = start_process([sys.executable, str(SRC_DIR / "server.py")], SERVER_LOG, SERVER_PID)
        self.info_status.set(f"Servidor iniciando (pid {pid}).")
        self.refresh_status()

    def stop_server(self) -> None:
        self.settings["server_autostart"] = False
        self.server_autostart_var.set(False)
        save_settings(self.settings)
        if stop_process(SERVER_PID, "server.py"):
            self.info_status.set("Servidor iniciado pela UI foi encerrado.")
        else:
            self.info_status.set("Servidor nao foi encerrado porque nao pertence a esta UI.")
        self.refresh_status()

    def open_review(self) -> None:
        state, _own, busy = self.review_state()
        if not busy:
            pid = start_process([sys.executable, str(SRC_DIR / "review_ui.py")], REVIEW_LOG, REVIEW_PID)
            self.info_status.set(f"Review UI iniciando (pid {pid}).")
            self.root.after(1800, lambda: webbrowser.open(REVIEW_URL))
        else:
            self.info_status.set(f"Review UI ja esta {state}; abrindo navegador.")
            webbrowser.open(REVIEW_URL)
        self.refresh_status()

    def stop_review(self) -> None:
        if stop_process(REVIEW_PID, "review_ui.py"):
            self.info_status.set("Review UI iniciado pela UI foi encerrado.")
        else:
            self.info_status.set("Review UI nao foi encerrado porque nao pertence a esta UI.")
        self.refresh_status()

    def tinder_url(self) -> str:
        try:
            cfg = load_config()
            browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
            return str(browser_cfg.get("tinder_url") or "https://tinder.com/app/recs")
        except Exception:
            return "https://tinder.com/app/recs"

    def open_tinder(self) -> None:
        try:
            chrome = open_url_in_main_chrome(self.tinder_url())
            self.info_status.set(f"Tinder aberto no Chrome principal ({chrome}).")
        except FileNotFoundError as exc:
            messagebox.showerror("Abrir Tinder", f"{exc}\n\nInstale o Google Chrome ou ajuste o PATH.")
            self.info_status.set("Nao encontrei o Google Chrome principal.")
        except Exception as exc:
            messagebox.showerror("Abrir Tinder", f"Nao consegui abrir o Tinder no Chrome:\n{exc}")
        self.refresh_status()

    def create_launcher(self) -> None:
        try:
            created = install_clickable_launchers()
            shown = ", ".join(str(path) for path in created)
            self.info_status.set(f"Icone clicavel criado: {shown}")
        except Exception as exc:
            messagebox.showerror("Icone clicavel", f"Nao consegui criar o icone:\n{exc}")

    def install_tray_support(self) -> None:
        try:
            ensure_dirs()
            log_file = open(TRAY_INSTALL_LOG, "a", encoding="utf-8")
            log_file.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(TRAY_INSTALL_COMMAND)}\n")
            log_file.flush()
            subprocess.Popen(
                TRAY_INSTALL_COMMAND,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            log_file.close()
            self.info_status.set(f"Instalacao do suporte de tray iniciada. Log: {TRAY_INSTALL_LOG}")
        except FileNotFoundError:
            messagebox.showerror("Tray", "pkexec nao foi encontrado neste sistema.")
        except Exception as exc:
            messagebox.showerror("Tray", f"Nao consegui iniciar a instalacao:\n{exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Central de controle local do Tinder-IA.")
    parser.add_argument("--minimized", action="store_true", help="Inicia oculto, pronto para o tray.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    root = Tk()
    ControlUI(root, minimized=args.minimized)
    root.mainloop()


if __name__ == "__main__":
    main()
