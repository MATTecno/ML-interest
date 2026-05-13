"""
Calibração: descobre as coordenadas do centro do card do Tinder na sua tela.

Execute: python3 src/calibrate.py

O script mostra a posição do mouse em tempo real.
Você posiciona o cursor no centro do card e pressiona Enter.
As coordenadas são salvas automaticamente no config.yaml.
"""

import sys
import time
import yaml
from pathlib import Path

import pyautogui

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def main():
    print()
    print("  Tinder-IA — Calibração do card")
    print("  " + "-" * 40)
    print("  1. Abra o Tinder no browser em tela cheia")
    print("  2. Deixe um perfil aparecendo na tela")
    print("  3. Volte aqui e posicione o mouse no CENTRO do card")
    print("  4. Pressione Enter para salvar a posição")
    print()
    print("  Posição atual do mouse (atualizando a cada segundo):")
    print("  (Ctrl+C para cancelar)")
    print()

    try:
        while True:
            x, y = pyautogui.position()
            print(f"\r  Mouse em: x={x:4d}, y={y:4d}  — pressione Enter para confirmar", end="", flush=True)

            # Checa se Enter foi pressionado sem bloquear
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                input()  # consome o Enter
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n  Cancelado.\n")
        return

    x, y = pyautogui.position()
    print(f"\n\n  Posição salva: x={x}, y={y}")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("swiper", {})["card_center"] = [x, y]

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"  Salvo em config.yaml → swiper.card_center: [{x}, {y}]")
    print()
    print("  Para ativar o swipe automático, mude swiper.enabled para true no config.yaml")
    print()


if __name__ == "__main__":
    main()
