"""Limpeza manual segura das fotos salvas.

Por padrao roda em dry-run e apenas mostra o que seria apagado.

Uso:
  python3 src/cleanup_photos.py
  python3 src/cleanup_photos.py --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_photos_config
from photo_storage import PHOTOS_DIR, eligible_photo_eviction_candidates


def _remove_with_sidecar(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".txt").unlink(missing_ok=True)


def _plan_folder(name: str, label: int, max_files: int) -> tuple[Path, list[Path], list[Path], int]:
    folder = PHOTOS_DIR / name
    files = sorted(folder.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    excess = max(0, len(files) - max_files)
    candidates = eligible_photo_eviction_candidates(folder, label)
    selected = candidates[:excess]
    return folder, files, selected, excess


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpa fotos antigas ja seguras para remocao.")
    parser.add_argument("--apply", action="store_true", help="Apaga de verdade. Sem isso, apenas simula.")
    parser.add_argument("--max-per-folder", type=int, default=None, help="Sobrescreve photos.max_per_folder.")
    args = parser.parse_args()

    cfg = get_photos_config()
    max_files = args.max_per_folder or int(cfg.get("max_per_folder", 50))
    mode = "APLICANDO" if args.apply else "DRY-RUN"
    print(f"\nLimpeza de fotos ({mode}) | limite por pasta: {max_files}\n")

    total_selected = 0
    for folder_name, label in [("liked", 1), ("disliked", 0)]:
        folder, files, selected, excess = _plan_folder(folder_name, label, max_files)
        total_selected += len(selected)
        print(
            f"{folder.name}: arquivos={len(files)} excesso={excess} "
            f"removiveis_agora={len(selected)}"
        )
        for path in selected[:12]:
            print(f"  - {path.name}")
        if len(selected) > 12:
            print(f"  ... +{len(selected) - 12} arquivo(s)")

        if args.apply:
            for path in selected:
                _remove_with_sidecar(path)

    if args.apply:
        print(f"\nRemovidos {total_selected} arquivo(s) .jpg e seus .txt correspondentes.")
    else:
        print("\nNenhum arquivo foi apagado. Use --apply para executar a limpeza.")


if __name__ == "__main__":
    main()
