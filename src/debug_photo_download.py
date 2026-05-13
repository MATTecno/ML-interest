"""
Diagnostica download das fotos do Tinder sem rodar DeepFace.

Uso:
  python3 src/debug_photo_download.py tinder_response.json --limit 12
  python3 src/debug_photo_download.py tinder_response.json --quality full --limit 4
  python3 src/debug_photo_download.py "https://images-ssl.gotinder.com/u/..."
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import sys
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))

from photo_features import analyze_photo
from profile_parser import calc_age, pick_photo_url


def _request(url: str, timeout: float) -> tuple[bool, str]:
    started = time.perf_counter()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": "https://tinder.com/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            elapsed = time.perf_counter() - started
            content_type = resp.headers.get("content-type", "?")

        try:
            img = PILImage.open(io.BytesIO(data))
            decoded = f"{img.format} {img.width}x{img.height}"
        except Exception as exc:
            decoded = f"decode falhou: {type(exc).__name__}: {exc}"

        return True, (
            f"OK HTTP {resp.status} | {len(data) / 1024:.0f}KB | "
            f"{content_type} | {decoded} | {elapsed:.2f}s"
        )
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        return False, f"HTTP {exc.code} {exc.reason} | {elapsed:.2f}s"
    except urllib.error.URLError as exc:
        elapsed = time.perf_counter() - started
        return False, f"REDE {exc.reason} | {elapsed:.2f}s"
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return False, f"ERRO {type(exc).__name__}: {exc} | {elapsed:.2f}s"


def _iter_urls_from_json(path: Path, quality: str, limit: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("data", {}).get("results", [])
    yielded = 0
    for item in results:
        if item.get("type") != "user":
            continue

        user = item.get("user", {})
        name = user.get("name", "?")
        age = calc_age(user.get("birth_date", "")) if user.get("birth_date") else 0
        for idx, photo in enumerate(user.get("photos") or [], 1):
            url = pick_photo_url([photo], quality)
            if not url:
                continue
            yield f"{name}, {age} anos | foto {idx}", url
            yielded += 1
            if yielded >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testa URLs de foto do Tinder e mostra se o problema é HTTP, rede, timeout ou decode."
    )
    parser.add_argument("input", help="Arquivo JSON de recs ou URL direta de imagem")
    parser.add_argument("--limit", type=int, default=12, help="Máximo de fotos ao ler JSON")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout por foto em segundos")
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high", "full"],
        default="low",
        help="Qualidade extraída do JSON",
    )
    parser.add_argument(
        "--deepface",
        action="store_true",
        help="Também roda a análise DeepFace nas URLs testadas",
    )
    args = parser.parse_args()

    source = args.input.strip()
    if source.startswith("http://") or source.startswith("https://"):
        items = [("URL direta", source)]
    else:
        path = Path(source)
        if not path.exists():
            raise SystemExit(f"Arquivo não encontrado: {path}")
        items = list(_iter_urls_from_json(path, args.quality, args.limit))

    if not items:
        raise SystemExit("Nenhuma URL de foto encontrada.")

    print(f"\nTestando {len(items)} foto(s) | quality={args.quality} | timeout={args.timeout}s\n")
    ok_count = 0
    for i, (label, url) in enumerate(items, 1):
        ok, message = _request(url, args.timeout)
        ok_count += int(ok)
        status = "OK" if ok else "FALHA"
        print(f"{i:02d}. [{status}] {label}")
        print(f"    {message}")
        print(f"    path: {urllib.parse.urlsplit(url).path}")
        if args.deepface and ok:
            started = time.perf_counter()
            features = analyze_photo(url)
            elapsed = time.perf_counter() - started
            face = features.get("photo_has_face")
            woman = int(features.get("photo_woman_confidence", 0.5) * 100)
            failed = features.get("_analysis_failed", False)
            reason = features.get("_failure_reason", "")
            print(
                f"    deepface: face={face} mulher={woman}% failed={failed} "
                f"reason={reason or '-'} | {elapsed:.2f}s"
            )

    print(f"\nResumo: {ok_count}/{len(items)} downloads OK\n")


if __name__ == "__main__":
    main()
