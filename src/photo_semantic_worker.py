"""Worker persistente para embeddings CLIP de fotos.

Mantem o SentenceTransformer fora do processo principal para isolar memoria
pesada de PyTorch/CUDA. O cache persistente continua no processo chamador.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
import atexit
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from logging_config import get_logger, setup_logging


logger = get_logger(__name__)


def _valid_embedding(value: Any) -> list[float] | None:
    try:
        import numpy as np

        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size < 16 or np.any(np.isnan(arr)):
            return None
        return [float(x) for x in arr]
    except Exception:
        return None


def _worker_main(request_q, response_q) -> None:
    setup_logging()
    worker_logger = get_logger(__name__)
    model = None
    loaded_model_name = ""
    loaded_device = ""
    worker_logger.info("Worker CLIP iniciado pid=%s", os.getpid())

    while True:
        request = request_q.get()
        if not isinstance(request, dict):
            continue
        command = request.get("command")
        if command == "shutdown":
            worker_logger.info("Worker CLIP encerrando por shutdown")
            return
        if command != "encode":
            continue

        request_id = str(request.get("request_id") or "")
        started = time.perf_counter()
        try:
            model_name = str(request.get("model_name") or "")
            device = str(request.get("device") or "cpu")
            batch_size = max(1, int(request.get("batch_size") or 1))
            images = list(request.get("images") or [])

            if model is None or loaded_model_name != model_name or loaded_device != device:
                from sentence_transformers import SentenceTransformer

                load_started = time.perf_counter()
                worker_logger.info("Carregando CLIP no worker: model=%s device=%s", model_name, device)
                try:
                    model = SentenceTransformer(model_name, device=device, local_files_only=True)
                    worker_logger.info("CLIP carregado do cache local no worker: model=%s", model_name)
                except TypeError:
                    worker_logger.warning(
                        "SentenceTransformer sem local_files_only; carregando CLIP pelo caminho padrao: model=%s",
                        model_name,
                    )
                    model = SentenceTransformer(model_name, device=device)
                except Exception:
                    worker_logger.warning(
                        "CLIP nao encontrado no cache local; tentando carga padrao: model=%s",
                        model_name,
                        exc_info=True,
                    )
                    model = SentenceTransformer(model_name, device=device)
                loaded_model_name = model_name
                loaded_device = device
                worker_logger.info(
                    "CLIP carregado no worker: model=%s device=%s elapsed=%.2fs",
                    model_name,
                    device,
                    time.perf_counter() - load_started,
                )

            from PIL import Image as PILImage
            import numpy as np

            labels: list[str] = []
            pil_images = []
            errors: dict[str, str] = {}
            for item in images:
                try:
                    label = str(item.get("label") or "")
                    width = int(item.get("width") or 0)
                    height = int(item.get("height") or 0)
                    rgb_bytes = item.get("rgb_bytes")
                    if not label or width <= 0 or height <= 0 or not isinstance(rgb_bytes, bytes):
                        errors[label or "?"] = "invalid_image"
                        continue
                    pil_images.append(PILImage.frombytes("RGB", (width, height), rgb_bytes))
                    labels.append(label)
                except Exception:
                    errors[str(item.get("label") or "?")] = "decode_failed"

            embeddings_by_label: dict[str, list[float] | None] = {}
            if pil_images:
                embeddings = model.encode(
                    pil_images,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                arr = np.array(embeddings, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                for label, row in zip(labels, arr[: len(labels)]):
                    embedding = _valid_embedding(row)
                    if embedding is None:
                        errors[label] = "invalid_embedding"
                    embeddings_by_label[label] = embedding

            response_q.put(
                {
                    "request_id": request_id,
                    "ok": True,
                    "device": loaded_device,
                    "embeddings": embeddings_by_label,
                    "errors": errors,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            worker_logger.info(
                "Batch CLIP no worker concluido: request=%s images=%s ok=%s errors=%s elapsed=%.2fs",
                request_id,
                len(images),
                len([v for v in embeddings_by_label.values() if v is not None]),
                len(errors),
                time.perf_counter() - started,
            )
        except BaseException as exc:
            worker_logger.exception("Worker CLIP falhou request=%s", request_id)
            try:
                response_q.put(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "device": loaded_device,
                        "embeddings": {},
                        "errors": {"_worker": f"{type(exc).__name__}: {exc}"},
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            except Exception:
                pass


class ClipWorkerClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._request_q = None
        self._response_q = None

    def _start_locked(self, start_timeout_seconds: float) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._terminate_locked("restart_before_start")
        self._request_q = self._ctx.Queue(maxsize=2)
        self._response_q = self._ctx.Queue(maxsize=2)
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._request_q, self._response_q),
            name="photo-semantic-clip-worker",
            daemon=True,
        )
        self._process.start()
        deadline = time.time() + max(0.1, float(start_timeout_seconds or 0.1))
        while time.time() < deadline:
            if self._process.is_alive():
                logger.info("Worker CLIP iniciado pid=%s", self._process.pid)
                return
            time.sleep(0.05)
        raise RuntimeError("worker CLIP nao iniciou")

    def _terminate_locked(self, reason: str = "") -> None:
        process = self._process
        request_q = self._request_q
        response_q = self._response_q
        self._process = None
        self._request_q = None
        self._response_q = None
        if process is None:
            for q in (request_q, response_q):
                if q is None:
                    continue
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
            return
        pid = process.pid
        if process.is_alive():
            logger.warning("Worker CLIP sendo terminado: pid=%s reason=%s", pid, reason or "unspecified")
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            logger.warning("Worker CLIP nao respondeu ao terminate; usando kill: pid=%s", pid)
            process.kill()
            process.join(timeout=5)
        for q in (request_q, response_q):
            if q is None:
                continue
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
        logger.info("Worker CLIP parado: pid=%s exitcode=%s reason=%s", pid, process.exitcode, reason or "unspecified")

    def encode_rgb_batch(
        self,
        images: list[dict],
        model_name: str,
        device: str,
        batch_size: int,
        start_timeout_seconds: float,
        request_timeout_seconds: float,
    ) -> dict:
        if not images:
            return {"ok": True, "device": device, "embeddings": {}, "errors": {}}
        request_id = uuid.uuid4().hex
        with self._lock:
            self._start_locked(start_timeout_seconds)
            assert self._request_q is not None
            assert self._response_q is not None
            try:
                logger.info(
                    "Enviando batch ao worker CLIP: request=%s pid=%s images=%s batch_size=%s model=%s device=%s timeout=%.1fs",
                    request_id,
                    self._process.pid if self._process is not None else "?",
                    len(images),
                    batch_size,
                    model_name,
                    device,
                    request_timeout_seconds,
                )
                self._request_q.put(
                    {
                        "command": "encode",
                        "request_id": request_id,
                        "model_name": model_name,
                        "device": device,
                        "batch_size": batch_size,
                        "images": images,
                    },
                    timeout=5,
                )
                deadline = time.time() + max(0.1, float(request_timeout_seconds or 0.1))
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(f"timeout do worker CLIP apos {request_timeout_seconds:.1f}s")
                    if self._process is None or not self._process.is_alive():
                        raise RuntimeError("worker CLIP morreu durante requisicao")
                    try:
                        response = self._response_q.get(timeout=min(0.5, remaining))
                    except queue.Empty:
                        continue
                    if isinstance(response, dict) and response.get("request_id") == request_id:
                        embeddings = response.get("embeddings") or {}
                        ok_count = len([value for value in embeddings.values() if value is not None])
                        errors = response.get("errors") or {}
                        logger.info(
                            "Resposta do worker CLIP recebida: request=%s ok=%s images=%s embeddings=%s errors=%s elapsed=%.2fs",
                            request_id,
                            bool(response.get("ok")),
                            len(images),
                            ok_count,
                            len(errors),
                            float(response.get("elapsed_seconds") or 0.0),
                        )
                        return response
                    stale_id = response.get("request_id") if isinstance(response, dict) else "?"
                    logger.warning("Resposta CLIP obsoleta ignorada: request=%s", stale_id)
            except queue.Empty:
                self._terminate_locked("response_queue_empty")
                raise RuntimeError("worker CLIP encerrou sem resposta")
            except Exception:
                self._terminate_locked("request_failed")
                raise

    def warmup(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        start_timeout_seconds: float,
        request_timeout_seconds: float,
    ) -> dict:
        request_id = uuid.uuid4().hex
        with self._lock:
            self._start_locked(start_timeout_seconds)
            assert self._request_q is not None
            assert self._response_q is not None
            try:
                logger.info(
                    "Preaquecendo worker CLIP: request=%s pid=%s model=%s device=%s timeout=%.1fs",
                    request_id,
                    self._process.pid if self._process is not None else "?",
                    model_name,
                    device,
                    request_timeout_seconds,
                )
                self._request_q.put(
                    {
                        "command": "encode",
                        "request_id": request_id,
                        "model_name": model_name,
                        "device": device,
                        "batch_size": batch_size,
                        "images": [],
                    },
                    timeout=5,
                )
                deadline = time.time() + max(0.1, float(request_timeout_seconds or 0.1))
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(f"timeout do prewarm CLIP apos {request_timeout_seconds:.1f}s")
                    if self._process is None or not self._process.is_alive():
                        raise RuntimeError("worker CLIP morreu durante prewarm")
                    try:
                        response = self._response_q.get(timeout=min(0.5, remaining))
                    except queue.Empty:
                        continue
                    if isinstance(response, dict) and response.get("request_id") == request_id:
                        logger.info(
                            "Prewarm do worker CLIP concluido: request=%s ok=%s elapsed=%.2fs",
                            request_id,
                            bool(response.get("ok")),
                            float(response.get("elapsed_seconds") or 0.0),
                        )
                        return response
            except Exception:
                self._terminate_locked("warmup_failed")
                raise

    def shutdown(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive() and self._request_q is not None:
                try:
                    logger.info("Solicitando shutdown do worker CLIP: pid=%s", self._process.pid)
                    self._request_q.put({"command": "shutdown"}, timeout=1)
                    self._process.join(timeout=5)
                except Exception:
                    pass
            self._terminate_locked("shutdown")


_CLIENT_LOCK = threading.Lock()
_CLIENT: ClipWorkerClient | None = None
_ATEXIT_REGISTERED = False


def _register_atexit_shutdown() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    _ATEXIT_REGISTERED = True
    atexit.register(shutdown_worker)


def get_client() -> ClipWorkerClient:
    global _CLIENT
    _register_atexit_shutdown()
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = ClipWorkerClient()
        return _CLIENT


def encode_rgb_batch(
    images: list[dict],
    model_name: str,
    device: str,
    batch_size: int,
    start_timeout_seconds: float,
    request_timeout_seconds: float,
) -> dict:
    return get_client().encode_rgb_batch(
        images,
        model_name,
        device,
        batch_size,
        start_timeout_seconds,
        request_timeout_seconds,
    )


def warmup_clip(
    model_name: str,
    device: str,
    batch_size: int,
    start_timeout_seconds: float,
    request_timeout_seconds: float,
) -> dict:
    return get_client().warmup(
        model_name,
        device,
        batch_size,
        start_timeout_seconds,
        request_timeout_seconds,
    )


def shutdown_worker() -> None:
    with _CLIENT_LOCK:
        client = _CLIENT
    if client is not None:
        client.shutdown()


_register_atexit_shutdown()
