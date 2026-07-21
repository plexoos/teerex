#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import threading
import time

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class EventSpec:
    event_id: int
    event_size_s: float
    cpu_fraction: float
    gpu_fraction: float

    cpu_work_s: float
    gpu_work_s: float
    dispatch_fraction: float
    cpu_pre_work_s: float
    cpu_post_work_s: float

    cpu_matrix_size: int
    gpu_matrix_size: int

    gpu_memory_mb: int
    h2d_mb: int
    d2h_mb: int

    num_cpus: float
    num_gpus: float


@dataclass
class GpuTaskHandle:
    event: Dict[str, Any]
    submit_time: float
    done: threading.Event
    result: Optional[Dict[str, Any]] = None
    exception: Optional[BaseException] = None

    def ready(self) -> bool:
        return self.done.is_set()

    def wait(self) -> Dict[str, Any]:
        self.done.wait()
        if self.exception is not None:
            raise self.exception
        if self.result is None:
            raise RuntimeError("GPU task completed without a result.")
        return self.result


class LocalGpuExecutor:
    """A single local GPU worker that runs Torch payloads in submission order."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch.")

        self._tasks: queue.Queue[GpuTaskHandle | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="simload-gpu", daemon=True)
        self._thread.start()

    def submit(self, event: Dict[str, Any]) -> GpuTaskHandle:
        handle = GpuTaskHandle(
            event=event,
            submit_time=time.perf_counter(),
            done=threading.Event(),
        )
        self._tasks.put(handle)
        return handle

    def close(self) -> None:
        self._tasks.put(None)
        self._thread.join()

    def _worker(self) -> None:
        stream = torch.cuda.Stream()

        while True:
            handle = self._tasks.get()
            if handle is None:
                return

            try:
                handle.result = _run_gpu_stage(handle.event, handle.submit_time, stream)
            except BaseException as exc:
                handle.exception = exc
            finally:
                handle.done.set()
                self._tasks.task_done()


def sample_event(event_id: int, params: Dict[str, Any]) -> EventSpec:
    """
    Generate one synthetic Geant4-like event record.

    The sampled CPU work is split into a pre-dispatch and post-dispatch phase.
    The GPU payload is submitted after the pre-dispatch phase.
    """

    seed = params.get("seed", 12345)
    rng = random.Random(seed + event_id)
    event_size_rng = (
        random.Random(int(params["event_size_seed"]) + event_id)
        if "event_size_seed" in params
        else rng
    )

    event_size_s = sample_event_size_s(event_size_rng, params)

    beta_alpha = float(params.get("cpu_fraction_beta_alpha", 4.0))
    beta_beta = float(params.get("cpu_fraction_beta_beta", 4.0))

    cpu_fraction = rng.betavariate(beta_alpha, beta_beta)
    gpu_fraction = 1.0 - cpu_fraction

    cpu_work_s = event_size_s * cpu_fraction
    gpu_work_s = event_size_s * gpu_fraction

    dispatch_fraction = sample_dispatch_fraction(rng, params)
    cpu_pre_work_s = cpu_work_s * dispatch_fraction
    cpu_post_work_s = cpu_work_s - cpu_pre_work_s

    cpu_matrix_base = int(params.get("cpu_matrix_base", 512))
    gpu_matrix_base = int(params.get("gpu_matrix_base", 2048))

    cpu_matrix_size = max(128, int(cpu_matrix_base * math.sqrt(max(cpu_work_s, 0.05))))
    gpu_matrix_size = max(256, int(gpu_matrix_base * math.sqrt(max(gpu_work_s, 0.05))))

    gpu_memory_base_mb = int(params.get("gpu_memory_base_mb", 256))
    gpu_memory_jitter_mb = int(params.get("gpu_memory_jitter_mb", 256))
    mean_event_size_s = float(params.get("mean_event_size_s", 3.0))
    size_scale = event_size_s / max(mean_event_size_s, 1.0e-9)

    gpu_memory_mb = int(
        (gpu_memory_base_mb + rng.uniform(0, gpu_memory_jitter_mb))
        * gpu_fraction
        * size_scale
    )

    h2d_mb = int(params.get("h2d_base_mb", 64) * gpu_fraction * size_scale)
    d2h_mb = int(params.get("d2h_base_mb", 16) * gpu_fraction * size_scale)

    return EventSpec(
        event_id=event_id,
        event_size_s=event_size_s,
        cpu_fraction=cpu_fraction,
        gpu_fraction=gpu_fraction,
        cpu_work_s=cpu_work_s,
        gpu_work_s=gpu_work_s,
        dispatch_fraction=dispatch_fraction,
        cpu_pre_work_s=cpu_pre_work_s,
        cpu_post_work_s=cpu_post_work_s,
        cpu_matrix_size=cpu_matrix_size,
        gpu_matrix_size=gpu_matrix_size,
        gpu_memory_mb=gpu_memory_mb,
        h2d_mb=h2d_mb,
        d2h_mb=d2h_mb,
        num_cpus=float(params.get("num_cpus_per_event", 1.0)),
        num_gpus=float(params.get("num_gpus_per_event", 1.0)),
    )


def sample_event_size_s(rng: random.Random, params: Dict[str, Any]) -> float:
    distribution = str(params.get("event_size_distribution", "gaussian")).lower()
    mean_event_size_s = float(params.get("mean_event_size_s", 3.0))
    event_size_spread_s = float(params.get("event_size_spread_s", 0.6))
    min_event_size_s = float(params.get("min_event_size_s", 0.05))

    if distribution == "fixed":
        event_size_s = mean_event_size_s
    elif distribution in {"uniform", "flat"}:
        event_size_s = rng.uniform(
            mean_event_size_s - event_size_spread_s,
            mean_event_size_s + event_size_spread_s,
        )
    elif distribution in {"gaussian", "normal"}:
        event_size_s = rng.gauss(mean_event_size_s, event_size_spread_s)
    else:
        raise ValueError(
            "event_size_distribution must be 'fixed', 'uniform', or 'gaussian' "
            f"(got {distribution!r})"
        )

    return max(min_event_size_s, event_size_s)


def sample_dispatch_fraction(rng: random.Random, params: Dict[str, Any]) -> float:
    mean = float(params.get("gpu_dispatch_fraction_mean", 0.5))
    jitter = float(params.get("gpu_dispatch_fraction_jitter", 0.1))
    lower = float(params.get("gpu_dispatch_fraction_min", 0.35))
    upper = float(params.get("gpu_dispatch_fraction_max", 0.65))

    if lower > upper:
        raise ValueError("gpu_dispatch_fraction_min must be <= gpu_dispatch_fraction_max")

    value = mean + rng.uniform(-jitter, jitter)
    return min(upper, max(lower, value))


def _run_cpu_payload(event: Dict[str, Any], work_key: str, stage: str) -> Dict[str, Any]:
    """
    Synthetic CPU payload for one event phase.

    The phase uses NumPy matrix multiplication in a time-bounded loop.
    """

    start = time.perf_counter()

    target_s = float(event[work_key])
    matrix_size = int(event["cpu_matrix_size"])

    if target_s <= 0.0:
        end = time.perf_counter()
        return {
            "stage": stage,
            "start_time": start,
            "end_time": end,
            "runtime_s": end - start,
            "iterations": 0,
            "checksum": 0.0,
        }

    a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    b = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    iterations = 0
    checksum = 0.0

    while time.perf_counter() - start < target_s:
        c = a @ b
        checksum += float(c[0, 0])
        iterations += 1

    end = time.perf_counter()

    return {
        "stage": stage,
        "start_time": start,
        "end_time": end,
        "runtime_s": end - start,
        "iterations": iterations,
        "checksum": checksum,
    }


def _run_gpu_stage(
    event: Dict[str, Any],
    submit_time: float,
    stream: torch.cuda.Stream,
) -> Dict[str, Any]:
    """
    Synthetic GPU stage.

    This is intentionally local: a dedicated Python worker submits Torch CUDA
    work to one stream and waits for that work to complete before accepting the
    next event.
    """

    start = time.perf_counter()

    target_s = float(event["gpu_work_s"])
    matrix_size = int(event["gpu_matrix_size"])
    gpu_memory_mb = int(event["gpu_memory_mb"])
    h2d_mb = int(event["h2d_mb"])
    d2h_mb = int(event["d2h_mb"])

    if target_s <= 0.0:
        end = time.perf_counter()
        return {
            "event_id": event["event_id"],
            "stage": "gpu",
            "submit_time": submit_time,
            "start_time": start,
            "end_time": end,
            "runtime_s": end - start,
            "queue_delay_s": start - submit_time,
            "iterations": 0,
            "checksum": 0.0,
        }

    device = torch.device("cuda")

    with torch.cuda.stream(stream):
        if h2d_mb > 0:
            n = h2d_mb * 1024 * 1024 // 4
            host_data = torch.rand((n,), dtype=torch.float32, pin_memory=True)
            device_data = host_data.to(device, non_blocking=True)
        else:
            device_data = None

        memory_block = None
        if gpu_memory_mb > 0:
            n = gpu_memory_mb * 1024 * 1024 // 4
            memory_block = torch.empty((n,), device=device, dtype=torch.float32)
            memory_block.fill_(1.0)

        a = torch.rand((matrix_size, matrix_size), device=device, dtype=torch.float32)
        b = torch.rand((matrix_size, matrix_size), device=device, dtype=torch.float32)

        iterations = 0
        checksum = 0.0

        while time.perf_counter() - start < target_s:
            c = a @ b
            stream.synchronize()
            checksum += float(c[0, 0].item())
            iterations += 1

        if d2h_mb > 0:
            n = d2h_mb * 1024 * 1024 // 4
            out = torch.empty((n,), device=device, dtype=torch.float32)
            host_out = out.cpu()
            stream.synchronize()
            checksum += float(host_out[0].item())

        if memory_block is not None:
            checksum += float(memory_block[0].item())

        if device_data is not None:
            checksum += float(device_data[0].item())

    stream.synchronize()
    end = time.perf_counter()

    return {
        "event_id": event["event_id"],
        "stage": "gpu",
        "submit_time": submit_time,
        "start_time": start,
        "end_time": end,
        "runtime_s": end - start,
        "queue_delay_s": start - submit_time,
        "iterations": iterations,
        "checksum": checksum,
    }


def _merge_event_results(
    event: Dict[str, Any],
    cpu_pre_result: Dict[str, Any],
    cpu_post_result: Dict[str, Any],
    gpu_result: Dict[str, Any],
    gpu_mode: str,
    gpu_wait_start_time: float,
    gpu_wait_end_time: float,
) -> Dict[str, Any]:
    """
    Lightweight event-level reduction.

    This represents merging CPU/GPU results into an event record.
    """

    start = time.perf_counter()

    time.sleep(float(event.get("reduce_work_s", 0.01)))

    end = time.perf_counter()
    cpu_runtime_s = cpu_pre_result["runtime_s"] + cpu_post_result["runtime_s"]

    return {
        "event_id": event["event_id"],
        "stage": "event",
        "gpu_mode": gpu_mode,
        "gpu_async": gpu_mode == "async",
        "start_time": cpu_pre_result["start_time"],
        "end_time": end,
        "runtime_s": end - cpu_pre_result["start_time"],
        "cpu_start_time": cpu_pre_result["start_time"],
        "cpu_end_time": cpu_post_result["end_time"],
        "cpu_runtime_s": cpu_runtime_s,
        "cpu_pre_start_time": cpu_pre_result["start_time"],
        "cpu_pre_end_time": cpu_pre_result["end_time"],
        "cpu_pre_runtime_s": cpu_pre_result["runtime_s"],
        "cpu_pre_iterations": cpu_pre_result["iterations"],
        "cpu_post_start_time": cpu_post_result["start_time"],
        "cpu_post_end_time": cpu_post_result["end_time"],
        "cpu_post_runtime_s": cpu_post_result["runtime_s"],
        "cpu_post_iterations": cpu_post_result["iterations"],
        "gpu_submit_time": gpu_result["submit_time"],
        "gpu_start_time": gpu_result["start_time"],
        "gpu_end_time": gpu_result["end_time"],
        "gpu_runtime_s": gpu_result["runtime_s"],
        "gpu_queue_delay_s": gpu_result["queue_delay_s"],
        "gpu_iterations": gpu_result["iterations"],
        "gpu_wait_start_time": gpu_wait_start_time,
        "gpu_wait_end_time": gpu_wait_end_time,
        "gpu_wait_runtime_s": gpu_wait_end_time - gpu_wait_start_time,
        "merge_start_time": start,
        "merge_end_time": end,
        "merge_runtime_s": end - start,
    }


def _submit_gpu_stage(event: Dict[str, Any], executor: LocalGpuExecutor) -> GpuTaskHandle:
    return executor.submit(event)


def _gpu_mode(params: Dict[str, Any]) -> str:
    if "gpu_mode" in params:
        value = str(params["gpu_mode"]).lower().replace("-", "_")
    else:
        legacy_async = params.get("gpu_async", False)
        if not isinstance(legacy_async, bool):
            raise ValueError(f"gpu_async must be a boolean value (got {legacy_async!r})")
        value = "async" if legacy_async else "blocking"

    aliases = {
        "sync": "blocking",
        "synchronous": "blocking",
        "wait_at_dispatch": "blocking",
        "barrier": "event_barrier",
        "event_sync": "event_barrier",
        "event_synchronous": "event_barrier",
        "wait_before_next": "event_barrier",
        "ordered": "event_barrier",
        "full_async": "async",
    }
    value = aliases.get(value, value)

    valid_modes = {"blocking", "event_barrier", "async"}
    if value not in valid_modes:
        raise ValueError(f"gpu_mode must be one of {sorted(valid_modes)} (got {value!r})")

    return value


def generate_events(params: Dict[str, Any]) -> List[EventSpec]:
    n_events = int(params.get("n_events", 20))
    return [sample_event(event_id, params) for event_id in range(n_events)]


def _wait_for_gpu(handle: GpuTaskHandle) -> tuple[Dict[str, Any], float, float]:
    wait_start = time.perf_counter()
    gpu_result = handle.wait()
    wait_end = time.perf_counter()
    return gpu_result, wait_start, wait_end


def _merge_pending_event(
    item: Dict[str, Any],
    gpu_mode: str,
) -> Dict[str, Any]:
    gpu_result, wait_start, wait_end = _wait_for_gpu(item["gpu_handle"])
    return _merge_event_results(
        item["event"],
        item["cpu_pre_result"],
        item["cpu_post_result"],
        gpu_result,
        gpu_mode,
        wait_start,
        wait_end,
    )


def _merge_ready_pending_events(
    pending: List[Dict[str, Any]],
    reduce_results: List[Dict[str, Any]],
    gpu_mode: str,
) -> None:
    ready_items = [item for item in pending if item["gpu_handle"].ready()]

    for item in ready_items:
        pending.remove(item)
        reduce_results.append(_merge_pending_event(item, gpu_mode))


def run_simulation(params: Dict[str, Any]) -> pd.DataFrame:
    events = generate_events(params)
    gpu_mode = _gpu_mode(params)
    async_merge_ready = bool(params.get("async_merge_ready", False))

    t0 = time.perf_counter()

    reduce_results: List[Dict[str, Any]] = []
    event_records: List[Dict[str, Any]] = []
    pending_gpu: List[Dict[str, Any]] = []

    executor = LocalGpuExecutor()

    try:
        for spec in events:
            event = asdict(spec)
            event["reduce_work_s"] = float(params.get("reduce_work_s", 0.01))

            cpu_pre_result = _run_cpu_payload(event, "cpu_pre_work_s", "cpu_pre")
            gpu_handle = _submit_gpu_stage(event, executor)

            if gpu_mode == "blocking":
                gpu_result, wait_start, wait_end = _wait_for_gpu(gpu_handle)
                cpu_post_result = _run_cpu_payload(event, "cpu_post_work_s", "cpu_post")
                reduce_results.append(
                    _merge_event_results(
                        event,
                        cpu_pre_result,
                        cpu_post_result,
                        gpu_result,
                        gpu_mode,
                        wait_start,
                        wait_end,
                    )
                )
            else:
                cpu_post_result = _run_cpu_payload(event, "cpu_post_work_s", "cpu_post")

                pending_item = {
                    "event": event,
                    "cpu_pre_result": cpu_pre_result,
                    "cpu_post_result": cpu_post_result,
                    "gpu_handle": gpu_handle,
                }

                if gpu_mode == "event_barrier":
                    reduce_results.append(_merge_pending_event(pending_item, gpu_mode))
                else:
                    pending_gpu.append(pending_item)
                    if async_merge_ready:
                        _merge_ready_pending_events(pending_gpu, reduce_results, gpu_mode)

            event_records.append(
                {
                    "event_id": spec.event_id,
                    **asdict(spec),
                }
            )

            print(
                f"processed CPU event={spec.event_id:04d} "
                f"at={cpu_pre_result['start_time'] - t0:8.3f}s "
                f"mode={gpu_mode:13s} "
                f"size={spec.event_size_s:6.3f}s "
                f"cpu_pre={spec.cpu_pre_work_s:6.3f}s "
                f"gpu={spec.gpu_work_s:6.3f}s "
                f"cpu_post={spec.cpu_post_work_s:6.3f}s "
                f"dispatch_frac={spec.dispatch_fraction:5.2f} "
                f"gpu_mem={spec.gpu_memory_mb:5d} MB",
                flush=True,
            )

        while pending_gpu:
            item = pending_gpu.pop(0)
            reduce_results.append(_merge_pending_event(item, gpu_mode))
    finally:
        executor.close()

    events_df = pd.DataFrame(event_records)
    results_df = pd.DataFrame(reduce_results)

    df = events_df.merge(results_df, on="event_id", how="left")

    for col in [
        "start_time",
        "end_time",
        "cpu_start_time",
        "cpu_end_time",
        "cpu_pre_start_time",
        "cpu_pre_end_time",
        "cpu_post_start_time",
        "cpu_post_end_time",
        "gpu_submit_time",
        "gpu_start_time",
        "gpu_end_time",
        "gpu_wait_start_time",
        "gpu_wait_end_time",
        "merge_start_time",
        "merge_end_time",
    ]:
        if col in df.columns:
            df[f"{col}_offset_s"] = df[col] - t0

    df["start_offset_s"] = df["start_time"] - t0
    df["end_offset_s"] = df["end_time"] - t0

    return df.sort_values("event_id")


def default_params() -> Dict[str, Any]:
    return {
        "seed": 12345,
        "n_events": 20,
        "event_size_distribution": "gaussian",
        "mean_event_size_s": 3.0,
        "event_size_spread_s": 0.6,
        "min_event_size_s": 0.05,
        "cpu_fraction_beta_alpha": 4.0,
        "cpu_fraction_beta_beta": 4.0,
        "gpu_dispatch_fraction_mean": 0.5,
        "gpu_dispatch_fraction_jitter": 0.1,
        "gpu_dispatch_fraction_min": 0.35,
        "gpu_dispatch_fraction_max": 0.65,
        "cpu_matrix_base": 512,
        "gpu_matrix_base": 2048,
        "gpu_memory_base_mb": 256,
        "gpu_memory_jitter_mb": 512,
        "h2d_base_mb": 64,
        "d2h_base_mb": 16,
        "num_cpus_per_event": 1.0,
        "num_gpus_per_event": 1.0,
        "gpu_async": False,
        "async_merge_ready": False,
        "reduce_work_s": 0.01,
    }


def resolve_output_path(
    config_path: Optional[str],
    out_path: Optional[str],
    out_dir: Optional[str],
) -> str:
    if out_path is not None:
        return out_path

    if config_path is None:
        filename = "default_params.csv"
    else:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        filename = f"{config_name}.csv"

    if out_dir is None:
        return filename

    return os.path.join(out_dir, filename)


def ensure_output_directory(out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON parameter file")
    parser.add_argument(
        "--mode",
        choices=["blocking", "event_barrier", "async"],
        default=None,
        help="override gpu_mode from config",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="accepted for CLI compatibility; local mode does not create a CPU resource pool",
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=None,
        help="accepted for CLI compatibility; local mode uses the current CUDA device",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=("full output CSV path; overrides --out-dir and defaults to <config-name>.csv with --config, otherwise default_params.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default="simload_runs",
        help="directory for the generated CSV filename when --out is not set",
    )
    args = parser.parse_args()

    params = default_params()

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            user_params = json.load(f)
        params.update(user_params)

    if args.mode is not None:
        params["gpu_mode"] = args.mode

    if args.num_cpus is not None:
        params["num_cpus"] = args.num_cpus

    if args.num_gpus is not None:
        params["num_gpus"] = args.num_gpus

    out_path = resolve_output_path(args.config, args.out, args.out_dir)

    print("Local Simload mode:")
    print(f"CUDA device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'unavailable'}")
    print(f"gpu_mode: {_gpu_mode(params)}")
    print()

    df = run_simulation(params)

    cols = [
        "event_id",
        "gpu_mode",
        "gpu_async",
        "start_offset_s",
        "end_offset_s",
        "runtime_s",
        "event_size_s",
        "dispatch_fraction",
        "cpu_pre_runtime_s",
        "cpu_post_runtime_s",
        "cpu_runtime_s",
        "gpu_runtime_s",
        "gpu_queue_delay_s",
        "gpu_wait_runtime_s",
        "merge_runtime_s",
        "cpu_fraction",
        "gpu_fraction",
        "cpu_pre_work_s",
        "gpu_work_s",
        "cpu_post_work_s",
        "gpu_memory_mb",
        "h2d_mb",
        "d2h_mb",
    ]

    print()
    print(df[cols].to_string(index=False))

    ensure_output_directory(out_path)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
