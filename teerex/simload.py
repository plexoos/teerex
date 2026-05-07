#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import time

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ray
import torch


@dataclass
class EventSpec:
    event_id: int
    event_size_s: float
    cpu_fraction: float
    gpu_fraction: float

    cpu_work_s: float
    gpu_work_s: float

    cpu_matrix_size: int
    gpu_matrix_size: int

    gpu_memory_mb: int
    h2d_mb: int
    d2h_mb: int

    num_cpus: float
    num_gpus: float


def sample_event(event_id: int, params: Dict[str, Any]) -> EventSpec:
    """
    Generate one synthetic Geant4-like event record.

    The event size is the total synthetic processing payload. The CPU/GPU split
    is then sampled independently with a Beta distribution.
    """

    seed = params.get("seed", 12345)
    rng = random.Random(seed + event_id)

    event_size_s = sample_event_size_s(rng, params)

    beta_alpha = float(params.get("cpu_fraction_beta_alpha", 4.0))
    beta_beta = float(params.get("cpu_fraction_beta_beta", 4.0))

    cpu_fraction = rng.betavariate(beta_alpha, beta_beta)
    gpu_fraction = 1.0 - cpu_fraction

    cpu_work_s = event_size_s * cpu_fraction
    gpu_work_s = event_size_s * gpu_fraction

    # Matrix sizes loosely correlated with work.
    cpu_matrix_base = int(params.get("cpu_matrix_base", 512))
    gpu_matrix_base = int(params.get("gpu_matrix_base", 2048))

    cpu_matrix_size = max(128, int(cpu_matrix_base * math.sqrt(max(cpu_work_s, 0.05))))
    gpu_matrix_size = max(256, int(gpu_matrix_base * math.sqrt(max(gpu_work_s, 0.05))))

    # Optional memory and transfer sizes correlated with GPU fraction.
    gpu_memory_base_mb = int(params.get("gpu_memory_base_mb", 256))
    gpu_memory_jitter_mb = int(params.get("gpu_memory_jitter_mb", 256))
    mean_event_size_s = float(params.get("mean_event_size_s", 3.0))
    size_scale = event_size_s / max(mean_event_size_s, 1.0e-9)

    gpu_memory_mb = int( ( gpu_memory_base_mb + rng.uniform(0, gpu_memory_jitter_mb) ) * gpu_fraction * size_scale )

    h2d_mb = int(params.get("h2d_base_mb", 64) * gpu_fraction * size_scale)
    d2h_mb = int(params.get("d2h_base_mb", 16) * gpu_fraction * size_scale)

    return EventSpec(
        event_id=event_id,
        event_size_s=event_size_s,
        cpu_fraction=cpu_fraction,
        gpu_fraction=gpu_fraction,
        cpu_work_s=cpu_work_s,
        gpu_work_s=gpu_work_s,
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


@ray.remote
def cpu_stage(event: Dict[str, Any]) -> Dict[str, Any]:
    return _run_cpu_stage(event)


def _run_cpu_stage(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthetic CPU stage.

    This represents Geant4-like CPU work:
      - tracking
      - stepping
      - geometry navigation
      - physics process bookkeeping
      - hit creation
    """

    start = time.perf_counter()

    target_s = float(event["cpu_work_s"])
    matrix_size = int(event["cpu_matrix_size"])

    if target_s <= 0.0:
        end = time.perf_counter()
        return {
            "event_id": event["event_id"],
            "stage": "cpu",
            "start_time": start,
            "end_time": end,
            "runtime_s": end - start,
            "iterations": 0,
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
        "event_id": event["event_id"],
        "stage": "cpu",
        "start_time": start,
        "end_time": end,
        "runtime_s": end - start,
        "iterations": iterations,
        "checksum": checksum,
    }


@ray.remote
def gpu_stage(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthetic GPU stage.

    This represents GPU-side work such as:
      - optical photon propagation
      - large vectorized scoring
      - batched hit processing
      - detector response kernels

    The simulation submits this task only after the single CPU manager has
    completed CPU work for the event. In async mode, GPU work from this event
    may overlap CPU work from later events.
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
            "start_time": start,
            "end_time": end,
            "runtime_s": end - start,
            "iterations": 0,
        }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    device = torch.device("cuda")

    # Simulate host-to-device transfer.
    if h2d_mb > 0:
        n = h2d_mb * 1024 * 1024 // 4
        host_data = torch.rand((n,), dtype=torch.float32, pin_memory=True)
        device_data = host_data.to(device, non_blocking=False)
        torch.cuda.synchronize()
    else:
        device_data = None

    # Simulate event-level GPU memory footprint.
    memory_block = None
    if gpu_memory_mb > 0:
        n = gpu_memory_mb * 1024 * 1024 // 4
        memory_block = torch.empty((n,), device=device, dtype=torch.float32)
        memory_block.fill_(1.0)
        torch.cuda.synchronize()

    a = torch.rand((matrix_size, matrix_size), device=device, dtype=torch.float32)
    b = torch.rand((matrix_size, matrix_size), device=device, dtype=torch.float32)

    iterations = 0
    checksum = 0.0

    while time.perf_counter() - start < target_s:
        c = a @ b
        torch.cuda.synchronize()
        checksum += float(c[0, 0].item())
        iterations += 1

    # Simulate device-to-host transfer.
    if d2h_mb > 0:
        n = d2h_mb * 1024 * 1024 // 4
        out = torch.empty((n,), device=device, dtype=torch.float32)
        host_out = out.cpu()
        torch.cuda.synchronize()
        checksum += float(host_out[0].item())

    if memory_block is not None:
        checksum += float(memory_block[0].item())

    if device_data is not None:
        checksum += float(device_data[0].item())

    end = time.perf_counter()

    return {
        "event_id": event["event_id"],
        "stage": "gpu",
        "start_time": start,
        "end_time": end,
        "runtime_s": end - start,
        "iterations": iterations,
        "checksum": checksum,
    }


def _merge_event_results(
    event: Dict[str, Any],
    cpu_result: Dict[str, Any],
    gpu_result: Dict[str, Any],
    gpu_async: bool,
) -> Dict[str, Any]:
    """
    Lightweight event-level reduction.

    This represents merging CPU/GPU results into an event record.
    """

    start = time.perf_counter()

    # Small bookkeeping delay.
    time.sleep(float(event.get("reduce_work_s", 0.01)))

    end = time.perf_counter()

    return {
        "event_id": event["event_id"],
        "stage": "event",
        "gpu_async": gpu_async,
        "start_time": cpu_result["start_time"],
        "end_time": end,
        "runtime_s": end - cpu_result["start_time"],
        "cpu_start_time": cpu_result["start_time"],
        "cpu_end_time": cpu_result["end_time"],
        "cpu_runtime_s": cpu_result["runtime_s"],
        "gpu_start_time": gpu_result["start_time"],
        "gpu_end_time": gpu_result["end_time"],
        "gpu_runtime_s": gpu_result["runtime_s"],
        "merge_start_time": start,
        "merge_end_time": end,
        "merge_runtime_s": end - start,
    }


def _submit_gpu_stage(event: Dict[str, Any], params: Dict[str, Any]) -> ray.ObjectRef:
    return gpu_stage.options(
        num_cpus=float(params.get("num_cpus_per_gpu_task", 0.25)),
        num_gpus=float(event["num_gpus"]),
    ).remote(event)


def _gpu_async_enabled(params: Dict[str, Any]) -> bool:
    value = params.get("gpu_async", False)

    if isinstance(value, bool):
        return value
    else:
        raise ValueError(f"gpu_async must be a boolean value (got {value!r})")


def generate_events(params: Dict[str, Any]) -> List[EventSpec]:
    n_events = int(params.get("n_events", 20))
    return [sample_event(event_id, params) for event_id in range(n_events)]


def run_simulation(params: Dict[str, Any]) -> pd.DataFrame:
    events = generate_events(params)
    gpu_async = _gpu_async_enabled(params)

    t0 = time.perf_counter()

    reduce_results = []
    event_records = []
    pending_gpu = []

    for spec in events:
        event = asdict(spec)
        event["reduce_work_s"] = float(params.get("reduce_work_s", 0.01))

        cpu_result = _run_cpu_stage(event)
        gpu_ref = _submit_gpu_stage(event, params)

        if gpu_async:
            pending_gpu.append(
                {
                    "event": event,
                    "cpu_result": cpu_result,
                    "gpu_ref": gpu_ref,
                }
            )
        else:
            gpu_result = ray.get(gpu_ref)
            reduce_results.append(
                _merge_event_results(event, cpu_result, gpu_result, gpu_async)
            )

        event_records.append(
            {
                "event_id": spec.event_id,
                **asdict(spec),
            }
        )

        print(
            f"processed CPU event={spec.event_id:04d} "
            f"at={cpu_result['start_time'] - t0:8.3f}s "
            f"async={str(gpu_async):5s} "
            f"size={spec.event_size_s:6.3f}s "
            f"cpu_frac={spec.cpu_fraction:5.2f} "
            f"gpu_frac={spec.gpu_fraction:5.2f} "
            f"cpu_work={spec.cpu_work_s:6.3f}s "
            f"gpu_work={spec.gpu_work_s:6.3f}s "
            f"gpu_mem={spec.gpu_memory_mb:5d} MB",
            flush=True,
        )

    while pending_gpu:
        refs = [item["gpu_ref"] for item in pending_gpu]
        ready_refs, _ = ray.wait(refs, num_returns=1)
        ready_ref = ready_refs[0]
        ready_index = next(i for i, item in enumerate(pending_gpu) if item["gpu_ref"] == ready_ref)
        ready_item = pending_gpu.pop(ready_index)
        gpu_result = ray.get(ready_ref)
        reduce_results.append(
            _merge_event_results(
                ready_item["event"],
                ready_item["cpu_result"],
                gpu_result,
                gpu_async,
            )
        )

    events_df = pd.DataFrame(event_records)
    results_df = pd.DataFrame(reduce_results)

    df = events_df.merge(results_df, on="event_id", how="left")

    df["start_offset_s"] = df["start_time"] - t0
    df["end_offset_s"] = df["end_time"] - t0

    return df.sort_values("event_id")


def default_params() -> Dict[str, Any]:
    return {
        "seed": 12345,

        "n_events": 20,
        # Event size is the total synthetic event payload in seconds before
        # splitting it into CPU and GPU work.
        "event_size_distribution": "gaussian",
        "mean_event_size_s": 3.0,
        "event_size_spread_s": 0.6,
        "min_event_size_s": 0.05,

        # Beta(alpha, beta) for CPU fraction.
        # alpha=beta gives balanced events.
        # alpha > beta gives CPU-heavy events.
        # alpha < beta gives GPU-heavy events.
        "cpu_fraction_beta_alpha": 4.0,
        "cpu_fraction_beta_beta": 4.0,

        "cpu_matrix_base": 512,
        "gpu_matrix_base": 2048,

        "gpu_memory_base_mb": 256,
        "gpu_memory_jitter_mb": 512,
        "h2d_base_mb": 64,
        "d2h_base_mb": 16,

        "num_cpus_per_event": 1.0,
        "num_gpus_per_event": 1.0,
        "num_cpus_per_gpu_task": 0.25,

        "gpu_async": False,
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
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=float, default=None)
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

    out_path = resolve_output_path(args.config, args.out, args.out_dir)

    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        object_store_memory=10_000_000_000,
    )

    print("Ray resources:")
    print(ray.cluster_resources())
    print()

    df = run_simulation(params)

    cols = [
        "event_id",
        "gpu_async",
        "start_offset_s",
        "end_offset_s",
        "runtime_s",
        "event_size_s",
        "cpu_runtime_s",
        "gpu_runtime_s",
        "merge_runtime_s",
        "cpu_fraction",
        "gpu_fraction",
        "cpu_work_s",
        "gpu_work_s",
        "gpu_memory_mb",
        "h2d_mb",
        "d2h_mb",
    ]

    print()
    print(df[cols].to_string(index=False))

    ensure_output_directory(out_path)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
