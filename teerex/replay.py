"""Offline CPU/GPU replay; supplied durations are service demands, not forecasts."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import heapq
import math
from numbers import Integral, Real

import pandas as pd


_DURATIONS = (
    "cpu_pre_runtime_s",
    "gpu_runtime_s",
    "cpu_post_locked_runtime_s",
    "cpu_post_unlocked_runtime_s",
)
_BOUNDARIES = (
    "release", "admission", "start", "end", "cpu_start", "cpu_end",
    "cpu_pre_start", "cpu_pre_end", "gpu_submit", "gpu_start", "gpu_end",
    "gpu_release", "gpu_wait_start", "gpu_wait_end", "cpu_post_start",
    "cpu_post_end", "cpu_post_locked_start", "cpu_post_locked_end",
    "cpu_post_unlocked_start", "cpu_post_unlocked_end",
)
_COLUMNS = (
    "scenario", "dispatch_mode", "gpu_mode", "gpu_async", "event_id",
    "workers", "max_inflight", "effective_max_inflight", "worker_id",
    "cpu_pre_worker_id", "cpu_post_worker_id", "num_photons",
    *_DURATIONS, "cpu_runtime_s", "cpu_post_runtime_s", "gpu_gate_runtime_s",
    "gpu_queue_delay_s", "gpu_wait_runtime_s", "cpu_post_queue_delay_s",
    "latency_s", "service_latency_s",
    *(f"{boundary}_time_offset_s" for boundary in _BOUNDARIES),
    "event_batch_size",
)


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate(events):
    try:
        inputs = list(events)
    except TypeError as error:
        raise ValueError("events must be a sequence of mappings") from error
    rows = []
    seen = set()
    for index, event in enumerate(inputs):
        if not isinstance(event, Mapping):
            raise ValueError(f"event {index} must be a mapping")
        if "event_id" not in event:
            raise ValueError(f"event {index} is missing event_id")
        event_id = _integer(event["event_id"], "event_id", 0)
        if event_id in seen:
            raise ValueError(f"duplicate event_id: {event_id}")
        seen.add(event_id)
        row = {"event_id": event_id}
        for name in _DURATIONS:
            value = event.get(name)
            message = f"event {event_id}: {name} must be finite and nonnegative"
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(message)
            try:
                duration = float(value)
            except OverflowError as error:
                raise ValueError(message) from error
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(message)
            row[name] = duration
        row["num_photons"] = (_integer(event["num_photons"], "num_photons", 0)
                              if "num_photons" in event else float("nan"))
        rows.append(row)
    return rows


def replay(events, *, workers=1, max_inflight=1, policy="blocking",
           event_batch_size=1) -> pd.DataFrame:
    """Replay a finite batch released at time zero, preserving input row order.

    Each event supplies a unique nonnegative integer ``event_id``, four finite
    nonnegative service durations named in ``_DURATIONS``, and optionally a
    nonnegative integer ``num_photons``. Input records are not modified.

    ``blocking`` admits one event. ``synchronous`` retains one worker from
    admission through queue/GPU waits and both post phases, and acquires the GPU
    in input order. ``async`` admits up to ``max_inflight`` events, releases its
    CPU worker after pre, and reacquires any free worker for post. Its GPU queue
    is FIFO by submission, with input order breaking simultaneous ties. Ready
    post work has priority over pre; CPU phases are nonpreemptive. Locked and
    unlocked post run contiguously on the same worker, releasing the exclusive
    GPU dispatch gate between them. Completed stages at a timestamp are handled
    before new starts, including zero-duration stages.

    ``event_batch_size > 1`` is supported only for synchronous execution. Each
    available worker reserves the next contiguous input chunk and runs its
    events serially on that worker, retaining ownership of the chunk between
    events. A new chunk is assigned when its first event can be admitted; lower
    worker IDs break simultaneous availability ties. Reserved future events do
    not count toward ``max_inflight``. Thus admission can leave global input
    order, while GPU acquisition still follows it. The in-flight bound limits
    participating workers when smaller than ``workers``. Batch size one retains
    ordinary FIFO event admission.

    Workers model both event ownership (synchronous) and the shared finite CPU
    pool. Durations are fixed and do not change with contention. Scheduling
    choices never rank events by future service durations. This is an offline
    counterfactual, not an asynchronous Simphony executor or hardware-utilization
    model. GPU service can include host transfer/synchronization time.

    Offsets use the common release origin. ``latency_s`` includes admission
    backlog; ``service_latency_s`` starts at admission. ``cpu_start/end`` are an
    envelope: CPU busy demand is the sum of pre and post service. GPU gate time
    includes waiting for a post worker plus locked post, beyond GPU service.
    Missing photon counts remain NaN. Empty input retains the output schema.
    """
    workers = _integer(workers, "workers", 1)
    max_inflight = _integer(max_inflight, "max_inflight", 1)
    event_batch_size = _integer(event_batch_size, "event_batch_size", 1)
    if not isinstance(policy, str) or policy not in ("blocking", "synchronous", "async"):
        raise ValueError("policy must be 'blocking', 'synchronous', or 'async'")
    if event_batch_size != 1 and policy != "synchronous":
        raise ValueError("event_batch_size > 1 requires synchronous policy")
    rows = _validate(events)
    asynchronous = policy == "async"
    capacity = (1 if policy == "blocking" else
                min(workers, max_inflight) if policy == "synchronous" else max_inflight)
    free_workers = list(range(workers))  # Already a min-heap.
    event_worker = {}
    worker_batches = {}
    pre_ready, post_ready = deque(), deque()
    gpu_ready, completions = [], []
    now = 0.0
    admitted = active = completed = next_gpu = sequence = 0
    gate = None

    def finish_after(index, stage, duration):
        nonlocal sequence
        end = now + duration
        if not math.isfinite(end):
            raise ValueError("replay timeline exceeds finite floating-point range")
        heapq.heappush(completions, (end, sequence, index, stage))
        sequence += 1
        return end

    while completed < len(rows):
        # Drain the entire timestamp, including zero-duration continuations.
        while completions and completions[0][0] == now:
            _, _, index, stage = heapq.heappop(completions)
            row = rows[index]
            if stage == "pre":
                if asynchronous:
                    heapq.heappush(free_workers, event_worker[index])
                priority = now if asynchronous else index
                heapq.heappush(gpu_ready, (priority, index))
            elif stage == "gpu":
                post_ready.append(index)
            elif stage == "locked":
                gate = None
                row["gpu_release_time_offset_s"] = now
                row["cpu_post_unlocked_start_time_offset_s"] = now
                row["cpu_post_unlocked_end_time_offset_s"] = finish_after(
                    index, "post", row["cpu_post_unlocked_runtime_s"])
            else:  # Unlocked post completes the event and frees its WIP slot.
                row["cpu_post_end_time_offset_s"] = now
                heapq.heappush(free_workers, event_worker.pop(index))
                active -= 1
                completed += 1

        # Chunk reservations do not occupy active-event slots. Once all input
        # is assigned, an idle worker without a reserved tail can retire.
        while active < capacity and (asynchronous or free_workers):
            if event_batch_size > 1:
                worker = heapq.heappop(free_workers)
                if not worker_batches.get(worker):
                    if admitted == len(rows):
                        continue
                    stop = min(len(rows), admitted + event_batch_size)
                    worker_batches[worker] = deque(range(admitted, stop))
                    admitted = stop
                index = worker_batches[worker].popleft()
                event_worker[index] = worker
            else:
                if admitted == len(rows):
                    break
                index = admitted
                admitted += 1
                if not asynchronous:
                    event_worker[index] = heapq.heappop(free_workers)
            rows[index]["admission_time_offset_s"] = now
            pre_ready.append(index)
            active += 1

        if post_ready and (not asynchronous or free_workers):
            index = post_ready.popleft()
            row = rows[index]
            if asynchronous:
                event_worker[index] = heapq.heappop(free_workers)
            row["cpu_post_worker_id"] = event_worker[index]
            row["cpu_post_start_time_offset_s"] = now
            row["cpu_post_locked_start_time_offset_s"] = now
            row["cpu_post_locked_end_time_offset_s"] = finish_after(
                index, "locked", row["cpu_post_locked_runtime_s"])
            continue

        if (gate is None and gpu_ready
                and (asynchronous or gpu_ready[0][1] == next_gpu)):
            _, index = heapq.heappop(gpu_ready)
            row = rows[index]
            gate = index
            next_gpu += 1
            row["gpu_start_time_offset_s"] = now
            row["gpu_end_time_offset_s"] = finish_after(index, "gpu", row["gpu_runtime_s"])
            continue

        if pre_ready and (not asynchronous or free_workers):
            index = pre_ready.popleft()
            row = rows[index]
            if asynchronous:
                event_worker[index] = heapq.heappop(free_workers)
            row["cpu_pre_worker_id"] = event_worker[index]
            row["cpu_pre_start_time_offset_s"] = now
            row["cpu_pre_end_time_offset_s"] = finish_after(
                index, "pre", row["cpu_pre_runtime_s"])
            continue

        if completed < len(rows):
            if not completions:
                raise RuntimeError("replay reached a resource deadlock")
            now = completions[0][0]

    scenario = f"replay_{policy}_w{workers}_inflight{max_inflight}"
    if event_batch_size != 1:
        scenario += f"_batch{event_batch_size}"
    for row in rows:
        pre_start = row["cpu_pre_start_time_offset_s"]
        submit = row["cpu_pre_end_time_offset_s"]
        gpu_start, gpu_end = row["gpu_start_time_offset_s"], row["gpu_end_time_offset_s"]
        release = row["gpu_release_time_offset_s"]
        end = row["cpu_post_end_time_offset_s"]
        post = row["cpu_post_locked_runtime_s"] + row["cpu_post_unlocked_runtime_s"]
        row.update(
            scenario=scenario, dispatch_mode=policy, gpu_mode=policy,
            gpu_async=asynchronous, workers=workers, max_inflight=max_inflight,
            effective_max_inflight=capacity, worker_id=row["cpu_pre_worker_id"],
            event_batch_size=event_batch_size,
            release_time_offset_s=0.0, start_time_offset_s=row["admission_time_offset_s"],
            end_time_offset_s=end, cpu_start_time_offset_s=pre_start,
            cpu_end_time_offset_s=end, gpu_submit_time_offset_s=submit,
            gpu_wait_start_time_offset_s=submit, gpu_wait_end_time_offset_s=gpu_end,
            cpu_post_runtime_s=post, cpu_runtime_s=row["cpu_pre_runtime_s"] + post,
            gpu_gate_runtime_s=release - gpu_start, gpu_queue_delay_s=gpu_start - submit,
            gpu_wait_runtime_s=gpu_end - submit,
            cpu_post_queue_delay_s=row["cpu_post_start_time_offset_s"] - gpu_end,
            latency_s=end, service_latency_s=end - row["admission_time_offset_s"],
        )
    return pd.DataFrame.from_records(rows, columns=_COLUMNS)
