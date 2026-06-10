#!/usr/bin/env python3

import tvm
import numpy as np
import csv
import argparse
import hashlib
import signal
from pathlib import Path

from tvm import tir
from tvm import meta_schedule as ms


class RecordTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise RecordTimeout("record timed out")


def trace_fingerprint(trace):
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)
    s = obj if isinstance(obj, str) else "\n".join(str(x) for x in obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def make_random_args(args_info, device):
    args = []
    for idx, t in enumerate(args_info):
        shape = [int(s) for s in t.shape]
        arr = np.random.rand(*shape).astype(t.dtype)
        args.append(tvm.nd.array(arr, device=device))
        print(f"[DEBUG] arg[{idx}] shape={shape}, dtype={t.dtype}", flush=True)
    return args


def extract_task_name(trace):
    for inst in trace.insts:
        if inst.kind.name == "GetBlock":
            block = inst.attrs[0]
            if block not in ["pad_temp", "root"]:
                return block
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--target", default=None)

    # In the new WarmupEvaluator:
    # total observed calls = number * repeat
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=20)

    # These are kept only to match the C++ function signature.
    # The debug WarmupEvaluator ignores them.
    parser.add_argument("--min-repeat-ms", type=int, default=0)

    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=160)

    args = parser.parse_args()

    signal.signal(signal.SIGALRM, _timeout_handler)

    dev = tvm.cpu(0)

    # The new C++ WarmupEvaluator does not use intervals,
    # but the function signature still requires an NDArray.
    dummy_intervals = tvm.nd.array(
        np.zeros((1, 2), dtype="float64"),
        device=dev,
    )

    if args.out is None:
        out_path = Path(f"warmup_debug_{Path(args.work_dir).name}.csv")
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("[DEBUG] warmup_evaluator.py started", flush=True)
    print(f"[DEBUG] work_dir      = {args.work_dir}", flush=True)
    print(f"[DEBUG] out_path      = {out_path}", flush=True)
    print(f"[DEBUG] target arg    = {args.target}", flush=True)
    print(f"[DEBUG] number        = {args.number}", flush=True)
    print(f"[DEBUG] repeat        = {args.repeat}", flush=True)
    print(f"[DEBUG] min_repeat_ms = {args.min_repeat_ms}", flush=True)
    print(f"[DEBUG] timeout_sec   = {args.timeout_sec}", flush=True)
    print("[DEBUG] mode          = per-kernel-call warmup power debug", flush=True)
    print("[DEBUG] C++ return     = avg_latency_ms, avg_power_w, total_energy_j", flush=True)
    print("=" * 60, flush=True)

    work_dir_path = Path(args.work_dir)
    if not work_dir_path.exists():
        print(f"[ERROR] work_dir does not exist: {args.work_dir}", flush=True)
        return

    print("[DEBUG] files in work_dir:", flush=True)
    for p in sorted(work_dir_path.iterdir()):
        print(f"  - {p.name}", flush=True)

    print(f"[DEBUG] device = {dev}", flush=True)

    print("[DEBUG] Opening JSONDatabase...", flush=True)
    db = ms.database.JSONDatabase(work_dir=args.work_dir)
    recs = db.get_all_tuning_records()
    print(f"[DEBUG] num_records = {len(recs)}", flush=True)

    if len(recs) == 0:
        print("[WARNING] No tuning records found. CSV may contain only header.", flush=True)

    mode = "a" if out_path.exists() else "w"
    print(f"[DEBUG] csv mode = {mode}", flush=True)

    with open(out_path, mode, newline="") as f:
        writer = csv.writer(f)
        header_written = out_path.exists()

        if not header_written:
            print("[DEBUG] Writing CSV header...", flush=True)
            writer.writerow([
                "record_id",
                "workload_hash",
                "trace_hash",
                "task_name",
                "avg_latency_ms",
                "avg_power_w",
                "total_energy_j",
                "number",
                "repeat",
                "total_calls",
            ])
            f.flush()
            header_written = True

        for i, r in enumerate(recs):
            if i < args.start_idx:
                continue

            print("-" * 60, flush=True)
            print(f"[DEBUG] Processing record {i}", flush=True)

            task_name = "unknown"
            workload_hash = "unknown"
            trace_hash = "unknown"

            try:
                mod = r.workload.mod
                workload_hash = int(tvm.ir.structural_hash(mod))
                trace_hash = trace_fingerprint(r.trace)
                task_name = extract_task_name(r.trace)
                target = tvm.target.Target(args.target) if args.target else r.target

                print(f"[DEBUG] task_name      = {task_name}", flush=True)
                print(f"[DEBUG] workload_hash = {workload_hash}", flush=True)
                print(f"[DEBUG] trace_hash    = {trace_hash}", flush=True)
                print(f"[DEBUG] target        = {target}", flush=True)

                if args.timeout_sec > 0:
                    signal.alarm(args.timeout_sec)

                print("[DEBUG] Creating schedule...", flush=True)
                sch = tir.Schedule(mod)

                print("[DEBUG] Applying trace to schedule...", flush=True)
                r.trace.apply_to_schedule(sch, remove_postproc=True)

                print("[DEBUG] Building module...", flush=True)
                rt_mod = tvm.build(sch.mod, target)

                print("[DEBUG] Creating random input args...", flush=True)
                args_nd = make_random_args(r.args_info, dev)

                print("[DEBUG] Loading global evaluator...", flush=True)
                f_eval = tvm.get_global_func(
                    "runtime.profiling.warmup_evaluator"
                )
                print("[DEBUG] Global evaluator loaded successfully", flush=True)

                print("[DEBUG] Creating evaluator instance...", flush=True)
                evaluator = f_eval(
                    rt_mod["main"],
                    dev,
                    args.number,
                    args.repeat,
                    args.min_repeat_ms,
                    50,              # limit_zero_time_iterations, ignored by new C++ debug version
                    dummy_intervals, # required only because C++ signature still has NDArray intervals
                )

                print("[DEBUG] Running evaluator...", flush=True)

                vals = evaluator(*args_nd)

                print(f"[DEBUG] Raw evaluator return = {vals}", flush=True)

                vals = [float(x) for x in vals]

                if len(vals) != 3:
                    raise RuntimeError(
                        f"Expected 3 return values "
                        f"(avg_latency_ms, avg_power_w, total_energy_j), "
                        f"but got {len(vals)}: {vals}"
                    )

                avg_latency_ms, avg_power_w, total_energy_j = vals
                total_calls = args.number * args.repeat

                row = [
                    i,
                    workload_hash,
                    trace_hash,
                    task_name,
                    avg_latency_ms,
                    avg_power_w,
                    total_energy_j,
                    args.number,
                    args.repeat,
                    total_calls,
                ]

                writer.writerow(row)
                f.flush()

                print(
                    f"[{i}] {task_name} | "
                    f"avg_latency={avg_latency_ms:.6f} ms | "
                    f"avg_power={avg_power_w:.6f} W | "
                    f"total_energy={total_energy_j:.9f} J | "
                    f"calls={total_calls}",
                    flush=True,
                )

            except RecordTimeout:
                print(
                    f"[{i}] TIMEOUT | {task_name} | workload_hash={workload_hash}",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"[{i}] FAILED: {type(e).__name__}: {e}",
                    flush=True,
                )

            finally:
                signal.alarm(0)

    print("=" * 60, flush=True)
    print("CSV written to:", out_path, flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()