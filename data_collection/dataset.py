#!/usr/bin/env python3

"""
Dataset extraction script for TVM MetaSchedule tuning logs.

This script:
1. Loads tuning records from a MetaSchedule database
2. Reconstructs each schedule
3. Extracts per-store features
4. Aggregates features into fixed-length vectors
5. Runs each schedule to measure latency, power, and CPU frequency
6. Saves valid measurements into a CSV dataset

Timeout behavior:
- Python-level signal timeout is NOT used.
- --timeout-sec is passed only to the C++ evaluator.
- If the C++ evaluator raises EVALUATOR_TIMEOUT, that record is skipped.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms


# ============================================================
# Trace fingerprint
# ============================================================
def trace_fingerprint(trace):
    """
    Create a unique hash for a schedule trace.

    This helps distinguish different schedules for the same workload.
    """
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)
    s = obj if isinstance(obj, str) else "\n".join(str(x) for x in obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ============================================================
# Feature aggregation
# ============================================================
def aggregate_stats(feat: np.ndarray) -> np.ndarray:
    """
    Convert per-store features into a fixed-length vector.

    Input:
        feat shape = (num_stores, feature_dim)

    Output:
        [mean, std, min, max] over stores
        shape = (4 * feature_dim,)
    """
    feat = feat.astype("float64")

    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    minv = feat.min(axis=0)
    maxv = feat.max(axis=0)

    return np.concatenate([mean, std, minv, maxv])


# ============================================================
# Random input generator
# ============================================================
def make_random_args(args_info, device):
    """
    Generate random input tensors for running the compiled module.

    The values do not matter for latency/power measurement.
    Only valid shapes and dtypes are needed.
    """
    args = []

    for t in args_info:
        shape = [int(s) for s in t.shape]
        arr = np.random.rand(*shape).astype(t.dtype)
        args.append(tvm.nd.array(arr, device=device))

    return args


# ============================================================
# Task name extraction
# ============================================================
def extract_task_name(trace):
    """
    Extract a human-readable task/block name from the schedule trace.
    """
    for inst in trace.insts:
        if inst.kind.name == "GetBlock":
            block = inst.attrs[0]
            if block not in ["pad_temp", "root"]:
                return block

    return "unknown"


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--target", default=None)

    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-repeat-ms", type=int, default=3000)

    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Skip records before this index.",
    )

    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help=(
            "Timeout passed to the C++ evaluator. "
            "If the evaluator exceeds this limit or detects a slow repeat, "
            "the record is skipped. 0 disables evaluator timeout."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------
    if args.out is None:
        out_path = Path(f"dataset_{Path(args.work_dir).name}.csv")
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CPU device
    dev = tvm.cpu(0)

    # --------------------------------------------------------
    # Load MetaSchedule tuning records
    # --------------------------------------------------------
    db = ms.database.JSONDatabase(work_dir=args.work_dir)
    recs = db.get_all_tuning_records()

    extractor = ms.feature_extractor.PerStoreFeature()

    # Append if the file already exists.
    mode = "a" if out_path.exists() else "w"

    with open(out_path, mode, newline="") as f:
        writer = csv.writer(f)
        header_written = out_path.exists()

        # ----------------------------------------------------
        # Process records
        # ----------------------------------------------------
        for i, r in enumerate(recs):
            if i < args.start_idx:
                continue

            mod = r.workload.mod
            workload_hash = int(tvm.ir.structural_hash(mod))
            trace_hash = trace_fingerprint(r.trace)
            task_name = extract_task_name(r.trace)

            target = tvm.target.Target(args.target) if args.target else r.target

            try:
                # ------------------------------------------------
                # Reconstruct schedule
                # ------------------------------------------------
                sch = tir.Schedule(mod)
                r.trace.apply_to_schedule(sch, remove_postproc=True)

                # ------------------------------------------------
                # Extract features
                # ------------------------------------------------
                cand = ms.MeasureCandidate(sch=sch, args_info=r.args_info)

                ctx = ms.TuneContext(
                    mod=mod,
                    target=target,
                    task_name=f"rec_{i}",
                )

                (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])
                feat = feat_nd.numpy()

                n_stores = feat.shape[0]
                feat_agg = aggregate_stats(feat)

                # ------------------------------------------------
                # Write CSV header once
                # ------------------------------------------------
                if not header_written:
                    num_feat = feat.shape[1] * 4

                    header = [
                        "record_id",
                        "workload_hash",
                        "trace_hash",
                        "task_name",
                        "n_stores",
                        "latency_ms",
                        "power_w",
                        "cpu_freq_khz",
                    ] + [f"f{i}" for i in range(num_feat)]

                    writer.writerow(header)
                    f.flush()
                    header_written = True

                # ------------------------------------------------
                # Build schedule
                # ------------------------------------------------
                rt_mod = tvm.build(sch.mod, target)
                args_nd = make_random_args(r.args_info, dev)

                # ------------------------------------------------
                # Run custom C++ evaluator
                # ------------------------------------------------
                f_eval = tvm.get_global_func(
                    "runtime.profiling.energy_latency_freq_evaluator"
                )

                evaluator = f_eval(
                    rt_mod["main"],
                    dev,
                    args.number,
                    args.repeat,
                    args.min_repeat_ms,
                    50,                # limit_zero_time_iterations
                    1,                 # cooldown_interval_ms
                    0,                 # repeats_to_cooldown
                    0,                 # cache_flush_bytes
                    args.timeout_sec,  # C++ evaluator timeout
                    None,              # f_preproc
                )

                latency, power, cpu_freq = evaluator(*args_nd)
                latency_ms = latency * 1000

                # ------------------------------------------------
                # Save valid measurement
                # ------------------------------------------------
                row = [
                    i,
                    workload_hash,
                    trace_hash,
                    task_name,
                    n_stores,
                    latency_ms,
                    power,
                    cpu_freq,
                ] + feat_agg.tolist()

                writer.writerow(row)
                f.flush()

                print(
                    f"[{i}] {task_name} | "
                    f"latency: {latency_ms:.3f} ms | "
                    f"power: {power:.4f} W | "
                    f"frequency: {cpu_freq / 1000:.1f} MHz | "
                    f"stores={n_stores}",
                    flush=True,
                )

            except Exception as e:
                msg = str(e)

                if "EVALUATOR_TIMEOUT" in msg:
                    print(
                        f"[{i}] TIMEOUT | {task_name} | "
                        f"workload_hash={workload_hash} | {msg}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{i}] FAILED | {task_name} | "
                        f"workload_hash={workload_hash} | {msg}",
                        flush=True,
                    )

    print("Dataset written to:", out_path, flush=True)


if __name__ == "__main__":
    main()