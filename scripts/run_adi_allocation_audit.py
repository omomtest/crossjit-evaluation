#!/usr/bin/env python3
"""Observe ADI NumPy data allocations in the unchanged historical worker.

GDB observes existing snapshot boundaries and successful allocation returns.
This x86-64 diagnostic uses 300 warmup calls and one recorded 128-expression
call per independent process. Its instrumented times are never performance data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time

from common import ROOT, canonical_hash, clean_env, digest, read_json, save, write_csv
from redesign_reporting import region_window


def summarize_cell(worker, observer, events):
    if not observer["complete"] or not worker["correct"]:
        raise ValueError("incomplete allocation diagnostic")
    if not all(e.get("successful") and e["requested_bytes"] > 0 for e in events):
        raise ValueError("allocation failed or return was not observed")
    window = region_window(worker["preparation"], worker["final"], worker["repeat"])
    evaluations = worker["repeat"] * worker["spec"]["iterations"]
    full_arrays = [e for e in events if e["requested_bytes"] == 98 * 8]
    scalars = [e for e in events if e["requested_bytes"] == 8]
    if len(full_arrays) + len(scalars) != len(events):
        raise ValueError("unexpected allocation size; cannot classify as array/scalar conversion")
    provider_arrays = [e for e in full_arrays if any(
        name and name.startswith("numpy_crossjit_execute_validated") for name in e["frames"])]
    operations = Counter()
    for e in full_arrays:
        if e in provider_arrays:
            operations["provider_output"] += 1
        elif "array_multiply" in e["frames"]:
            operations["ordinary_multiply"] += 1
        elif "array_add" in e["frames"]:
            operations["ordinary_add"] += 1
        elif "array_true_divide" in e["frames"]:
            operations["ordinary_divide"] += 1
        else:
            raise ValueError("array allocation lacks an identified operation stack")
    mode = worker["mode"]
    expected_arrays = evaluations * (1 if mode == "on" else 3)
    expected_hits = evaluations if mode in ("single", "on") else 0
    if len(full_arrays) != expected_arrays or window["provider"].get("exec_hit", 0) != expected_hits:
        raise ValueError("ordinary prefix or changed execution path in allocation window")
    expected_operations = ({"provider_output": evaluations} if mode == "on" else
        {"provider_output": evaluations, "ordinary_add": evaluations, "ordinary_divide": evaluations}
        if mode == "single" else
        {"ordinary_multiply": evaluations, "ordinary_add": evaluations, "ordinary_divide": evaluations})
    if dict(operations) != expected_operations:
        raise ValueError("allocation operation stacks disagree with execution modes")
    if mode in ("on", "single"):
        if window["provider"]["output_allocations"] != len(provider_arrays):
            raise ValueError("external allocations disagree with Provider output counter")
        if any(window["provider"].get(k, 0) for k in (
            "guard_miss", "exec_miss", "exec_error", "internal_allocations",
            "internal_object_allocations", "packing_allocations")):
            raise ValueError("MISS, ERROR or unexpected private allocation")
        histogram = {int(n): h for n, h in window["node_histogram"].items()}
        if histogram != ({3: evaluations} if mode == "on" else {1: evaluations}):
            raise ValueError("wrong executed node inventory")
    return dict(mode={"on": "FULL", "single": "SINGLE", "off": "OFF"}[mode],
        trial=worker["trial"], recorded_calls=worker["repeat"], expression_evaluations=evaluations,
        successful_array_data_allocations=len(events), array_98_allocations=len(full_arrays),
        scalar_0d_allocations=len(scalars), operation_allocations=dict(operations),
        requested_data_bytes=sum(e["requested_bytes"] for e in events),
        arrays_per_expression=len(full_arrays)/evaluations,
        internal_arrays_per_expression=(len(full_arrays)-evaluations)/evaluations,
        output_arrays_per_expression=1,
        scalar_0d_arrays_per_expression=len(scalars)/evaluations,
        executed_hits=expected_hits, allocation_provider_corroborated=mode != "off",
        actual_signatures=window["actual_signatures"], provider_window=window["provider"],
        observation_complete=True, timing_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/rq2-4-microbench-20261008")
    parser.add_argument("--python", type=Path, default=ROOT/".runtime/bin/python")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if platform.machine() != "x86_64" or not shutil.which("gdb"):
        parser.error("requires x86-64 System V ABI and GDB with Python support")
    if args.trials < 1 or args.repeat < 1:
        parser.error("trials and repeat must be positive")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("use a fresh output directory; raw records are immutable")
    args.output.mkdir(parents=True)
    original = args.results.resolve()/"microbench"
    build = read_json(args.results/"rq2/manifest.json")["identity"]["build"]
    for path, expected in build["binary_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("historical diagnostic binary drift: " + path)
    if digest(args.python.resolve()) not in build["binary_sha256"].values():
        raise ValueError("Python is not the historical diagnostic binary")
    scripts = [ROOT/"scripts"/s for s in (
        "run_adi_allocation_audit.py", "adi_allocation_gdb.py", "redesign_worker.py",
        "mechanism_cases.py", "redesign_reporting.py")]
    manifest = dict(schema_version=1, experiment="untimed_adi_allocation_audit",
        original_results=str(original), original_manifest_sha256=digest(original/"manifest.json"),
        binary_sha256=build["binary_sha256"], script_sha256={str(p): digest(p) for p in scripts},
        trials=args.trials, warmup=300, repeat=args.repeat, iterations=128, blas_threads=6,
        jit_stress=1, default_allocator_preserved=True, python_tracing_enabled=False,
        observer="GDB PyDataMem_UserNEW/NEW_ZEROED entry and successful return",
        window="after warmup invocation snapshot return to final invocation snapshot entry",
        timing_eligible=False, protocol_change="one recorded call by default; allocation diagnostic only")
    save(args.output/"manifest.json", manifest)
    rows = []
    started = time.perf_counter()
    for trial in range(args.trials):
        modes = ["off", "single", "on"]
        random.Random(20261014+trial).shuffle(modes)
        for mode in modes:
            label = f"{mode}.t{trial}"
            task_source = next((original/"tasks").glob(f"m1_adi_arithmetic-*.{mode}.t0.diag.json"))
            task = read_json(task_source)
            task.update(trial=trial, repeat=args.repeat)
            task["task_hash"] = canonical_hash({k: v for k, v in task.items() if k != "task_hash"})
            task_path = args.output/"tasks"/(label+".json")
            save(task_path, task)
            worker_path = args.output/"raw"/(label+".worker.json")
            worker_path.parent.mkdir(exist_ok=True)
            events_path = args.output/"raw"/(label+".events.jsonl")
            observer_path = args.output/"raw"/(label+".observer.json")
            env = clean_env(6, mode != "off", True, {"PYTHON_JIT_STRESS": "1",
                **({"PYTHON_JIT_CROSSJIT_MAX_REGION_NODES": "1"} if mode == "single" else {})})
            env.update(CJ_ALLOCATION_WARMUP="300", CJ_ALLOCATION_REPEAT=str(args.repeat),
                CJ_ALLOCATION_EVENTS=str(events_path), CJ_ALLOCATION_OBSERVER=str(observer_path))
            command = ["gdb", "-q", "-nx", "--batch", "-iex", "set auto-load python-scripts off",
                "-ex", "source " + str(ROOT/"scripts/adi_allocation_gdb.py"), "-ex", "run", "--args",
                str(args.python.absolute()), "-B", str(ROOT/"scripts/redesign_worker.py"),
                "--task", str(task_path), "--output", str(worker_path)]
            save(args.output/"tasks"/(label+".command.json"), dict(command=command,
                task_source=str(task_source), task_source_sha256=digest(task_source),
                controlled_environment={k: v for k, v in env.items() if k.startswith(("CJ_", "PYTHON_JIT", "NPY_CROSSJIT"))}))
            log = args.output/"logs"/(label+".log")
            log.parent.mkdir(exist_ok=True)
            with log.open("w") as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=args.timeout, check=True)
            worker = read_json(worker_path)
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            row = summarize_cell(worker, read_json(observer_path), events)
            row["raw_sources"] = {"worker": str(worker_path), "events": str(events_path),
                                  "observer": str(observer_path), "task": str(task_path), "log": str(log)}
            rows.append(row)
            print(f"{label}: 98-element arrays/expression={row['arrays_per_expression']:g}, "
                  f"0d conversion arrays/expression={row['scalar_0d_arrays_per_expression']:g}; "
                  f"HIT={row['executed_hits']}", flush=True)
    summary = dict(schema_version=1, complete=len(rows) == args.trials*3, manifest=manifest,
        rows=rows, wall_seconds=time.perf_counter()-started,
        limitations=["Array-data allocation requests and non-null returns, not all Python object allocations.",
            "Two internal 98-element arrays in OFF/SINGLE exclude 0d scalar-conversion arrays.",
            "No time or byte-saving performance attribution from GDB-instrumented runs."])
    save(args.output/"summary.json", summary)
    write_csv(args.output/"summary.csv", [{k: v for k, v in row.items() if k not in (
        "actual_signatures", "provider_window", "raw_sources")} for row in rows])
    print(f"Completed {len(rows)} independent diagnostics in {summary['wall_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
