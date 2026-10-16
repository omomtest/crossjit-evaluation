"""Isolated worker for redesign experiments, with mechanism identity/alias checks."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import sys
import time

from common import apply_affinity, digest, save


def probe():
    import sysconfig
    import numpy as np
    from numpy._core import _multiarray_umath as umath
    try:
        import _testinternalcapi as host
    except ImportError:
        host = None
    files = {str(Path(sys.executable).resolve()), umath.__file__}
    files.update(str(p) for p in Path(np.__file__).parent.rglob("*.so"))
    if host is not None and getattr(host, "__file__", None):
        files.add(host.__file__)
    lib = Path(sysconfig.get_config_var("LIBDIR") or "") / (sysconfig.get_config_var("LDLIBRARY") or "")
    if lib.is_file():
        files.add(str(lib.resolve()))
    return dict(python=sys.executable, python_version=sys.version, python_git=getattr(sys, "_git", None),
                python_config=sysconfig.get_config_var("CONFIG_ARGS"), cflags=sysconfig.get_config_var("CFLAGS"),
                machine=os.uname().machine, jit_enabled=bool(hasattr(sys, "_jit") and sys._jit.is_enabled()),
                jit_stress=int(os.environ.get("PYTHON_JIT_STRESS", "0")),
                numpy_path=np.__file__, numpy_version=np.__version__, numpy_commit=np.version.git_revision,
                blas=np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas"),
                provider_api=hasattr(umath, "_crossjit_stats"),
                provider_abi_version=umath._crossjit_stats().get("provider_abi_version") if hasattr(umath, "_crossjit_stats") else None,
                host_api=bool(host and hasattr(host, "crossjit_stats")),
                binary_files=sorted(files))


def runtimes(no_provider=False):
    try:
        import _testinternalcapi as host
    except ImportError:
        host = None
    provider = None
    if not no_provider:
        from numpy._core import _multiarray_umath as provider
    return host, provider


def snapshot(host, provider):
    return dict(host=host.crossjit_stats() if host and hasattr(host, "crossjit_stats") else {},
                provider=provider._crossjit_stats() if provider and hasattr(provider, "_crossjit_stats") else {})


def output_record(value):
    import numpy as np
    a = np.asarray(value)
    if a.dtype.hasobject:
        return dict(shape=list(a.shape), dtype=a.dtype.str, objects=repr(a.tolist()))
    return dict(shape=list(a.shape), dtype=a.dtype.str, bytes=a.tobytes().hex())


def validate_record(value, ref, info):
    import numpy as np
    a = np.asarray(value)
    if list(a.shape) != ref["shape"] or a.dtype.str != ref["dtype"]:
        raise AssertionError("output dtype/shape changed")
    if a.dtype.hasobject:
        if repr(a.tolist()) != ref["objects"]:
            raise AssertionError("object output changed")
        return
    raw, expected = a.tobytes(), bytes.fromhex(ref["bytes"])
    if raw == expected:
        return
    fmt = {("f", 2): "=e", ("f", 4): "=f", ("f", 8): "=d", ("c", 8): "=ff", ("c", 16): "=dd"}.get((a.dtype.kind, a.dtype.itemsize))
    if fmt is None:
        raise AssertionError("exact output changed")
    rtol, atol = info.get("rtol", 1e-5), info.get("atol", 1e-8)
    for left, right in zip(struct.iter_unpack(fmt, raw), struct.iter_unpack(fmt, expected), strict=True):
        for x, y in zip(left, right, strict=True):
            if math.isnan(x) and math.isnan(y):
                continue
            if x == y:
                if x == 0. and math.copysign(1., x) != math.copysign(1., y):
                    raise AssertionError("signed zero changed")
                continue
            if not math.isfinite(x) or not math.isfinite(y) or abs(x-y) > atol + rtol*abs(y):
                raise AssertionError("floating output changed")


def persist_inputs(directory, data):
    import numpy as np
    arrays, descriptors, scalars = {}, {}, {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                # Only the controlled object fixture uses this path; retain its type explicitly.
                descriptors[key] = dict(objects=value.tolist(), dtype="object", shape=list(value.shape))
            else:
                arrays[key] = value
                descriptors[key] = dict(shape=list(value.shape), dtype=value.dtype.str, strides=list(value.strides))
        elif isinstance(value, np.generic):
            scalars[key] = dict(numpy_dtype=value.dtype.str, value=value.item())
        else:
            scalars[key] = dict(value=value)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "arrays.npz", **arrays)
    save(directory / "inputs.json", dict(arrays=descriptors, scalars=scalars))


def restore_inputs(directory):
    import numpy as np
    from common import read_json
    info = read_json(Path(directory) / "inputs.json")
    result = {}
    with np.load(Path(directory) / "arrays.npz", allow_pickle=False) as arrays:
        for key, desc in info["arrays"].items():
            if "objects" in desc:
                result[key] = np.asarray(desc["objects"], dtype=object)
                continue
            shape, strides = desc["shape"], desc["strides"]
            dtype = np.dtype(desc["dtype"])
            low = sum(min(0, (n-1)*s) for n, s in zip(shape, strides))
            high = sum(max(0, (n-1)*s) for n, s in zip(shape, strides))
            storage = np.empty(max(dtype.itemsize, high-low+dtype.itemsize), dtype=np.uint8)
            value = np.ndarray(tuple(shape), dtype=dtype, buffer=storage, offset=-low, strides=tuple(strides))
            value[...] = arrays[key]
            result[key] = value
        for key, desc in info["scalars"].items():
            result[key] = np.dtype(desc["numpy_dtype"]).type(desc["value"]) if "numpy_dtype" in desc else desc["value"]
    return result


def run(task):
    from benchmarks import load_npbench, load_micro, load_overhead
    from mechanism_cases import load_case
    from common import read_json
    spec = task["spec"]
    no_provider = spec["kind"] == "overhead" and spec["name"] == "no_provider"
    seed = task.get("seed",20261002)
    random.seed(seed)
    if not no_provider:
        import numpy as np
        np.random.seed(seed)
    if spec["kind"] == "npbench":
        function, data, info = load_npbench(task["npbench"], spec["name"], spec["preset"], spec.get("overrides"))
    elif spec["kind"] == "mechanism":
        function, data, info = load_case(spec)
    elif spec["kind"] == "micro":
        function, data, info = load_micro(spec)
    else:
        function, data, info = load_overhead(spec, task["npbench"])
    input_dir = Path(task["inputs"])
    if no_provider:
        if input_dir.exists():
            data = read_json(input_dir / "inputs.json")
        else:
            save(input_dir / "inputs.json", data)
    elif (input_dir / "inputs.json").is_file():
        data = restore_inputs(input_dir)
    else:
        persist_inputs(input_dir, data)
    host, provider = runtimes(no_provider)
    if host and hasattr(host, "crossjit_reset_stats"):
        host.crossjit_reset_stats()
    if provider and hasattr(provider, "_crossjit_reset_stats"):
        provider._crossjit_reset_stats()
    ref_path = Path(task["reference"])
    expected = read_json(ref_path) if ref_path.exists() else None
    if not task["is_reference"] and expected is None:
        raise RuntimeError("independent baseline reference missing")
    warmup, repeat = task["warmup"], task["repeat"]
    checkpoints = set(task.get("checkpoints", []))
    snapshots, warm_times, times, sequence = {}, [], [], []
    initial = snapshot(host, provider)
    preparation = initial if warmup == 0 else None
    validated = 0
    import_started = time.perf_counter_ns()
    outputs = None
    prior_output = None
    semantic_checks = 0
    original_input_bytes = {k: v.tobytes() for k, v in data.items() if hasattr(v, "tobytes")}
    for i in range(warmup + repeat):
        if no_provider:
            values = data
        else:
            import numpy as np
            values = dict(data)
            for name in info["array_args"]:
                values[name] = np.copy(data[name])
        call_args = [values[n] for n in info["input_args"]]
        started = time.perf_counter_ns()
        result = function(*call_args)
        elapsed = time.perf_counter_ns() - started
        outputs = list(result) if isinstance(result, (tuple, list)) else ([] if result is None else [result])
        outputs.extend(values[n] for n in info.get("output_args", []))
        if spec["kind"] == "mechanism":
            if type(result[0] if isinstance(result, tuple) else result) is not np.ndarray:
                raise AssertionError("mechanism output type changed")
            final_output = result[0] if isinstance(result, tuple) else result
            if prior_output is not None and (final_output is prior_output or np.shares_memory(final_output, prior_output)):
                raise AssertionError("final output is not fresh")
            for value in call_args:
                if isinstance(value, np.ndarray) and np.shares_memory(final_output, value):
                    raise AssertionError("output aliases a borrowed input")
            prior_output = final_output
            for key, original in original_input_bytes.items():
                if isinstance(values[key], np.ndarray) and values[key].tobytes() != original:
                    raise AssertionError("mechanism modified input")
            semantic_checks += 1
        if expected:
            if len(outputs) != len(expected["outputs"]):
                raise AssertionError("output arity changed")
            for value, ref in zip(outputs, expected["outputs"], strict=True):
                if no_provider:
                    if value != ref:
                        raise AssertionError("scalar result changed")
                else:
                    validate_record(value, ref, info)
            validated += 1
        (warm_times if i < warmup else times).append(elapsed)
        if i + 1 == warmup:
            preparation = snapshot(host, provider)
        if i + 1 in checkpoints:
            snapshots[str(i + 1)] = snapshot(host, provider)
        if task["diagnostics"]:
            state = snapshot(host, provider)
            sequence.append(dict(invocation=i+1, generated_exec_hit=state["provider"].get("generated_exec_hit", 0)))
    final = snapshot(host, provider)
    if task["is_reference"]:
        save(ref_path, dict(outputs=outputs if no_provider else [output_record(v) for v in outputs],
                            checking="external fixed-baseline output; numeric tolerance from metadata; signed zero checked"))
    input_hashes = {str(p): digest(p) for p in input_dir.iterdir() if p.is_file()}
    return dict(schema_version=1, task_hash=task["task_hash"], spec=spec, mode=task["mode"], trial=task["trial"],
                correct=True, semantic_checks=semantic_checks, validated_invocations=validated, warmup=warmup, repeat=repeat,
                jit_stress=int(os.environ.get("PYTHON_JIT_STRESS", "0")),
                diagnostics=task["diagnostics"], runtime_ns=times, warmup_runtime_ns=warm_times,
                median_ns=statistics.median(times), first_call_ns=(warm_times + times)[0],
                initial=initial, preparation=preparation, final=final, checkpoints=snapshots,
                invocation_hits=sequence, reference_sha256=digest(ref_path), input_hashes=input_hashes,
                source=info.get("source"),
                source_bytecode=info.get("bytecode"),
                inputs_metadata={n: dict(shape=list(v.shape), dtype=str(v.dtype), strides=list(v.strides), bytes=v.nbytes)
                                 for n, v in data.items() if hasattr(v, "shape")},
                numpy_imported="numpy" in sys.modules, python=sys.executable,
                input_seed=seed, input_seed_protocol="global RNG seed; upstream initializer-local seeds unchanged; saved inputs shared",
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                total_loop_wall_ns=time.perf_counter_ns()-import_started)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--task", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    apply_affinity()
    if args.probe:
        result = probe()
    else:
        from common import read_json
        result = run(read_json(args.task))
    save(args.output, result)


if __name__ == "__main__":
    main()
