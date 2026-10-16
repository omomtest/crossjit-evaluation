"""Production guard/replay probes and separately labelled injected ERROR."""
import argparse
import os
from pathlib import Path
import sys
import warnings

import numpy as np
from common import save
from mechanism_cases import case_specs, load_case
from reporting import numeric_delta
from worker import output_record
from numpy._core import _multiarray_umath as provider
import _testinternalcapi as host


def state():
    return dict(host=host.crossjit_stats(), provider=provider._crossjit_stats())


def fresh(name="m1_adi_arithmetic"):
    spec = next(s for s in case_specs() if s["name"] == name)
    function, data, info = load_case(spec)
    # Independent diagnostic activation budget; never used as timing samples.
    for _ in range(20):
        function(*data.values())
    return function, data, info


def call_capture(function, data):
    result, error, locals_record = None, None, None
    before_inputs = {k: v.tobytes() for k, v in data.items() if isinstance(v, np.ndarray)}
    refs = sys.getrefcount(data["left"])
    with warnings.catch_warnings(record=True) as events:
        warnings.simplefilter("always")
        try:
            result = function(*data.values())
        except (FloatingPointError, RuntimeError) as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            tb = exc.__traceback__
            while tb:
                if tb.tb_frame.f_code is function.__code__:
                    local = tb.tb_frame.f_locals.get("intermediate")
                    locals_record = output_record(local) if local is not None else None
                    error["source_line"] = tb.tb_lineno
                tb = tb.tb_next
    for k, original in before_inputs.items():
        if data[k].tobytes() != original:
            raise AssertionError("probe altered an original input")
    if sys.getrefcount(data["left"]) != refs:
        raise AssertionError("probe leaked a borrowed input reference")
    return dict(output=output_record(result) if result is not None else None, error=error,
                exception_visible_intermediate=locals_record,
                events=[dict(type=w.category.__name__, message=str(w.message)) for w in events],
                inputs_unchanged=True, borrowed_references_preserved=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    np.seterr(all="warn")
    provider._crossjit_reset_stats(); host.crossjit_reset_stats()
    probes = []
    for name in ("shape_guard", "execution_miss_replay", "injected_error", "exception_visible_local"):
        # Build a different function for each change so a preceding side exit
        # cannot invalidate the next probe's installed executor.
        function, data, info = fresh("m2_exception_visible_local" if name == "exception_visible_local" else "m1_adi_arithmetic")
        established = state()
        data["count"] = 2
        if name == "shape_guard":
            data["left"] = (np.arange(9900., dtype=np.float64) + 2)[::100]
        elif name in ("execution_miss_replay", "exception_visible_local"):
            data["left"][0] = -1.5
        before = state()
        if name == "injected_error" and os.environ.get("PYTHON_JIT_CROSSJIT") == "1":
            os.environ["NPY_CROSSJIT_TEST_EXEC_ERROR"] = "1"
        if name == "exception_visible_local":
            np.seterr(all="raise")
        try:
            result = call_capture(function, data)
        finally:
            os.environ.pop("NPY_CROSSJIT_TEST_EXEC_ERROR", None)
            np.seterr(all="warn")
        after = state()
        probes.append(dict(name=name, result=result, established=established, before=before, after=after,
            delta={side: numeric_delta(before[side], after[side]) for side in ("host", "provider")},
            scope="test-only ERROR injection" if name == "injected_error" else "production guarded execution",
            source=info["source"]))
    save(args.output, dict(complete=True, probes=probes, diagnostic_warmup_calls=20,
                          inner_iterations=128, mode=os.environ.get("PYTHON_JIT_CROSSJIT")))


if __name__ == "__main__":
    main()
