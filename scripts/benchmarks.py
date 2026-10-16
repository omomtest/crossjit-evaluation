"""Load upstream NumPy kernels without importing npbench.infrastructure."""
import importlib
import json
from pathlib import Path


def load_npbench(root, name, preset, overrides=None):
    import sys
    sys.path.insert(0, str(Path(root).absolute()))
    info = json.loads((Path(root) / "bench_info" / (name + ".json")).read_text())["benchmark"]
    parameters = dict(info["parameters"][preset])
    overrides = overrides or {}
    if not overrides.keys() <= parameters.keys():
        raise ValueError("parameter override is not in NPBench metadata")
    parameters.update(overrides)
    prefix = "npbench.benchmarks." + info["relative_path"].replace("/", ".") + "." + info["module_name"]
    initialize = info.get("init")
    if initialize:
        value = getattr(importlib.import_module(prefix), initialize["func_name"])(
            *(parameters[k] for k in initialize["input_args"]))
        names = initialize["output_args"]
        values = [value] if len(names) == 1 else value
        parameters.update(zip(names, values, strict=True))
    function = getattr(importlib.import_module(prefix + "_numpy"), info["func_name"])
    return function, parameters, info


def micro_source(spec):
    k = spec.get("nodes", 2)
    observer = spec.get("observer", False)
    if spec.get("structure", "chain") == "diamond":
        body = ["left = a + b", "right = a - b", "result = left * right"]
    else:
        body = ["v0 = a + b"]
        for index in range(1, k):
            op, rhs = ("*", "c") if index % 3 == 1 else ("-", "b") if index % 3 == 2 else ("+", "a")
            body.append(f"v{index} = v{index-1} {op} {rhs}")
        body.append(f"result = v{k-1}")
    if observer:
        # Container retains all intermediates through the return, making escape observable.
        intermediates = ["left", "right"] if spec.get("structure") == "diamond" else [f"v{i}" for i in range(k-1)]
        body.insert(-1, "observed = [" + ", ".join(intermediates) + "]")
        return_value = "(result, *observed)"
    else:
        return_value = "result"
    return "def kernel(a, b, c):\n" + "".join("    " + line + "\n" for line in body) + "    return " + return_value + "\n"


def load_micro(spec):
    import numpy as np
    n = spec["elements"]
    layout = spec.get("layout", "contiguous")
    def array(offset):
        if layout == "stride2":
            return (np.arange(n * 2, dtype=np.float64) / (n * 2) + offset)[::2]
        if layout == "negative_stride":
            return (np.arange(n, dtype=np.float64) / n + offset)[::-1]
        if layout == "unsupported_object":
            return np.arange(n, dtype=object) + offset
        return np.arange(n, dtype=np.float64) / n + offset
    data = dict(a=array(1.), b=array(1.25), c=array(.75))
    scalar = spec.get("scalar")
    if scalar:
        data["c"] = np.float64(.75) if scalar == "numpy" else .75
    namespace = {}
    source = micro_source(spec)
    exec(compile(source, "<crossjit-evaluation:" + spec["name"] + ">", "exec"), namespace)
    return namespace["kernel"], data, dict(input_args=["a", "b", "c"], array_args=[], output_args=[],
                                         rtol=0., atol=0., source=source)


def load_overhead(spec, npbench=None):
    # The no-Provider fixture deliberately never imports NumPy.
    if spec["name"] == "no_provider":
        def kernel(a, b, c):
            total = 0
            for i in range(64):
                total += (a + i) * b - c
            return total
        return kernel, dict(a=3, b=5, c=2), dict(input_args=["a", "b", "c"], array_args=[], output_args=[])
    if spec["name"] == "generated_active":
        # Declare a real-workload positive control instead of assuming a tiny
        # expression has a profitable production artifact on every platform.
        return load_npbench(npbench, "adi", "S")
    import numpy as np
    if spec["name"] == "registered_unmatched":
        def kernel(a, b, c):
            return np.sort(a)
    else:
        namespace = {}
        exec(micro_source(dict(name=spec["name"], nodes=2)), namespace)
        kernel = namespace["kernel"]
    dtype = np.float64 if spec["name"] == "generated_active" else object
    if spec["name"] == "unmeasured":
        dtype = np.float64
    # 17 is outside the predeclared production calibration grid. Actual path
    # evidence still determines whether this is a valid unmeasured control.
    size = 17 if spec["name"] == "unmeasured" else 64
    data = {n: np.arange(size, dtype=dtype) + offset for n, offset in (("a", 1), ("b", 2), ("c", 3))}
    return kernel, data, dict(input_args=["a", "b", "c"], array_args=[], output_args=[], rtol=0., atol=0.)
