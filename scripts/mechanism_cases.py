"""Fixed-input, loop-shaped production cases extracted from ADI arithmetic.

The 98-element, stride-100 domain and scalar value class have frozen admission
evidence. No force/min-work/calibration switch is used by these cases.
"""
import dis
import io


def case_specs():
    return [dict(kind="mechanism", name=name, elements=98, stride=100,
                 iterations=128, source_workload="adi", source_nodes=3)
            for name in ("m1_adi_arithmetic", "m2_observable_intermediate",
                         "m2_exception_visible_local")]


def load_case(spec):
    import numpy as np
    if spec["name"] == "m1_adi_arithmetic":
        body = "        result = numerator / (scale * left + offset)\n"
        result = "result"
    elif spec["name"] == "m2_observable_intermediate":
        body = ("        intermediate = scale * left\n"
                "        observed.append(intermediate)\n"
                "        result = numerator / (intermediate + offset)\n")
        result = "(result, observed[-1])"
    elif spec["name"] == "m2_exception_visible_local":
        body = ("        intermediate = scale * left\n"
                "        try:\n"
                "            result = numerator / (intermediate + offset)\n"
                "        except FloatingPointError:\n"
                "            raise\n")
        result = "result"
    else:
        raise ValueError("unknown mechanism case")
    source = ("def kernel(scale, left, offset, numerator, count):\n"
              "    observed = []\n"
              "    for _ in range(count):\n" + body + "    return " + result + "\n")
    namespace = {}
    exec(compile(source, "<redesign:" + spec["name"] + ">", "exec"), namespace)
    n, stride = spec["elements"], spec["stride"]
    data = dict(scale=2., left=(np.arange(n * stride, dtype=np.float64) + 2)[::stride],
                offset=3., numerator=4., count=spec["iterations"])
    bytecode = io.StringIO()
    dis.dis(namespace["kernel"], file=bytecode)
    return namespace["kernel"], data, dict(
        input_args=list(data), array_args=[], output_args=[], rtol=0., atol=0.,
        source=source, bytecode=bytecode.getvalue())
