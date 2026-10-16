"""Check actual execution of the frozen acceptance paths, keeping raw results."""
from common import save


def check_coverage(manifest, summary):
    build = manifest["identity"].get("build") or {}
    policy = build.get("frozen_optimization_policy")
    cfg = manifest["configuration"]
    if not policy or cfg.get("preset") != "S" or cfg.get("jit_stress") != 1 or cfg.get("warmup", 0) < 200:
        return dict(status="not_applicable", reason="requires frozen v32, S, stress1 and W>=200")
    expected = policy["expected_S_stress1_activity"]["workloads"]
    failures = []
    checked = []
    for row in summary["rows"]:
        name = row["workload"]
        if row["spec"].get("overrides"):
            continue  # A different shape is a separate experiment.
        backends = [backend for backend in ("generated", "native", "mixed") if name in expected[backend]]
        checked.append(name)
        diagnostics = row.get("diagnostics", [row["timed_window"]] if "timed_window" in row else [])
        if backends and not diagnostics:
            failures.append(dict(workload=name, reason="missing execution diagnostics"))
        for trial, diagnostic in enumerate(diagnostics):
            for backend in backends:
                if diagnostic["backend_hits"].get(backend, 0) <= 0:
                    failures.append(dict(workload=name, diagnostic=trial, missing_backend=backend))
            provider = diagnostic.get("provider", {})
            if provider.get("exec_miss", 0) or provider.get("exec_error", 0):
                failures.append(dict(workload=name, diagnostic=trial, reason="provider execution miss/error"))
    return dict(status="failed" if failures else "passed", checked_workloads=checked,
                expected_activity=policy["expected_S_stress1_activity"], failures=failures,
                performance_threshold_checked=False)


def verify_coverage(output, manifest, summary):
    result = check_coverage(manifest, summary)
    save(output / "optimization-coverage.json", result)
    if result["status"] == "failed":
        raise RuntimeError("frozen v32 optimization paths did not execute; inspect optimization-coverage.json (raw results preserved)")
