#!/usr/bin/env python3
"""RQ2/RQ3 ADI fragment: actual HITs, observations, recovery and scope ablation."""
from pathlib import Path
import subprocess

from common import ROOT, clean_env, cli, configuration, freeze, parser, read_json, runtime_environment, save, verify_identity
from mechanism_cases import case_specs
from redesign_common import SINGLE, same_binary
from redesign_execution import run_matrix
from redesign_reporting import composition_report, reach_report


def run_probes(output, manifest):
    cfg = manifest["configuration"]
    for mode, enabled in (("off", False), ("on", True)):
        env = clean_env(cfg["blas_threads"], enabled, True,
                        {**runtime_environment(cfg), "NPY_CROSSJIT_EVALUATION_AUDIT": "1"}, cfg["cpu_affinity"])
        command = [cfg["candidate_python"], "-B", str(ROOT / "scripts/mechanism_probe.py"),
                   "--output", str(output / f"probes-{mode}.json")]
        with (output / f"probes-{mode}.log").open("w") as log:
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=cfg["timeout"], check=True)
    env = clean_env(cfg["blas_threads"], True, True, runtime_environment(cfg), cfg["cpu_affinity"])
    with (output / "host-fixtures.log").open("w") as log:
        subprocess.run([cfg["candidate_python"], "-B", str(ROOT / "scripts/host_fixture_worker.py"),
                        "--output", str(output / "host-fixtures.json")], cwd=output, env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=cfg["timeout"], check=True)
    off = {p["name"]: p for p in read_json(output / "probes-off.json")["probes"]}
    on = {p["name"]: p for p in read_json(output / "probes-on.json")["probes"]}
    checks = {}
    for name in ("shape_guard", "execution_miss_replay", "exception_visible_local"):
        checks[name + "_semantics"] = on[name]["result"] == off[name]["result"]
    checks["guard_miss_reached"] = on["shape_guard"]["delta"]["provider"].get("guard_miss", 0) > 0
    checks["execution_miss_reached"] = on["execution_miss_replay"]["delta"]["provider"].get("exec_miss", 0) > 0
    injected = on["injected_error"]
    checks["error_once"] = (injected["delta"]["provider"].get("exec_error", 0) == 1 and
                            injected["result"]["error"] is not None and
                            injected["result"]["error"]["type"] == "RuntimeError" and
                            not injected["result"]["events"])
    checks["exception_local_retained"] = on["exception_visible_local"]["result"]["exception_visible_intermediate"] is not None
    checks["host_fixtures"] = read_json(output / "host-fixtures.json")["complete"]
    verify_identity(manifest["identity"])
    return checks


def case_checks(reach, trials):
    checks = {}
    for mode in ("single", "on"):
        main = [r for r in reach["rows"] if r["workload"] == "m1_adi_arithmetic" and r["mode"] == mode]
        checks[mode + "_main_reached"] = len(main) == trials and all(
            r["multi_node_hits"] > 0 if mode == "on" else r["single_node_hits"] > 0 and not r["multi_node_hits"]
            for r in main)
    observer = [r for r in reach["rows"] if r["workload"] == "m2_observable_intermediate" and r["mode"] == "on"]
    checks["observer_boundary"] = len(observer) == trials and all(r["max_executed_nodes"] < 3 for r in observer)
    checks["correct_and_complete"] = reach["complete"]
    return checks


def main():
    p = parser(__doc__)
    args = p.parse_args()
    cfg = same_binary(configuration(args), require_single=True)
    output, manifest = freeze(args.output, cfg, "microbench-case-study", case_specs(), [SINGLE],
        extra=dict(fixed_input=True, legacy_130_grid_is_not_case_study=True,
                   evidence_layers=["numerical production region", "synthetic Host fixture", "injected ERROR"]))
    run_matrix(output, manifest, modes=("off", "single", "on"))
    reach = reach_report(output, ("off", "single", "on"))
    composition = composition_report(output, reach)
    checks = case_checks(reach, cfg["trials"])
    checks.update(run_probes(output, manifest))
    report = dict(complete=composition["complete"] and all(checks.values()), checks=checks,
        gaps=["Trace/RegionView/guarded summary/certificate identities are not fully exported by the current Host ABI.",
              "General concurrent/refcount/finalizer/resource-exhaustion semantics are outside this evidence."],
        replay_scope="Real optimized execution MISS, warning-event sequence and exception-visible local checked against OFF.",
        interpretation="SINGLE/FULL may use different bindings and cover different admitted nodes; joint-region effect only.")
    save(output / "case-study.json", report)
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    cli(main)
