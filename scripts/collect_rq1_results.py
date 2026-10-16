#!/usr/bin/env python3
"""Verify and gather machine bundles; export tables and optionally plot each machine separately."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from common import canonical_hash, digest, read_json, save, write_csv
from rq1_bundle import unpack
from run_rq1_machine import label


def import_bundle(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary) / "run"
        index = unpack(archive, stage)
        machine, run_id = label(index["machine"]), label(index["run_id"])
        destination = output / "machines" / machine / run_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if read_json(destination / "bundle.json") != index:
                raise ValueError("different evidence already imported for " + machine + "/" + run_id)
            for name, row in index["files"].items():
                if digest(destination / name) != row["sha256"]:
                    raise ValueError("previously imported evidence changed: " + name)
            print("Already verified: " + machine + "/" + run_id)
        else:
            stage.rename(destination)
            print("Imported: " + machine + "/" + run_id)
    return destination


def tables(output):
    runs, workloads = [], []
    for path in sorted((output / "machines").glob("*/*/bundle.json")):
        root = path.parent
        index = read_json(path)
        record = dict(machine=index["machine"], run_id=index["run_id"], directory=str(root.resolve()),
                      bundle_inventory_hash=canonical_hash(index), status="incomplete")
        host = read_json(root / "metadata/host.json")
        record.update(host_os=host["host_os"], hardware_label=host.get("hardware_label"),
                      notes=host.get("notes"))
        summary_path = root / "experiment/rq1/summary.json"
        if not summary_path.is_file():
            runs.append(record)
            continue
        manifest = read_json(root / "experiment/rq1/manifest.json")
        summary = read_json(summary_path)
        cfg = manifest["configuration"]
        build = manifest["identity"].get("build") or {}
        group = dict(source_lock=build.get("source_lock"), toolchain_lock=build.get("toolchain_lock"),
                     optimization_preset=build.get("optimization_preset", "legacy-generated"),
                     frozen_policy_bundle=(build.get("frozen_optimization_policy") or {}).get("bundle_sha256"),
                     policy_hashes=build.get("policy_hashes", {}),
                     scripts_hash=manifest["identity"]["scripts_hash"], suite_hash=manifest["identity"]["suite_hash"],
                     protocol={key: cfg.get(key) for key in ("warmup", "jit_stress", "repeat", "trials",
                               "blas_threads", "preset", "seed", "timeout")}, policy_mode=manifest["policy_mode"])
        pipeline = read_json(root / "experiment/pipeline.json")
        record.update(comparison_group=canonical_hash(group), measurement_status=summary["measurement_status"],
                      optimization_preset=group["optimization_preset"],
                      policy_mode=manifest["policy_mode"], architecture=manifest["system"]["machine"],
                      **group["protocol"], complete=summary["complete"], full53=summary["full_53_npbench"],
                      GM_all=summary["GM_all"], GM_requested=summary["GM_requested"],
                      suite_GM_ci95=summary["suite_GM_ci95"], active_total=summary["active_total"],
                      platform_execution_confirmed=summary["platform_execution_confirmed"],
                      pipeline_complete=pipeline["complete"])
        eligible = (pipeline["complete"] and summary["complete"] and summary["full_53_npbench"]
                    and summary["equivalent_performance_comparison_eligible"]
                    and cfg["warmup"] == 300 and cfg.get("jit_stress") == 1 and cfg["trials"] >= 10)
        record.update(status="full53_fixed_protocol" if eligible else "incomplete_subset_or_unconfirmed_execution",
                      fixed_protocol_comparison_eligible=eligible)
        runs.append(record)
        for row in summary["rows"]:
            workloads.append(dict(machine=index["machine"], run_id=index["run_id"],
                                  comparison_group=record["comparison_group"],
                                  fixed_protocol_comparison_eligible=eligible,
                                  **{key: value for key, value in row.items() if key != "diagnostics"}))
    save(output / "index.json", dict(schema_version=1, runs=runs,
         aggregation="Separate machine results; no pooling of trials or cross-machine GM"))
    write_csv(output / "machines.csv", runs)
    write_csv(output / "workloads.csv", workloads)
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archives", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-python", default=sys.executable, help="Independent Python with Matplotlib installed")
    args = p.parse_args()
    output = args.output.expanduser().resolve()
    for archive in args.archives:
        import_bundle(archive, output)
    runs = tables(output)
    if args.plot:
        for record in runs:
            root = Path(record["directory"])
            summary = root / "experiment/rq1/summary.json"
            if summary.is_file():
                # Run the plotter archived with this experiment. Do not re-probe
                # measurement binaries or alter frozen imported evidence.
                figure = output / "figures" / f"{record['machine']}--{record['run_id']}.svg"
                subprocess.run([args.plot_python, str(root / "evaluation/scripts/plot_results.py"),
                                str(summary), "--output", str(figure)], check=True)
    print("Machine table: " + str(output / "machines.csv"))
    print("Workload table: " + str(output / "workloads.csv"))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
