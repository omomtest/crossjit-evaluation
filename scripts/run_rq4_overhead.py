#!/usr/bin/env python3
"""RQ4: controlled non-execution cost and full inactive set."""
from pathlib import Path
from redesign_common import same_binary

from common import ROOT, canonical_hash, cli, configuration, freeze, parser, read_json, save
from execution import run_matrix
from reporting import coverage_report, summarize_matrix
from statistics_tools import gm, percentile


def cost_report(output, manifest, summaries):
    rows = []
    for path in sorted((output / "raw").glob("*.timing.json")):
        cell = read_json(path)
        if cell["trial"] == "reference" or not cell.get("correct"):
            continue
        samples = cell["runtime_ns"]
        rows.append(dict(workload=cell["spec"]["name"], mode=cell["mode"], trial=cell["trial"],
            group="inactive_npbench" if cell["spec"]["kind"] == "npbench" else "controlled_path",
            first_call_ns=cell["first_call_ns"], process_wall_ns=cell["process_wall_ns"],
            repeated_call_distribution_ns=dict(min=min(samples), median=percentile(samples,.5),
                                                p10=percentile(samples,.1), p90=percentile(samples,.9), max=max(samples)),
            lifetime_host=cell["final"]["host"], lifetime_provider=cell["final"]["provider"],
            numpy_imported=cell["numpy_imported"]))
    groups = []
    for summary in summaries:
        inactive = [r for r in summary["rows"] if r["spec"]["kind"] == "npbench"]
        groups.append(dict(comparison=summary["comparison"], inactive_workloads=len(inactive),
            GM_inactive_npbench=gm(r["speedup"] for r in inactive) if inactive and all(r["complete"] for r in inactive) else None,
            controlled_paths="Report each declared control separately; no mixed control/workload GM."))
    save(output / "overhead-costs.json", dict(rows=rows, groups=groups,
         binary_sizes_bytes={path:Path(path).stat().st_size for path in manifest["identity"]["file_hashes"]
                             if path.endswith((".so", ".a"))},
         distribution_scope="Within-process repeated calls are descriptive; CI uses independent process pairs only.",
         plan_memory_bytes=None, plan_memory_status="No dedicated plan-memory accounting ABI in the selected source."))


def main():
    p = parser(__doc__)
    p.add_argument("--coverage", type=Path, help="RQ2 coverage.json identifying the full inactive set")
    args = p.parse_args()
    cfg = same_binary(configuration(args))
    specs = [dict(kind="overhead", name=name) for name in
             ("no_provider", "registered_unmatched", "unsupported_rejected", "unmeasured", "generated_active")]
    missing = []
    if args.coverage:
        data = read_json(args.coverage)
        origin = read_json(args.coverage.parent / "manifest.json")
        suite_names = {row["name"] for row in read_json(ROOT / "config/suite.json")["benchmarks"]}
        if origin["experiment"] != "rq2" or len(origin["specs"]) != 53 or {s["name"] for s in origin["specs"]} != suite_names:
            raise ValueError("inactive inventory requires the full 53-workload RQ2 suite")
        if data["complete_cells"] != data["expected_cells"]:
            raise ValueError("inactive inventory requires a complete RQ2 diagnostic matrix")
        by_spec = {}
        for row in data["rows"]:
            key = canonical_hash(row["spec"])
            by_spec.setdefault(key, dict(spec=row["spec"], active=False))["active"] |= row["active"]
        specs += [v["spec"] for v in by_spec.values() if not v["active"]]
    else:
        missing.append("NPBench inactive set requires --coverage from matching RQ2")
    extra = dict(missing=missing,
                 generated_positive_control="NPBench ADI S, fixed before measurement; requires actual HIT",
                 unsupported_control="object dtype; ordinary production rejection, no force flags")
    output, manifest = freeze(args.output, cfg, "rq4-overhead", specs, extra=extra)
    if args.coverage:
        origin = read_json(args.coverage.parent / "manifest.json")
        if origin["identity"] != manifest["identity"] or origin["configuration"] != cfg:
            raise ValueError("inactive set came from a different identity/protocol")
    run_matrix(output, manifest, modes=("baseline", "off", "on"))
    summaries = [summarize_matrix(output), summarize_matrix(output, "off", "on")]
    coverage = coverage_report(output)
    checks = []
    from execution import spec_id
    for row in coverage["rows"]:
        name = row["workload"]
        if row["mode"] != "on":
            continue
        active = row["active"]
        expected = name == "generated_active"
        diag = read_json(output / "raw" / f"{spec_id(row['spec'])}.on.t{row['trial']}.diag.json")
        imported_ok = name != "no_provider" or not diag["numpy_imported"]
        checks.append(dict(workload=name, trial=row["trial"], expected_active=expected, actual_active=active,
                           valid=imported_ok and (active == expected if row["spec"]["kind"] == "overhead" else not active)))
    paths_valid = len(checks) == len(specs)*cfg["trials"] and all(row["valid"] for row in checks)
    cost_report(output, manifest, summaries)
    save(output / "overhead-status.json", dict(missing=missing, path_checks=checks,
                                                complete=not missing and all(s["complete"] for s in summaries) and paths_valid,
                                                path_controls_valid=paths_valid,
                                                note="An inactive on path may still pay lookup/analysis/preparation costs."))
    if not all(s["complete"] for s in summaries) or not paths_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    cli(main)
