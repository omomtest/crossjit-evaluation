#!/usr/bin/env python3
"""RQ2: coverage and frequency of actual replacements, traversal fusion and storage effects."""
from common import cli, configuration, freeze, parser, workload_specs
from redesign_execution import RunProgress, run_cell
from reporting import coverage_report
from optimization_coverage import verify_coverage
from redesign_common import same_binary
from redesign_reporting import reach_report


def main():
    p = parser(__doc__)
    p.add_argument("--cost-universe")
    args = p.parse_args()
    cfg = same_binary(configuration(args))
    output, manifest = freeze(args.output, cfg, "rq2", workload_specs(args, cfg))
    progress = RunProgress(manifest["experiment"], len(manifest["specs"]) * (1 + cfg["trials"]))
    for spec in manifest["specs"]:
        reference = run_cell(output, manifest, spec, "baseline", "reference", reference=True, progress=progress)
        for trial in range(cfg["trials"]):
            if reference is None:
                progress.skip(spec, "on", trial, diagnostic=True)
            else:
                run_cell(output, manifest, spec, "on", trial, diagnostic=True, progress=progress)
    report = coverage_report(output, args.cost_universe)
    reach = reach_report(output)
    if not reach["complete"] or report["complete_cells"] != report["expected_cells"]:
        raise SystemExit(2)
    verify_coverage(output, manifest, report)


if __name__ == "__main__":
    cli(main)
