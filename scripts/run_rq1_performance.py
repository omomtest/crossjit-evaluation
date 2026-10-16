#!/usr/bin/env python3
"""RQ1: fixed-baseline end-to-end NPBench performance, independent scale and cold runs."""
from common import cli, configuration, freeze, parser, read_json, workload_specs
from execution import run_matrix
from reporting import summarize_matrix
from optimization_coverage import verify_coverage


def main():
    p = parser(__doc__)
    p.add_argument("--parameter-grid", help="JSON array of {workload, parameters}; separate result identity")
    p.add_argument("--cold", action="store_true", help="Separate short-lived processes, warmup=0, repeat=1")
    args = p.parse_args()
    cfg = configuration(args, require_warmup=not args.cold)
    specs = workload_specs(args, cfg)
    if args.parameter_grid:
        grid = read_json(args.parameter_grid)
        allowed = {s["name"] for s in specs}
        specs = [dict(kind="npbench", name=row["workload"], preset=cfg["preset"], overrides=row["parameters"])
                 for row in grid if row["workload"] in allowed]
        if not specs:
            raise ValueError("empty parameter grid")
    if args.cold:
        cfg.update(warmup=0, repeat=1, warmup_protocol=None)
    output, manifest = freeze(args.output, cfg, "rq1-cold" if args.cold else "rq1", specs)
    run_matrix(output, manifest)
    summary = summarize_matrix(output)
    if not summary["complete"]:
        raise SystemExit(2)
    verify_coverage(output, manifest, summary)


if __name__ == "__main__":
    cli(main)
