#!/usr/bin/env python3
"""RQ3: same-binary OFF/SINGLE/FULL region-scope ablation on all 53 workloads."""
from common import cli, configuration, freeze, parser, workload_specs
from redesign_common import SINGLE, admission_inventory, archive_policy, same_binary
from redesign_execution import run_matrix
from redesign_reporting import composition_report, reach_report


def main():
    p = parser(__doc__)
    p.add_argument("--policy-evidence", help="Frozen policy/evidence directory; never recalibrated per mode")
    args = p.parse_args()
    cfg = same_binary(configuration(args), require_single=True)
    output, manifest = freeze(args.output, cfg, "rq3-composition", workload_specs(args, cfg), [SINGLE],
        extra=dict(scope="OFF/SINGLE/FULL share the integrated binary and frozen policy", off_is_reference=True))
    archive_policy(output, args.policy_evidence, manifest)
    run_matrix(output, manifest, modes=("off", "single", "on"))
    reach = reach_report(output, ("off", "single", "on"))
    admission_inventory(output, reach)
    report = composition_report(output, reach)
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    cli(main)
