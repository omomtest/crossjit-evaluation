#!/usr/bin/env python3
"""Collect the current paper RQ2–RQ4, including the RQ2/RQ3 ADI fragment."""
from pathlib import Path
import shlex
import subprocess
import sys

from common import ROOT, canonical_hash, cli, configuration, parser, read_json, save
from redesign_common import same_binary


def stages(args, cfg):
    output = Path(args.output).absolute()
    # Materialize a common config: every stage uses the integrated OFF binary
    # as reference, including the historical callback/overhead utilities.
    config_path = output / "configuration.json"
    options = ["--config", str(config_path)]
    phases = []
    def add(name, script, extra=()):
        phases.append((name, [sys.executable, "-B", str(ROOT / "scripts" / script),
                             *options, "--output", str(output / name), *extra]))
    add("rq2-adi", "run_microbench_case_study.py")  # activation gate before expensive full suite
    subset = ["--workloads", *args.workloads] if args.workloads else []
    add("rq2", "run_rq2_coverage.py", subset)
    add("rq3", "run_rq3_composition.py", [*subset, *(["--policy-evidence", args.policy_evidence] if args.policy_evidence else [])])
    add("rq3-storage", "run_rq3_storage.py", ["--reach", str(output / "rq2/reach.json")])
    if not args.skip_secondary:
        if args.workloads:
            raise ValueError("RQ4 inactive-cost supplement needs the full suite; use --skip-secondary for subset tests")
        add("rq4-callback", "run_rq4_attribution.py")
        add("rq4-inactive", "run_rq4_overhead.py", ["--coverage", str(output / "rq2/coverage.json")])
    if not args.skip_secondary:
        phases.append(("rq4-guard", [sys.executable, "-B", str(ROOT / "scripts/run_rq4_guard.py"),
            "--scope", str(output / "rq3"), "--output", str(output / "rq4-guard")]))
    return phases


def main():
    p = parser(__doc__)
    p.set_defaults(config=ROOT / "config/experiment.json")
    p.add_argument("--skip-secondary", action="store_true", help="Omit RQ4 callback and inactive-cost supplements")
    p.add_argument("--policy-evidence")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = same_binary(configuration(args), require_single=not args.dry_run)
    phases = stages(args, cfg)
    if args.dry_run:
        for name, command in phases:
            print(name + ": " + shlex.join(command))
        return
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = output / "configuration.json"
    if cfg_path.exists() and read_json(cfg_path) != cfg:
        raise RuntimeError("protocol changed: choose a new output directory")
    save(cfg_path, cfg)
    status = dict(configuration=cfg, phases=[], complete=False, rq1_scheduled=False)
    save(output / "pipeline.json", status)
    for name, command in phases:
        print(name + ": " + shlex.join(command), flush=True)
        path = output / "logs" / f"{name}.log"
        path.parent.mkdir(exist_ok=True)
        with path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        status["phases"].append(dict(name=name, exitcode=code, command=command, log=str(path)))
        save(output / "pipeline.json", status)
        if code:
            raise SystemExit(code)
    status["complete"] = True
    save(output / "pipeline.json", status)


if __name__ == "__main__":
    cli(main)
